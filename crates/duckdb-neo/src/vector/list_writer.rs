use std::{borrow::Borrow, ptr::NonNull};

use super::{
    Elements, NativeElement, PlainNative, ReadableNativeElement, RowLayout, RowWriter, Unknown, Vector, VectorReader,
    VectorWriter, WithWriter, WritableNativeElement, out_of_bounds,
};
use crate::{Result, error::Error, types::List, types::ListIndex};

/// Writes `LIST` rows into a writable flat vector whose element child is already sized.
///
/// Size the element child with [`Vector::set_child_size`] first: the writer fills it from the first
/// element, through the element type's own [`RowWriter`], so lists of nested types compose. [`Self::set_elements`] writes a whole row; columnar writers fill [`Self::elements`] and
/// point rows at them with [`Self::set_entry`]. Both move [`Self::child_offset`] past the row's elements.
///
/// Every writer starts at the first element, so write all rows through one writer: a second writer
/// overwrites the elements of the rows the first one wrote.
///
/// ```
/// # use duckdb_neo::{DuckDBType, Result, data_chunk::DataChunk, environment::{Environment, StorageLocation}, types::List};
/// # fn main() -> Result<()> {
/// # let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
/// let mut chunk = DataChunk::create(&conn, &[List::<i32>::logical_type(&conn)?], true)?;
/// let mut vector = chunk.get_vector_at_mut::<List<i32>>(0)?;
/// vector.set_size(2)?;
/// vector.set_child_size(2)?;
/// let mut writer = vector.list_writer()?;
/// writer.set_elements(0, [Some(&1), None])?;
/// writer.set_null(1)?;
/// assert!(writer.set_elements(1, [Some(&3)]).is_err());
/// assert_eq!(vector.get_value_slow(0)?.get::<List<i32>>()?, Some(List(vec![Some(1), None])));
/// # Ok(())
/// # }
/// ```
///
/// Columnar writers fill the elements directly and then point rows at them; entries are checked
/// against the child's size:
///
/// ```
/// # use duckdb_neo::{DuckDBType, Result, data_chunk::DataChunk, environment::{Environment, StorageLocation}, types::{List, ListIndex}};
/// # fn main() -> Result<()> {
/// # let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
/// # let mut chunk = DataChunk::create(&conn, &[List::<i32>::logical_type(&conn)?], true)?;
/// # let mut vector = chunk.get_vector_at_mut::<List<i32>>(0)?;
/// vector.set_size(2)?;
/// vector.set_child_size(3)?;
/// let mut writer = vector.list_writer()?;
/// let start = writer.child_offset();
/// for (index, value) in [7, 8, 9].iter().enumerate() {
///     writer.elements().set(start + index, value)?;
/// }
/// writer.set_entry(0, ListIndex::new(start as u64, 2))?;
/// writer.set_entry(1, ListIndex::new(start as u64 + 2, 1))?;
/// assert!(writer.set_entry(1, ListIndex::new(2, 2)).is_err());
/// assert_eq!(writer.child_offset(), 3);
/// # Ok(())
/// # }
/// ```
pub struct ListWriter<'w, T: WithWriter> {
    parent: VectorWriter<'w, List<T>>,
    /// The element child, whose size bounds every entry; the element writer could be swapped out.
    child: NonNull<Vector<'w, Unknown>>,
    elements: T::Writer<'w>,
    /// The first element no row written by this writer uses.
    offset: usize,
}

impl<'w, T: WithWriter + 'w> ListWriter<'w, T> {
    /// # Safety
    /// As for [`WithWriter::row_writer`].
    pub(super) unsafe fn from_raw(mut vector: NonNull<Vector<'w, List<T>>>) -> Result<Self> {
        // SAFETY: the caller guarantees an exclusive borrow of a validated, writable vector.
        let borrowed = unsafe { vector.as_mut() };
        let child = borrowed
            .children
            .as_mut()
            .expect("validated LIST vector has a child")
            .cached_iter_mut()?
            .next()
            .expect("validated LIST vector has a child");
        let child = NonNull::from(child).cast::<Vector<'w, Unknown>>();
        // SAFETY: the writer keeps `vector` exclusively borrowed and only touches its child through
        // `elements`, which `List<T>::validate` checked as `T`; resizing the parent keeps the child wrapper.
        let (parent, elements) = unsafe { (VectorWriter::from_raw(vector)?, T::row_writer(child.cast())?) };
        Ok(Self {
            parent,
            child,
            elements,
            offset: 0,
        })
    }

    #[inline]
    fn child_len(&self) -> usize {
        // SAFETY: the child wrapper lives as long as the borrowed vector.
        unsafe { self.child.as_ref() }.len()
    }

    /// Return the number of rows.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Return whether the vector has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the first element no row written by this writer uses, where [`Self::set_elements`] writes next.
    pub fn child_offset(&self) -> usize {
        self.offset
    }

    /// Borrow the element writer, for columnar writes; point rows at the elements with [`Self::set_entry`].
    pub fn elements(&mut self) -> &mut T::Writer<'w> {
        &mut self.elements
    }

    /// Point a row at elements of the child, checked against the child's size.
    #[inline]
    pub fn set_entry(&mut self, row: usize, entry: ListIndex) -> Result<()> {
        if row >= self.len() {
            return Err(out_of_bounds(row, self.len()));
        }
        let child_len = self.child_len();
        let Some(end) = entry.end().filter(|end| *end <= child_len as u64) else {
            return Err(entry_past_child(entry, child_len));
        };
        self.offset = self.offset.max(end as usize);
        // SAFETY: the row is in bounds, and the entry covers initialized child elements.
        unsafe { self.write_entry(row, entry) };
        Ok(())
    }

    /// Make a row SQL NULL.
    pub fn set_null(&mut self, row: usize) -> Result<()> {
        if row >= self.len() {
            return Err(out_of_bounds(row, self.len()));
        }
        // SAFETY: the row is in bounds.
        unsafe { self.parent.mark_null_unchecked(row) }
    }

    /// Write a whole row from its elements, at [`Self::child_offset`].
    ///
    /// Fails, leaving the row unchanged, if the elements do not fit the child.
    #[inline]
    pub fn set_elements<'v, I>(&mut self, row: usize, values: I) -> Result<()>
    where
        I: IntoIterator<Item = Option<&'v <T::Writer<'w> as RowWriter>::Value>>,
        <T::Writer<'w> as RowWriter>::Value: 'v,
    {
        if row >= self.len() {
            return Err(out_of_bounds(row, self.len()));
        }
        let (start, capacity) = (self.offset, self.child_len());
        let mut end = start;
        for value in values {
            if end >= capacity {
                return Err(Error::invalid_input(format!(
                    "LIST elements do not fit the {capacity} child elements; size the child first"
                )));
            }
            self.elements.set_option(end, value)?;
            end += 1;
        }
        self.offset = end;
        // SAFETY: the row is in bounds, and the entry covers the elements just written.
        unsafe { self.write_entry(row, ListIndex::new(start as u64, (end - start) as u64)) };
        Ok(())
    }

    #[inline(always)]
    unsafe fn write_entry(&mut self, row: usize, entry: ListIndex) {
        // SAFETY: the caller guarantees the row's bounds and that the entry covers initialized elements.
        unsafe {
            let _ = self.parent.write_internal(row, Some(entry));
            self.parent.mark_valid_unchecked(row);
        }
    }
}

#[cold]
fn entry_past_child(entry: ListIndex, child_len: usize) -> Error {
    Error::invalid_input(format!(
        "LIST entry {}..+{} is past the {child_len} child elements",
        entry.offset, entry.length
    ))
}

impl<'w, T> RowWriter for ListWriter<'w, T>
where
    T: WithWriter + Borrow<<T::Writer<'w> as RowWriter>::Value> + 'w,
{
    type Value = List<T>;

    fn len(&self) -> usize {
        ListWriter::len(self)
    }

    #[inline]
    fn set(&mut self, row: usize, value: &List<T>) -> Result<()> {
        self.set_elements(row, value.0.iter().map(|value| value.as_ref().map(Borrow::borrow)))
    }

    fn set_null(&mut self, row: usize) -> Result<()> {
        ListWriter::set_null(self, row)
    }

    unsafe fn resize_rows(&mut self, len: usize) -> Result<()> {
        self.parent.resize(len)?;
        self.elements.refresh()
    }

    fn refresh(&mut self) -> Result<()> {
        self.parent.refresh_all()?;
        self.elements.refresh()
    }
}

// SAFETY: rows only reference elements below the child's size, checked against the child itself.
unsafe impl<T> WithWriter for List<T>
where
    T: WithWriter + for<'w> Borrow<<T::Writer<'w> as RowWriter>::Value>,
{
    type Writer<'w>
        = ListWriter<'w, T>
    where
        Self: 'w;

    unsafe fn row_writer<'w>(vector: NonNull<Vector<'w, Self>>) -> Result<Self::Writer<'w>> {
        // SAFETY: the caller upholds the borrow contract.
        unsafe { ListWriter::from_raw(vector) }
    }
}

impl<T: WithWriter> std::fmt::Debug for ListWriter<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ListWriter")
            .field("len", &self.len())
            .field("child_offset", &self.offset)
            .field("child_len", &self.child_len())
            .finish()
    }
}

impl<T: WithWriter> Vector<'_, List<T>> {
    /// Prepare a [`ListWriter`] that fills the sized element child from its first element.
    pub fn list_writer(&mut self) -> Result<ListWriter<'_, T>> {
        // SAFETY: the exclusive borrow covers the writer, and validation checked the type.
        unsafe { ListWriter::from_raw(NonNull::from(self).cast()) }
    }
}

/// Writes `LIST` rows whose elements it appends to the element child, borrowed by [`Vector::list_appender`].
///
/// The child grows as rows are written, without initializing elements first, and ends with exactly the
/// elements of the rows written, after the ones it held before:
///
/// ```
/// # use duckdb_neo::{DuckDBType, Result, data_chunk::DataChunk, environment::{Environment, StorageLocation}, types::List};
/// # fn main() -> Result<()> {
/// # let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
/// let mut chunk = DataChunk::create(&conn, &[List::<String>::logical_type(&conn)?], true)?;
/// let mut vector = chunk.get_vector_at_mut::<List<String>>(0)?;
/// vector.set_size(3)?;
/// vector.list_appender(|lists| {
///     lists.set_elements(0, "a,b".split(',').map(Some))?;
///     lists.set_null(1)?;
///     lists.set_elements(2, [Some("c"), None])
/// })?;
/// assert_eq!(vector.child_len()?, 4);
/// assert_eq!(vector.get_value_slow(2)?.get::<List<String>>()?, Some(List(vec![Some("c".into()), None])));
/// # Ok(())
/// # }
/// ```
///
/// Plain elements are copied from a slice or computed in one loop per row:
///
/// ```
/// # use duckdb_neo::{DuckDBType, Result, data_chunk::DataChunk, environment::{Environment, StorageLocation}, types::List};
/// # fn main() -> Result<()> {
/// # let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
/// # let mut chunk = DataChunk::create(&conn, &[List::<i32>::logical_type(&conn)?], true)?;
/// # let mut vector = chunk.get_vector_at_mut::<List<i32>>(0)?;
/// vector.set_size(2)?;
/// vector.list_appender(|lists| {
///     lists.set_slice(0, &[7, 8])?;
///     lists.set_with(1, 3, |index| index as i32)
/// })?;
/// assert_eq!(vector.get_value_slow(1)?.get::<List<i32>>()?, Some(List(vec![Some(0), Some(1), Some(2)])));
/// # Ok(())
/// # }
/// ```
pub struct ListAppender<'w, T: NativeElement> {
    parent: VectorWriter<'w, List<T>>,
    elements: VectorWriter<'w, T>,
    /// The child elements in use: the earlier ones and those of every row written.
    used: usize,
    /// The child's size, of which elements from `used` are not initialized.
    reserved: usize,
    /// Bytes per element; zero bytes are a valid value.
    element_size: usize,
}

impl<T: NativeElement> Vector<'_, List<T>> {
    /// Write rows through a [`ListAppender`] in `f`, appending their elements after the child's.
    ///
    /// Size the rows first. The child ends with the elements of the rows written, also when `f` fails or
    /// panics. Fails for element types that [`Vector::append`] rejects, e.g. nested ones.
    pub fn list_appender<R>(&mut self, f: impl FnOnce(&mut ListAppender<'_, T>) -> Result<R>) -> Result<R> {
        // SAFETY: the exclusive borrow covers both writers, and `List<T>::validate` checked the child as `T`.
        let mut appender = unsafe { ListAppender::from_raw(NonNull::from(self).cast())? };
        let value = f(&mut appender.0)?;
        appender.0.finish()?;
        Ok(value)
    }

    /// Return the number of elements in the child; fails if the vector has none, e.g. after [`Vector::make_sequence`].
    pub fn child_len(&self) -> Result<usize> {
        let children = self
            .children()
            .ok_or_else(|| Error::invalid_input("the LIST vector has no element child"))?;
        Ok(children.get_untyped_vector_at(0)?.len())
    }
}

impl<'w, T: NativeElement> ListAppender<'w, T> {
    /// # Safety
    /// `vector` must be validated, exclusively borrowed for `'w` and outlive the appender.
    unsafe fn from_raw(mut vector: NonNull<Vector<'w, List<T>>>) -> Result<FinishList<'w, T>> {
        // SAFETY: the caller guarantees an exclusive borrow of a validated vector.
        let borrowed = unsafe { vector.as_mut() };
        if !borrowed.is_writable() {
            return Err(super::not_writable());
        }
        let child = borrowed
            .children
            .as_mut()
            .expect("validated LIST vector has a child")
            .cached_iter_mut()?
            .next()
            .expect("validated LIST vector has a child");
        let element_size = child
            .zeroed_row_size()?
            .ok_or_else(|| Error::invalid_input(format!("cannot append {} elements in place", child.logical_type())))?;
        let child = NonNull::from(child).cast::<Vector<'w, T>>();
        // SAFETY: the appender keeps `vector` exclusively borrowed and only touches its child through
        // `elements`; the child was validated as `T`, and resizing it keeps the parent's buffers.
        let (parent, elements) = unsafe { (VectorWriter::from_raw(vector)?, VectorWriter::from_raw(child)?) };
        let used = elements.len();
        Ok(FinishList(Self {
            parent,
            elements,
            used,
            reserved: used,
            element_size,
        }))
    }

    /// Return the number of rows.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Return whether the vector has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of child elements in use, where the next row's elements start.
    pub fn child_len(&self) -> usize {
        self.used
    }

    /// Make a row SQL NULL.
    #[inline]
    pub fn set_null(&mut self, row: usize) -> Result<()> {
        self.check_row(row)?;
        // SAFETY: the row is in bounds.
        unsafe { self.parent.mark_null_unchecked(row) }
    }

    #[inline(always)]
    fn check_row(&self, row: usize) -> Result<()> {
        if row >= self.len() {
            return Err(out_of_bounds(row, self.len()));
        }
        Ok(())
    }

    /// Make room for `additional` elements after those in use, so rows of known total length grow the child once.
    ///
    /// The child still ends with only the elements in use.
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        self.claim(additional).map(|_| ())
    }

    #[inline(always)]
    fn claim(&mut self, count: usize) -> Result<usize> {
        let start = self.used;
        let end = start
            .checked_add(count)
            .ok_or_else(|| Error::invalid_input("LIST element count overflows"))?;
        if end > self.reserved {
            self.grow(end)?;
        }
        Ok(start)
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self, end: usize) -> Result<()> {
        let reserved = end.max(self.reserved.saturating_mul(2)).max(64);
        // SAFETY: elements from `used` are written before a row's entry covers them, and `finish` cuts the
        // child to `used` before anything can read the vector.
        let result = unsafe { self.elements.vector_mut().set_size_uninit(reserved) };
        // A failed resize can still replace the buffers, so refresh either way; `reserved` moves only once the
        // writer's pointers are current, which `finish` relies on.
        if let Err(error) = self.elements.refresh_all() {
            self.reserved = self.used;
            // SAFETY: shrinking exposes no new elements.
            if unsafe { self.elements.vector_mut().set_size_uninit(self.used) }.is_err() {
                // Elements past `used` would be exposed uninitialized, behind stale pointers.
                std::process::abort();
            }
            return Err(error);
        }
        let old = self.reserved;
        self.reserved = self.elements.len();
        // SAFETY: the new elements are in bounds, and are written before a row's entry covers them.
        unsafe { self.elements.mark_valid_range(old.min(self.reserved), self.reserved) };
        result
    }

    /// Point `row` at the `count` elements from `start`, which were just written, and take them into use.
    ///
    /// # Safety
    /// `row` must be in bounds, and elements `start..start + count` initialized, with `start` the old `used`.
    #[inline(always)]
    unsafe fn commit(&mut self, row: usize, start: usize, count: usize) {
        self.used = start + count;
        // SAFETY: the caller guarantees the row's bounds and that the entry covers initialized elements.
        unsafe {
            let _ = self
                .parent
                .write_internal(row, Some(ListIndex::new(start as u64, count as u64)));
            self.parent.mark_valid_unchecked(row);
        }
    }

    /// Cut the child to the elements in use, or zero the others if that fails.
    fn finish(&mut self) -> Result<()> {
        let (used, reserved) = (self.used, self.reserved);
        if used == reserved {
            return Ok(());
        }
        self.reserved = used;
        // SAFETY: shrinking exposes no new elements.
        let result = unsafe { self.elements.vector_mut().set_size_uninit(used) };
        if result.is_err() {
            // SAFETY: elements `used..reserved` are in bounds of the buffer the writer holds, current since the last
            // `grow`, which DuckDB only replaces when it grows; zero bytes are a valid value.
            unsafe {
                self.elements
                    .as_mut_ptr()
                    .cast::<u8>()
                    .add(used * self.element_size)
                    .write_bytes(0, (reserved - used) * self.element_size);
            }
        }
        result
    }
}

impl<T: WritableNativeElement> ListAppender<'_, T> {
    /// Write a row from its elements; `None` elements are SQL NULL.
    ///
    /// Fails, leaving the row and the child's elements in use unchanged, if an element is invalid.
    #[inline]
    pub fn set_elements<V, I>(&mut self, row: usize, values: I) -> Result<()>
    where
        V: Borrow<T::Write>,
        I: IntoIterator<Item = Option<V>>,
    {
        self.check_row(row)?;
        let values = values.into_iter();
        let start = self.claim(values.size_hint().0)?;
        let mut written = Remark {
            appender: self,
            start,
            end: start,
        };
        for value in values {
            written.appender.set_element(written.end, value)?;
            written.end += 1;
        }
        let end = written.end;
        std::mem::forget(written);
        // SAFETY: the row is in bounds, and the elements were just written.
        unsafe { self.commit(row, start, end - start) };
        Ok(())
    }

    #[inline(always)]
    fn set_element<V: Borrow<T::Write>>(&mut self, index: usize, value: Option<V>) -> Result<()> {
        if index == self.reserved {
            self.grow(index + 1)?;
        }
        // SAFETY: the element is reserved, flat and writable; on `Ok`, `write` has initialized the whole slot, as
        // `WritableNativeElement` requires, and zero bytes are a valid value.
        unsafe {
            match value {
                Some(value) => T::write(&mut self.elements, index, value.borrow()),
                None => {
                    self.elements
                        .as_mut_ptr()
                        .cast::<u8>()
                        .add(index * self.element_size)
                        .write_bytes(0, self.element_size);
                    self.elements.mark_null_unchecked(index)
                }
            }
        }
    }
}

impl<T: PlainNative> ListAppender<'_, T> {
    /// Write a row from a slice of non-NULL elements.
    #[inline]
    pub fn set_slice(&mut self, row: usize, values: &[T]) -> Result<()> {
        self.check_row(row)?;
        let start = self.claim(values.len())?;
        // SAFETY: the elements are reserved and marked valid, and `T` is the slot type.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.elements.as_mut_ptr().cast::<T>().add(start),
                values.len(),
            );
            self.commit(row, start, values.len());
        }
        Ok(())
    }

    /// Write a row of `count` non-NULL elements, the `i`-th from `f(i)`, in one loop.
    #[inline]
    pub fn set_with(&mut self, row: usize, count: usize, mut f: impl FnMut(usize) -> T) -> Result<()> {
        self.check_row(row)?;
        let start = self.claim(count)?;
        let data = self.elements.as_mut_ptr().cast::<T>();
        for index in 0..count {
            // SAFETY: the elements are reserved and marked valid, and `T` is the slot type. If `f` panics,
            // `used` has not moved, so `finish` cuts these elements off.
            unsafe { data.add(start + index).write(f(index)) };
        }
        // SAFETY: the row is in bounds, and the elements were just written.
        unsafe { self.commit(row, start, count) };
        Ok(())
    }

    /// Write a row from the elements of a `LIST` or `ARRAY` row, `f(value)` for each; NULL elements stay NULL
    /// without a call. Storage is resolved once, so flat elements without NULLs run a vectorizable loop.
    ///
    /// ```
    /// # use duckdb_neo::{Parameters, Result, environment::{Environment, StorageLocation}, scalar::MicroScalar, types::List, vector::{RowReader, Vector}};
    /// # fn main() -> Result<()> {
    /// # let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
    /// MicroScalar::create("doubled", |_ctx, input: Vector<'_, List<i64>>, mut output: Vector<'_, List<i64>>| {
    ///     let lists = input.list_reader()?;
    ///     output.list_appender(|doubled| {
    ///         for (row, list) in lists.iter().enumerate() {
    ///             match list {
    ///                 Some(list) => doubled.map_elements(row, list, |value| 2 * value)?,
    ///                 None => doubled.set_null(row)?,
    ///             }
    ///         }
    ///         Ok(())
    ///     })
    /// })
    /// .register(&conn)?;
    /// let chunk = conn.query("SELECT doubled([1, NULL, 3])", Parameters::None)?.next_chunk()?.unwrap();
    /// let row = chunk.get_vector_at::<List<i64>>(0)?.get_value_slow(0)?.get::<List<i64>>()?;
    /// assert_eq!(row, Some(List(vec![Some(2), None, Some(6)])));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn map_elements<'a, I: ReadableNativeElement + 'a>(
        &mut self,
        row: usize,
        elements: Elements<'_, VectorReader<'a, I>>,
        mut f: impl FnMut(I::Ref<'a>) -> T,
    ) -> Result<()> {
        self.check_row(row)?;
        let (reader, rows) = elements.child_rows();
        let count = rows.len();
        let start = self.claim(count)?;
        let data = self.elements.as_mut_ptr().cast::<T>();
        let layout = reader.layout();
        if matches!(layout, RowLayout::Flat { validity } if validity.is_null()) {
            for (index, source) in rows.enumerate() {
                // SAFETY: `source` is a valid flat row of the reader, its own slot; the element is reserved and
                // marked valid, and `T` is the slot type. If `f` panics, `used` has not moved, so `finish` cuts
                // these elements off.
                unsafe { data.add(start + index).write(f(I::get(reader, source, source))) };
            }
        } else {
            let written = Remark {
                appender: self,
                start,
                end: start + count,
            };
            for (index, source) in rows.enumerate() {
                // SAFETY: `source` is a row of the reader, read only when valid at its physical slot; the element
                // is reserved, and zero bytes are a valid `T`, as `PlainNative` requires. If `f` panics or the
                // mask fails, `written` re-marks the elements valid and `finish` cuts them off.
                unsafe {
                    match layout.valid_physical(source) {
                        Some(physical) => data.add(start + index).write(f(I::get(reader, physical, source))),
                        None => {
                            data.add(start + index).write_bytes(0, 1);
                            written.appender.elements.mark_null_unchecked(start + index)?;
                        }
                    }
                }
            }
            std::mem::forget(written);
        }
        // SAFETY: the row is in bounds, and the elements were just written.
        unsafe { self.commit(row, start, count) };
        Ok(())
    }
}

impl<T: NativeElement> std::fmt::Debug for ListAppender<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ListAppender")
            .field("len", &self.len())
            .field("child_len", &self.used)
            .finish()
    }
}

/// Re-marks the elements of a row that failed or panicked valid, as later rows expect of unused elements.
struct Remark<'a, 'w, T: NativeElement> {
    appender: &'a mut ListAppender<'w, T>,
    start: usize,
    /// The element being written.
    end: usize,
}

impl<T: NativeElement> Drop for Remark<'_, '_, T> {
    fn drop(&mut self) {
        let end = (self.end + 1).min(self.appender.reserved);
        // SAFETY: the elements are in bounds and unused, so `finish` cuts them off unless a row overwrites them.
        unsafe { self.appender.elements.mark_valid_range(self.start, end) };
    }
}

/// Finishes the appender when its closure fails or panics.
struct FinishList<'w, T: NativeElement>(ListAppender<'w, T>);

impl<T: NativeElement> Drop for FinishList<'_, T> {
    fn drop(&mut self) {
        let _ = self.0.finish();
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        DuckDBType, Result,
        connection::Connection,
        data_chunk::DataChunk,
        environment::{Environment, StorageLocation},
        types::{List, ListIndex},
        vector::{RowReader, Vector},
    };

    fn connection() -> Result<Connection> {
        Environment::new()?.open(StorageLocation::InMemory)?.connect()
    }

    fn rows<T: crate::FromValue + crate::vector::NativeElement>(
        vector: &Vector<'_, List<T>>,
    ) -> Result<Vec<Option<List<T>>>> {
        (0..vector.len())
            .map(|row| vector.get_value_slow(row)?.get::<List<T>>())
            .collect()
    }

    fn child_len<T: crate::vector::NativeElement>(vector: &Vector<'_, List<T>>) -> Result<usize> {
        Ok(vector.children().unwrap().get_untyped_vector_at(0)?.len())
    }

    #[test]
    fn list_writer_fills_the_sized_child() -> Result<()> {
        let conn = connection()?;
        let mut chunk = DataChunk::create(&conn, &[List::<String>::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<List<String>>(0)?;
        vector.set_size(3)?;
        vector.set_child_size(5002)?;
        let long = "a string longer than the inline storage".to_owned();
        let many = (0..5000).map(|index| index.to_string()).collect::<Vec<_>>();
        {
            let mut writer = vector.list_writer()?;
            writer.set_elements(1, many.iter().map(|value| Some(value.as_str())))?;
            writer.set_null(2)?;
            writer.set_elements(0, [Some(long.as_str()), None])?;
            assert_eq!(writer.child_offset(), 5002);
            let error = writer.set_elements(2, [Some("no room")]).unwrap_err();
            assert!(error.message.contains("do not fit the 5002 child elements"), "{error}");
        }
        assert_eq!(child_len(&vector)?, 5002);
        let read = rows(&vector)?;
        assert_eq!(read[0], Some(List(vec![Some(long), None])));
        assert_eq!(read[1], Some(List(many.into_iter().map(Some).collect())));
        assert_eq!(read[2], None);
        Ok(())
    }

    #[test]
    fn list_writer_checks_entries_against_the_child() -> Result<()> {
        let conn = connection()?;
        let mut chunk = DataChunk::create(&conn, &[List::<i32>::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<List<i32>>(0)?;
        vector.set_size(2)?;
        vector.set_child_size(3)?;
        {
            let mut writer = vector.list_writer()?;
            assert_eq!(writer.elements().len(), 3);
            writer.elements().set(0, &7)?;
            writer.elements().set(1, &8)?;
            writer.set_entry(0, ListIndex::new(0, 2))?;
            // Unwritten elements below the child's size are initialized.
            writer.set_entry(1, ListIndex::new(2, 1))?;
            assert_eq!(writer.child_offset(), 3);
            for entry in [ListIndex::new(2, 2), ListIndex::new(4, 0), ListIndex::new(u64::MAX, 2)] {
                assert!(writer.set_entry(1, entry).is_err(), "{entry:?}");
            }
            assert!(writer.set_entry(2, ListIndex::new(0, 0)).is_err());
            assert!(writer.set_null(2).is_err());
            assert!(writer.set_elements(2, []).is_err());
        }
        assert_eq!(
            rows(&vector)?,
            [Some(List(vec![Some(7), Some(8)])), Some(List(vec![Some(0)]))]
        );
        Ok(())
    }

    #[test]
    fn set_entry_moves_the_offset_past_its_elements() -> Result<()> {
        let conn = connection()?;
        let mut chunk = DataChunk::create(&conn, &[List::<i32>::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<List<i32>>(0)?;
        vector.set_size(2)?;
        vector.set_child_size(3)?;
        {
            let mut writer = vector.list_writer()?;
            writer.elements().set(0, &7)?;
            writer.elements().set(1, &8)?;
            writer.set_entry(0, ListIndex::new(0, 2))?;
            writer.set_elements(1, [Some(&9)])?;
        }
        assert_eq!(
            rows(&vector)?,
            [Some(List(vec![Some(7), Some(8)])), Some(List(vec![Some(9)]))]
        );
        Ok(())
    }

    #[test]
    fn list_writer_writes_nested_lists() -> Result<()> {
        let conn = connection()?;
        let mut chunk = DataChunk::create(&conn, &[List::<List<i32>>::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<List<List<i32>>>(0)?;
        vector.set_size(2)?;
        vector.set_child_size(3)?;
        vector
            .children_mut()
            .unwrap()
            .get_vector_at_mut::<List<i32>>(0)?
            .set_child_size(3)?;
        let inner = [List(vec![Some(1), Some(2)]), List(vec![Some(3)])];
        {
            let mut writer = vector.list_writer()?;
            writer.set_elements(0, inner.iter().map(Some))?;
            writer.set_elements(1, [None])?;
        }
        assert_eq!(
            vector.get_value_slow(0)?.get::<List<List<i32>>>()?,
            Some(List(inner.into_iter().map(Some).collect()))
        );
        assert_eq!(
            vector.get_value_slow(1)?.get::<List<List<i32>>>()?,
            Some(List(vec![None]))
        );
        Ok(())
    }

    #[test]
    fn unwritten_elements_read_as_initialized() -> Result<()> {
        let conn = connection()?;
        let mut chunk = DataChunk::create(&conn, &[List::<String>::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<List<String>>(0)?;
        vector.set_size(1)?;
        vector.set_child_size(100)?;
        let mut writer = vector.list_writer()?;
        writer.elements().set(0, "written")?;
        writer.set_entry(0, ListIndex::new(0, 100))?;
        let reader = vector.list_reader()?;
        let row = reader.get(0)?.unwrap();
        assert_eq!(row.get(0), Some(Some("written")));
        assert!((1..100).all(|index| row.get(index) == Some(Some(""))));
        Ok(())
    }

    #[test]
    fn untyped_structs_and_unions_compose_into_lists() -> Result<()> {
        use crate::{
            logical_type::LogicalType,
            types::{Struct, Union},
            vector::RowReader,
        };
        let conn = connection()?;
        let structs = LogicalType::from_text(&conn, "STRUCT(a INTEGER)[]")?;
        let unions = LogicalType::from_text(&conn, "UNION(n INTEGER, s VARCHAR)[]")?;
        let mut chunk = DataChunk::create(&conn, &[structs, unions], true)?;
        let mut vector = chunk.get_vector_at_mut::<List<Struct>>(0)?;
        vector.set_size(1)?;
        vector.set_child_size(2)?;
        vector.list_writer()?.set_elements(0, [Some(&()), None])?;
        assert_eq!(vector.get_value_slow(0)?.dbg_string()?, "[{'a': 0}, NULL]");
        let reader = vector.list_reader()?;
        assert_eq!(reader.get(0)?.unwrap().iter().collect::<Vec<_>>(), [Some(()), None]);
        let mut unions = chunk.get_vector_at_mut::<List<Union>>(1)?;
        unions.set_size(1)?;
        let reader = unions.list_reader()?;
        assert_eq!(reader.get(0)?.unwrap().len(), 0);
        Ok(())
    }

    #[test]
    fn list_appender_grows_the_child_as_rows_are_written() -> Result<()> {
        let conn = connection()?;
        let mut chunk = DataChunk::create(&conn, &[List::<String>::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<List<String>>(0)?;
        vector.set_size(300)?;
        let long = "a string longer than the inline storage";
        vector.list_appender(|lists| {
            assert_eq!((lists.len(), lists.child_len()), (300, 0));
            for row in 0..300 {
                match row % 3 {
                    0 => lists.set_null(row)?,
                    1 => lists.set_elements(row, (0..row).map(|index| (index % 2 == 0).then_some(long)))?,
                    _ => lists.set_elements(row, std::iter::empty::<Option<&str>>())?,
                }
            }
            assert!(lists.set_null(300).is_err());
            assert!(lists.set_elements(300, [Some("x")]).is_err());
            Ok(())
        })?;
        let expected = (1..300).step_by(3).sum::<usize>();
        assert_eq!(vector.child_len()?, expected);
        let read = rows(&vector)?;
        for (row, list) in read.into_iter().enumerate() {
            match row % 3 {
                0 => assert_eq!(list, None),
                1 => assert_eq!(
                    list,
                    Some(
                        (0..row)
                            .map(|index| (index % 2 == 0).then(|| long.to_owned()))
                            .collect()
                    )
                ),
                _ => assert_eq!(list, Some(List(vec![]))),
            }
        }
        Ok(())
    }

    #[test]
    fn list_appender_appends_after_existing_elements() -> Result<()> {
        let conn = connection()?;
        let mut chunk = DataChunk::create(&conn, &[List::<i64>::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<List<i64>>(0)?;
        vector.set_size(4)?;
        vector.set_child_size(2)?;
        vector.list_writer()?.set_elements(0, [Some(&1), None])?;
        vector.list_appender(|lists| {
            assert_eq!(lists.child_len(), 2);
            lists.reserve(1000)?;
            lists.set_slice(1, &[2, 3])?;
            lists.set_with(2, 3, |index| 10 * index as i64)?;
            assert_eq!(lists.child_len(), 7);
            Ok(())
        })?;
        assert_eq!(vector.child_len()?, 7);
        vector.list_appender(|lists| lists.set_elements(3, [None, Some(4_i64)]))?;
        assert_eq!(
            rows(&vector)?,
            [
                Some(List(vec![Some(1), None])),
                Some(List(vec![Some(2), Some(3)])),
                Some(List(vec![Some(0), Some(10), Some(20)])),
                Some(List(vec![None, Some(4)])),
            ]
        );
        Ok(())
    }

    #[test]
    fn list_appender_keeps_only_finished_rows_after_a_panic_or_error() -> Result<()> {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let conn = connection()?;
        let mut chunk = DataChunk::create(&conn, &[List::<i64>::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<List<i64>>(0)?;
        vector.set_size(3)?;
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            vector.list_appender(|lists| {
                lists.set_slice(0, &[1, 2])?;
                lists.set_with(1, 100, |index| if index == 50 { panic!("in set_with") } else { 7 })
            })
        }));
        assert!(panicked.is_err());
        assert_eq!(vector.child_len()?, 2);
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            vector.list_appender(|lists| {
                let values = (0..100).map(|index| {
                    if index == 70 {
                        panic!("in the iterator")
                    } else {
                        Some(index)
                    }
                });
                lists.set_elements(1, values)
            })
        }));
        assert!(panicked.is_err());
        assert_eq!(vector.child_len()?, 2);
        let error = vector
            .list_appender(|lists| {
                lists.set_slice(1, &[3])?;
                Err::<(), _>(crate::Error::invalid_input("stop"))
            })
            .unwrap_err();
        assert!(error.message.contains("stop"), "{error}");
        assert_eq!(vector.child_len()?, 3);
        assert_eq!(
            rows(&vector)?,
            [
                Some(List(vec![Some(1), Some(2)])),
                Some(List(vec![Some(3)])),
                Some(List(vec![]))
            ]
        );
        Ok(())
    }

    #[test]
    fn list_appender_leaves_rows_unchanged_when_an_element_fails() -> Result<()> {
        use crate::{logical_type::LogicalType, types::EnumIndex};
        let conn = connection()?;
        let types = [LogicalType::from_text(&conn, "ENUM('a', 'b')[]")?];
        let mut chunk = DataChunk::create(&conn, &types, true)?;
        let mut vector = chunk.get_vector_at_mut::<List<EnumIndex<u8>>>(0)?;
        vector.set_size(2)?;
        vector.list_appender(|lists| {
            let error = lists
                .set_elements(0, [None, Some(EnumIndex(1)), Some(EnumIndex(2))])
                .unwrap_err();
            assert!(error.message.contains('2'), "{error}");
            assert_eq!(lists.child_len(), 0);
            // The failed row's NULL element does not leak into the next row.
            lists.set_elements(1, [Some(EnumIndex(0)), Some(EnumIndex(1))])
        })?;
        assert_eq!(vector.get_value_slow(0)?.dbg_string()?, "[]");
        assert_eq!(vector.get_value_slow(1)?.dbg_string()?, "[a, b]");
        Ok(())
    }

    #[test]
    fn list_appender_rejects_nested_elements() -> Result<()> {
        use crate::{logical_type::LogicalType, types::Struct};
        let conn = connection()?;
        let types = [
            List::<List<i32>>::logical_type(&conn)?,
            LogicalType::from_text(&conn, "STRUCT(a INTEGER)[]")?,
        ];
        let mut chunk = DataChunk::create(&conn, &types, true)?;
        let mut lists = chunk.get_vector_at_mut::<List<List<i32>>>(0)?;
        let error = lists.list_appender(|_| Ok(())).unwrap_err();
        assert!(error.message.contains("cannot append"), "{error}");
        let mut structs = chunk.get_vector_at_mut::<List<Struct>>(1)?;
        assert!(structs.list_appender(|_| Ok(())).is_err());
        Ok(())
    }

    #[test]
    fn list_appender_remarks_elements_of_a_row_that_panicked() -> Result<()> {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let conn = connection()?;
        let mut chunk = DataChunk::create(&conn, &[List::<i64>::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<List<i64>>(0)?;
        vector.set_size(2)?;
        vector.list_appender(|lists| {
            let values = [None::<i64>, None]
                .into_iter()
                .chain(std::iter::once_with(|| panic!("in the iterator")));
            assert!(catch_unwind(AssertUnwindSafe(|| lists.set_elements(0, values))).is_err());
            lists.set_slice(1, &[1, 2])
        })?;
        assert_eq!(rows(&vector)?, [Some(List(vec![])), Some(List(vec![Some(1), Some(2)]))]);
        Ok(())
    }

    #[test]
    fn list_appender_rewrites_rows() -> Result<()> {
        let conn = connection()?;
        let mut chunk = DataChunk::create(&conn, &[List::<String>::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<List<String>>(0)?;
        vector.set_size(3)?;
        vector.list_appender(|lists| {
            lists.set_elements(0, [Some("a"), None])?;
            lists.set_null(0)?;
            lists.set_elements(1, [Some("b")])?;
            lists.set_elements(1, [None, Some("c")])?;
            // An iterator whose lower bound over-reports only reserves more.
            struct Overstated(std::vec::IntoIter<Option<&'static str>>);
            impl Iterator for Overstated {
                type Item = Option<&'static str>;
                fn next(&mut self) -> Option<Self::Item> {
                    self.0.next()
                }
                fn size_hint(&self) -> (usize, Option<usize>) {
                    (1000, None)
                }
            }
            lists.set_elements(2, Overstated(vec![Some("d")].into_iter()))
        })?;
        assert_eq!(vector.child_len()?, 6);
        assert_eq!(
            rows(&vector)?,
            [
                None,
                Some(List(vec![None, Some("c".to_owned())])),
                Some(List(vec![Some("d".to_owned())])),
            ]
        );
        Ok(())
    }

    #[test]
    fn list_appender_keeps_string_rows_finished_before_a_panic() -> Result<()> {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let conn = connection()?;
        let mut chunk = DataChunk::create(&conn, &[List::<String>::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<List<String>>(0)?;
        vector.set_size(2)?;
        let long = "a string longer than the inline storage";
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            vector.list_appender(|lists| {
                lists.set_elements(0, [Some(long), None])?;
                let values = (0..100).map(|index| {
                    if index == 70 {
                        panic!("in the iterator")
                    } else {
                        Some(long)
                    }
                });
                lists.set_elements(1, values)
            })
        }));
        assert!(panicked.is_err());
        assert_eq!(vector.child_len()?, 2);
        assert_eq!(
            rows(&vector)?,
            [Some(List(vec![Some(long.to_owned()), None])), Some(List(vec![]))]
        );
        Ok(())
    }

    #[test]
    fn child_lists_use_the_appender() -> Result<()> {
        use crate::{logical_type::LogicalType, types::Struct};
        let conn = connection()?;
        let types = [
            LogicalType::from_text(&conn, "STRUCT(l INTEGER[])")?,
            List::<List<i32>>::logical_type(&conn)?,
        ];
        let mut chunk = DataChunk::create(&conn, &types, true)?;
        let mut structs = chunk.get_vector_at_mut::<Struct>(0)?;
        structs.set_size(2)?;
        structs
            .children_mut()
            .unwrap()
            .get_vector_at_mut::<List<i32>>(0)?
            .list_appender(|lists| {
                lists.set_slice(0, &[1, 2])?;
                lists.set_null(1)
            })?;
        assert_eq!(structs.get_value_slow(0)?.dbg_string()?, "{'l': [1, 2]}");
        assert_eq!(structs.get_value_slow(1)?.dbg_string()?, "{'l': NULL}");
        let mut nested = chunk.get_vector_at_mut::<List<List<i32>>>(1)?;
        nested.set_size(1)?;
        nested.set_child_size(2)?;
        nested
            .children_mut()
            .unwrap()
            .get_vector_at_mut::<List<i32>>(0)?
            .list_appender(|inner| {
                inner.set_with(0, 3, |index| index as i32)?;
                inner.set_elements(1, [None, Some(7)])
            })?;
        nested.list_writer()?.set_entry(0, ListIndex::new(0, 2))?;
        assert_eq!(nested.get_value_slow(0)?.dbg_string()?, "[[0, 1, 2], [NULL, 7]]");
        Ok(())
    }

    #[test]
    fn child_len_fails_without_a_child() -> Result<()> {
        let conn = connection()?;
        let mut chunk = DataChunk::create(&conn, &[List::<i32>::logical_type(&conn)?], true)?;
        let sequence = chunk.get_vector_at_mut::<List<i32>>(0)?.make_sequence(0, 1, 3)?;
        assert!(sequence.child_len().is_err());
        Ok(())
    }

    fn copy_lists(conn: &Connection, sql: &str) -> Result<usize> {
        let mut result = conn.query(sql, crate::Parameters::None)?;
        let mut rows = 0;
        while let Some(chunk) = result.next_chunk()? {
            let input = chunk.get_vector_at::<List<i64>>(0)?;
            let lists = input.list_reader()?;
            let mut output = DataChunk::create(conn, &[List::<i64>::logical_type(conn)?], true)?;
            let mut doubled = output.get_vector_at_mut::<List<i64>>(0)?;
            doubled.set_size(lists.len())?;
            doubled.list_appender(|appender| {
                for (row, list) in lists.iter().enumerate() {
                    match list {
                        Some(list) => appender.map_elements(row, list, |value| value.wrapping_mul(2))?,
                        None => appender.set_null(row)?,
                    }
                }
                Ok(())
            })?;
            for row in 0..lists.len() {
                let expected = input.get_value_slow(row)?.get::<List<i64>>()?.map(|list| {
                    list.0
                        .into_iter()
                        .map(|value| value.map(|value| value.wrapping_mul(2)))
                        .collect::<List<_>>()
                });
                assert_eq!(doubled.get_value_slow(row)?.get::<List<i64>>()?, expected, "row {row}");
                rows += 1;
            }
        }
        Ok(rows)
    }

    #[test]
    fn map_elements_reads_every_storage_layout() -> Result<()> {
        let conn = connection()?;
        let layouts = "SELECT * FROM test_vector_types(NULL::BIGINT[], all_flat := false)";
        assert!(copy_lists(&conn, layouts)? > 0);
        let sliced = "SELECT l FROM (SELECT i, CASE WHEN i % 7 = 0 THEN NULL ELSE list_transform(range(i % 5), \
                      lambda x: CASE WHEN (x + i) % 3 = 0 THEN NULL ELSE x * i END) END AS l FROM range(3000) r(i)) \
                      WHERE i % 4 <> 1";
        assert!(copy_lists(&conn, sliced)? > 2000);
        let without_nulls = "SELECT list_transform(range(i % 9), lambda x: x + i) FROM range(3000) r(i)";
        assert_eq!(copy_lists(&conn, without_nulls)?, 3000);
        assert_eq!(copy_lists(&conn, "SELECT [1] WHERE false")?, 0);
        Ok(())
    }

    #[test]
    fn map_elements_maps_array_rows() -> Result<()> {
        use crate::types::Array;
        let conn = connection()?;
        let chunk = conn
            .query(
                "SELECT * FROM (VALUES ([1, NULL, 3]::DOUBLE[3]), (NULL), ([4, 5, 6]::DOUBLE[3])) t(a)",
                crate::Parameters::None,
            )?
            .next_chunk()?
            .unwrap();
        let input = chunk.get_vector_at::<Array<f64, 3>>(0)?;
        let arrays = input.array_reader()?;
        let mut output = DataChunk::create(&conn, &[List::<f64>::logical_type(&conn)?], true)?;
        let mut lists = output.get_vector_at_mut::<List<f64>>(0)?;
        lists.set_size(3)?;
        lists.list_appender(|lists| {
            for (row, array) in arrays.iter().enumerate() {
                match array {
                    Some(array) => lists.map_elements(row, array, |value| value / 2.0)?,
                    None => lists.set_null(row)?,
                }
            }
            Ok(())
        })?;
        assert_eq!(lists.child_len()?, 6);
        assert_eq!(
            rows(&lists)?,
            [
                Some(List(vec![Some(0.5), None, Some(1.5)])),
                None,
                Some(List(vec![Some(2.0), Some(2.5), Some(3.0)])),
            ]
        );
        Ok(())
    }

    #[test]
    fn map_elements_keeps_only_finished_rows_after_a_panic_or_error() -> Result<()> {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let conn = connection()?;
        let chunk = conn
            .query(
                "SELECT [1, NULL, 3, NULL, 5]::BIGINT[], [6, 7, 8]::BIGINT[]",
                crate::Parameters::None,
            )?
            .next_chunk()?
            .unwrap();
        let (with_nulls, without_nulls) = (
            chunk.get_vector_at::<List<i64>>(0)?,
            chunk.get_vector_at::<List<i64>>(1)?,
        );
        let (with_nulls, without_nulls) = (with_nulls.list_reader()?, without_nulls.list_reader()?);
        let (with_nulls, without_nulls) = (with_nulls.get(0)?.unwrap(), without_nulls.get(0)?.unwrap());
        let mut output = DataChunk::create(&conn, &[List::<i64>::logical_type(&conn)?], true)?;
        let mut vector = output.get_vector_at_mut::<List<i64>>(0)?;
        vector.set_size(4)?;
        vector.list_appender(|lists| {
            lists.map_elements(0, without_nulls, |value| *value)?;
            for list in [with_nulls, without_nulls] {
                let panicked = catch_unwind(AssertUnwindSafe(|| {
                    lists.map_elements(1, list, |value| if *value > 3 { panic!("in f") } else { *value })
                }));
                assert!(panicked.is_err());
                assert_eq!(lists.child_len(), 3);
            }
            // The panicked row's NULL elements do not leak into the next row.
            lists.set_slice(1, &[9, 9, 9, 9, 9])?;
            Ok(())
        })?;
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            vector.list_appender(|lists| {
                lists.map_elements(2, with_nulls, |value| -value)?;
                lists.map_elements(3, with_nulls, |_| panic!("in f"))
            })
        }));
        assert!(panicked.is_err());
        assert_eq!(vector.child_len()?, 13);
        let error = vector
            .list_appender(|lists| {
                lists.map_elements(3, without_nulls, |value| value + 1)?;
                assert!(lists.map_elements(4, without_nulls, |value| *value).is_err());
                Err::<(), _>(crate::Error::invalid_input("stop"))
            })
            .unwrap_err();
        assert!(error.message.contains("stop"), "{error}");
        assert_eq!(vector.child_len()?, 16);
        assert_eq!(
            rows(&vector)?,
            [
                Some(List(vec![Some(6), Some(7), Some(8)])),
                Some(List(vec![Some(9); 5])),
                Some(List(vec![Some(-1), None, Some(-3), None, Some(-5)])),
                Some(List(vec![Some(7), Some(8), Some(9)])),
            ]
        );
        Ok(())
    }
}
