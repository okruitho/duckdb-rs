use std::{marker::PhantomData, ptr::NonNull};

use crate::raw::RawExt;
use crate::{Error, Result, check_api_call, ffi, logical_type::LogicalTypeID};

use super::{
    Arena, BytesBuffer, DuckDBBytes, HasDuckDBBytes, NativeElement, PlainNative, ReadableNativeElement, Selection,
    StorageKind, StrBuffer, Vector, VectorReader, WritableNativeElement, not_flat, not_writable, out_of_bounds,
    reader::RowLayout,
};

/// Exclusive access to a writable flat vector.
///
/// [`Self::set`] encodes typed values and maintains their validity. Raw data
/// access leaves validity to the caller; raw slices and manual validity changes
/// are unsafe because DuckDB trusts initialized, correctly typed values.
/// Child writer state is prepared on the first child write and invalidated by reshaping
/// or slow-path writes that can replace buffers.
///
/// A vector without a validity mask has only valid rows. The writer allocates the mask on the
/// first NULL, so output without NULLs keeps the all-valid fast paths of whatever reads it.
///
/// Resize the vector before acquiring its writer:
///
/// ```
/// # use duckdb_neo::{DuckDBType, Result, data_chunk::DataChunk, environment::{Environment, StorageLocation}};
/// # fn main() -> Result<()> {
/// # let env = Environment::new()?;
/// # let conn = env.open(StorageLocation::InMemory)?.connect()?;
/// let mut chunk = DataChunk::create(&conn, &[i32::logical_type(&conn)?], true)?;
/// let mut vector = chunk.get_vector_at_mut::<i32>(0)?;
/// vector.set_size(3)?;
/// {
///     let mut writer = vector.writer()?;
///     let data = writer.as_mut_ptr();
///     // SAFETY: All three slots are in bounds and contain INTEGER values.
///     unsafe {
///         for index in 0..writer.len() {
///             data.add(index).write(index as i32);
///         }
///         writer.fill_validity(true)?;
///         writer.set_row_validity(1, false)?;
///     }
/// }
/// assert_eq!(vector.reader()?.iter().collect::<Vec<_>>(), [Some(&0), None, Some(&2)]);
/// # Ok(())
/// # }
/// ```
///
/// The vector cannot be resized while its writer is in use:
///
/// ```compile_fail,E0499
/// use duckdb_neo::{Result, vector::Vector};
///
/// fn resize(vector: &mut Vector<'_, i32>) -> Result<()> {
///     let mut writer = vector.writer()?;
///     vector.set_size(10)?;
///     writer.set(0, &42)?;
///     Ok(())
/// }
/// ```
///
/// The writer cannot outlive the vector wrapper it borrows:
///
/// ```compile_fail,E0515
/// use duckdb_neo::{Result, vector::{Vector, VectorWriter}};
///
/// fn escape<'a>(mut vector: Vector<'a, i32>) -> Result<VectorWriter<'a, i32>> {
///     vector.writer()
/// }
/// ```
#[repr(C)]
#[derive(Debug)]
pub struct VectorWriter<'writer, T: NativeElement> {
    vector: NonNull<Vector<'writer, T>>,
    len: usize,
    /// The writable mask, or null while the vector has none.
    validity: *mut u64,
    data: *mut std::ffi::c_void,
    _borrow: PhantomData<&'writer mut T>,
}

impl<'writer, T: NativeElement> VectorWriter<'writer, T> {
    #[inline]
    pub(super) fn new(vector: &'writer mut Vector<'_, T>) -> Result<Self> {
        // SAFETY: the exclusive borrow covers this writer and every cached descendant.
        unsafe { Self::from_raw(NonNull::from(vector).cast()) }
    }

    /// # Safety
    /// The vector must stay exclusively borrowed, and not be rewrapped, for the writer's lifetime.
    #[inline]
    pub(crate) unsafe fn from_raw(mut vector: NonNull<Vector<'writer, T>>) -> Result<Self> {
        // SAFETY: the caller exclusively borrows this vector for the writer's lifetime.
        let borrowed = unsafe { vector.as_mut() };
        if !borrowed.is_writable() {
            return Err(not_writable());
        }
        if borrowed.storage_kind() != StorageKind::Flat {
            return Err(not_flat());
        }

        let (data, validity) = writable_buffers(borrowed)?;

        Ok(Self {
            vector,
            len: borrowed.len(),
            validity,
            data,
            _borrow: PhantomData,
        })
    }

    /// Return the number of writable rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether the writer contains no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the mutable data pointer, or null when the type exposes no data.
    ///
    /// Writing through this pointer does not change validity. Only access rows
    /// within [`Self::len`], using the physical representation of the logical
    /// type. Typed writes may replace storage; reacquire the pointer afterwards.
    /// The caller chooses `'data` and must uphold it when dereferencing the pointer.
    pub fn as_mut_ptr<'data>(&mut self) -> *mut T::Native<'data> {
        self.data.cast()
    }

    /// Borrow mutable physical storage, or `None` when no data is exposed.
    ///
    /// Changes to these values do not change validity.
    ///
    /// # Safety
    /// Every slot must already be initialized, including NULL slots. Written
    /// values must match the logical type, including UTF-8 and nested offsets,
    /// and any referenced storage must outlive all reads from the vector.
    ///
    /// ```compile_fail,E0499
    /// use duckdb_neo::{Result, vector::Vector};
    ///
    /// fn overwrite(vector: &mut Vector<'_, i32>) -> Result<()> {
    ///     let mut writer = vector.writer()?;
    ///     writer.set(0, &1)?;
    ///     let data = unsafe { writer.as_slice_mut() }.unwrap();
    ///     writer.set(0, &2)?;
    ///     data[0] = 3;
    ///     Ok(())
    /// }
    /// ```
    pub unsafe fn as_slice_mut(&mut self) -> Option<&mut [T::Native<'_>]> {
        let len = self.len();
        let data = self.as_mut_ptr();
        if data.is_null() {
            None
        } else {
            // SAFETY: The writer owns flat storage, and the caller guarantees initialization.
            Some(unsafe { std::slice::from_raw_parts_mut(data, len) })
        }
    }

    /// Borrow the writable validity mask, allocating it if the vector has none.
    ///
    /// Bit `index % 64` of word `index / 64` marks a non-NULL row.
    ///
    /// This only changes the parent's mask; DuckDB expects NULL STRUCT, ARRAY, and UNION rows to have
    /// NULL descendants, which [`Self::set_null`] maintains.
    ///
    /// # Safety
    /// Rows marked valid must contain initialized values of the logical type before any read.
    ///
    /// A borrowed mask cannot survive a typed write that might replace it:
    ///
    /// ```compile_fail,E0499
    /// use duckdb_neo::{Result, vector::Vector};
    ///
    /// fn overwrite(vector: &mut Vector<'_, i32>) -> Result<()> {
    ///     let mut writer = vector.writer()?;
    ///     writer.set(0, &1)?;
    ///     let mask = unsafe { writer.validity_mut()? };
    ///     writer.set_null(0)?;
    ///     mask[0] = 0;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// ```compile_fail,E0133
    /// use duckdb_neo::{Result, vector::Vector};
    ///
    /// fn validity(vector: &mut Vector<'_, i32>) -> Result<()> {
    ///     vector.writer()?.validity_mut()?;
    ///     Ok(())
    /// }
    /// ```
    pub unsafe fn validity_mut(&mut self) -> Result<&mut [u64]> {
        let len = self.len().div_ceil(64);
        let validity = self.allocated_validity()?;
        // SAFETY: DuckDB supplied a writable mask, exclusively borrowed through the writer.
        Ok(unsafe { std::slice::from_raw_parts_mut(validity, len) })
    }

    /// Set a row's validity without changing its data or descendants.
    ///
    /// DuckDB expects NULL STRUCT, ARRAY, and UNION rows to have NULL descendants, which
    /// [`Self::set_null`] maintains.
    ///
    /// # Safety
    /// A row marked valid must contain an initialized value of the logical type before any read.
    #[inline]
    pub unsafe fn set_row_validity(&mut self, index: usize, is_valid: bool) -> Result<()> {
        if index >= self.len() {
            return Err(out_of_bounds(index, self.len()));
        }
        // SAFETY: The writer is flat and writable, and the row is in bounds.
        unsafe { self.set_validity_unchecked(index, is_valid) }
    }

    /// Set every row's validity, including padding bits in the last word.
    ///
    /// DuckDB expects NULL STRUCT, ARRAY, and UNION rows to have NULL descendants, which
    /// [`Self::fill_null`] maintains.
    ///
    /// # Safety
    /// Rows marked valid must contain initialized values of the logical type before any read.
    pub unsafe fn fill_validity(&mut self, is_valid: bool) -> Result<()> {
        if is_valid && self.validity.is_null() {
            return Ok(());
        }
        // SAFETY: the caller guarantees rows marked valid hold initialized values.
        unsafe { self.validity_mut()? }.fill(if is_valid { u64::MAX } else { 0 });
        Ok(())
    }

    pub(crate) fn set_null_slow(&mut self, index: usize) -> Result<()> {
        let result = self.vector_mut().set_null_slow(index);
        self.refresh_buffers()?;
        result
    }

    pub(crate) fn write_value_slow(&mut self, index: usize, value: &crate::value::Value) -> Result<()> {
        let result = self.vector_mut().write_value_slow(index, value);
        self.refresh_buffers()?;
        result
    }

    #[inline(always)]
    pub(crate) unsafe fn write_internal<U>(&mut self, index: usize, value: Option<U>) -> Result<()> {
        if let Some(value) = value {
            let data = self.data.cast::<U>();
            // SAFETY: The caller guarantees the row's bounds and physical representation.
            unsafe { data.add(index).write(value) };
        }
        Ok(())
    }

    /// Reacquire the buffers and size after an ancestor resized or rewrote this vector.
    pub(crate) fn refresh_all(&mut self) -> Result<()> {
        self.len = self.vector().len();
        self.refresh_buffers()
    }

    /// Resize the vector like [`Vector::set_size`] and reacquire its buffers.
    pub(crate) fn resize(&mut self, len: usize) -> Result<()> {
        let result = self.vector_mut().set_size(len);
        self.len = self.vector().len();
        self.refresh_buffers()?;
        result
    }

    pub(crate) fn vector(&self) -> &Vector<'writer, T> {
        // SAFETY: construction retains the vector's borrow, and this reference cannot outlive self.
        unsafe { self.vector.as_ref() }
    }

    pub(crate) fn vector_mut(&mut self) -> &mut Vector<'writer, T> {
        // SAFETY: construction retains the vector's exclusive borrow, and this reference cannot outlive self.
        unsafe { self.vector.as_mut() }
    }

    /// NULL rows `start..end` of child `child` and their descendants.
    pub(crate) fn null_child_rows(&mut self, child: usize, start: usize, end: usize) -> Result<()> {
        self.vector_mut()
            .children_collection_mut()
            .expect("nested vectors have children")
            .cached_iter_mut()?
            .nth(child)
            .expect("validated child exists")
            .null_rows(start, end)
    }

    /// NULL `row` of every child, as DuckDB expects of a NULL STRUCT or UNION row.
    pub(crate) fn null_children_row(&mut self, row: usize) -> Result<()> {
        if let Some(children) = self.vector_mut().children_collection_mut() {
            for child in children.cached_iter_mut()? {
                child.null_rows(row, row + 1)?;
            }
        }
        Ok(())
    }
}

impl<T: WritableNativeElement> VectorWriter<'_, T> {
    /// Copy a borrowed value into the vector and mark the row valid.
    /// Logical value checks remain enforced.
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: &T::Write) -> Result<()> {
        if index >= self.len() {
            return Err(out_of_bounds(index, self.len()));
        }
        // SAFETY: The writer is flat and writable, and the row is in bounds.
        unsafe { self.set_unchecked(index, value) }
    }

    /// Make a row SQL NULL, along with the descendants DuckDB requires to be NULL.
    #[inline(always)]
    pub fn set_null(&mut self, index: usize) -> Result<()> {
        if index >= self.len() {
            return Err(out_of_bounds(index, self.len()));
        }
        // SAFETY: The writer is flat and writable, and the row is in bounds.
        unsafe { self.set_null_unchecked(index) }
    }

    /// Write a row; `None` writes SQL NULL.
    #[inline(always)]
    pub fn set_option(&mut self, index: usize, value: Option<&T::Write>) -> Result<()> {
        match value {
            Some(value) => self.set(index, value),
            None => self.set_null(index),
        }
    }

    /// Encode a value and mark the row valid without checking bounds.
    ///
    /// Logical value checks still apply.
    ///
    /// # Safety
    /// `index` must be less than [`Self::len`].
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: &T::Write) -> Result<()> {
        // SAFETY: The writer guarantees flat, writable storage; the caller guarantees bounds.
        unsafe { T::write(self, index, value)? };
        // VARIANT's value API determines nullness after casting the supplied Value.
        if T::TYPE_ID != LogicalTypeID::Variant {
            // SAFETY: as above; the row now holds a value.
            unsafe { self.mark_valid_unchecked(index) };
        }
        Ok(())
    }

    /// Make a row SQL NULL without checking bounds.
    ///
    /// # Safety
    /// `index` must be less than [`Self::len`].
    #[inline(always)]
    pub unsafe fn set_null_unchecked(&mut self, index: usize) -> Result<()> {
        // SAFETY: The writer guarantees flat, writable storage; the caller guarantees bounds.
        unsafe { T::write_null(self, index)? };
        if T::TYPE_ID != LogicalTypeID::Variant {
            // SAFETY: as above.
            unsafe { self.mark_null_unchecked(index)? };
        }
        Ok(())
    }

    /// Write a row without checking bounds; `None` writes SQL NULL.
    ///
    /// # Safety
    /// `index` must be less than [`Self::len`].
    #[inline(always)]
    pub unsafe fn set_option_unchecked(&mut self, index: usize, value: Option<&T::Write>) -> Result<()> {
        // SAFETY: the caller guarantees bounds.
        unsafe {
            match value {
                Some(value) => self.set_unchecked(index, value),
                None => self.set_null_unchecked(index),
            }
        }
    }
}

impl<T: NativeElement> VectorWriter<'_, T> {
    #[inline]
    fn refresh_buffers(&mut self) -> Result<()> {
        (self.data, self.validity) = writable_buffers(self.vector())?;
        Ok(())
    }

    #[inline(always)]
    fn allocated_validity(&mut self) -> Result<*mut u64> {
        if self.validity.is_null() {
            self.allocate_validity()
        } else {
            Ok(self.validity)
        }
    }

    #[cold]
    #[inline(never)]
    fn allocate_validity(&mut self) -> Result<*mut u64> {
        self.validity = writable_validity(self.vector_mut())?;
        Ok(self.validity)
    }

    /// Mark rows `start..end` valid, a whole word at a time where possible.
    ///
    /// # Safety
    /// `end` must not exceed [`Self::len`], and the rows must hold initialized values or be written before
    /// anything reads them.
    pub(crate) unsafe fn mark_valid_range(&mut self, start: usize, end: usize) {
        if start >= end || self.validity.is_null() {
            return;
        }
        let (first, last) = (start / 64, (end - 1) / 64);
        let low = u64::MAX << (start % 64);
        let high = u64::MAX >> (63 - (end - 1) % 64);
        // SAFETY: the caller guarantees the rows are within the mask.
        unsafe {
            if first == last {
                *self.validity.add(first) |= low & high;
                return;
            }
            *self.validity.add(first) |= low;
            std::slice::from_raw_parts_mut(self.validity.add(first + 1), last - first - 1).fill(u64::MAX);
            *self.validity.add(last) |= high;
        }
    }

    #[inline(always)]
    unsafe fn set_validity_unchecked(&mut self, index: usize, is_valid: bool) -> Result<()> {
        // SAFETY: the caller guarantees bounds.
        unsafe {
            if is_valid {
                self.mark_valid_unchecked(index);
                Ok(())
            } else {
                self.mark_null_unchecked(index)
            }
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn mark_valid_unchecked(&mut self, index: usize) {
        if self.validity.is_null() {
            return;
        }
        // SAFETY: The owning writer guarantees flat, writable storage; the caller guarantees bounds.
        let word = unsafe { &mut *self.validity.add(index / 64) };
        let mask = 1u64 << (index % 64);
        // Rows are mostly valid already, so storing only to revive a NULL row keeps this branch predictable.
        if *word & mask == 0 {
            *word |= mask;
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn mark_null_unchecked(&mut self, index: usize) -> Result<()> {
        let validity = self.allocated_validity()?;
        // SAFETY: The owning writer guarantees flat, writable storage; the caller guarantees bounds.
        unsafe { *validity.add(index / 64) &= !(1u64 << (index % 64)) };
        Ok(())
    }
}

impl<T: NativeElement> VectorWriter<'_, T> {
    /// Copy `values` into rows `offset..offset + values.len()` and mark those rows valid.
    ///
    /// # Safety
    /// `U` must be the physical slot type, and every value must be a valid row.
    pub(crate) unsafe fn write_slice_unchecked<U>(&mut self, offset: usize, values: &[U]) -> Result<()> {
        let end = offset
            .checked_add(values.len())
            .filter(|end| *end <= self.len())
            .ok_or_else(|| out_of_bounds(offset.saturating_add(values.len()).saturating_sub(1), self.len()))?;
        // SAFETY: the rows are in bounds, and the caller guarantees `T` and the values are valid rows.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), self.data.cast::<U>().add(offset), values.len());
            self.mark_valid_range(offset, end);
        }
        Ok(())
    }

    /// Copy `values` into rows `offset..offset + values.len()` and mark those rows valid, when DuckDB
    /// stores this vector's rows as `U`: e.g. `i64` into `BIGINT`, `TIMESTAMP` or `DECIMAL(18, 3)`.
    /// Values are not checked against the logical type's range; see [`PlainNative`].
    pub fn write_slice_as<U: PlainNative>(&mut self, offset: usize, values: &[U]) -> Result<()> {
        let logical_type = self.vector().logical_type();
        if super::element::plain_slot(logical_type)? != Some(U::SLOT) {
            return Err(Error::invalid_input(format!(
                "{logical_type} rows are not stored as {}",
                std::any::type_name::<U>()
            )));
        }
        // SAFETY: `U` has the physical type of this vector's slots, all of whose values are valid rows.
        unsafe { self.write_slice_unchecked(offset, values) }
    }
}

impl<T: PlainNative> VectorWriter<'_, T> {
    /// Copy `values` into rows `offset..offset + values.len()` and mark those rows valid.
    ///
    /// ```
    /// # use duckdb_neo::{DuckDBType, Result, data_chunk::DataChunk, environment::{Environment, StorageLocation}};
    /// # fn main() -> Result<()> {
    /// # let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
    /// let mut chunk = DataChunk::create(&conn, &[i32::logical_type(&conn)?], true)?;
    /// let mut vector = chunk.get_vector_at_mut::<i32>(0)?;
    /// vector.set_size(4)?;
    /// let mut writer = vector.writer()?;
    /// writer.write_slice(1, &[7, 8, 9])?;
    /// writer.set_null(2)?;
    /// # drop(writer);
    /// assert_eq!(vector.reader()?.iter().collect::<Vec<_>>(), [Some(&0), Some(&7), None, Some(&9)]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn write_slice(&mut self, offset: usize, values: &[T]) -> Result<()> {
        // SAFETY: every value of a `PlainNative` type is a valid row.
        unsafe { self.write_slice_unchecked(offset, values) }
    }

    /// Copy every row of `source`, values and NULLs, into the first rows of this vector.
    ///
    /// Flat sources copy whole buffers; dictionary and constant sources copy row by row.
    pub fn copy_from(&mut self, source: &VectorReader<'_, T>) -> Result<()> {
        let len = source.len();
        if len > self.len() {
            return Err(out_of_bounds(len - 1, self.len()));
        }
        if len == 0 {
            return Ok(());
        }
        let data = self.data.cast::<T>();
        match source.selection() {
            Selection::Identity => {
                // SAFETY: both vectors hold at least `len` slots of `T`.
                unsafe { std::ptr::copy_nonoverlapping(source.as_ptr().cast::<T>(), data, len) };
                match source.validity() {
                    // SAFETY: the rows are in bounds and hold the values just copied.
                    None => unsafe { self.mark_valid_range(0, len) },
                    // SAFETY: as above; the source mask covers `len` rows.
                    Some(mask) => unsafe { self.copy_validity(mask, len)? },
                }
            }
            selection => {
                let (slots, validity) = (source.as_ptr().cast::<T>(), source.validity());
                for row in 0..len {
                    let physical = selection.index(row);
                    let valid = validity.is_none_or(|mask| mask[physical / 64] & (1 << (physical % 64)) != 0);
                    // SAFETY: the row is in bounds, the selection maps it to a source slot, and `T` is
                    // the physical slot type.
                    unsafe {
                        if valid {
                            data.add(row).write(*slots.add(physical));
                            self.mark_valid_unchecked(row);
                        } else {
                            self.mark_null_unchecked(row)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// # Safety
    /// `mask` must cover `len` rows, and `len` must not exceed [`Self::len`].
    unsafe fn copy_validity(&mut self, mask: &[u64], len: usize) -> Result<()> {
        let (words, tail) = (len / 64, len % 64);
        let validity = self.allocated_validity()?;
        // SAFETY: the caller guarantees both masks cover `len` rows.
        unsafe {
            std::ptr::copy_nonoverlapping(mask.as_ptr(), validity, words);
            if tail != 0 {
                let keep = u64::MAX << tail;
                let word = validity.add(words);
                *word = (*word & keep) | (mask[words] & !keep);
            }
        }
        Ok(())
    }

    /// Write `f(value)` for each row of `input` into the first rows, NULL rows staying NULL without a call; fails if
    /// `input` is longer. Storage is resolved once, so a flat input without NULLs runs a vectorizable loop.
    ///
    /// ```
    /// # use duckdb_neo::{Parameters, Result, environment::{Environment, StorageLocation}, scalar::MicroScalar, vector::Vector};
    /// # fn main() -> Result<()> {
    /// # let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
    /// MicroScalar::create("plus_one", |_ctx, input: Vector<'_, i64>, mut output: Vector<'_, i64>| {
    ///     output.writer()?.map_from(&input.reader()?, |value| value.wrapping_add(1))
    /// })
    /// .register(&conn)?;
    /// let chunk = conn.query("SELECT plus_one(v) FROM (VALUES (1), (NULL)) t(v)", Parameters::None)?
    ///     .next_chunk()?.unwrap();
    /// let rows = chunk.get_vector_at::<i64>(0)?.reader()?.iter().collect::<Vec<_>>();
    /// assert_eq!(rows, [Some(&2), None]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn map_from<'v, I: ReadableNativeElement + 'v>(
        &mut self,
        input: &VectorReader<'v, I>,
        mut f: impl FnMut(I::Ref<'v>) -> T,
    ) -> Result<()> {
        self.map_reader(input, true, |value| Ok(value.map(&mut f)))
    }

    /// As [`Self::map_from`], but `f` gets `None` for NULL rows and its `None` makes the row NULL.
    ///
    /// ```
    /// # use duckdb_neo::{Parameters, Result, environment::{Environment, StorageLocation}, scalar::MicroScalar, vector::Vector};
    /// # fn main() -> Result<()> {
    /// # let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
    /// MicroScalar::create("swap_null", |_ctx, input: Vector<'_, i64>, mut output: Vector<'_, i64>| {
    ///     output.writer()?.map_option_from(&input.reader()?, |value| match value {
    ///         Some(0) => None,
    ///         Some(value) => Some(*value),
    ///         None => Some(0),
    ///     })
    /// })
    /// .set_property(duckdb_neo::enums::FunctionProperty::HasSpecialNullHandling(true))
    /// .register(&conn)?;
    /// let chunk = conn.query("SELECT swap_null(v) FROM (VALUES (0), (NULL), (5)) t(v)", Parameters::None)?
    ///     .next_chunk()?.unwrap();
    /// let rows = chunk.get_vector_at::<i64>(0)?.reader()?.iter().collect::<Vec<_>>();
    /// assert_eq!(rows, [None, Some(&0), Some(&5)]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn map_option_from<'v, I: ReadableNativeElement + 'v>(
        &mut self,
        input: &VectorReader<'v, I>,
        mut f: impl FnMut(Option<I::Ref<'v>>) -> Option<T>,
    ) -> Result<()> {
        self.map_reader(input, false, |value| Ok(f(value)))
    }

    /// As [`Self::map_from`] with a fallible `f`, returning its first error; each row of `input` then holds its old
    /// or its new value.
    #[inline]
    pub fn try_map_from<'v, I: ReadableNativeElement + 'v>(
        &mut self,
        input: &VectorReader<'v, I>,
        mut f: impl FnMut(I::Ref<'v>) -> Result<T>,
    ) -> Result<()> {
        self.map_reader(input, true, |value| value.map(&mut f).transpose())
    }

    #[inline(always)]
    fn map_reader<'v, I: ReadableNativeElement + 'v>(
        &mut self,
        input: &VectorReader<'v, I>,
        propagate_nulls: bool,
        f: impl FnMut(Option<I::Ref<'v>>) -> Result<Option<T>>,
    ) -> Result<()> {
        let reader = input.clone();
        let len = reader.len();
        if len > self.len() {
            return Err(out_of_bounds(len - 1, self.len()));
        }
        // SAFETY: the layout describes the reader's `len` rows, and `I::get` reads the slot of a valid row.
        unsafe {
            self.map_layout(
                len,
                reader.layout(),
                propagate_nulls,
                |physical, row| I::get(&reader, physical, row),
                f,
            )
        }
    }

    /// Write rows `0..len` with `f` over the rows of `layout`, each read as `read(physical, row)`; with
    /// `propagate_nulls`, NULL rows become NULL without calling `f`, otherwise `f` gets `None`.
    ///
    /// # Safety
    /// `len` must not exceed [`Self::len`], `layout` must describe at least `len` rows, and `read` must be sound for
    /// each valid row and its slot.
    #[inline(always)]
    pub(crate) unsafe fn map_layout<V>(
        &mut self,
        len: usize,
        layout: RowLayout,
        propagate_nulls: bool,
        read: impl Fn(usize, usize) -> V,
        mut f: impl FnMut(Option<V>) -> Result<Option<T>>,
    ) -> Result<()> {
        // SAFETY (every arm): the caller guarantees the layout covers `len` rows and `read` is sound for valid ones;
        // the validity words only report rows the layout marks valid.
        unsafe {
            match layout {
                RowLayout::Flat { validity } if validity.is_null() => self.write_rows(
                    len,
                    propagate_nulls,
                    |_, _| u64::MAX,
                    |row, valid| f(valid.then(|| read(row, row))),
                ),
                RowLayout::Flat { validity } => self.write_rows(
                    len,
                    propagate_nulls,
                    |start, _| *validity.add(start / 64),
                    |row, valid| f(valid.then(|| read(row, row))),
                ),
                RowLayout::Constant { is_null } => self.write_rows(
                    len,
                    propagate_nulls,
                    |_, _| if is_null { 0 } else { u64::MAX },
                    |row, valid| f(valid.then(|| read(0, row))),
                ),
                RowLayout::Dictionary { selection, .. } => self.write_rows(
                    len,
                    propagate_nulls,
                    |start, end| layout.validity_word(start, end),
                    |row, valid| f(valid.then(|| read(*selection.add(row) as usize, row))),
                ),
            }
        }
    }

    /// Write rows `0..len` from `row(index, valid)` 64 at a time, `None` writing NULL; `valid(start, end)` has bit `i`
    /// set when input row `start + i` is valid, and with `propagate_nulls` other rows become NULL without a call.
    ///
    /// # Safety
    /// `len` must not exceed [`Self::len`], and both closures must be sound for every row below `len`.
    #[inline(always)]
    pub(crate) unsafe fn write_rows(
        &mut self,
        len: usize,
        propagate_nulls: bool,
        valid: impl Fn(usize, usize) -> u64,
        mut row: impl FnMut(usize, bool) -> Result<Option<T>>,
    ) -> Result<()> {
        let data = self.data.cast::<T>();
        let mut start = 0;
        while start < len {
            let end = len.min(start + 64);
            let rows = u64::MAX >> (64 - (end - start));
            let input = valid(start, end) & rows;
            let mut output;
            if input == rows {
                output = rows;
                for index in start..end {
                    match row(index, true)? {
                        // SAFETY: the caller guarantees the row is in bounds, and `T` is the slot type.
                        Some(value) => unsafe { data.add(index).write(value) },
                        None => output &= !(1 << (index - start)),
                    }
                }
            } else if propagate_nulls {
                output = input;
                let mut bits = input;
                while bits != 0 {
                    let index = start + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    match row(index, true)? {
                        // SAFETY: as above.
                        Some(value) => unsafe { data.add(index).write(value) },
                        None => output &= !(1 << (index - start)),
                    }
                }
            } else {
                output = rows;
                for index in start..end {
                    match row(index, input >> (index - start) & 1 != 0)? {
                        // SAFETY: as above.
                        Some(value) => unsafe { data.add(index).write(value) },
                        None => output &= !(1 << (index - start)),
                    }
                }
            }
            // SAFETY: the rows are in bounds, and the valid ones were just written.
            unsafe { self.store_validity_word(start / 64, output, rows)? };
            start = end;
        }
        Ok(())
    }

    /// Resize to `len` rows without zeroing them, then run `write`; if it fails or panics, zero every row.
    ///
    /// # Safety
    /// When `write` succeeds, it must have written or marked NULL each of the `len` rows.
    #[inline(always)]
    pub(crate) unsafe fn resize_and_write(
        &mut self,
        len: usize,
        write: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        let old = self.len();
        // SAFETY: the new rows are written by `write`, or zeroed, before this returns.
        let resized = unsafe { self.vector_mut().set_size_uninit(len) };
        self.len = self.vector().len();
        if let Err(error) = resized.and_then(|()| self.refresh_buffers()) {
            let len = self.len;
            self.vector_mut().init_rows(old, len)?;
            self.refresh_buffers()?;
            return Err(error);
        }
        let zero = ZeroRowsOnDrop {
            data: self.data.cast::<T>(),
            len: self.len,
        };
        write(self)?;
        std::mem::forget(zero);
        Ok(())
    }
}

struct ZeroRowsOnDrop<T: PlainNative> {
    data: *mut T,
    len: usize,
}

impl<T: PlainNative> Drop for ZeroRowsOnDrop<T> {
    fn drop(&mut self) {
        if self.len != 0 && !self.data.is_null() {
            // SAFETY: the slots are the writer's flat rows of `T`, and `PlainNative` guarantees that all-zero bytes are
            // a valid `T` and row.
            unsafe { self.data.write_bytes(0, self.len) };
        }
    }
}

impl<T: NativeElement> VectorWriter<'_, T> {
    /// Set the bits `rows` of validity word `word` to `bits`, allocating the mask only for a NULL row.
    ///
    /// # Safety
    /// The word must be within the mask, and rows marked valid must hold initialized values.
    #[inline(always)]
    unsafe fn store_validity_word(&mut self, word: usize, bits: u64, rows: u64) -> Result<()> {
        if bits == rows && self.validity.is_null() {
            return Ok(());
        }
        let validity = self.allocated_validity()?;
        // SAFETY: the caller guarantees the word is within the writable mask.
        unsafe {
            let slot = validity.add(word);
            *slot = (*slot & !rows) | bits;
        }
        Ok(())
    }
}

impl<T: NativeElement> VectorWriter<'_, T> {
    /// Make every row SQL NULL, along with the descendants DuckDB requires to be NULL.
    pub fn fill_null(&mut self) -> Result<()> {
        let len = self.len();
        let result = self.vector_mut().null_rows(0, len);
        self.refresh_buffers()?;
        result
    }
}

impl VectorWriter<'_, String> {
    /// Write concatenated UTF-8 fragments; `None` writes NULL and an empty slice writes an empty string.
    /// Fails if `index` is out of bounds or the combined byte length exceeds `u32::MAX`.
    #[inline(always)]
    pub fn set_parts(&mut self, index: usize, parts: Option<&[&str]>) -> Result<()> {
        if index >= self.len() {
            return Err(out_of_bounds(index, self.len()));
        }
        if let Some(parts) = parts {
            // SAFETY: The row is in bounds, and concatenated UTF-8 is valid VARCHAR storage.
            unsafe { self.write_byte_parts(index, parts)? };
        }
        // SAFETY: Non-NULL rows are initialized; VARCHAR has no descendants.
        unsafe { self.set_validity_unchecked(index, parts.is_some()) }
    }
}

impl VectorWriter<'_, String> {
    /// Build a row's text in place, in at most `capacity` bytes, and mark the row valid.
    ///
    /// `fill` pushes the text; pushes beyond `capacity` fail. Short results are stored inline, longer
    /// ones in this vector's arena, where unused capacity stays reserved until the vector is reset.
    /// If `fill` fails or panics, the row holds an empty string and keeps its validity. A failure
    /// before `fill` runs (`capacity` above `u32::MAX`, or the arena allocation) leaves the row unchanged.
    ///
    /// ```
    /// # use duckdb_neo::{DuckDBType, Result, data_chunk::DataChunk, environment::{Environment, StorageLocation}};
    /// # fn main() -> Result<()> {
    /// # let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
    /// let mut chunk = DataChunk::create(&conn, &[String::logical_type(&conn)?], true)?;
    /// let mut vector = chunk.get_vector_at_mut::<String>(0)?;
    /// vector.set_size(1)?;
    /// let mut writer = vector.writer()?;
    /// let input = "a string longer than twelve bytes";
    /// writer.set_with(0, input.len(), |buffer| {
    ///     buffer.push_str(input)?;
    ///     buffer.as_mut_str().make_ascii_uppercase();
    ///     Ok(())
    /// })?;
    /// # drop(writer);
    /// assert_eq!(vector.reader()?.get(0)?, Some("A STRING LONGER THAN TWELVE BYTES"));
    /// # Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn set_with(
        &mut self,
        index: usize,
        capacity: usize,
        fill: impl FnOnce(&mut StrBuffer<'_>) -> Result<()>,
    ) -> Result<()> {
        if index >= self.len() {
            return Err(out_of_bounds(index, self.len()));
        }
        // SAFETY: The row is in bounds; a fresh buffer is empty, and a `StrBuffer` only holds UTF-8.
        unsafe {
            self.write_byte_buffer(index, capacity, |bytes| fill(StrBuffer::from_bytes_mut(bytes)))?;
            self.mark_valid_unchecked(index);
        }
        Ok(())
    }
}

impl VectorWriter<'_, crate::types::UncheckedString> {
    /// Build a row's text in place, as with [`VectorWriter::<String>::set_with`].
    #[inline(always)]
    pub fn set_with(
        &mut self,
        index: usize,
        capacity: usize,
        fill: impl FnOnce(&mut StrBuffer<'_>) -> Result<()>,
    ) -> Result<()> {
        if index >= self.len() {
            return Err(out_of_bounds(index, self.len()));
        }
        // SAFETY: The row is in bounds; a fresh buffer is empty, and a `StrBuffer` only holds UTF-8.
        unsafe {
            self.write_byte_buffer(index, capacity, |bytes| fill(StrBuffer::from_bytes_mut(bytes)))?;
            self.mark_valid_unchecked(index);
        }
        Ok(())
    }

    /// Concatenate UTF-8 fragments, as with [`VectorWriter::<String>::set_parts`].
    #[inline(always)]
    pub fn set_parts(&mut self, index: usize, parts: Option<&[&str]>) -> Result<()> {
        if index >= self.len() {
            return Err(out_of_bounds(index, self.len()));
        }
        if let Some(parts) = parts {
            // SAFETY: The row is in bounds, and concatenated UTF-8 is valid VARCHAR storage.
            unsafe { self.write_byte_parts(index, parts)? };
        }
        // SAFETY: Non-NULL rows are initialized; VARCHAR has no descendants.
        unsafe { self.set_validity_unchecked(index, parts.is_some()) }
    }
}

impl VectorWriter<'_, crate::types::BlobValue> {
    /// Write concatenated BLOB fragments; `None` writes NULL and an empty slice writes an empty blob.
    /// Fails if `index` is out of bounds or the combined byte length exceeds `u32::MAX`.
    #[inline(always)]
    pub fn set_parts(&mut self, index: usize, parts: Option<&[&[u8]]>) -> Result<()> {
        if index >= self.len() {
            return Err(out_of_bounds(index, self.len()));
        }
        if let Some(parts) = parts {
            // SAFETY: The row is in bounds, and arbitrary bytes are valid BLOB storage.
            unsafe { self.write_byte_parts(index, parts)? };
        }
        // SAFETY: Non-NULL rows are initialized; BLOB has no descendants.
        unsafe { self.set_validity_unchecked(index, parts.is_some()) }
    }
}

impl VectorWriter<'_, crate::types::BlobValue> {
    /// Build a row's bytes in place, as with [`VectorWriter::<String>::set_with`].
    #[inline(always)]
    pub fn set_with(
        &mut self,
        index: usize,
        capacity: usize,
        fill: impl FnOnce(&mut BytesBuffer<'_>) -> Result<()>,
    ) -> Result<()> {
        if index >= self.len() {
            return Err(out_of_bounds(index, self.len()));
        }
        // SAFETY: The row is in bounds, and arbitrary bytes are valid BLOB storage.
        unsafe {
            self.write_byte_buffer(index, capacity, fill)?;
            self.mark_valid_unchecked(index);
        }
        Ok(())
    }
}

impl<T: HasDuckDBBytes> VectorWriter<'_, T> {
    /// Borrow the vector's byte arena for raw string or binary encoding.
    ///
    /// Take [`Self::as_mut_ptr`] before borrowing the arena to fill data slots.
    ///
    /// ```compile_fail,E0599
    /// use duckdb_neo::{Result, vector::Vector};
    ///
    /// fn arena(vector: &mut Vector<'_, i32>) -> Result<()> {
    ///     let mut writer = vector.writer()?;
    ///     let _ = writer.arena()?;
    ///     Ok(())
    /// }
    /// ```
    pub fn arena(&mut self) -> Result<&Arena> {
        let vector = self.vector_mut();
        if vector.heap.is_none() {
            let handle = check_api_call!(ffi::duckdb_v2_vector_get_arena, vector.handle, RET)?;
            vector.heap = Some(Arena { handle });
        }
        Ok(vector.heap.as_ref().unwrap())
    }

    #[inline]
    pub(crate) unsafe fn write_bytes(&mut self, index: usize, value: Option<&[u8]>) -> Result<()> {
        if let Some(value) = value {
            // SAFETY: The caller guarantees the row's bounds and logical encoding.
            unsafe { self.write_byte_parts(index, &[value])? };
        }
        Ok(())
    }

    #[inline(always)]
    unsafe fn write_byte_parts<P: AsRef<[u8]> + ?Sized>(&mut self, index: usize, parts: &[&P]) -> Result<()> {
        // SAFETY: The row is in bounds, and long bytes are copied into this vector's arena.
        unsafe {
            let slot = self.data.cast::<DuckDBBytes<'_>>().add(index);
            DuckDBBytes::write_parts(slot, parts, || self.arena())
        }
    }

    /// # Safety
    /// `index` must be in bounds, and every value `fill` can build must be valid for the logical type.
    #[inline(always)]
    unsafe fn write_byte_buffer(
        &mut self,
        index: usize,
        capacity: usize,
        fill: impl FnOnce(&mut BytesBuffer<'_>) -> Result<()>,
    ) -> Result<()> {
        // SAFETY: The row is in bounds, and long bytes are built in this vector's arena.
        unsafe {
            let slot = self.data.cast::<DuckDBBytes<'_>>().add(index);
            DuckDBBytes::write_with(slot, capacity, || self.arena(), fill)
        }
    }
}

impl<T: HasDuckDBBytes + WritableNativeElement> VectorWriter<'_, T> {
    /// Copy row `source_row` of `source`, value or NULL, into row `index` without decoding it.
    ///
    /// Inline values are copied as they are, longer ones into this vector's arena. Both element types
    /// must store the same logical type, e.g. any `VARCHAR` marker into [`String`]. The bytes are
    /// copied unvalidated, so they are exactly as valid as the source's (see [`VarcharBytes`]).
    ///
    /// [`VarcharBytes`]: crate::types::VarcharBytes
    ///
    /// A vector cannot be copied into itself:
    ///
    /// ```compile_fail,E0502
    /// use duckdb_neo::{Result, vector::Vector};
    ///
    /// fn copy(vector: &mut Vector<'_, String>) -> Result<()> {
    ///     let reader = vector.reader()?;
    ///     vector.writer()?.copy_row(0, &reader, 1)
    /// }
    /// ```
    #[inline]
    pub fn copy_row<S: HasDuckDBBytes>(
        &mut self,
        index: usize,
        source: &VectorReader<'_, S>,
        source_row: usize,
    ) -> Result<()> {
        check_same_bytes_type::<S, T>()?;
        if index >= self.len() {
            return Err(out_of_bounds(index, self.len()));
        }
        if source_row >= source.len() {
            return Err(out_of_bounds(source_row, source.len()));
        }
        // SAFETY: both rows are in bounds, and the types share their storage.
        unsafe { self.copy_row_unchecked(index, source, source_row) }
    }

    /// Copy every row of `source`, values and NULLs, into the first rows of this vector, as with
    /// [`Self::copy_row`].
    pub fn copy_bytes_from<S: HasDuckDBBytes>(&mut self, source: &VectorReader<'_, S>) -> Result<()> {
        check_same_bytes_type::<S, T>()?;
        let len = source.len();
        if len > self.len() {
            return Err(out_of_bounds(len - 1, self.len()));
        }
        for row in 0..len {
            // SAFETY: the row is in bounds of both, and the types share their storage.
            unsafe { self.copy_row_unchecked(row, source, row)? };
        }
        Ok(())
    }

    /// # Safety
    /// `index` and `source_row` must be rows of this writer and `source`, whose types share storage.
    #[inline(always)]
    unsafe fn copy_row_unchecked<S: HasDuckDBBytes>(
        &mut self,
        index: usize,
        source: &VectorReader<'_, S>,
        source_row: usize,
    ) -> Result<()> {
        // SAFETY: the caller guarantees the rows; a valid physical slot holds a value of the shared
        // logical type, which is copied as stored, long bytes into this vector's arena.
        unsafe {
            match source.physical_index_unchecked(source_row) {
                Some(physical) => {
                    let value = &*source.as_ptr().cast::<DuckDBBytes<'_>>().add(physical);
                    let slot = self.data.cast::<DuckDBBytes<'_>>().add(index);
                    DuckDBBytes::write_copy(slot, value, || self.arena())?;
                    self.mark_valid_unchecked(index);
                    Ok(())
                }
                None => self.set_null_unchecked(index),
            }
        }
    }
}

fn check_same_bytes_type<S: NativeElement, T: NativeElement>() -> Result<()> {
    if S::TYPE_ID != T::TYPE_ID {
        return Err(Error::invalid_input(format!(
            "cannot copy {:?} rows into a {:?} vector",
            S::TYPE_ID,
            T::TYPE_ID
        )));
    }
    Ok(())
}

#[inline]
fn writable_buffers<T: NativeElement>(vector: &Vector<'_, T>) -> Result<(*mut std::ffi::c_void, *mut u64)> {
    // A flat view borrows the buffers `get_data_mutable` and `flat_get_validity_mutable` return once the
    // mask exists, so one call replaces both, and a missing mask stays unallocated. Copying the whole
    // view into the cache would reload DuckDB's separate field stores as one, stalling store forwarding.
    let view = check_api_call!(ffi::duckdb_v2_vector_get_view, vector.raw(), RET)?;
    Ok((view.data.cast_mut(), view.validity.cast_mut()))
}

fn writable_validity<T: NativeElement>(vector: &mut Vector<'_, T>) -> Result<*mut u64> {
    // Allocating the mask replaces the validity pointer of the cached view.
    vector.view.take();
    check_api_call!(ffi::duckdb_v2_vector_flat_get_validity_mutable, vector.raw(), RET)
}

#[cfg(test)]
mod tests;
