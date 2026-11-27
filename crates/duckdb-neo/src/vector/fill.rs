//! Appending rows in order, so a vector is sized without initializing rows that are written anyway.

use super::{
    NativeElement, PlainNative, StorageKind, Vector, VectorWriter, WritableNativeElement, not_flat, not_writable,
};
use crate::{Result, error::Error};

/// Pushes rows, in order, onto a vector borrowed by [`Vector::append`].
///
/// ```
/// # use duckdb_neo::{DuckDBType, Result, data_chunk::DataChunk, environment::{Environment, StorageLocation}};
/// # fn main() -> Result<()> {
/// # let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
/// let mut chunk = DataChunk::create(&conn, &[String::logical_type(&conn)?], true)?;
/// let mut vector = chunk.get_vector_at_mut::<String>(0)?;
/// vector.append(3, |rows| {
///     rows.push("a")?;
///     rows.push_null()?;
///     assert_eq!(rows.remaining(), 1);
///     Ok(())
/// })?;
/// let reader = vector.reader()?;
/// assert_eq!(reader.iter().collect::<Vec<_>>(), [Some("a"), None]);
/// # Ok(())
/// # }
/// ```
///
/// An appender cannot leave its closure, so the vector always ends with only the rows pushed:
///
/// ```compile_fail,E0521
/// # use duckdb_neo::{Result, vector::Vector};
/// fn escape(vector: &mut Vector<'_, i64>, other: &mut Vector<'_, i64>) -> Result<()> {
///     vector.append(1, |a| other.append(1, |b| {
///         std::mem::swap(a, b);
///         Ok(())
///     }))
/// }
/// ```
#[derive(Debug)]
pub struct VectorAppender<'w, T: NativeElement> {
    writer: VectorWriter<'w, T>,
    /// The row the next push writes.
    next: usize,
    /// One past the last row the vector was sized for.
    end: usize,
    /// Bytes per row; zero bytes are a valid value.
    row_size: usize,
}

impl<T: NativeElement> Vector<'_, T> {
    /// Push up to `additional` rows after the existing ones through `f`, without initializing them first.
    ///
    /// The vector ends with the rows `f` pushed, also when it fails or panics. Fails for nested types
    /// and types whose zeroed storage is not a valid value, e.g. `STRUCT`, `LIST` or `VARIANT`.
    #[inline]
    pub fn append<R>(
        &mut self,
        additional: usize,
        f: impl FnOnce(&mut VectorAppender<'_, T>) -> Result<R>,
    ) -> Result<R> {
        if !self.is_writable() {
            return Err(not_writable());
        }
        if self.storage_kind() != StorageKind::Flat {
            return Err(not_flat());
        }
        let row_size = self
            .zeroed_row_size()?
            .ok_or_else(|| Error::invalid_input(format!("cannot append to {} rows in place", self.logical_type())))?;
        let start = self.len();
        let end = start
            .checked_add(additional)
            .ok_or_else(|| Error::invalid_input("appended row count overflows"))?;
        // SAFETY: the appender below writes or zeroes every new row before the vector can be read again, or
        // the error path initializes them.
        unsafe { self.set_size_uninit(end)? };
        let writer = match VectorWriter::new(self) {
            Ok(writer) => writer,
            Err(error) => {
                let len = self.len();
                self.init_rows(start, len)?;
                return Err(error);
            }
        };
        let mut appender = Finish(VectorAppender {
            writer,
            next: start,
            end,
            row_size,
        });
        // SAFETY: rows `start..end` are in bounds; pushes write their values and NULL rows clear their bits.
        unsafe { appender.0.writer.mark_valid_range(start, end) };
        let value = f(&mut appender.0)?;
        appender.0.finish()?;
        Ok(value)
    }
}

impl<T: WritableNativeElement> VectorAppender<'_, T> {
    /// Push a non-NULL row; fails, pushing nothing, when the vector is full or the value is invalid.
    #[inline(always)]
    pub fn push(&mut self, value: &T::Write) -> Result<()> {
        let row = self.claim()?;
        // SAFETY: the row is in bounds of the sized, flat and writable vector; it is marked valid already.
        unsafe { T::write(&mut self.writer, row, value)? };
        self.next = row + 1;
        Ok(())
    }

    /// Push a SQL NULL row; fails when the vector is full.
    #[inline(always)]
    pub fn push_null(&mut self) -> Result<()> {
        let row = self.claim()?;
        // SAFETY: as for `push`; zero bytes are a valid value, so the NULL row's slot is initialized too.
        unsafe {
            self.writer
                .as_mut_ptr()
                .cast::<u8>()
                .add(row * self.row_size)
                .write_bytes(0, self.row_size);
            self.writer.mark_null_unchecked(row)?;
        }
        self.next = row + 1;
        Ok(())
    }

    /// Push a row; `None` pushes SQL NULL.
    #[inline(always)]
    pub fn push_option(&mut self, value: Option<&T::Write>) -> Result<()> {
        match value {
            Some(value) => self.push(value),
            None => self.push_null(),
        }
    }
}

impl<T: PlainNative> VectorAppender<'_, T> {
    /// Push `count` non-NULL rows, the `i`-th from `f(i)`; fails, pushing nothing, when they do not fit.
    ///
    /// Unlike a loop of [`Self::push`], the loop compiles like one over a slice.
    ///
    /// ```
    /// # use duckdb_neo::{DuckDBType, Result, data_chunk::DataChunk, environment::{Environment, StorageLocation}};
    /// # fn main() -> Result<()> {
    /// # let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
    /// let mut chunk = DataChunk::create(&conn, &[i64::logical_type(&conn)?], true)?;
    /// let mut vector = chunk.get_vector_at_mut::<i64>(0)?;
    /// vector.append(4, |rows| rows.push_with(4, |i| 10 * i as i64))?;
    /// assert_eq!(vector.reader()?.iter().collect::<Vec<_>>(), [Some(&0), Some(&10), Some(&20), Some(&30)]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn push_with(&mut self, count: usize, mut f: impl FnMut(usize) -> T) -> Result<()> {
        if count > self.remaining() {
            return Err(full(self.end));
        }
        let (data, start) = (self.writer.as_mut_ptr().cast::<T>(), self.next);
        for index in 0..count {
            // SAFETY: the rows are in bounds, `T` is the slot type, and the rows are marked valid already. If `f`
            // panics, `next` has not moved, so `finish` zeroes the rows written so far again.
            unsafe { data.add(start + index).write(f(index)) };
        }
        self.next = start + count;
        Ok(())
    }
}

impl<T: NativeElement> VectorAppender<'_, T> {
    /// Return how many more rows fit.
    pub fn remaining(&self) -> usize {
        self.end - self.next
    }

    #[inline(always)]
    fn claim(&self) -> Result<usize> {
        if self.next == self.end {
            return Err(full(self.end));
        }
        Ok(self.next)
    }

    /// Zero the rows not pushed, then cut the vector to the rows pushed.
    fn finish(&mut self) -> Result<()> {
        if self.next == self.end {
            return Ok(());
        }
        // SAFETY: rows `next..end` are in bounds, and zero bytes are a valid value, so the vector is fully
        // initialized even when the resize below fails.
        unsafe {
            self.writer
                .as_mut_ptr()
                .cast::<u8>()
                .add(self.next * self.row_size)
                .write_bytes(0, (self.end - self.next) * self.row_size);
        }
        let end = self.next;
        self.end = end;
        // SAFETY: shrinking exposes no new rows.
        unsafe { self.writer.vector_mut().set_size_uninit(end) }
    }
}

/// Finishes the appender when its closure fails or panics.
struct Finish<'w, T: NativeElement>(VectorAppender<'w, T>);

impl<T: NativeElement> Drop for Finish<'_, T> {
    fn drop(&mut self) {
        let _ = self.0.finish();
    }
}

#[cold]
fn full(len: usize) -> Error {
    Error::invalid_input(format!("the vector is full at {len} rows"))
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use crate::{
        DuckDBType, Result, ToValue,
        connection::Connection,
        data_chunk::DataChunk,
        environment::{Environment, StorageLocation},
        logical_type::LogicalType,
        types::{List, Struct},
    };

    fn connection() -> Result<Connection> {
        Environment::new()?.open(StorageLocation::InMemory)?.connect()
    }

    fn chunk<T: DuckDBType>(conn: &Connection) -> Result<DataChunk<'_>> {
        DataChunk::create(conn, &[T::logical_type(conn)?], true)
    }

    const LONG: &str = "a string longer than the inline storage";

    #[test]
    fn append_pushes_values_and_nulls() -> Result<()> {
        let conn = connection()?;
        let mut chunk = chunk::<i64>(&conn)?;
        let mut vector = chunk.get_vector_at_mut::<i64>(0)?;
        let pushed = vector.append(6, |rows| {
            rows.push(&7)?;
            rows.push_null()?;
            rows.push_option(Some(&8))?;
            rows.push_option(None)?;
            rows.push_with(2, |index| index as i64 + 100)?;
            assert_eq!(rows.remaining(), 0);
            assert!(rows.push(&9).is_err());
            assert!(rows.push_null().is_err());
            assert!(rows.push_with(1, |_| 9).is_err());
            rows.push_with(0, |_| unreachable!())?;
            Ok(6 - rows.remaining())
        })?;
        assert_eq!(pushed, 6);
        assert_eq!(vector.len(), 6);
        assert_eq!(
            vector.reader()?.iter().collect::<Vec<_>>(),
            [Some(&7), None, Some(&8), None, Some(&100), Some(&101)]
        );
        Ok(())
    }

    #[test]
    fn append_follows_existing_rows_and_their_mask() -> Result<()> {
        let conn = connection()?;
        let mut chunk = chunk::<i32>(&conn)?;
        let mut vector = chunk.get_vector_at_mut::<i32>(0)?;
        vector.set_size(2)?;
        vector.writer()?.set_null(1)?;
        vector.set_size(70)?;
        vector.writer()?.set_null(69)?;
        vector.set_size(2)?;
        // The mask exists, and rows 2..70 of it were left NULL or valid by earlier writes.
        vector.append(100, |rows| rows.push_with(100, |index| index as i32))?;
        let read = vector.reader()?.iter().map(|value| value.copied()).collect::<Vec<_>>();
        assert_eq!(read.len(), 102);
        assert_eq!(read[..2], [Some(0), None]);
        assert!(
            read[2..]
                .iter()
                .enumerate()
                .all(|(index, value)| *value == Some(index as i32))
        );
        Ok(())
    }

    #[test]
    fn append_ends_with_the_rows_pushed() -> Result<()> {
        let conn = connection()?;
        let mut chunk = chunk::<String>(&conn)?;
        let mut vector = chunk.get_vector_at_mut::<String>(0)?;
        vector.append(2048, |rows| {
            rows.push("short")?;
            rows.push_null()?;
            rows.push(LONG)
        })?;
        assert_eq!(vector.len(), 3);
        vector.append(0, |_| Ok(()))?;
        assert_eq!(vector.len(), 3);
        vector.append(5, |rows| rows.push(""))?;
        assert_eq!(
            vector.reader()?.iter().collect::<Vec<_>>(),
            [Some("short"), None, Some(LONG), Some("")]
        );
        Ok(())
    }

    #[test]
    fn append_keeps_the_rows_pushed_before_an_error() -> Result<()> {
        let conn = connection()?;
        let mut chunk = chunk::<String>(&conn)?;
        let mut vector = chunk.get_vector_at_mut::<String>(0)?;
        let error = vector
            .append(10, |rows| {
                rows.push(LONG)?;
                Err::<(), _>(crate::Error::invalid_input("stop"))
            })
            .unwrap_err();
        assert!(error.message.contains("stop"), "{error}");
        assert_eq!(vector.reader()?.iter().collect::<Vec<_>>(), [Some(LONG)]);
        Ok(())
    }

    #[test]
    fn append_keeps_the_rows_pushed_before_a_panic() -> Result<()> {
        let conn = connection()?;
        let mut chunk = DataChunk::create(&conn, &[String::logical_type(&conn)?, i64::logical_type(&conn)?], true)?;
        let mut strings = chunk.get_vector_at_mut::<String>(0)?;
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            strings.append(10, |rows| {
                rows.push(LONG)?;
                rows.push_null()?;
                panic!("in the closure");
                #[allow(unreachable_code)]
                Ok(())
            })
        }));
        assert!(panicked.is_err());
        assert_eq!(strings.reader()?.iter().collect::<Vec<_>>(), [Some(LONG), None]);
        let mut numbers = chunk.get_vector_at_mut::<i64>(1)?;
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            numbers.append(10, |rows| {
                rows.push(&1)?;
                rows.push_with(5, |index| if index == 3 { panic!("in push_with") } else { 2 })
            })
        }));
        assert!(panicked.is_err());
        assert_eq!(numbers.reader()?.iter().collect::<Vec<_>>(), [Some(&1)]);
        Ok(())
    }

    #[test]
    fn append_rejects_nested_and_read_only_vectors() -> Result<()> {
        let conn = connection()?;
        let types = [
            List::<i32>::logical_type(&conn)?,
            LogicalType::from_text(&conn, "STRUCT(a INTEGER)")?,
            i32::logical_type(&conn)?,
        ];
        let mut chunk = DataChunk::create(&conn, &types, true)?;
        let mut lists = chunk.get_vector_at_mut::<List<i32>>(0)?;
        let error = lists.append(1, |_| Ok(())).unwrap_err();
        assert!(error.message.contains("cannot append"), "{error}");
        assert_eq!(lists.len(), 0);
        let mut structs = chunk.get_vector_at_mut::<Struct>(1)?;
        assert!(structs.append(1, |_| Ok(())).is_err());
        let mut constant = chunk
            .get_vector_at_mut::<i32>(2)?
            .make_constant(42_i32.value(&conn)?, 1)?;
        assert!(constant.append(1, |rows| rows.push(&1)).is_err());
        Ok(())
    }
}
