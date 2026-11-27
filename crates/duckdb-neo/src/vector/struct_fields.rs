//! `STRUCT` vectors of plain fields, written a whole row at a time through raw columns.

use std::ptr::NonNull;

use super::{NativeElement, PlainNative, Unknown, Vector, VectorWriter, out_of_bounds};
use crate::{Result, data_chunk::VectorCollection, error::Error, logical_type::LogicalTypeID};

/// Plain field types of a `STRUCT`, in order, as a tuple such as `(f64, f64)`, for [`StructFieldsWriter`] and
/// [`StructListAppender`](super::StructListAppender).
///
/// Fields match by position and type, not by name.
///
/// # Safety
/// The writers trust these methods: `writers` and `refresh` must check every field's type and row count, and
/// `columns` must return each field's data as slots of its type. The trait is sealed to the tuples below.
pub unsafe trait PlainFields: sealed::Sealed + Copy {
    #[doc(hidden)]
    type Writers<'w>;
    #[doc(hidden)]
    type Columns: Copy;
    #[doc(hidden)]
    const FIELDS: usize;

    /// Prepare one writer per field, failing unless the fields are exactly `Self`'s types, with `len` rows each.
    ///
    /// # Safety
    /// `fields` must be the children of a `STRUCT` vector, exclusively borrowed for `'w`.
    #[doc(hidden)]
    unsafe fn writers<'w>(fields: &mut VectorCollection, len: usize) -> Result<Self::Writers<'w>>;

    /// Reacquire every field's buffers, failing unless each has `len` rows.
    #[doc(hidden)]
    fn refresh(writers: &mut Self::Writers<'_>, len: usize) -> Result<()>;

    #[doc(hidden)]
    fn columns(writers: &mut Self::Writers<'_>) -> Self::Columns;

    /// # Safety
    /// `end` must not exceed any field's length, and the rows must be written before anything reads them.
    #[doc(hidden)]
    unsafe fn mark_valid_range(writers: &mut Self::Writers<'_>, start: usize, end: usize);

    /// # Safety
    /// `index` must be a row of every column.
    #[doc(hidden)]
    unsafe fn write(columns: Self::Columns, index: usize, value: Self);

    /// # Safety
    /// `index` must be a row of every field, holding a written value.
    #[doc(hidden)]
    unsafe fn mark_valid(writers: &mut Self::Writers<'_>, index: usize);

    /// # Safety
    /// `index` must be a row of every field.
    #[doc(hidden)]
    unsafe fn mark_null(writers: &mut Self::Writers<'_>, index: usize) -> Result<()>;

    /// Allocate every field's validity mask, so that stores to it cannot fail.
    #[doc(hidden)]
    fn allocate_validity(writers: &mut Self::Writers<'_>) -> Result<()>;

    /// # Safety
    /// As for [`VectorWriter::store_validity_word`], for every field.
    #[doc(hidden)]
    unsafe fn store_validity_word(writers: &mut Self::Writers<'_>, word: usize, bits: u64, rows: u64) -> Result<()>;

    /// # Safety
    /// `start..end` must be rows of every column.
    #[doc(hidden)]
    unsafe fn zero(columns: Self::Columns, start: usize, end: usize);
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! plain_fields {
    ($($field:ident $index:tt),+) => {
        impl<$($field: PlainNative + 'static),+> sealed::Sealed for ($($field,)+) {}

        // SAFETY: `writers` validates each field as its type and checks the row counts, as `refresh` does again, and
        // `columns` returns each writer's data.
        unsafe impl<$($field: PlainNative + 'static),+> PlainFields for ($($field,)+) {
            type Writers<'w> = ($(VectorWriter<'w, $field>,)+);
            type Columns = ($(*mut $field,)+);
            const FIELDS: usize = [$($index),+].len();

            unsafe fn writers<'w>(fields: &mut VectorCollection, len: usize) -> Result<Self::Writers<'w>> {
                let mut children = fields.cached_iter_mut()?;
                let writers = ($({
                    let child = children.next().ok_or_else(|| field_count_mismatch(Self::FIELDS))?;
                    child.validate_as::<$field>()?;
                    // SAFETY: the caller guarantees an exclusive borrow of the STRUCT, whose field wrappers outlive the
                    // writer; the child was just validated as the field's type.
                    unsafe { VectorWriter::from_raw(NonNull::from(child).cast::<Vector<'w, $field>>())? }
                },)+);
                if children.next().is_some() {
                    return Err(field_count_mismatch(Self::FIELDS));
                }
                $(check_field_len(writers.$index.len(), len)?;)+
                Ok(writers)
            }

            fn refresh(writers: &mut Self::Writers<'_>, len: usize) -> Result<()> {
                $(
                    writers.$index.refresh_all()?;
                    check_field_len(writers.$index.len(), len)?;
                )+
                Ok(())
            }

            #[inline(always)]
            fn columns(writers: &mut Self::Writers<'_>) -> Self::Columns {
                ($(writers.$index.as_mut_ptr().cast::<$field>(),)+)
            }

            unsafe fn mark_valid_range(writers: &mut Self::Writers<'_>, start: usize, end: usize) {
                // SAFETY: the caller guarantees the rows are in bounds and written before any read.
                unsafe { $(writers.$index.mark_valid_range(start, end);)+ }
            }

            #[inline(always)]
            unsafe fn write(columns: Self::Columns, index: usize, value: Self) {
                // SAFETY: the caller guarantees the row is in bounds, and each column holds slots of its field's type.
                unsafe { $(columns.$index.add(index).write(value.$index);)+ }
            }

            #[inline(always)]
            unsafe fn mark_valid(writers: &mut Self::Writers<'_>, index: usize) {
                // SAFETY: the caller guarantees the row is in bounds and holds a value.
                unsafe { $(writers.$index.mark_valid_unchecked(index);)+ }
            }

            #[inline(always)]
            unsafe fn mark_null(writers: &mut Self::Writers<'_>, index: usize) -> Result<()> {
                // SAFETY: the caller guarantees the row is in bounds.
                unsafe { $(writers.$index.mark_null_unchecked(index)?;)+ }
                Ok(())
            }

            fn allocate_validity(writers: &mut Self::Writers<'_>) -> Result<()> {
                $(writers.$index.allocated_validity()?;)+
                Ok(())
            }

            #[inline(always)]
            unsafe fn store_validity_word(
                writers: &mut Self::Writers<'_>,
                word: usize,
                bits: u64,
                rows: u64,
            ) -> Result<()> {
                // SAFETY: the caller upholds the contract for every field.
                unsafe { $(writers.$index.store_validity_word(word, bits, rows)?;)+ }
                Ok(())
            }

            unsafe fn zero(columns: Self::Columns, start: usize, end: usize) {
                // SAFETY: the caller guarantees the rows are in bounds; zero bytes are a valid `PlainNative` value.
                unsafe { $(columns.$index.add(start).write_bytes(0, end - start);)+ }
            }
        }
    };
}

plain_fields!(A 0);
plain_fields!(A 0, B 1);
plain_fields!(A 0, B 1, C 2);
plain_fields!(A 0, B 1, C 2, D 3);
plain_fields!(A 0, B 1, C 2, D 3, E 4);
plain_fields!(A 0, B 1, C 2, D 3, E 4, G 5);

pub(super) fn check_field_len(field: usize, len: usize) -> Result<()> {
    if field != len {
        return Err(Error::invalid_input(format!(
            "a STRUCT field has {field} rows, not the STRUCT's {len}"
        )));
    }
    Ok(())
}

#[cold]
pub(super) fn field_count_mismatch(fields: usize) -> Error {
    Error::invalid_input(format!("the STRUCT does not have exactly {fields} fields"))
}

/// Whether vectors of this type are `STRUCT`s of their child vectors; `TUPLE` shares the layout.
pub(super) fn is_struct(id: LogicalTypeID) -> bool {
    matches!(id, LogicalTypeID::Struct | LogicalTypeID::Tuple)
}

/// Writes the rows of a `STRUCT` vector whose fields are the plain types `F`, each row as one tuple, borrowed by
/// [`Vector::struct_fields_writer`].
///
/// Fields match by position and type, not by name: `(f64, f64)` writes `STRUCT(y DOUBLE, x DOUBLE)` with `y` first.
///
/// ```
/// # use duckdb_neo::{Result, data_chunk::DataChunk, environment::{Environment, StorageLocation}, logical_type::LogicalType, types::Struct};
/// # fn main() -> Result<()> {
/// # let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
/// let mut chunk = DataChunk::create(&conn, &[LogicalType::from_text(&conn, "STRUCT(x DOUBLE, n INTEGER)")?], true)?;
/// let mut vector = chunk.get_vector_at_mut::<Struct>(0)?;
/// vector.set_size(3)?;
/// let mut writer = vector.struct_fields_writer()?;
/// writer.set(0, (0.5, 1))?;
/// writer.write_from([None, Some((2.5, 3))])?;
/// # drop(writer);
/// assert_eq!(vector.get_value_slow(1)?.dbg_string()?, "{'x': 2.5, 'n': 3}");
/// assert_eq!(vector.get_value_slow(0)?.dbg_string()?, "NULL");
/// # Ok(())
/// # }
/// ```
pub struct StructFieldsWriter<'w, F: PlainFields> {
    rows: VectorWriter<'w, Unknown>,
    fields: F::Writers<'w>,
    /// The fields' data, which no write replaces.
    columns: F::Columns,
}

impl<T: NativeElement> Vector<'_, T> {
    /// Prepare a [`StructFieldsWriter`]; fails unless this is a writable `STRUCT` or `TUPLE` whose fields are
    /// exactly `F`'s types, in order.
    pub fn struct_fields_writer<F: PlainFields>(&mut self) -> Result<StructFieldsWriter<'_, F>> {
        if !is_struct(self.logical_type().id()?) {
            return Err(Error::invalid_input(format!("{} is not a STRUCT", self.logical_type())));
        }
        let len = self.len();
        let mut vector = NonNull::from(self).cast::<Vector<'_, Unknown>>();
        // SAFETY: the exclusive borrow covers the writer, which touches the fields only through their own writers;
        // `F::writers` checks their types and sizes.
        let (rows, mut fields) = unsafe {
            let rows = VectorWriter::from_raw(vector)?;
            let fields = vector
                .as_mut()
                .children_collection_mut()
                .ok_or_else(|| field_count_mismatch(F::FIELDS))?;
            (rows, F::writers(fields, len)?)
        };
        Ok(StructFieldsWriter {
            rows,
            columns: F::columns(&mut fields),
            fields,
        })
    }
}

impl<F: PlainFields> StructFieldsWriter<'_, F> {
    /// Return the number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Return whether the vector has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write a valid row from its fields.
    #[inline(always)]
    pub fn set(&mut self, row: usize, value: F) -> Result<()> {
        if row >= self.len() {
            return Err(out_of_bounds(row, self.len()));
        }
        // SAFETY: the row is in bounds of the STRUCT and of every field, which `F::writers` checked has as many rows,
        // and the fields hold its values once written.
        unsafe {
            F::write(self.columns, row, value);
            F::mark_valid(&mut self.fields, row);
            self.rows.mark_valid_unchecked(row);
        }
        Ok(())
    }

    /// Make a row SQL NULL, along with its fields.
    #[inline]
    pub fn set_null(&mut self, row: usize) -> Result<()> {
        if row >= self.len() {
            return Err(out_of_bounds(row, self.len()));
        }
        // SAFETY: as for `set`.
        unsafe {
            self.rows.mark_null_unchecked(row)?;
            F::mark_null(&mut self.fields, row)
        }
    }

    /// Write a row; `None` writes SQL NULL.
    #[inline(always)]
    pub fn set_option(&mut self, row: usize, value: Option<F>) -> Result<()> {
        match value {
            Some(value) => self.set(row, value),
            None => self.set_null(row),
        }
    }

    /// Write the first rows from `values`, in one loop; `None` writes SQL NULL. Fails if there are more values
    /// than rows: before writing any when the iterator's size hint says so, else after writing the rows that fit.
    #[inline]
    pub fn write_from(&mut self, values: impl IntoIterator<Item = Option<F>>) -> Result<()> {
        let (len, columns) = (self.len(), self.columns);
        let mut values = values.into_iter();
        if values.size_hint().0 > len {
            return Err(out_of_bounds(len, len));
        }
        let mut start = 0;
        while start < len {
            let end = len.min(start + 64);
            let mut bits = 0_u64;
            let written = values.by_ref().take(end - start).fold(start, |row, value| {
                if let Some(value) = value {
                    // SAFETY: the row is in bounds of the STRUCT and of every field, which `F::writers` checked has
                    // as many rows; `flush_validity` marks it valid.
                    unsafe { F::write(columns, row, value) };
                    bits |= 1 << (row - start);
                }
                row + 1
            });
            // SAFETY: rows `start..written` are in bounds, and the valid ones were just written.
            unsafe { self.flush_validity(start, written, bits)? };
            if written < end {
                return Ok(());
            }
            start = end;
        }
        match values.next() {
            Some(_) => Err(out_of_bounds(len, len)),
            None => Ok(()),
        }
    }

    /// Mark rows `start..end`, at most 64 from a multiple of 64, valid where `bits` has bit `row - start` set and
    /// NULL elsewhere. If it fails, every row keeps its validity.
    ///
    /// # Safety
    /// The rows must be in bounds, and the valid ones written.
    #[inline(always)]
    unsafe fn flush_validity(&mut self, start: usize, end: usize, bits: u64) -> Result<()> {
        if start == end {
            return Ok(());
        }
        let rows = u64::MAX >> (64 - (end - start));
        if bits != rows {
            self.rows.allocated_validity()?;
            F::allocate_validity(&mut self.fields)?;
        }
        // SAFETY: the caller guarantees the rows are in bounds of the STRUCT and of every field, which `F::writers`
        // checked has as many rows, and that the valid ones were written. Every mask a NULL row needs exists, so
        // these stores allocate nothing and cannot fail part way.
        unsafe {
            self.rows.store_validity_word(start / 64, bits, rows)?;
            F::store_validity_word(&mut self.fields, start / 64, bits, rows)
        }
    }
}

impl<F: PlainFields> std::fmt::Debug for StructFieldsWriter<'_, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructFieldsWriter").field("len", &self.len()).finish()
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use crate::{
        Parameters, Result,
        connection::Connection,
        data_chunk::DataChunk,
        environment::{Environment, StorageLocation},
        logical_type::LogicalType,
        scalar::MicroScalar,
        signature::{FunctionParameter, SignatureBuilder},
        types::Struct,
        vector::{CombinedVectorIterator, Vector},
    };

    fn connection() -> Result<Connection> {
        Environment::new()?.open(StorageLocation::InMemory)?.connect()
    }

    fn pairs(conn: &Connection) -> Result<DataChunk<'_>> {
        DataChunk::create(
            conn,
            &[LogicalType::from_text(conn, "STRUCT(x DOUBLE, n INTEGER)")?],
            true,
        )
    }

    fn read(vector: &Vector<'_, Struct>) -> Result<Vec<String>> {
        (0..vector.len())
            .map(|row| vector.get_value_slow(row)?.dbg_string())
            .collect()
    }

    #[test]
    fn struct_fields_writer_writes_rows_and_nulls() -> Result<()> {
        let conn = connection()?;
        let mut chunk = pairs(&conn)?;
        let mut vector = chunk.get_vector_at_mut::<Struct>(0)?;
        vector.set_size(4)?;
        {
            let mut writer = vector.struct_fields_writer()?;
            assert_eq!((writer.len(), writer.is_empty()), (4, false));
            writer.set(0, (0.5, 1))?;
            writer.set_null(1)?;
            writer.set_option(2, Some((2.5, 3)))?;
            writer.set_option(3, None)?;
            // A NULL row becomes valid again, fields included.
            writer.set(1, (1.5, 2))?;
            assert!(writer.set(4, (0.0, 0)).is_err());
            assert!(writer.set_null(4).is_err());
        }
        assert_eq!(
            read(&vector)?,
            ["{'x': 0.5, 'n': 1}", "{'x': 1.5, 'n': 2}", "{'x': 2.5, 'n': 3}", "NULL"]
        );
        let fields = vector.children().unwrap();
        assert_eq!(fields.get_vector_at::<i32>(1)?.reader()?.get(3)?, None);
        Ok(())
    }

    #[test]
    fn write_from_fills_rows_in_order() -> Result<()> {
        let conn = connection()?;
        let mut chunk = pairs(&conn)?;
        let mut vector = chunk.get_vector_at_mut::<Struct>(0)?;
        vector.set_size(300)?;
        let value = |row: usize| (row % 7 != 3).then_some((row as f64, row as i32));
        vector.struct_fields_writer()?.write_from((0..300).map(value))?;
        let expected = |row: usize| match value(row) {
            Some((x, n)) => format!("{{'x': {x:.1}, 'n': {n}}}"),
            None => "NULL".to_owned(),
        };
        assert_eq!(read(&vector)?, (0..300).map(expected).collect::<Vec<_>>());
        // Rewriting keeps the mask: rows written NULL before become valid, and unwritten rows stay as they were.
        vector
            .struct_fields_writer()?
            .write_from((0..100).map(|row| Some((-(row as f64), 0))))?;
        let read = read(&vector)?;
        assert!((0..100).all(|row| read[row] == format!("{{'x': {:.1}, 'n': 0}}", -(row as f64))));
        assert!((100..300).all(|row| read[row] == expected(row)));
        let mut writer = vector.struct_fields_writer::<(f64, i32)>()?;
        writer.write_from(std::iter::empty())?;
        let error = writer.write_from((0..301).map(|_| None)).unwrap_err();
        assert!(error.message.contains("300"), "{error}");
        Ok(())
    }

    #[test]
    fn write_from_keeps_rows_valid_after_a_panic() -> Result<()> {
        let conn = connection()?;
        let mut chunk = pairs(&conn)?;
        let mut vector = chunk.get_vector_at_mut::<Struct>(0)?;
        vector.set_size(100)?;
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            vector.struct_fields_writer()?.write_from((0..100).map(|row| {
                if row == 80 {
                    panic!("in the iterator")
                }
                (row % 2 == 0).then_some((1.0, 1))
            }))
        }));
        assert!(panicked.is_err());
        let read = read(&vector)?;
        // The first 64 rows were finished; the rest hold zeroed or written fields, valid as they were.
        assert!((0..64).all(|row| read[row] == if row % 2 == 0 { "{'x': 1.0, 'n': 1}" } else { "NULL" }));
        assert!((64..100).all(|row| read[row] == "{'x': 1.0, 'n': 1}" || read[row] == "{'x': 0.0, 'n': 0}"));
        Ok(())
    }

    #[test]
    fn struct_fields_writer_rejects_other_vectors() -> Result<()> {
        let conn = connection()?;
        let types = [
            "STRUCT(x DOUBLE, n INTEGER)",
            "DOUBLE[]",
            "UNION(x DOUBLE, n INTEGER)",
            "STRUCT(x DOUBLE[])",
        ]
        .map(|name| LogicalType::from_text(&conn, name));
        let mut chunk = DataChunk::create(&conn, &types.into_iter().collect::<Result<Vec<_>>>()?, true)?;
        let mut vector = chunk.get_vector_at_mut::<Struct>(0)?;
        let message = |error: crate::Error| error.message;
        let too_few = message(vector.struct_fields_writer::<(f64,)>().unwrap_err());
        assert!(too_few.contains("does not have exactly 1 fields"), "{too_few}");
        let too_many = message(vector.struct_fields_writer::<(f64, i32, i32)>().unwrap_err());
        assert!(too_many.contains("does not have exactly 3 fields"), "{too_many}");
        assert!(vector.struct_fields_writer::<(f64, i64)>().is_err());
        assert!(vector.struct_fields_writer::<(f64, i32)>().is_ok());
        // A constant vector is not writable.
        vector.set_size(1)?;
        let value = vector.get_value_slow(0)?;
        let mut constant = vector.make_constant(&value, 1)?;
        assert!(constant.struct_fields_writer::<(f64, i32)>().is_err());
        for column in 1..4 {
            let mut vector = chunk.get_vector_at_mut::<crate::vector::Unknown>(column)?;
            assert!(vector.struct_fields_writer::<(f64,)>().is_err(), "column {column}");
            assert!(vector.struct_fields_writer::<(f64, i32)>().is_err(), "column {column}");
        }
        Ok(())
    }

    #[test]
    fn write_from_stores_each_validity_word_exactly() -> Result<()> {
        let conn = connection()?;
        let mut chunk = pairs(&conn)?;
        let mut vector = chunk.get_vector_at_mut::<Struct>(0)?;
        vector.set_size(200)?;
        vector
            .struct_fields_writer()?
            .write_from((0..200).map(|row| Some((row as f64, 1))))?;
        // Without NULLs no mask is allocated, so readers keep their all-valid paths.
        let masks = |vector: &Vector<'_, Struct>| -> Result<[bool; 3]> {
            let fields = vector.children().unwrap();
            Ok([
                vector.reader()?.validity().is_some(),
                fields.get_vector_at::<f64>(0)?.reader()?.validity().is_some(),
                fields.get_vector_at::<i32>(1)?.reader()?.validity().is_some(),
            ])
        };
        assert_eq!(masks(&vector)?, [false; 3]);
        let first = |row: usize| row.is_multiple_of(5);
        vector
            .struct_fields_writer()?
            .write_from((0..200).map(|row| (!first(row)).then_some((row as f64, 2))))?;
        assert_eq!(masks(&vector)?, [true; 3]);
        // Rows NULL before become valid, and valid rows become NULL, in the STRUCT and in every field.
        vector
            .struct_fields_writer()?
            .write_from((0..200).map(|row| first(row).then_some((row as f64, 3))))?;
        let fields = vector.children().unwrap();
        let (rows, xs, ns) = (
            vector.reader()?,
            fields.get_vector_at::<f64>(0)?,
            fields.get_vector_at::<i32>(1)?,
        );
        let (xs, ns) = (xs.reader()?, ns.reader()?);
        for row in 0..200 {
            let valid = first(row);
            assert_eq!(!rows.is_null(row)?, valid, "row {row}");
            assert_eq!(xs.get(row)?.copied(), valid.then_some(row as f64), "row {row}");
            assert_eq!(ns.get(row)?.copied(), valid.then_some(3), "row {row}");
        }
        Ok(())
    }

    #[test]
    fn struct_fields_writer_handles_zero_rows_six_fields_and_tuples() -> Result<()> {
        let conn = connection()?;
        let six = LogicalType::from_text(
            &conn,
            "STRUCT(a TINYINT, b SMALLINT, c INTEGER, d BIGINT, e FLOAT, f DOUBLE)",
        )?;
        let tuple = <(f64, i32) as crate::DuckDBType>::logical_type(&conn)?;
        assert_eq!(tuple.id()?, crate::logical_type::LogicalTypeID::Tuple);
        let mut chunk = DataChunk::create(&conn, &[six, tuple], true)?;
        let mut empty = chunk.get_vector_at_mut::<Struct>(0)?;
        {
            let mut writer = empty.struct_fields_writer::<(i8, i16, i32, i64, f32, f64)>()?;
            assert!(writer.is_empty());
            writer.write_from(std::iter::empty())?;
            assert!(writer.set(0, (1, 2, 3, 4, 5.0, 6.0)).is_err());
            assert!(writer.set_null(0).is_err());
            assert!(writer.write_from([None]).is_err());
        }
        empty.set_size(2)?;
        empty
            .struct_fields_writer()?
            .write_from([Some((1_i8, 2_i16, 3_i32, 4_i64, 5.5_f32, 6.5_f64)), None])?;
        assert_eq!(
            empty.get_value_slow(0)?.dbg_string()?,
            "{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5.5, 'f': 6.5}"
        );
        assert_eq!(empty.get_value_slow(1)?.dbg_string()?, "NULL");
        let mut tuples = chunk.get_vector_at_mut::<Struct>(1)?;
        tuples.set_size(2)?;
        tuples.struct_fields_writer()?.write_from([None, Some((1.5, 7))])?;
        assert_eq!(tuples.get_value_slow(0)?.get::<(Option<f64>, Option<i32>)>()?, None);
        assert_eq!(
            tuples.get_value_slow(1)?.get::<(Option<f64>, Option<i32>)>()?,
            Some((Some(1.5), Some(7)))
        );
        Ok(())
    }

    #[test]
    fn write_from_fails_before_writing_when_the_size_hint_is_too_long() -> Result<()> {
        let conn = connection()?;
        let mut chunk = pairs(&conn)?;
        let mut vector = chunk.get_vector_at_mut::<Struct>(0)?;
        vector.set_size(2)?;
        let error = vector
            .struct_fields_writer()?
            .write_from([Some((1.0, 1)), Some((2.0, 2)), None])
            .unwrap_err();
        assert!(error.message.contains('2'), "{error}");
        assert_eq!(read(&vector)?, ["{'x': 0.0, 'n': 0}", "{'x': 0.0, 'n': 0}"]);
        Ok(())
    }

    #[test]
    fn child_structs_use_the_fields_writer() -> Result<()> {
        let conn = connection()?;
        let types = [LogicalType::from_text(&conn, "STRUCT(x DOUBLE, n INTEGER)[]")?];
        let mut chunk = DataChunk::create(&conn, &types, true)?;
        let mut lists = chunk.get_vector_at_mut::<crate::types::List<Struct>>(0)?;
        lists.set_size(1)?;
        lists.set_child_size(2)?;
        lists.list_writer()?.set_entry(0, crate::types::ListIndex::new(0, 2))?;
        lists
            .children_mut()
            .unwrap()
            .get_vector_at_mut::<Struct>(0)?
            .struct_fields_writer()?
            .write_from([Some((1.0, 2)), None])?;
        assert_eq!(lists.get_value_slow(0)?.dbg_string()?, "[{'x': 1.0, 'n': 2}, NULL]");
        Ok(())
    }

    #[test]
    fn write_from_reads_constant_and_dictionary_inputs() -> Result<()> {
        let conn = connection()?;
        let signature = SignatureBuilder::new()
            .param(FunctionParameter::new::<f64>("x"))
            .param(FunctionParameter::new::<i32>("n"))
            .return_type(LogicalType::from_text(&conn, "STRUCT(x DOUBLE, n INTEGER)")?);
        MicroScalar::with_signature("make_pair", signature, |_ctx, input, mut output: Vector<'_, Struct>| {
            let [xs, ns] = input.get_vector_array_mut([0, 1])?;
            let (xs, ns) = (xs.cast::<f64>()?, ns.cast::<i32>()?);
            let rows = CombinedVectorIterator::new((xs.reader()?, ns.reader()?))?;
            output
                .struct_fields_writer()?
                .write_from(rows.map(|(x, n)| Some((*x?, *n?))))
        })
        .register(&conn)?;
        let chunk = conn
            .query(
                "SELECT count(*) FILTER (WHERE make_pair(x, 7) IS DISTINCT FROM
                                         CASE WHEN x IS NOT NULL THEN {'x': x, 'n': 7} END)
                      + count(*) FILTER (WHERE make_pair(x, n) IS DISTINCT FROM
                                         CASE WHEN x IS NOT NULL AND n IS NOT NULL THEN {'x': x, 'n': n} END)
                      + count(*) FILTER (WHERE make_pair(NULL, n) IS NOT NULL)
                 FROM (SELECT CASE WHEN i % 5 = 0 THEN NULL ELSE i / 4 END::DOUBLE AS x,
                              CASE WHEN i % 11 = 0 THEN NULL ELSE i END::INTEGER AS n, i
                       FROM range(5000) r(i)) WHERE i % 3 <> 1",
                Parameters::None,
            )?
            .next_chunk()?
            .unwrap();
        assert_eq!(chunk.get_vector_at::<i64>(0)?.reader()?.get(0)?, Some(&0));
        Ok(())
    }
}
