//! Arrow C Data Interface conversion.
//!
//! Exported `ArrowSchema` and `ArrowArray` values follow Arrow's `release`
//! callback ownership convention. Importing an array transfers its buffers to
//! the resulting [`crate::data_chunk::DataChunk`].

use libduckdb_sys::DuckDBStr;

use crate::{
    Result, check_api_call, check_api_call_no_err,
    connection::Context,
    data_chunk::{DataChunk, DataChunkRef},
    ffi,
    logical_type::LogicalType,
    schema::Schema,
};

pub struct ArrowExporter {
    handle: ffi::duckdb_v2_arrow_exporter_handle,
}

impl ArrowExporter {
    pub fn new(
        context: &Context,
        logical_types: &[LogicalType],
        names: &[String],
        count: usize,
        batch_size: Option<usize>,
    ) -> Result<Self> {
        let logical_type_handles = logical_types.iter().map(|x| x.handle).collect::<Vec<_>>();
        let name_ptrs = names.iter().map(|x| x.into()).collect::<Vec<DuckDBStr>>();

        let handle = check_api_call!(
            ffi::duckdb_v2_arrow_exporter_create,
            **context,
            logical_type_handles.as_ptr(),
            name_ptrs.as_ptr(),
            count as u64,
            batch_size.unwrap_or(0) as u64,
            RET
        )?;

        Ok(ArrowExporter { handle })
    }

    pub fn append(&mut self, mut chunk: DataChunkRef, flush: bool) -> Result<()> {
        check_api_call!(
            ffi::duckdb_v2_arrow_exporter_append,
            self.handle,
            &mut chunk.handle,
            false,
            flush
        )
    }

    pub fn get_schema(&self) -> Result<ffi::ArrowSchema> {
        check_api_call!(ffi::duckdb_v2_arrow_exporter_get_schema, self.handle, RET)
    }

    pub fn next_array(&self) -> Result<ffi::ArrowArray> {
        check_api_call!(ffi::duckdb_v2_arrow_exporter_next_array, self.handle, RET)
    }
}

impl Drop for ArrowExporter {
    fn drop(&mut self) {
        check_api_call_no_err!(ffi::duckdb_v2_arrow_exporter_destroy, &mut self.handle).unwrap();
    }
}

pub struct ArrowImporter {
    handle: ffi::duckdb_v2_arrow_importer_handle,
}

impl ArrowImporter {
    pub fn new(context: Context, schema: &mut ffi::ArrowSchema, batch_size: Option<usize>) -> Result<Self> {
        let handle = check_api_call!(
            ffi::duckdb_v2_arrow_importer_create,
            *context,
            schema,
            batch_size.unwrap_or(0) as u64,
            RET
        )?;
        Ok(ArrowImporter { handle })
    }

    pub fn append(&self, array: &mut ffi::ArrowArray, flush: bool) -> Result<()> {
        check_api_call!(ffi::duckdb_v2_arrow_importer_append, self.handle, array, false, flush)
    }

    pub fn schema(&self) -> Result<Schema> {
        Ok(Schema {
            handle: check_api_call!(ffi::duckdb_v2_arrow_importer_get_schema, self.handle, RET)?,
        })
    }

    pub fn chunk(&self) -> Result<DataChunk> {
        Ok(DataChunk::new(
            check_api_call!(ffi::duckdb_v2_arrow_importer_next_chunk, self.handle, RET)?,
            true,
        ))
    }
}

impl Drop for ArrowImporter {
    fn drop(&mut self) {
        check_api_call_no_err!(ffi::duckdb_v2_arrow_importer_destroy, &mut self.handle).unwrap();
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(false)]
mod tests {
    use crate::{
        DuckDBType, Environment, Parameters, StorageLocation,
        arrow::{ConversionPlan, logical_types_to_arrow_schema},
        builder_helpers::scalar_callback,
        scalar::ScalarFunctionBuilder,
        signature::{Parameter, SignatureBuilder},
    };

    scalar_callback!(ToArrowTest, i64, |input, result, ctx, user_data| {
        let logical_types = input
            .vectors()?
            .iter()
            .map(|v| v.logical_type().clone())
            .collect::<Vec<_>>();

        let mut result = result;

        let mut arrow_schema = logical_types_to_arrow_schema(&ctx, &logical_types)?;

        let conversion_plan = ConversionPlan::new(&ctx, &mut arrow_schema)?;

        unsafe {
            arrow_schema.release.unwrap()(&mut arrow_schema);
        }

        let schema = conversion_plan.schema()?;

        println!("Schema: {:?}", schema.get_all()?);

        let mut arrow_array = input.to_arrow_array(&ctx)?;

        dbg!(arrow_array);

        let data_chunk = conversion_plan.to_data_chunk(&ctx, &mut arrow_array)?;

        assert_eq!(arrow_array.release.is_none(), true);

        assert_eq!(data_chunk.row_count()?, input.row_count()?);
        assert_eq!(data_chunk.vectors_count()?, input.vectors_count()?);

        result.set_size(data_chunk.row_count()?)?;

        let vec1 = data_chunk.get_vector_at::<i64>(0)?;
        let vec2 = data_chunk.get_vector_at::<bool>(1)?;
        let vec3 = data_chunk.get_vector_at::<String>(2)?;

        for i in 0..data_chunk.row_count()? {
            let val1 = vec1.get(i)?;
            let val2 = vec2.get(i)?;
            let val3 = vec3.get(i)?;

            println!("Row {}: val1={:?}, val2={:?}, val3={:?}", i, val1, val2, val3);

            result.write(
                i,
                Some(*val1.unwrap_or(&0) + *val2.unwrap_or(&false) as i64 + val3.unwrap_or_default().len() as i64),
            )?;
        }

        Ok(())
    });

    #[test]
    fn test_arrow_conversion() -> crate::Result<()> {
        let env = Environment::new()?;
        let db = env.open(StorageLocation::InMemory)?;
        let conn = db.connect()?;

        ScalarFunctionBuilder::new(
            "to_arrow",
            SignatureBuilder::new(
                [
                    Parameter::normal("val1", i64::logical_type(&conn)?),
                    Parameter::normal("val2", bool::logical_type(&conn)?),
                    Parameter::normal("val3", String::logical_type(&conn)?),
                ],
                i64::logical_type(&conn)?,
            ),
            ToArrowTest,
        )
        .register(&conn)?;

        let result = conn.query(
            "SELECT to_arrow(a, b, c) FROM (VALUES (2, true, 'hello'), (1, false, 'world')) AS t(a, b, c)",
            Parameters::None,
        )?;

        for chunk in result {
            let chunk = chunk?;

            let res = chunk.get_vector_at::<i64>(0)?;

            assert_eq!(*res.get(0)?.unwrap_or(&0), 2 + 1 + 5);
            assert_eq!(*res.get(1)?.unwrap_or(&0), 1 + 0 + 5);
        }

        Ok(())
    }
}
