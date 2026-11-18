use std::marker::PhantomData;

use crate::{
    AsRaw, FromRaw, connection::HasFactory, error::check_api_call_no_err, logical_type::LogicalType, raw::RawExt,
    value::Value,
};

use crate::{Result, error::check_api_call, ffi};

/// Nullability statistics of a function argument, function result, or table column.
///
/// Borrowed from DuckDB for the duration of a stats callback. Argument statistics
/// are read-only, and their setters fail.
#[derive(Debug, AsRaw, FromRaw)]
pub struct Stats<'a> {
    pub(crate) handle: ffi::duckdb_v2_stats_handle,
    _marker: PhantomData<&'a ()>,
    owned: bool,
}

impl Stats<'_> {
    pub fn new<C: HasFactory>(link: &C, logical_type: &LogicalType) -> Result<Self> {
        check_api_call!(
            ffi::duckdb_v2_stats_create,
            link.factory()?.raw(),
            logical_type.raw(),
            RET
        )
        .map(|handle| Stats {
            handle,
            _marker: PhantomData,
            owned: true,
        })
    }

    /// Whether a value can be NULL; `false` guarantees none is.
    pub fn can_have_null(&self) -> Result<bool> {
        check_api_call!(ffi::duckdb_v2_stats_can_have_null, self.raw(), RET)
    }

    /// Whether a value can be non-NULL; `false` guarantees every value is NULL.
    pub fn can_have_valid(&self) -> Result<bool> {
        check_api_call!(ffi::duckdb_v2_stats_can_have_valid, self.raw(), RET)
    }

    /// Set whether a value can be non-NULL; `false` is a guarantee, while `true` is always safe.
    pub fn set_have_valid(&self, value: bool) -> Result<()> {
        check_api_call!(ffi::duckdb_v2_stats_set_can_have_valid, self.raw(), value)
    }

    /// Set whether a value can be NULL; `false` is a guarantee, while `true` is always safe.
    pub fn set_have_null(&self, value: bool) -> Result<()> {
        check_api_call!(ffi::duckdb_v2_stats_set_can_have_null, self.raw(), value)
    }

    pub fn assign(&mut self, other: &Stats<'_>) -> Result<()> {
        check_api_call!(ffi::duckdb_v2_stats_assign, self.raw(), other.raw())
    }

    pub fn merge(&mut self, other: &Stats<'_>) -> Result<()> {
        check_api_call!(ffi::duckdb_v2_stats_merge, self.raw(), other.raw())
    }

    pub fn value(&self) -> Result<Value> {
        check_api_call!(ffi::duckdb_v2_stats_get_value, self.raw(), RET).map(|handle| Value { handle })
    }

    pub fn set_value(&mut self, value: &Value) -> Result<()> {
        check_api_call!(ffi::duckdb_v2_stats_set_value, self.raw(), value.raw())
    }
}

impl Drop for Stats<'_> {
    fn drop(&mut self) {
        if self.owned {
            check_api_call_no_err!(ffi::duckdb_v2_stats_destroy, &mut self.raw()).unwrap()
        }
    }
}
