//! Replacing unresolved table references with table functions, column data collections, or subqueries.

use crate::{
    Result,
    builder_helpers::{OpaqueHandle, get_user_data, handle_unwind},
    check_api_call,
    column_data_collection::ColumnDataCollection,
    connection::Context,
    ffi,
    handles::{ReplacementScanBuilderHandle, ReplacementScanBuilderLink},
    qualified_name::QualifiedName,
    value::Value,
};

/// Callback-scoped controls for claiming an unresolved table reference.
///
/// Use [`Self::set_reference`] to replace it with a table function, column data
/// collection, or subquery. Parameters can be added only after selecting a
/// table function.
pub struct ReplacementHandle<'a> {
    info: &'a ffi::duckdb_v2_replacement_scan_info_handle,
}

/// A replacement source for an unresolved table reference, selected with
/// [`ReplacementHandle::set_reference`].
pub enum ReplacementType<'a> {
    /// A table function's name, optionally qualified by schema and catalog.
    Table(QualifiedName),
    ColumnDataCollection((&'a ColumnDataCollection, Vec<String>)),
    Subquery(String),
}

impl<'a> ReplacementHandle<'a> {
    /// Append a positional table-function parameter.
    pub fn add_parameter(&self, value: Value) -> Result<()> {
        check_api_call!(ffi::duckdb_v2_replacement_scan_add_argument, *self.info, value.handle,)
    }

    /// Add a named table-function parameter.
    pub fn add_parameter_with_name(&self, name: &str, value: Value) -> Result<()> {
        check_api_call!(
            ffi::duckdb_v2_replacement_scan_add_named_argument,
            *self.info,
            name.into(),
            value.handle,
        )
    }

    /// Claim the reference with a table function, column data collection, or SELECT subquery.
    ///
    /// Returns an error if a different replacement kind has already claimed the
    /// reference. A subquery must contain exactly one SELECT statement.
    pub fn set_reference<'b>(&'a self, replacement_type: ReplacementType<'b>) -> Result<()>
    where
        'b: 'a,
    {
        match replacement_type {
            ReplacementType::Table(name) => {
                check_api_call!(ffi::duckdb_v2_replacement_scan_set_function_name, *self.info, *name)
            }
            ReplacementType::ColumnDataCollection((cdc, names)) => check_api_call!(
                ffi::duckdb_v2_replacement_scan_set_collection,
                *self.info,
                **cdc,
                names.iter().map(|n| n.into()).collect::<Vec<_>>().as_ptr(),
                names.len() as u64
            ),
            ReplacementType::Subquery(query) => {
                check_api_call!(
                    ffi::duckdb_v2_replacement_scan_set_subquery,
                    *self.info,
                    (&query).into()
                )
            }
        }
    }

    /// Set the replacement's alias unless the query supplies one.
    pub fn set_alias(&self, name: &str) -> Result<()> {
        check_api_call!(ffi::duckdb_v2_replacement_scan_set_alias, *self.info, name.into())
    }
}

unsafe extern "C" fn replacement_callback<T: ReplacementScanCallbacks>(
    info: ffi::duckdb_v2_replacement_scan_info_handle,
    context: ffi::duckdb_v2_context_handle,
    err: *mut ffi::duckdb_v2_error_info_handle,
) {
    handle_unwind(
        || {
            let user_data = get_user_data!(ffi::duckdb_v2_replacement_scan_get_user_data, info);

            let qname = QualifiedName {
                handle: check_api_call!(ffi::duckdb_v2_replacement_scan_get_name, info, RET)?,
            };

            T::scan(user_data, Context(context), &qname, ReplacementHandle { info: &info })
        },
        err,
    );
}

/// Registers a replacement scan callback.
///
/// Callbacks are consulted when binding cannot resolve a table name, in
/// registration order within each scope. Connection-local scans run before
/// database-wide scans; the first to claim the reference wins.
///
/// Registering through a connection keeps the scan local to that connection
/// until it closes. Registering through a database or extension makes the scan
/// visible to all connections until the database closes.
pub struct ReplacementScanBuilder<T> {
    implementation: OpaqueHandle<T>,
}

impl<T> ReplacementScanBuilder<T>
where
    T: ReplacementScanCallbacks,
{
    /// Create a builder from its callback implementation.
    pub fn new(implementation: T) -> Self {
        Self {
            implementation: OpaqueHandle::new(implementation),
        }
    }

    fn build(&self, handle: &ReplacementScanBuilderHandle) -> Result<()> {
        check_api_call!(
            ffi::duckdb_v2_replacement_scan_set_callback,
            **handle,
            Some(replacement_callback::<T>)
        )?;

        check_api_call!(
            ffi::duckdb_v2_replacement_scan_set_user_data,
            **handle,
            &mut self.implementation.to_handle()
        )?;

        Ok(())
    }

    /// Register through a connection, extension, or database, consuming the builder.
    #[allow(private_bounds)]
    pub fn register<C: ReplacementScanBuilderLink>(self, link: &C) -> Result<()> {
        let handle = link.create_replacement_scan_handle()?;
        self.build(&handle)?;

        check_api_call!(ffi::duckdb_v2_replacement_scan_register, *handle)
    }
}

/// Binding callback for unresolved table references.
pub trait ReplacementScanCallbacks: Send + Sync + 'static {
    /// **Bind:** claim, decline, or reject an unresolved table reference.
    ///
    /// Call [`ReplacementHandle::set_reference`] to claim it, return without
    /// doing so to let the next replacement scan try, or return an error to
    /// reject the query.
    fn scan(&self, context: Context, name: &QualifiedName, parameters: ReplacementHandle) -> Result<()>;
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use libduckdb_sys::DUCKDB_V2_LOGICAL_TYPE_ID::{
        DUCKDB_V2_LOGICAL_TYPE_ID_BIGINT, DUCKDB_V2_LOGICAL_TYPE_ID_BOOLEAN,
    };

    use crate::{
        Parameters, Result, ToValue,
        connection::Context,
        environment::{Environment, StorageLocation},
        qualified_name::QualifiedName,
        replacement_scan::{ReplacementHandle, ReplacementScanBuilder, ReplacementScanCallbacks, ReplacementType},
    };

    struct CustomReplacementScan {
        count: i32,
    }

    impl ReplacementScanCallbacks for CustomReplacementScan {
        fn scan(&self, context: Context, name: &QualifiedName, replacement: ReplacementHandle) -> Result<()> {
            let view = name.get_view()?;

            if let Some(table) = view.table
                && table.starts_with("num")
            {
                assert!(view.catalog == Some("test".to_string()));
                assert!(view.schema == Some("main".to_string()));

                let split = table
                    .replace("num_", "")
                    .replace('\'', "")
                    .split('_')
                    .map(|x| x.parse::<i32>().unwrap())
                    .collect::<Vec<_>>();

                dbg!(&split);

                replacement.set_reference(ReplacementType::Table("range".try_into()?))?;
                replacement.add_parameter(split[0].value(&context)?)?;
                replacement.add_parameter((split[1] + self.count).value(&context)?)?;
            }

            Ok(())
        }
    }

    struct CustomNamedParameters {}

    impl ReplacementScanCallbacks for CustomNamedParameters {
        fn scan(&self, context: Context, name: &QualifiedName, replacement: ReplacementHandle) -> Result<()> {
            let view = name.get_view()?;

            if view.table.is_some_and(|t| t.starts_with("alltypes")) {
                assert!(view.catalog.is_none());
                assert!(view.schema.is_none());

                replacement.set_reference(ReplacementType::Table("test_all_types".try_into()?))?;
                replacement.add_parameter_with_name("use_large_bignum", true.value(&context)?)?;
                replacement.add_parameter_with_name("use_large_enum", false.value(&context)?)?;
            }

            Ok(())
        }
    }

    #[test]
    fn test_replacement_scan() -> crate::Result<()> {
        let env = Environment::new()?;
        let db = env.open(StorageLocation::InMemory)?;
        let conn = db.connect()?;

        ReplacementScanBuilder::new(CustomReplacementScan { count: 42 }).register(&db)?;

        ReplacementScanBuilder::new(CustomNamedParameters {}).register(&conn)?;

        let mut query = conn.query("SELECT * FROM test.main.num_10_20", Parameters::None)?;

        let chunk = query.next().unwrap()?;

        assert!(chunk.get_vector_at::<i64>(0)?.logical_type().type_id() == DUCKDB_V2_LOGICAL_TYPE_ID_BIGINT);

        assert_eq!(chunk.row_count()?, 10 + 42);

        assert!(query.next().is_none());

        let mut query = conn.query("SELECT * FROM alltypes", Parameters::None)?;

        let chunk = query.next().unwrap()?;

        assert!(chunk.get_vector_at::<bool>(0)?.logical_type().type_id() == DUCKDB_V2_LOGICAL_TYPE_ID_BOOLEAN);

        assert_eq!(chunk.vectors_count()?, 59);

        Ok(())
    }
}
