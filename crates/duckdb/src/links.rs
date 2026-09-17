//! Internal dispatch for operations available through multiple FFI link types.

use crate::{
    connection::{Connection, Context},
    ffi,
};

// Share the Rust signature and FFI arguments; only the link type and C function vary.
macro_rules! define_link {
    (
        name: $name:ident,
        method: fn $method:ident $params:tt -> $ret:ty,
        args: $args:tt,
        implementations: {
            $( $type:ty => $func:expr ),* $(,)?
        } $(,)?
    ) => {
        define_link!(@trait $name, $method, $params, $ret);
        $(
            define_link!(@impl $name, $type, $method, $params, $ret, $func, $args);
        )*
    };
    (@trait $name:ident, $method:ident, ($($arg:ident: $arg_ty:ty),* $(,)?), $ret:ty) => {
        pub(crate) trait $name {
            fn $method(&self, $($arg: $arg_ty),*) -> $crate::Result<$ret>;
        }
    };
    (
        @impl $name:ident, $type:ty, $method:ident,
        ($($arg:ident: $arg_ty:ty),* $(,)?), $ret:ty,
        $func:expr, ($($args:tt)*)
    ) => {
        impl $name for $type {
            fn $method(&self, $($arg: $arg_ty),*) -> $crate::Result<$ret> {
                $crate::check_api_call!($func, **self, $($args)*)
            }
        }
    };
}

define_link! {
    name: ColumnDataCollectionLink,
    method: fn create_column_data_collection(types: &[ffi::duckdb_v2_logical_type_handle])
        -> ffi::duckdb_v2_column_data_collection_handle,
    args: (types.as_ptr(), types.len() as u64, RET),
    implementations: {
        Connection => ffi::duckdb_v2_column_data_collection_create_with_connection,
        Context => ffi::duckdb_v2_column_data_collection_create_with_context,
    },
}
