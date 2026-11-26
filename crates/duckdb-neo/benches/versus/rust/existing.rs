//! The existing duckdb-neo benchmark functions from `benches/support`, registered unchanged:
//! `merge_struct`, `dash_split`, `row_hash`, `int_sum` and the `base56` COPY format. Their
//! C++ ports are in `../cpp/existing.cpp`.

// The modules write some output through duckdb-neo's `set_size_uninit`; see `../findings/existing.md`.
#[allow(dead_code, unsafe_code)]
#[path = "../../support/mod.rs"]
mod support;

use duckdb_neo::{Result, connection::Connection};

pub fn register(conn: &Connection) -> Result<()> {
    support::merge_struct::register(conn)?;
    support::dash_split::register(conn)?;
    support::row_hash::register(conn)?;
    support::int_sum::register(conn)?;
    support::base56::register(conn)
}
