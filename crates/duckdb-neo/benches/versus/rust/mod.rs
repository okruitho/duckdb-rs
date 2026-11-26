//! The duckdb-neo side of every family. Each family registers its `vs_*` functions on the
//! runner's connection; the C++ side is in `../cpp/<family>.cpp`.

mod aggregate_heap;
mod aggregate_num;
mod cast;
mod copy;
mod existing;
mod scalar_nested;
mod scalar_num;
mod scalar_str;
mod table;

use duckdb_neo::{Result, connection::Connection};

pub fn register_all(conn: &Connection) -> Result<()> {
    aggregate_heap::register(conn)?;
    aggregate_num::register(conn)?;
    cast::register(conn)?;
    copy::register(conn)?;
    existing::register(conn)?;
    scalar_nested::register(conn)?;
    scalar_num::register(conn)?;
    scalar_str::register(conn)?;
    table::register(conn)?;
    Ok(())
}
