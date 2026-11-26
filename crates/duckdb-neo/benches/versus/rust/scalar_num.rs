use duckdb_neo::{Result, connection::Connection, scalar::MicroScalar, vector::Vector};

fn register_add_one(conn: &Connection) -> Result<()> {
    MicroScalar::create(
        "vs_add_one",
        |_context, input: Vector<'_, i64>, mut output: Vector<'_, i64>| {
            let mut writer = output.writer()?;
            for (row, value) in input.reader()?.iter().enumerate() {
                writer.set_option(row, value.map(|value| value.wrapping_add(1)).as_ref())?;
            }
            Ok(())
        },
    )
    .register(conn)
}

pub fn register(conn: &Connection) -> Result<()> {
    register_add_one(conn)
}
