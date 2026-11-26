//! `int_sum(v)`: a `BIGINT` sum, `NULL` without non-`NULL` rows. Its plain state and cheap rows
//! leave the cost of each update and finalize call to dominate, especially in window queries
//! that call them for a few rows at a time.

use duckdb_neo::{
    DuckDBType, Parameters, Result,
    aggregate::{AggregateCallbacks, AggregateFunctionBuilder, ReturnTypeHandle, States},
    bind_arguments::BindArguments,
    connection::{Connection, Context},
    data_chunk::VectorCollection,
    signature::{FunctionParameter, SignatureBuilder},
    vector::{Unknown, Vector},
};

use super::{nulls::Nulls, readable};

#[derive(Debug, Default)]
pub struct SumState {
    sum: i64,
    rows: u64,
}

pub struct IntSum;

impl AggregateCallbacks for IntSum {
    type BindData = ();
    type StateItem = SumState;

    fn bind(
        &self,
        _context: &Context,
        _arguments: BindArguments<'_>,
        _result_type_handle: ReturnTypeHandle<'_>,
    ) -> Result<Self::BindData> {
        Ok(())
    }

    fn init(&self, _bind_data: Option<&()>) -> Self::StateItem {
        SumState::default()
    }

    fn update(
        &self,
        _bind_data: Option<&()>,
        data: &mut VectorCollection,
        mut states: States<'_, SumState>,
    ) -> Result<()> {
        let values = readable::<i64>(data.get_untyped_vector_at_mut(0)?)?;
        for (row, value) in values.reader()?.iter().enumerate() {
            if let Some(value) = value {
                let state = &mut states[row];
                state.sum = state.sum.wrapping_add(*value);
                state.rows += 1;
            }
        }
        Ok(())
    }

    fn combine(
        &self,
        _bind_data: Option<&()>,
        source: &States<'_, SumState>,
        mut target: States<'_, SumState>,
    ) -> Result<()> {
        for index in 0..target.len() {
            let source = &source[index];
            let target = &mut target[index];
            target.sum = target.sum.wrapping_add(source.sum);
            target.rows += source.rows;
        }
        Ok(())
    }

    fn finalize(
        &self,
        _bind_data: Option<&()>,
        states: &[&SumState],
        result: Vector<'_, Unknown>,
        result_offset: usize,
    ) -> Result<()> {
        let mut result = result.cast::<i64>()?;
        let mut writer = result.writer()?;
        for (index, state) in states.iter().enumerate() {
            writer.set_option(result_offset + index, (state.rows > 0).then_some(&state.sum))?;
        }
        Ok(())
    }
}

pub fn register(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<i64>("v"))
        .return_type(i64::logical_type(conn)?);
    AggregateFunctionBuilder::new("int_sum", signature, IntSum).register(conn)
}

/// Number of groups in `sum_input`.
pub const GROUPS: usize = 1024;

/// Create `sum_input` with `rows` rows of `int_sum` arguments in [`GROUPS`] groups.
pub fn create_input(conn: &Connection, rows: usize) -> Result<()> {
    create_input_with_nulls(conn, rows, Nulls::from_env()?)
}

pub fn create_input_with_nulls(conn: &Connection, rows: usize, nulls: Nulls) -> Result<()> {
    conn.execute(
        format!(
            "CREATE OR REPLACE TABLE sum_input AS
             SELECT i::BIGINT AS i, (i % {GROUPS})::INTEGER AS g, {}::BIGINT AS v
             FROM range({rows}) t(i)",
            nulls.expression("i * 7919 % 100003 - 50000", 30),
        )
        .as_str(),
        Parameters::None,
    )?;
    Ok(())
}

pub const UNGROUPED_QUERY: &str = "SELECT int_sum(v) FROM sum_input";
pub const GROUPED_QUERY: &str = "SELECT g, int_sum(v) FROM sum_input GROUP BY g";
/// A running sum without `ORDER BY`: DuckDB's streaming window updates and finalizes once per row.
pub const RUNNING_QUERY: &str = "SELECT int_sum(v) OVER (ROWS UNBOUNDED PRECEDING) FROM sum_input";
/// A sliding sum over partitions: DuckDB's segment tree updates and finalizes a few rows at a time.
pub const SLIDING_QUERY: &str =
    "SELECT int_sum(v) OVER (PARTITION BY g ORDER BY i ROWS BETWEEN 8 PRECEDING AND CURRENT ROW) FROM sum_input";

/// Check every query shape against DuckDB's `sum`.
pub fn verify(conn: &Connection) -> Result<()> {
    let mismatches = |select: &str| -> Result<i64> {
        super::query_i64(
            conn,
            &format!("SELECT count(*) FROM ({select}) WHERE a IS DISTINCT FROM b::BIGINT"),
        )
    };
    let checks = [
        (
            "ungrouped",
            "SELECT int_sum(v) AS a, sum(v) AS b FROM sum_input".to_owned(),
        ),
        (
            "grouped",
            "SELECT int_sum(v) AS a, sum(v) AS b FROM sum_input GROUP BY g".to_owned(),
        ),
        (
            "running",
            "SELECT int_sum(v) OVER w AS a, sum(v) OVER w AS b FROM sum_input WINDOW w AS (ROWS UNBOUNDED PRECEDING)"
                .to_owned(),
        ),
        (
            "sliding",
            "SELECT int_sum(v) OVER w AS a, sum(v) OVER w AS b FROM sum_input
             WINDOW w AS (PARTITION BY g ORDER BY i ROWS BETWEEN 8 PRECEDING AND CURRENT ROW)"
                .to_owned(),
        ),
        (
            "empty",
            "SELECT int_sum(v) AS a, sum(v) AS b FROM sum_input WHERE false".to_owned(),
        ),
    ];
    for (shape, select) in checks {
        let count = mismatches(&select)?;
        super::ensure(count == 0, || {
            format!("int_sum differs from sum in {count} {shape} rows")
        })?;
    }
    Ok(())
}
