//! Aggregates with plain states:
//! - `vs_sum(v BIGINT) -> BIGINT`: wrapping sum, NULL without non-NULL rows.
//! - `vs_avg(d DOUBLE) -> DOUBLE`: mean of the non-NULL rows, NULL without any.
//! - `vs_count_if(b BOOLEAN) -> BIGINT`: number of true rows, 0 without any.
//! - `vs_spread(v BIGINT) -> BIGINT`: wrapping `max - min` of the non-NULL rows, NULL without any.
//! - `vs_wavg(v DOUBLE, w DOUBLE) -> DOUBLE`: `sum(v * w) / sum(w)` over rows where both are
//!   non-NULL, NULL without any.
//! - `vs_sum_scaled(v BIGINT, factor BIGINT) -> BIGINT`: wrapping sum of `v * factor`, NULL without
//!   non-NULL `v`; `factor` must be a non-NULL constant and is read once at bind time.

use duckdb_neo::{
    DuckDBType, Error, Result,
    aggregate::{AggregateCallbacks, AggregateFunctionBuilder, ReturnTypeHandle, States},
    bind_arguments::BindArguments,
    connection::{Connection, Context},
    data_chunk::VectorCollection,
    signature::{FunctionParameter, SignatureBuilder},
    vector::{CombinedVectorIterator, NativeElement, StorageKind, Unknown, Vector},
};

fn readable<T: NativeElement>(vector: Vector<'_, Unknown>) -> Result<Vector<'_, T>> {
    let mut vector = vector.cast::<T>()?;
    if vector.storage_kind() == StorageKind::Other {
        vector.flatten()?;
    }
    Ok(vector)
}

#[derive(Debug, Default)]
struct SumState {
    sum: i64,
    rows: u64,
}

struct Sum;

impl AggregateCallbacks for Sum {
    type BindData = ();
    type StateItem = SumState;

    fn bind(&self, _context: &Context, _arguments: BindArguments<'_>, _result: ReturnTypeHandle<'_>) -> Result<()> {
        Ok(())
    }

    fn init(&self, _bind_data: Option<&()>) -> SumState {
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

fn register_sum(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<i64>("v"))
        .return_type(i64::logical_type(conn)?);
    AggregateFunctionBuilder::new("vs_sum", signature, Sum).register(conn)
}

#[derive(Debug, Default)]
struct AvgState {
    sum: f64,
    rows: u64,
}

struct Avg;

impl AggregateCallbacks for Avg {
    type BindData = ();
    type StateItem = AvgState;

    fn bind(&self, _context: &Context, _arguments: BindArguments<'_>, _result: ReturnTypeHandle<'_>) -> Result<()> {
        Ok(())
    }

    fn init(&self, _bind_data: Option<&()>) -> AvgState {
        AvgState::default()
    }

    fn update(
        &self,
        _bind_data: Option<&()>,
        data: &mut VectorCollection,
        mut states: States<'_, AvgState>,
    ) -> Result<()> {
        let values = readable::<f64>(data.get_untyped_vector_at_mut(0)?)?;
        for (row, value) in values.reader()?.iter().enumerate() {
            if let Some(value) = value {
                let state = &mut states[row];
                state.sum += *value;
                state.rows += 1;
            }
        }
        Ok(())
    }

    fn combine(
        &self,
        _bind_data: Option<&()>,
        source: &States<'_, AvgState>,
        mut target: States<'_, AvgState>,
    ) -> Result<()> {
        for index in 0..target.len() {
            let source = &source[index];
            let target = &mut target[index];
            target.sum += source.sum;
            target.rows += source.rows;
        }
        Ok(())
    }

    fn finalize(
        &self,
        _bind_data: Option<&()>,
        states: &[&AvgState],
        result: Vector<'_, Unknown>,
        result_offset: usize,
    ) -> Result<()> {
        let mut result = result.cast::<f64>()?;
        let mut writer = result.writer()?;
        for (index, state) in states.iter().enumerate() {
            let mean = state.sum / state.rows as f64;
            writer.set_option(result_offset + index, (state.rows > 0).then_some(&mean))?;
        }
        Ok(())
    }
}

fn register_avg(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<f64>("d"))
        .return_type(f64::logical_type(conn)?);
    AggregateFunctionBuilder::new("vs_avg", signature, Avg).register(conn)
}

#[derive(Debug, Default)]
struct CountIfState {
    count: i64,
}

struct CountIf;

impl AggregateCallbacks for CountIf {
    type BindData = ();
    type StateItem = CountIfState;

    fn bind(&self, _context: &Context, _arguments: BindArguments<'_>, _result: ReturnTypeHandle<'_>) -> Result<()> {
        Ok(())
    }

    fn init(&self, _bind_data: Option<&()>) -> CountIfState {
        CountIfState::default()
    }

    fn update(
        &self,
        _bind_data: Option<&()>,
        data: &mut VectorCollection,
        mut states: States<'_, CountIfState>,
    ) -> Result<()> {
        let values = readable::<bool>(data.get_untyped_vector_at_mut(0)?)?;
        for (row, value) in values.reader()?.iter().enumerate() {
            if value == Some(&true) {
                states[row].count += 1;
            }
        }
        Ok(())
    }

    fn combine(
        &self,
        _bind_data: Option<&()>,
        source: &States<'_, CountIfState>,
        mut target: States<'_, CountIfState>,
    ) -> Result<()> {
        for index in 0..target.len() {
            target[index].count += source[index].count;
        }
        Ok(())
    }

    fn finalize(
        &self,
        _bind_data: Option<&()>,
        states: &[&CountIfState],
        result: Vector<'_, Unknown>,
        result_offset: usize,
    ) -> Result<()> {
        let mut result = result.cast::<i64>()?;
        let mut writer = result.writer()?;
        for (index, state) in states.iter().enumerate() {
            writer.set(result_offset + index, &state.count)?;
        }
        Ok(())
    }
}

fn register_count_if(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<bool>("b"))
        .return_type(i64::logical_type(conn)?);
    AggregateFunctionBuilder::new("vs_count_if", signature, CountIf).register(conn)
}

#[derive(Debug)]
struct SpreadState {
    min: i64,
    max: i64,
    rows: u64,
}

impl Default for SpreadState {
    fn default() -> Self {
        SpreadState {
            min: i64::MAX,
            max: i64::MIN,
            rows: 0,
        }
    }
}

struct Spread;

impl AggregateCallbacks for Spread {
    type BindData = ();
    type StateItem = SpreadState;

    fn bind(&self, _context: &Context, _arguments: BindArguments<'_>, _result: ReturnTypeHandle<'_>) -> Result<()> {
        Ok(())
    }

    fn init(&self, _bind_data: Option<&()>) -> SpreadState {
        SpreadState::default()
    }

    fn update(
        &self,
        _bind_data: Option<&()>,
        data: &mut VectorCollection,
        mut states: States<'_, SpreadState>,
    ) -> Result<()> {
        let values = readable::<i64>(data.get_untyped_vector_at_mut(0)?)?;
        for (row, value) in values.reader()?.iter().enumerate() {
            if let Some(&value) = value {
                let state = &mut states[row];
                state.min = state.min.min(value);
                state.max = state.max.max(value);
                state.rows += 1;
            }
        }
        Ok(())
    }

    fn combine(
        &self,
        _bind_data: Option<&()>,
        source: &States<'_, SpreadState>,
        mut target: States<'_, SpreadState>,
    ) -> Result<()> {
        for index in 0..target.len() {
            let source = &source[index];
            let target = &mut target[index];
            target.min = target.min.min(source.min);
            target.max = target.max.max(source.max);
            target.rows += source.rows;
        }
        Ok(())
    }

    fn finalize(
        &self,
        _bind_data: Option<&()>,
        states: &[&SpreadState],
        result: Vector<'_, Unknown>,
        result_offset: usize,
    ) -> Result<()> {
        let mut result = result.cast::<i64>()?;
        let mut writer = result.writer()?;
        for (index, state) in states.iter().enumerate() {
            let spread = state.max.wrapping_sub(state.min);
            writer.set_option(result_offset + index, (state.rows > 0).then_some(&spread))?;
        }
        Ok(())
    }
}

fn register_spread(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<i64>("v"))
        .return_type(i64::logical_type(conn)?);
    AggregateFunctionBuilder::new("vs_spread", signature, Spread).register(conn)
}

#[derive(Debug, Default)]
struct WavgState {
    weighted_sum: f64,
    weight_sum: f64,
    rows: u64,
}

struct Wavg;

impl AggregateCallbacks for Wavg {
    type BindData = ();
    type StateItem = WavgState;

    fn bind(&self, _context: &Context, _arguments: BindArguments<'_>, _result: ReturnTypeHandle<'_>) -> Result<()> {
        Ok(())
    }

    fn init(&self, _bind_data: Option<&()>) -> WavgState {
        WavgState::default()
    }

    fn update(
        &self,
        _bind_data: Option<&()>,
        data: &mut VectorCollection,
        mut states: States<'_, WavgState>,
    ) -> Result<()> {
        let [v, w] = data.get_vector_array_mut([0, 1])?;
        let (v, w) = (readable::<f64>(v)?, readable::<f64>(w)?);
        let rows = CombinedVectorIterator::new((v.reader()?, w.reader()?))?;
        for (row, (v, w)) in rows.enumerate() {
            if let (Some(&v), Some(&w)) = (v, w) {
                let state = &mut states[row];
                state.weighted_sum += v * w;
                state.weight_sum += w;
                state.rows += 1;
            }
        }
        Ok(())
    }

    fn combine(
        &self,
        _bind_data: Option<&()>,
        source: &States<'_, WavgState>,
        mut target: States<'_, WavgState>,
    ) -> Result<()> {
        for index in 0..target.len() {
            let source = &source[index];
            let target = &mut target[index];
            target.weighted_sum += source.weighted_sum;
            target.weight_sum += source.weight_sum;
            target.rows += source.rows;
        }
        Ok(())
    }

    fn finalize(
        &self,
        _bind_data: Option<&()>,
        states: &[&WavgState],
        result: Vector<'_, Unknown>,
        result_offset: usize,
    ) -> Result<()> {
        let mut result = result.cast::<f64>()?;
        let mut writer = result.writer()?;
        for (index, state) in states.iter().enumerate() {
            let mean = state.weighted_sum / state.weight_sum;
            writer.set_option(result_offset + index, (state.rows > 0).then_some(&mean))?;
        }
        Ok(())
    }
}

fn register_wavg(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<f64>("v"))
        .param(FunctionParameter::new::<f64>("w"))
        .return_type(f64::logical_type(conn)?);
    AggregateFunctionBuilder::new("vs_wavg", signature, Wavg).register(conn)
}

struct SumScaled;

fn factor(bind_data: Option<&i64>) -> Result<i64> {
    bind_data
        .copied()
        .ok_or_else(|| Error::invalid_input("vs_sum_scaled: missing bind data"))
}

impl AggregateCallbacks for SumScaled {
    type BindData = i64;
    type StateItem = SumState;

    fn bind(&self, _context: &Context, arguments: BindArguments<'_>, _result: ReturnTypeHandle<'_>) -> Result<i64> {
        match arguments.arg(1)?.resolved_value::<i64>()? {
            Some(Some(factor)) => Ok(factor),
            _ => Err(Error::invalid_input(
                "vs_sum_scaled: factor must be a non-NULL constant",
            )),
        }
    }

    fn init(&self, _bind_data: Option<&i64>) -> SumState {
        SumState::default()
    }

    fn update(
        &self,
        bind_data: Option<&i64>,
        data: &mut VectorCollection,
        mut states: States<'_, SumState>,
    ) -> Result<()> {
        let factor = factor(bind_data)?;
        let values = readable::<i64>(data.get_untyped_vector_at_mut(0)?)?;
        for (row, value) in values.reader()?.iter().enumerate() {
            if let Some(value) = value {
                let state = &mut states[row];
                state.sum = state.sum.wrapping_add(value.wrapping_mul(factor));
                state.rows += 1;
            }
        }
        Ok(())
    }

    fn combine(
        &self,
        _bind_data: Option<&i64>,
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
        _bind_data: Option<&i64>,
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

fn register_sum_scaled(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<i64>("v"))
        .param(FunctionParameter::new::<i64>("factor"))
        .return_type(i64::logical_type(conn)?);
    AggregateFunctionBuilder::new("vs_sum_scaled", signature, SumScaled).register(conn)
}

pub fn register(conn: &Connection) -> Result<()> {
    register_sum(conn)?;
    register_avg(conn)?;
    register_count_if(conn)?;
    register_spread(conn)?;
    register_wavg(conn)?;
    register_sum_scaled(conn)
}
