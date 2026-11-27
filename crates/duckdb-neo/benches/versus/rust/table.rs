//! Table functions. A NULL row count produces no rows, as does a negative one.
//! - `vs_series(n BIGINT) -> (i BIGINT)`: `0 .. n-1`.
//! - `vs_grid(n BIGINT) -> (i BIGINT, sq BIGINT, half DOUBLE, tag VARCHAR, even BOOLEAN, opt INTEGER)`:
//!   `sq = i*i` (wrapping), `half = i / 2`, `tag = 'row-' || i`, `even = i % 2 = 0`, `opt` NULL when `i % 5 = 0`
//!   else `i % 1000`; projection pushdown.
//! - `vs_words(n BIGINT, len INTEGER) -> (word VARCHAR)`: `lpad(i::VARCHAR, len, '.')` for `i` in `0 .. n-1`;
//!   NULL `len` produces no rows, a negative one is an error.
//! - `vs_repeat_row(s VARCHAR, k BIGINT, n BIGINT) -> (s VARCHAR, k BIGINT)`: `(s, k)` `n` times as constant
//!   vectors, NULLs included; projection pushdown.
//! - `vs_series_step(start BIGINT, stop BIGINT, step := 1) -> (i BIGINT)`: `range(start, stop, step)`; a zero step is
//!   an error, any NULL argument produces no rows.
//! - `vs_series_filter(n BIGINT) -> (i BIGINT)`: `vs_series` that applies `i <op> constant` filters itself.
//! - `vs_series_par(n BIGINT) -> (i BIGINT)`: `vs_series` scanned by several threads, in no particular order.

use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicI64, Ordering},
};

use duckdb_neo::{
    DuckDBType, Error, FromValue, Result, ToValue,
    bind_arguments::BindArguments,
    connection::{Connection, Context},
    data_chunk::DataChunk,
    expression::ExpressionType,
    logical_type::LogicalTypeID,
    signature::{FunctionParameter, FunctionParameterKind, SignatureBuilder},
    table_function::{
        BindFunctionHandle, ExecColumnInfo, InitColumnData, OrderPreservation, PushdownData,
        TableFilterPushdownCallbacks, TableFunctionBuilder, TableFunctionCallbacks,
    },
    value::Value,
    vector::{NativeElement, VectorWriter, WritableNativeElement},
};

/// Rows a parallel scan claims at a time, which also sizes its thread count.
const PAR_BLOCK_ROWS: i64 = 2048;

fn constant<T: FromValue>(arguments: &BindArguments<'_>, index: usize) -> Result<Option<T>> {
    arguments
        .arg(index)?
        .resolved_value::<T>()?
        .ok_or_else(|| Error::invalid_input("table function arguments must be constant"))
}

fn row_count(n: Option<i64>) -> i64 {
    n.map_or(0, |n| n.max(0))
}

fn chunk_rows(output: &DataChunk<'_>, remaining: i64) -> Result<usize> {
    Ok(remaining.clamp(0, output.capacity()? as i64) as usize)
}

fn fill_column<T>(
    output: &mut DataChunk<'_>,
    position: usize,
    rows: usize,
    mut value: impl FnMut(usize) -> T,
) -> Result<()>
where
    T: NativeElement + WritableNativeElement<Write = T>,
{
    let mut vector = output.get_vector_at_mut::<T>(position)?;
    vector.set_size(rows)?;
    let mut writer = vector.writer()?;
    for row in 0..rows {
        writer.set(row, &value(row))?;
    }
    Ok(())
}

fn fill_series(output: &mut DataChunk<'_>, rows: usize, first: i64) -> Result<()> {
    fill_column::<i64>(output, 0, rows, |row| first + row as i64)
}

/// The decimal digits of `value`, right-aligned in `buffer`.
fn format_decimal(buffer: &mut [u8; 20], value: u64) -> &[u8] {
    let mut start = buffer.len();
    let mut rest = value;
    loop {
        start -= 1;
        buffer[start] = b'0' + (rest % 10) as u8;
        rest /= 10;
        if rest == 0 {
            break;
        }
    }
    &buffer[start..]
}

fn write_tag(writer: &mut VectorWriter<'_, String>, row: usize, i: i64) -> Result<()> {
    let mut digits = [0u8; 20];
    let digits = std::str::from_utf8(format_decimal(&mut digits, i as u64)).expect("decimal digits are ASCII");
    writer.set_with(row, 4 + digits.len(), |tag| {
        tag.push_str("row-")?;
        tag.push_str(digits)
    })
}

// Like `lpad(i::VARCHAR, len, '.')`, which keeps the leading `len` digits of a longer number.
fn write_word(writer: &mut VectorWriter<'_, String>, row: usize, i: i64, padding: &str) -> Result<()> {
    let mut digits = [0u8; 20];
    let digits = format_decimal(&mut digits, i as u64);
    let shown = digits.len().min(padding.len());
    writer.set_with(row, padding.len(), |word| {
        word.push_str(&padding[..padding.len() - shown])?;
        word.push_str(std::str::from_utf8(&digits[..shown]).expect("decimal digits are ASCII"))
    })
}

/// Where a single-threaded scan continues.
pub struct SeriesScan {
    position: i64,
}

fn init_series_scan() -> Result<Option<SeriesScan>> {
    Ok(Some(SeriesScan { position: 0 }))
}

pub struct CountBind {
    count: i64,
}

fn bind_series(
    context: &Context,
    arguments: BindArguments<'_>,
    bind_handle: &BindFunctionHandle<'_>,
) -> Result<CountBind> {
    let count = row_count(constant::<i64>(&arguments, 0)?);
    bind_handle.add_result_column("i", &i64::logical_type(context)?)?;
    bind_handle.set_cardinality(count as usize, true)?;
    Ok(CountBind { count })
}

pub struct Series;

impl TableFunctionCallbacks for Series {
    type BindData = CountBind;
    type GlobalState = ();
    type LocalState = SeriesScan;

    fn bind(
        &self,
        context: &Context,
        arguments: BindArguments<'_>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        bind_series(context, arguments, &bind_handle)
    }

    fn init_local_state(
        &self,
        _context: &Context,
        _bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        _column_data: InitColumnData<'_>,
    ) -> Result<Option<Self::LocalState>> {
        init_series_scan()
    }

    fn exec(
        &self,
        _context: &Context,
        bind_data: &Self::BindData,
        local_state: Option<&mut Self::LocalState>,
        _global_state: Option<&Self::GlobalState>,
        output: &mut DataChunk<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let scan = local_state.expect("init_local_state sets the scan");
        let rows = chunk_rows(output, bind_data.count - scan.position)?;
        fill_series(output, rows, scan.position)?;
        scan.position += rows as i64;
        Ok(())
    }
}

pub struct Grid;

impl TableFunctionCallbacks for Grid {
    type BindData = CountBind;
    type GlobalState = ();
    type LocalState = SeriesScan;

    fn bind(
        &self,
        context: &Context,
        arguments: BindArguments<'_>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        let count = row_count(constant::<i64>(&arguments, 0)?);
        bind_handle.add_result_column("i", &i64::logical_type(context)?)?;
        bind_handle.add_result_column("sq", &i64::logical_type(context)?)?;
        bind_handle.add_result_column("half", &f64::logical_type(context)?)?;
        bind_handle.add_result_column("tag", &String::logical_type(context)?)?;
        bind_handle.add_result_column("even", &bool::logical_type(context)?)?;
        bind_handle.add_result_column("opt", &i32::logical_type(context)?)?;
        bind_handle.set_cardinality(count as usize, true)?;
        Ok(CountBind { count })
    }

    fn init_local_state(
        &self,
        _context: &Context,
        _bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        _column_data: InitColumnData<'_>,
    ) -> Result<Option<Self::LocalState>> {
        init_series_scan()
    }

    fn exec(
        &self,
        _context: &Context,
        bind_data: &Self::BindData,
        local_state: Option<&mut Self::LocalState>,
        _global_state: Option<&Self::GlobalState>,
        output: &mut DataChunk<'_>,
        column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let scan = local_state.expect("init_local_state sets the scan");
        let rows = chunk_rows(output, bind_data.count - scan.position)?;
        let first = scan.position;
        let at = |row: usize| first + row as i64;
        for position in 0..column_info.count()? {
            match column_info.get_column_index(position)? {
                0 => fill_column::<i64>(output, position, rows, at)?,
                1 => fill_column::<i64>(output, position, rows, |row| at(row).wrapping_mul(at(row)))?,
                2 => fill_column::<f64>(output, position, rows, |row| at(row) as f64 / 2.0)?,
                3 => {
                    let mut vector = output.get_vector_at_mut::<String>(position)?;
                    vector.set_size(rows)?;
                    let mut writer = vector.writer()?;
                    for row in 0..rows {
                        write_tag(&mut writer, row, at(row))?;
                    }
                }
                4 => fill_column::<bool>(output, position, rows, |row| at(row) % 2 == 0)?,
                5 => {
                    let mut vector = output.get_vector_at_mut::<i32>(position)?;
                    vector.set_size(rows)?;
                    let mut writer = vector.writer()?;
                    for row in 0..rows {
                        let i = at(row);
                        if i % 5 == 0 {
                            writer.set_null(row)?;
                        } else {
                            writer.set(row, &((i % 1000) as i32))?;
                        }
                    }
                }
                column => return Err(Error::invalid_input(format!("vs_grid has no column {column}"))),
            }
        }
        scan.position += rows as i64;
        Ok(())
    }
}

/// `padding` is `len` dots, the longest prefix a word needs.
pub struct WordsBind {
    count: i64,
    padding: String,
}

/// Where a scan continues.
pub struct WordsScan {
    position: i64,
}

pub struct Words;

impl TableFunctionCallbacks for Words {
    type BindData = WordsBind;
    type GlobalState = ();
    type LocalState = WordsScan;

    fn bind(
        &self,
        context: &Context,
        arguments: BindArguments<'_>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        let mut count = row_count(constant::<i64>(&arguments, 0)?);
        let len = match constant::<i32>(&arguments, 1)? {
            Some(len) => usize::try_from(len).map_err(|_| Error::invalid_input("vs_words len must not be negative"))?,
            None => {
                count = 0;
                0
            }
        };
        bind_handle.add_result_column("word", &String::logical_type(context)?)?;
        bind_handle.set_cardinality(count as usize, true)?;
        Ok(WordsBind {
            count,
            padding: ".".repeat(len),
        })
    }

    fn init_local_state(
        &self,
        _context: &Context,
        _bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        _column_data: InitColumnData<'_>,
    ) -> Result<Option<Self::LocalState>> {
        Ok(Some(WordsScan { position: 0 }))
    }

    fn exec(
        &self,
        _context: &Context,
        bind_data: &Self::BindData,
        local_state: Option<&mut Self::LocalState>,
        _global_state: Option<&Self::GlobalState>,
        output: &mut DataChunk<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let scan = local_state.expect("init_local_state sets the scan");
        let rows = chunk_rows(output, bind_data.count - scan.position)?;
        let mut vector = output.get_vector_at_mut::<String>(0)?;
        vector.set_size(rows)?;
        let mut writer = vector.writer()?;
        for row in 0..rows {
            write_word(&mut writer, row, scan.position + row as i64, &bind_data.padding)?;
        }
        scan.position += rows as i64;
        Ok(())
    }
}

pub struct RepeatBind {
    s: Value,
    k: Value,
    count: i64,
}

pub struct RepeatRow;

impl TableFunctionCallbacks for RepeatRow {
    type BindData = RepeatBind;
    type GlobalState = ();
    type LocalState = SeriesScan;

    fn bind(
        &self,
        context: &Context,
        arguments: BindArguments<'_>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        let s = constant::<String>(&arguments, 0)?;
        let k = constant::<i64>(&arguments, 1)?;
        let count = row_count(constant::<i64>(&arguments, 2)?);
        bind_handle.add_result_column("s", &String::logical_type(context)?)?;
        bind_handle.add_result_column("k", &i64::logical_type(context)?)?;
        bind_handle.set_cardinality(count as usize, true)?;
        Ok(RepeatBind {
            s: s.value(context)?,
            k: k.value(context)?,
            count,
        })
    }

    fn init_local_state(
        &self,
        _context: &Context,
        _bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        _column_data: InitColumnData<'_>,
    ) -> Result<Option<Self::LocalState>> {
        init_series_scan()
    }

    fn exec(
        &self,
        _context: &Context,
        bind_data: &Self::BindData,
        local_state: Option<&mut Self::LocalState>,
        _global_state: Option<&Self::GlobalState>,
        output: &mut DataChunk<'_>,
        column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let scan = local_state.expect("init_local_state sets the scan");
        let rows = chunk_rows(output, bind_data.count - scan.position)?;
        if rows == 0 {
            return Ok(());
        }
        for position in 0..column_info.count()? {
            match column_info.get_column_index(position)? {
                0 => {
                    output
                        .get_vector_at_mut::<String>(position)?
                        .make_constant(&bind_data.s, rows)?;
                }
                1 => {
                    output
                        .get_vector_at_mut::<i64>(position)?
                        .make_constant(&bind_data.k, rows)?;
                }
                column => return Err(Error::invalid_input(format!("vs_repeat_row has no column {column}"))),
            }
        }
        scan.position += rows as i64;
        Ok(())
    }
}

pub struct StepBind {
    start: i64,
    step: i64,
    count: u64,
}

/// How many values `start, start + step, ...` lie before `stop`.
fn step_count(start: i64, stop: i64, step: i64) -> u64 {
    if step > 0 && start < stop {
        (stop.wrapping_sub(start) as u64 - 1) / step as u64 + 1
    } else if step < 0 && start > stop {
        (start.wrapping_sub(stop) as u64 - 1) / step.unsigned_abs() + 1
    } else {
        0
    }
}

pub struct StepScan {
    position: u64,
}

pub struct SeriesStep;

impl TableFunctionCallbacks for SeriesStep {
    type BindData = StepBind;
    type GlobalState = ();
    type LocalState = StepScan;

    fn bind(
        &self,
        context: &Context,
        arguments: BindArguments<'_>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        let start = constant::<i64>(&arguments, 0)?;
        let stop = constant::<i64>(&arguments, 1)?;
        let step = constant::<i64>(&arguments, 2)?;
        if step == Some(0) {
            return Err(Error::invalid_input("vs_series_step step must not be zero"));
        }
        let bind = match (start, stop, step) {
            (Some(start), Some(stop), Some(step)) => StepBind {
                start,
                step,
                count: step_count(start, stop, step),
            },
            _ => StepBind {
                start: 0,
                step: 1,
                count: 0,
            },
        };
        bind_handle.add_result_column("i", &i64::logical_type(context)?)?;
        bind_handle.set_cardinality(bind.count as usize, true)?;
        Ok(bind)
    }

    fn init_local_state(
        &self,
        _context: &Context,
        _bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        _column_data: InitColumnData<'_>,
    ) -> Result<Option<Self::LocalState>> {
        Ok(Some(StepScan { position: 0 }))
    }

    fn exec(
        &self,
        _context: &Context,
        bind_data: &Self::BindData,
        local_state: Option<&mut Self::LocalState>,
        _global_state: Option<&Self::GlobalState>,
        output: &mut DataChunk<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let scan = local_state.expect("init_local_state sets the scan");
        let rows = (bind_data.count - scan.position).min(output.capacity()? as u64) as usize;
        let first = scan.position;
        fill_column::<i64>(output, 0, rows, |row| {
            bind_data
                .start
                .wrapping_add(((first + row as u64) as i64).wrapping_mul(bind_data.step))
        })?;
        scan.position += rows as u64;
        Ok(())
    }
}

/// The pushed-down filters narrow `[lo, hi)`, possibly over several pushdown calls.
pub struct FilterBind {
    count: i64,
    lo: AtomicI64,
    hi: AtomicI64,
}

/// The rows `[position, end)` left to produce.
pub struct FilterScan {
    position: i64,
    end: i64,
}

pub struct SeriesFilter;

impl TableFunctionCallbacks for SeriesFilter {
    type BindData = FilterBind;
    type GlobalState = ();
    type LocalState = FilterScan;

    fn bind(
        &self,
        context: &Context,
        arguments: BindArguments<'_>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        let CountBind { count } = bind_series(context, arguments, &bind_handle)?;
        Ok(FilterBind {
            count,
            lo: AtomicI64::new(0),
            hi: AtomicI64::new(count),
        })
    }

    fn init_local_state(
        &self,
        _context: &Context,
        bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        _column_data: InitColumnData<'_>,
    ) -> Result<Option<Self::LocalState>> {
        Ok(Some(FilterScan {
            position: bind_data.lo.load(Ordering::Relaxed),
            end: bind_data.hi.load(Ordering::Relaxed).min(bind_data.count),
        }))
    }

    fn exec(
        &self,
        _context: &Context,
        _bind_data: &Self::BindData,
        local_state: Option<&mut Self::LocalState>,
        _global_state: Option<&Self::GlobalState>,
        output: &mut DataChunk<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let scan = local_state.expect("init_local_state sets the scan");
        let rows = chunk_rows(output, scan.end - scan.position)?;
        fill_series(output, rows, scan.position)?;
        scan.position += rows as i64;
        Ok(())
    }
}

impl TableFilterPushdownCallbacks for SeriesFilter {
    fn pushdown_filter(
        &self,
        _context: &Context,
        bind_data: &Self::BindData,
        pushdown_data: PushdownData<'_>,
    ) -> Result<()> {
        for index in 0..pushdown_data.filter_count()? {
            let filter = pushdown_data.filter(index)?;
            let comparison = filter.expression_type()?;
            if !matches!(
                comparison,
                ExpressionType::CompareLessThan
                    | ExpressionType::CompareLessThanOrEqualTo
                    | ExpressionType::CompareGreaterThan
                    | ExpressionType::CompareGreaterThanOrEqualTo
                    | ExpressionType::CompareEqual
            ) {
                continue;
            }
            let column = filter.child(0)?;
            let bound = filter.child(1)?;
            if column.expression_type()? != ExpressionType::BoundColumnRef
                || bound.expression_type()? != ExpressionType::ValueConstant
                || pushdown_data.get_column_index(column.reference_index()?)? != 0
                || bound.return_type()?.id()? != LogicalTypeID::BigInt
            {
                continue;
            }
            let Some(bound) = bound.get_constant_value()?.get::<i64>()? else {
                continue;
            };
            let (lo, hi) = match comparison {
                ExpressionType::CompareLessThan => (None, Some(bound)),
                ExpressionType::CompareLessThanOrEqualTo => (None, Some(bound.saturating_add(1))),
                ExpressionType::CompareGreaterThan => (Some(bound.saturating_add(1)), None),
                ExpressionType::CompareGreaterThanOrEqualTo => (Some(bound), None),
                _ => (Some(bound), Some(bound.saturating_add(1))),
            };
            if let Some(lo) = lo {
                bind_data.lo.fetch_max(lo, Ordering::Relaxed);
            }
            if let Some(hi) = hi {
                bind_data.hi.fetch_min(hi, Ordering::Relaxed);
            }
            pushdown_data.accept_pushdown(index)?;
        }
        Ok(())
    }
}

/// The next row a scanning thread claims.
pub struct ParScan {
    next: AtomicI64,
}

pub struct SeriesPar;

impl TableFunctionCallbacks for SeriesPar {
    type BindData = CountBind;
    type GlobalState = ParScan;
    type LocalState = ();

    fn bind(
        &self,
        context: &Context,
        arguments: BindArguments<'_>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        let bind = bind_series(context, arguments, &bind_handle)?;
        bind_handle.set_order(OrderPreservation::None)?;
        Ok(bind)
    }

    fn init_global_state(
        &self,
        _context: &Context,
        _bind_data: &Self::BindData,
        _column_data: InitColumnData<'_>,
    ) -> Result<Option<Self::GlobalState>> {
        Ok(Some(ParScan {
            next: AtomicI64::new(0),
        }))
    }

    fn max_threads(
        &self,
        _context: &Context,
        bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        _column_data: InitColumnData<'_>,
    ) -> Result<Option<NonZeroUsize>> {
        let blocks = (bind_data.count + PAR_BLOCK_ROWS - 1) / PAR_BLOCK_ROWS;
        Ok(NonZeroUsize::new(blocks.max(1) as usize))
    }

    fn exec(
        &self,
        _context: &Context,
        bind_data: &Self::BindData,
        _local_state: Option<&mut Self::LocalState>,
        global_state: Option<&Self::GlobalState>,
        output: &mut DataChunk<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let scan = global_state.expect("init_global_state sets the scan");
        let first = scan.next.fetch_add(PAR_BLOCK_ROWS, Ordering::Relaxed);
        let rows = chunk_rows(output, (bind_data.count - first).min(PAR_BLOCK_ROWS))?;
        fill_series(output, rows, first)
    }
}

fn count_signature() -> SignatureBuilder {
    SignatureBuilder::new().param(FunctionParameter::new::<i64>("n"))
}

pub fn register(conn: &Connection) -> Result<()> {
    TableFunctionBuilder::new("vs_series", count_signature(), Series).register(conn)?;
    TableFunctionBuilder::new("vs_grid", count_signature(), Grid)
        .with_projection_pushdown(true)
        .register(conn)?;
    let words = SignatureBuilder::new()
        .param(FunctionParameter::new::<i64>("n"))
        .param(FunctionParameter::new::<i32>("len"));
    TableFunctionBuilder::new("vs_words", words, Words).register(conn)?;
    let repeat_row = SignatureBuilder::new()
        .param(FunctionParameter::new::<String>("s"))
        .param(FunctionParameter::new::<i64>("k"))
        .param(FunctionParameter::new::<i64>("n"));
    TableFunctionBuilder::new("vs_repeat_row", repeat_row, RepeatRow)
        .with_projection_pushdown(true)
        .register(conn)?;
    let series_step = SignatureBuilder::new()
        .param(FunctionParameter::new::<i64>("start"))
        .param(FunctionParameter::new::<i64>("stop"))
        .param(FunctionParameter::with_default("step", 1_i64).kind(FunctionParameterKind::Named));
    TableFunctionBuilder::new("vs_series_step", series_step, SeriesStep).register(conn)?;
    TableFunctionBuilder::new("vs_series_filter", count_signature(), SeriesFilter)
        .with_filter_pushdown()
        .register(conn)?;
    TableFunctionBuilder::new("vs_series_par", count_signature(), SeriesPar).register(conn)
}
