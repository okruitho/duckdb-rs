//! Aggregates whose states own heap memory, so state construction, destruction and combine are
//! part of every call:
//!
//! - `vs_longest(VARCHAR) -> VARCHAR`: the longest string in bytes, ties broken by the smallest in
//!   byte order; NULL without non-NULL rows.
//! - `vs_distinct_count(BIGINT) -> BIGINT`: the number of distinct non-NULL values.
//! - `vs_median(DOUBLE) -> DOUBLE`: the lower median of the non-NULL values; NULL without any.
//! - `vs_sorted_list(BIGINT) -> BIGINT[]`: the non-NULL values sorted ascending; NULL without any.
//! - `vs_top3(BIGINT) -> BIGINT[]`: the three largest values, descending; NULL without any.
//! - `vs_concat_sorted(VARCHAR) -> VARCHAR`: the non-NULL strings sorted in byte order and joined
//!   with ','; NULL without any.

use std::collections::HashSet;

use duckdb_neo::{
    DuckDBType, Result,
    aggregate::{AggregateCallbacks, AggregateFunctionBuilder, ReturnTypeHandle, States},
    bind_arguments::BindArguments,
    connection::{Connection, Context},
    data_chunk::VectorCollection,
    enums::AggregateFunctionProperty,
    logical_type::LogicalType,
    signature::{FunctionParameter, SignatureBuilder},
    types::{List, UncheckedString},
    vector::{NativeElement, StorageKind, Unknown, Vector},
};

fn argument<T: NativeElement>(data: &mut VectorCollection) -> Result<Vector<'_, T>> {
    let mut vector = data.get_vector_at_mut::<T>(0)?;
    if vector.storage_kind() == StorageKind::Other {
        vector.flatten()?;
    }
    Ok(vector)
}

fn register_aggregate<T: AggregateCallbacks<BindData = ()>>(
    conn: &Connection,
    name: &str,
    parameter: FunctionParameter,
    return_type: LogicalType,
    callbacks: T,
) -> Result<()> {
    let signature = SignatureBuilder::new().param(parameter).return_type(return_type);
    AggregateFunctionBuilder::new(name, signature, callbacks)
        .set_property(AggregateFunctionProperty::AggregateOrderDependent(false))
        .register(conn)
}

#[derive(Debug, Default)]
struct LongestState {
    value: Option<String>,
}

/// Whether `candidate` beats `current`: longer in bytes, or as long and smaller in byte order.
fn is_longer(candidate: &str, current: &str) -> bool {
    candidate.len() > current.len() || (candidate.len() == current.len() && candidate < current)
}

impl LongestState {
    fn offer(&mut self, candidate: &str) {
        match &mut self.value {
            Some(current) => {
                if is_longer(candidate, current) {
                    candidate.clone_into(current);
                }
            }
            None => self.value = Some(candidate.to_owned()),
        }
    }
}

struct Longest;

impl AggregateCallbacks for Longest {
    type BindData = ();
    type StateItem = LongestState;

    fn bind(&self, _context: &Context, _arguments: BindArguments<'_>, _result: ReturnTypeHandle<'_>) -> Result<()> {
        Ok(())
    }

    fn init(&self, _bind_data: Option<&()>) -> LongestState {
        LongestState::default()
    }

    fn update(
        &self,
        _bind_data: Option<&()>,
        data: &mut VectorCollection,
        mut states: States<'_, LongestState>,
    ) -> Result<()> {
        let values = argument::<UncheckedString>(data)?;
        // `for_each` resolves the vector's storage once; a `for` loop dispatches on it per row.
        values.reader()?.iter().enumerate().for_each(|(row, value)| {
            if let Some(value) = value {
                states[row].offer(value);
            }
        });
        Ok(())
    }

    fn combine(
        &self,
        _bind_data: Option<&()>,
        source: &States<'_, LongestState>,
        mut target: States<'_, LongestState>,
    ) -> Result<()> {
        for index in 0..target.len() {
            if let Some(value) = &source[index].value {
                target[index].offer(value);
            }
        }
        Ok(())
    }

    fn finalize(
        &self,
        _bind_data: Option<&()>,
        states: &[&LongestState],
        result: Vector<'_, Unknown>,
        result_offset: usize,
    ) -> Result<()> {
        let mut result = result.cast::<String>()?;
        let mut writer = result.writer()?;
        for (index, state) in states.iter().enumerate() {
            writer.set_option(result_offset + index, state.value.as_deref())?;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct DistinctCountState {
    values: HashSet<i64>,
}

struct DistinctCount;

impl AggregateCallbacks for DistinctCount {
    type BindData = ();
    type StateItem = DistinctCountState;

    fn bind(&self, _context: &Context, _arguments: BindArguments<'_>, _result: ReturnTypeHandle<'_>) -> Result<()> {
        Ok(())
    }

    fn init(&self, _bind_data: Option<&()>) -> DistinctCountState {
        DistinctCountState::default()
    }

    fn update(
        &self,
        _bind_data: Option<&()>,
        data: &mut VectorCollection,
        mut states: States<'_, DistinctCountState>,
    ) -> Result<()> {
        let values = argument::<i64>(data)?;
        values.reader()?.iter().enumerate().for_each(|(row, value)| {
            if let Some(value) = value {
                states[row].values.insert(*value);
            }
        });
        Ok(())
    }

    fn combine(
        &self,
        _bind_data: Option<&()>,
        source: &States<'_, DistinctCountState>,
        mut target: States<'_, DistinctCountState>,
    ) -> Result<()> {
        for index in 0..target.len() {
            target[index].values.extend(source[index].values.iter().copied());
        }
        Ok(())
    }

    fn finalize(
        &self,
        _bind_data: Option<&()>,
        states: &[&DistinctCountState],
        result: Vector<'_, Unknown>,
        result_offset: usize,
    ) -> Result<()> {
        let mut result = result.cast::<i64>()?;
        let mut writer = result.writer()?;
        for (index, state) in states.iter().enumerate() {
            writer.set(result_offset + index, &(state.values.len() as i64))?;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct MedianState {
    values: Vec<f64>,
}

struct Median;

impl AggregateCallbacks for Median {
    type BindData = ();
    type StateItem = MedianState;

    fn bind(&self, _context: &Context, _arguments: BindArguments<'_>, _result: ReturnTypeHandle<'_>) -> Result<()> {
        Ok(())
    }

    fn init(&self, _bind_data: Option<&()>) -> MedianState {
        MedianState::default()
    }

    fn update(
        &self,
        _bind_data: Option<&()>,
        data: &mut VectorCollection,
        mut states: States<'_, MedianState>,
    ) -> Result<()> {
        let values = argument::<f64>(data)?;
        values.reader()?.iter().enumerate().for_each(|(row, value)| {
            if let Some(value) = value {
                states[row].values.push(*value);
            }
        });
        Ok(())
    }

    fn combine(
        &self,
        _bind_data: Option<&()>,
        source: &States<'_, MedianState>,
        mut target: States<'_, MedianState>,
    ) -> Result<()> {
        for index in 0..target.len() {
            target[index].values.extend_from_slice(&source[index].values);
        }
        Ok(())
    }

    // Finalize only sees shared states, so the selection runs on a copy.
    fn finalize(
        &self,
        _bind_data: Option<&()>,
        states: &[&MedianState],
        result: Vector<'_, Unknown>,
        result_offset: usize,
    ) -> Result<()> {
        let mut result = result.cast::<f64>()?;
        let mut writer = result.writer()?;
        let mut scratch = Vec::new();
        for (index, state) in states.iter().enumerate() {
            if state.values.is_empty() {
                writer.set_null(result_offset + index)?;
                continue;
            }
            scratch.clear();
            scratch.extend_from_slice(&state.values);
            let middle = (scratch.len() - 1) / 2;
            let (_, median, _) = scratch.select_nth_unstable_by(middle, f64::total_cmp);
            writer.set(result_offset + index, median)?;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct SortedListState {
    values: Vec<i64>,
}

struct SortedList;

impl AggregateCallbacks for SortedList {
    type BindData = ();
    type StateItem = SortedListState;

    fn bind(&self, _context: &Context, _arguments: BindArguments<'_>, _result: ReturnTypeHandle<'_>) -> Result<()> {
        Ok(())
    }

    fn init(&self, _bind_data: Option<&()>) -> SortedListState {
        SortedListState::default()
    }

    fn update(
        &self,
        _bind_data: Option<&()>,
        data: &mut VectorCollection,
        mut states: States<'_, SortedListState>,
    ) -> Result<()> {
        let values = argument::<i64>(data)?;
        values.reader()?.iter().enumerate().for_each(|(row, value)| {
            if let Some(value) = value {
                states[row].values.push(*value);
            }
        });
        Ok(())
    }

    fn combine(
        &self,
        _bind_data: Option<&()>,
        source: &States<'_, SortedListState>,
        mut target: States<'_, SortedListState>,
    ) -> Result<()> {
        for index in 0..target.len() {
            target[index].values.extend_from_slice(&source[index].values);
        }
        Ok(())
    }

    // Window operators finalize into a result whose child already holds earlier rows, so append
    // after them. Finalize only sees shared states, so the sort runs on a copy.
    fn finalize(
        &self,
        _bind_data: Option<&()>,
        states: &[&SortedListState],
        result: Vector<'_, Unknown>,
        result_offset: usize,
    ) -> Result<()> {
        let mut sorted = Vec::new();
        result.cast::<List<i64>>()?.list_appender(|lists| {
            for (index, state) in states.iter().enumerate() {
                let row = result_offset + index;
                if state.values.is_empty() {
                    lists.set_null(row)?;
                    continue;
                }
                sorted.clear();
                sorted.extend_from_slice(&state.values);
                sorted.sort_unstable();
                lists.set_slice(row, &sorted)?;
            }
            Ok(())
        })
    }
}

/// The largest values seen, descending, in `values[..len]`.
#[derive(Debug, Default)]
struct Top3State {
    values: [i64; 3],
    len: usize,
}

impl Top3State {
    fn offer(&mut self, value: i64) {
        let mut slot = self.len;
        if self.len == self.values.len() {
            if value <= self.values[slot - 1] {
                return;
            }
            slot -= 1;
        } else {
            self.len += 1;
        }
        while slot > 0 && self.values[slot - 1] < value {
            self.values[slot] = self.values[slot - 1];
            slot -= 1;
        }
        self.values[slot] = value;
    }
}

struct Top3;

impl AggregateCallbacks for Top3 {
    type BindData = ();
    type StateItem = Top3State;

    fn bind(&self, _context: &Context, _arguments: BindArguments<'_>, _result: ReturnTypeHandle<'_>) -> Result<()> {
        Ok(())
    }

    fn init(&self, _bind_data: Option<&()>) -> Top3State {
        Top3State::default()
    }

    fn update(
        &self,
        _bind_data: Option<&()>,
        data: &mut VectorCollection,
        mut states: States<'_, Top3State>,
    ) -> Result<()> {
        let values = argument::<i64>(data)?;
        values.reader()?.iter().enumerate().for_each(|(row, value)| {
            if let Some(value) = value {
                states[row].offer(*value);
            }
        });
        Ok(())
    }

    fn combine(
        &self,
        _bind_data: Option<&()>,
        source: &States<'_, Top3State>,
        mut target: States<'_, Top3State>,
    ) -> Result<()> {
        for index in 0..target.len() {
            let source = &source[index];
            let target = &mut target[index];
            for &value in &source.values[..source.len] {
                target.offer(value);
            }
        }
        Ok(())
    }

    fn finalize(
        &self,
        _bind_data: Option<&()>,
        states: &[&Top3State],
        result: Vector<'_, Unknown>,
        result_offset: usize,
    ) -> Result<()> {
        result.cast::<List<i64>>()?.list_appender(|lists| {
            for (index, state) in states.iter().enumerate() {
                let row = result_offset + index;
                if state.len == 0 {
                    lists.set_null(row)?;
                    continue;
                }
                lists.set_slice(row, &state.values[..state.len])?;
            }
            Ok(())
        })
    }
}

#[derive(Debug, Default)]
struct ConcatSortedState {
    values: Vec<String>,
}

struct ConcatSorted;

impl AggregateCallbacks for ConcatSorted {
    type BindData = ();
    type StateItem = ConcatSortedState;

    fn bind(&self, _context: &Context, _arguments: BindArguments<'_>, _result: ReturnTypeHandle<'_>) -> Result<()> {
        Ok(())
    }

    fn init(&self, _bind_data: Option<&()>) -> ConcatSortedState {
        ConcatSortedState::default()
    }

    fn update(
        &self,
        _bind_data: Option<&()>,
        data: &mut VectorCollection,
        mut states: States<'_, ConcatSortedState>,
    ) -> Result<()> {
        let values = argument::<UncheckedString>(data)?;
        values.reader()?.iter().enumerate().for_each(|(row, value)| {
            if let Some(value) = value {
                states[row].values.push(value.to_owned());
            }
        });
        Ok(())
    }

    fn combine(
        &self,
        _bind_data: Option<&()>,
        source: &States<'_, ConcatSortedState>,
        mut target: States<'_, ConcatSortedState>,
    ) -> Result<()> {
        for index in 0..target.len() {
            target[index].values.extend_from_slice(&source[index].values);
        }
        Ok(())
    }

    // Finalize only sees shared states, so the sort runs on borrowed strings.
    fn finalize(
        &self,
        _bind_data: Option<&()>,
        states: &[&ConcatSortedState],
        result: Vector<'_, Unknown>,
        result_offset: usize,
    ) -> Result<()> {
        let mut result = result.cast::<String>()?;
        let mut writer = result.writer()?;
        let mut sorted: Vec<&str> = Vec::new();
        let mut joined = String::new();
        for (index, state) in states.iter().enumerate() {
            if state.values.is_empty() {
                writer.set_null(result_offset + index)?;
                continue;
            }
            sorted.clear();
            sorted.extend(state.values.iter().map(String::as_str));
            sorted.sort_unstable();
            joined.clear();
            for (position, value) in sorted.iter().enumerate() {
                if position > 0 {
                    joined.push(',');
                }
                joined.push_str(value);
            }
            writer.set(result_offset + index, &joined)?;
        }
        Ok(())
    }
}

pub fn register(conn: &Connection) -> Result<()> {
    let varchar = || FunctionParameter::new::<String>("value");
    let bigint = || FunctionParameter::new::<i64>("value");
    let double = FunctionParameter::new::<f64>("value");
    register_aggregate(conn, "vs_longest", varchar(), String::logical_type(conn)?, Longest)?;
    register_aggregate(
        conn,
        "vs_distinct_count",
        bigint(),
        i64::logical_type(conn)?,
        DistinctCount,
    )?;
    register_aggregate(conn, "vs_median", double, f64::logical_type(conn)?, Median)?;
    register_aggregate(
        conn,
        "vs_sorted_list",
        bigint(),
        List::<i64>::logical_type(conn)?,
        SortedList,
    )?;
    register_aggregate(conn, "vs_top3", bigint(), List::<i64>::logical_type(conn)?, Top3)?;
    register_aggregate(
        conn,
        "vs_concat_sorted",
        varchar(),
        String::logical_type(conn)?,
        ConcatSorted,
    )
}
