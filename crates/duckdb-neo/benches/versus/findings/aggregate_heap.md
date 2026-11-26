# Findings: aggregate_heap

Aggregates whose states own heap memory (`vs_longest`, `vs_distinct_count`, `vs_median`, `vs_sorted_list`,
`vs_top3`, `vs_concat_sorted`). Round 1 is a correctness pass: the timings below are a smoke test on a busy machine
and support no conclusion yet.

## 1. `finalize` sees states only as shared references, so sorting or selecting needs a copy

`AggregateCallbacks::finalize` receives `states: &[&Self::StateItem]`. The C++ API hands finalize the same `void **`
as update (`FinalizeInput::GetStates`), and DuckDB's own aggregates (`quantile`, `list`, `string_agg`) reorder their
state in place in finalize.

- C++ (`cpp/aggregate_heap.cpp`): `MedianFinalize` runs `std::nth_element` on the state's `std::vector<double>`;
  `SortedListFinalize` and `ConcatSortedFinalize` run `std::sort` on the state's vector.
- Rust (`rust/aggregate_heap.rs`): `Median::finalize` and `SortedList::finalize` copy every state's values into a
  scratch `Vec` (reused across the batch) and select or sort the copy; `ConcatSorted::finalize` collects a scratch
  `Vec<&str>` of the state's strings and sorts that.

So every finalize copies all of a state's values once more: 8 bytes per value for median and sorted_list, 16 per
string for concat_sorted. Interior mutability is not a real way out: `StateItem: Send + Sync` rules out `RefCell`,
and a `Mutex` per state only to sort it in finalize is not what anyone would write. A fix could pass finalize a
`States<'_, T>` (mutable one state at a time through `IndexMut`, as update and combine get), or a separate
`finalize_mut` hook.

## 2. A `LIST` result cannot be appended to with `ListWriter::set_elements`

Finalize's contract is "state `i` goes to row `result_offset + i`", so a `LIST` result may already hold the elements
of earlier rows (DuckDB's own `list` aggregate appends at `ListVector::GetListSize(result)`). Two gaps:

- There is no public accessor for a `LIST` vector's child length. `Vector::list_child_len` is `pub(crate)`; the
  family writes `result.children().expect(..).get_untyped_vector_at(0)?.len()` (helper `list_child_len`).
- `ListWriter` always starts at child element 0 ("every writer starts at the first element"), and `child_offset` only
  moves forward through `set_entry` or `set_elements`. So `set_elements(row, values)`, the convenient whole-row
  call, would overwrite earlier rows' elements when the child was not empty. `SortedList::finalize` and
  `Top3::finalize` use the columnar form instead: `writer.elements().set(offset + j, value)` per element, then
  `writer.set_entry(row, ListIndex::new(offset, len))`.

The C++ variant does the natural thing for a raw API: `child.GetSize()`, `child.SetSize(old + n)`, then plain
`int64_t *` and `list_entry_t *` writes.

In practice, every shape the cases exercise (ungrouped, grouped, running, sliding, partitioned and `RANGE` windows)
finalizes with `result_offset = 0` into an empty child: a temporary variant that used `set_elements` from offset 0
passed every check. The workaround is about the contract, not about something that fails today. A `ListWriter` that
starts at the child's current length (or a `list_writer_at(offset)`) plus a public `child_len()` would remove it.

## 3. Growing a `LIST` child zero-fills it first

`Vector<List<T>>::set_child_size` initializes the new elements (zeroed, like `set_size`), and they are then
overwritten one by one. `set_child_size_uninit` skips that but is `unsafe`. The C++ `Vector::SetSize` only reserves.
Every `vs_sorted_list` and `vs_top3` element is therefore written twice on the Rust side. Profile this in round 2
(`sorted_list/ungrouped/1000000` and `sorted_list/groups_quarter/1000000` write about 1M elements per query).

## 4. Reading `VARCHAR` without UTF-8 validation needs `UncheckedString`

The C++ variant reads `varchar_t::view()`, which is unchecked. The idiomatic duckdb-neo marker for `VARCHAR`,
`String`, runs `std::str::from_utf8` on every row and panics on failure. To do the same work as the C++ side, the Rust
update callbacks read with `UncheckedString`, as `benches/support/row_hash.rs` does. It is safe to name, but its
docs say invalid UTF-8 is undefined behaviour. That is fine here because DuckDB guarantees valid `VARCHAR`, but a
safe API with a "must" clause is worth a second look.

For output the C++ side fetches the result's heap once (`GetHeap`) and stores `Arena::AddStringUnsafe` values.
`Vector::AssignString`, which the header recommends, fetches the heap and the logical type and validates UTF-8 on
every call. The Rust writers take `&str` and do not validate, so both sides skip validation.

## 5. Language differences that are not API costs

- `vs_distinct_count` uses each language's standard set: Rust `HashSet<i64>` (SipHash, open addressing) against
  C++ `std::unordered_set<int64_t>` (identity hash, one node per element). Their costs differ for reasons that have
  nothing to do with either DuckDB API, so do not read the distinct_count ratios as duckdb-neo overhead.
- `vs_median` selects with `f64::total_cmp` in Rust and `operator<` in C++. They agree on the cases' NaN-free input.
- duckdb-neo registers a destroy callback only when `StateItem` needs drop. The C++ variant mirrors that with
  `if constexpr (!std::is_trivially_destructible_v<STATE>)`. So `vs_top3` (an inline `[i64; 3]` state) has no destroy
  callback on either side, and DuckDB can stream `top3/running`.

## 6. Ergonomics

Every `AggregateCallbacks` implementation has to spell out `bind` (returning `Ok(())`) and `init` (returning
`Default::default()`), even with `BindData = ()` and a `Default` state. Six aggregates means twelve boilerplate
methods. Default methods for those cases would remove them. The C++ API has no state helpers
at all (raw `void **`), so the C++ variant adds `StateSize`/`StateInit`/`StateDestroy` templates of its own.
