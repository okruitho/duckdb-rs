# Findings: aggregate_heap

These aggregates have states that own heap memory: `vs_longest` (an `Option<String>` / `std::optional<std::string>`),
`vs_distinct_count` (`HashSet<i64>` / `std::unordered_set<int64_t>`), `vs_median` (`Vec<f64>` / `std::vector<double>`),
`vs_sorted_list` (`Vec<i64>` / `std::vector<int64_t>`, LIST result), `vs_top3` (an inline 3-slot array, LIST result,
no destructor) and `vs_concat_sorted` (`Vec<String>` / `std::vector<std::string>`). Both sides read VARCHAR without
UTF-8 validation: Rust through `duckdb_neo::types::UncheckedString`, C++ through `varchar_t::view()`. They also write
results without validation: Rust `&str`, C++ `Arena::AddStringUnsafe` on a heap fetched once per finalize.

Ratios are Rust median / C++ median: above 1 means duckdb-neo is slower. The "before" column comes from the central
round-1 baseline, run on a quiet machine. The "after" column is a 3-repetition `run` on CPU 6 while other agents were
building, so differences below about 5% are noise. Rust's absolute times were mostly 0 to 15% higher than the
baseline (up to 30% in allocation-heavy cases), including cases whose Rust code did not change.

## Changes

- **C++: `input.GetRowCount()` and `input.GetStateCount()` are hoisted out of every loop.** Both are out-of-line
  calls into the v2 C API. In a loop condition they ran once per row. In `top3/ungrouped/1000000`, `UpdateInput::GetRowCount`
  was 39.6% self and `duckdb_v2_aggregate_function_update_get_row_count` 12.1%, against 19.5% for the loop body.
  This is the only change that moved the ratios.
- **Rust: update loops use `reader.iter().enumerate().for_each(..)` instead of `for`.** `VectorReaderIter::fold`
  resolves the storage kind (flat, dictionary or constant) once. `next()` dispatches on it for every row. The
  annotated `Top3::update` loop lost its per-row selection branch (`testb %r10b; je`, 36% of the loop's samples, and
  `cmpl $0x1`). The measured effect is within noise (`top3/ungrouped/1000000`: 1.04 ms before, 0.99 ms after, on the
  same machine).
- **Rust: LIST results are written with `writer.elements().write_slice(offset, &values)`** instead of one
  `elements().set` per element. This mirrors the C++ `std::copy`. The effect is within noise.
- **Cases:** unchanged since round 1.

| Case (rows) | Ratio before | Ratio after | What changed |
| --- | --- | --- | --- |
| `top3/ungrouped/100000` | 0.51 | 1.06 | C++ hoisted the row count; Rust `for_each` |
| `top3/ungrouped/1000000` | 0.29 | 1.06 | same |
| `top3/groups_1024_nulls_10/1000000` | 0.61 | 1.04 | same |
| `top3/running/1000000` | 0.73 | 1.03 | C++ hoisting (update, combine and finalize) |
| `top3/partitioned_sliding_9/1000000` | 0.84 | 1.05 | same |
| `longest/ungrouped/1000000` | 0.74 | 0.96 | C++ hoisting |
| `longest/groups_1024/1000000` | 0.75 | 0.94 | same |
| `longest/nulls_50/1000000` | 0.75 | 0.99 | same |
| `longest/whole_partition/1000000` | 0.83 | 1.00 | same |
| `longest/running/100000` | 0.83 | 1.09 | same; the remaining gap is §5.2 |
| `sorted_list/all_null/1000000` | 0.60 | 0.96 | same |
| `sorted_list/sliding_2/1000000` | 0.92 | 1.03 | same |
| `median/ungrouped/1000000` | 0.46 | 0.62 | same; the remaining gap is §5.3 |
| `median/groups_1024_nulls_10/1000000` | 0.63 | 0.73 | same |
| `distinct_count/low_card/1000000` | 2.30 | 3.71 | C++ hoisting made C++ faster; the gap is SipHash (§5.1) |
| `distinct_count/partitioned_sliding_9/1000000` | 0.97 | 1.08 | C++ hoisting |
| `concat_sorted/sliding_9/100000` | 1.10 | 1.15 | C++ hoisting; the gap is small-string optimization (§5.2) |
| `sorted_list/ungrouped/1000000` | 0.28 | 0.29 | unchanged: Rust's sort is faster (§5.3) |
| `median/sliding_257/100000` | 0.55 | 0.57 | unchanged: Rust's select is faster (§5.3) |
| `distinct_count/high_card/1000000` | 0.61 | 0.70 | unchanged: `hashbrown` beats node-based `std::unordered_set` (§5.1) |
| `concat_sorted/groups_quarter/1000000` | 0.68 | 0.71 | unchanged |

Every other case at 2,048 rows and above is within ±10% of 1, apart from the standard-library effects of §5.

## 1. Remaining gaps attributable to duckdb-neo

None of them is larger than a few percent of a case here. The family's large ratios all come from the standard
library (§5). In order of importance:

### 1.1 `finalize` only gets shared states, so sorting or selecting needs a copy

`AggregateCallbacks::finalize` receives `states: &[&Self::StateItem]` (`crates/duckdb-neo/src/aggregate/mod.rs`,
trait method `finalize`, fed by `finalize_callback`). The C++ API passes `void **`, and DuckDB's own `quantile`,
`list` and `string_agg` reorder their state in place in finalize.

- `Median::finalize` and `SortedList::finalize` copy every state's values into a reused scratch `Vec` and
  select or sort the copy. `ConcatSorted::finalize` sorts a scratch `Vec<&str>`.
- The C++ variants run `std::nth_element` / `std::sort` on the state's own vector.
- Evidence, `median/ungrouped/1000000`: `__memmove_avx512_unaligned_erms` is 14.9% self in Rust against 10.7% in C++.
  The difference is the scratch copy, which costs one more pass over 8 MB. It is hidden today only because Rust's
  `select_nth_unstable` beats libstdc++'s `__introselect` (25.2% against 43.3% self).
- Suggestion: give finalize the same mutable view that update and combine already get, for example
  `fn finalize(&self, bind_data: Option<&Self::BindData>, states: States<'_, Self::StateItem>, result: Vector<'_, Unknown>, result_offset: usize)`.
  It would offer mutable access one state at a time through `IndexMut`, so repeated state pointers stay sound. A
  separate opt-in `finalize_mut` would also work.

### 1.2 `for` over a `VectorReader` dispatches on the storage kind per row; the flat path still tests validity per row

`VectorReaderIter::next` (`crates/duckdb-neo/src/vector/reader.rs`) goes through `get_unchecked`, which runs
`physical_index_unchecked` for every row. Only `fold` (and so `for_each`) hoists the storage match, through
`fold_rows`.

- The natural `for (row, value) in reader.iter().enumerate()` therefore keeps a per-row branch in
  `Top3::update` (`testb %r10b,%r10b; je`: 36% of the loop's samples).
- Even with `for_each`, the flat branch of `fold_rows` calls `FlatReader::is_valid_unchecked(view.validity, row)`
  per row. LLVM does not unswitch the null check (`je` at 18% of the loop). clang compiles the C++
  `view.IsValid(i)` loop into a validity-free and selection-free loop.
- After the change, `top3/ungrouped` is at 1.03 to 1.06, so this is small, but it applies to every user who writes
  a plain `for`.
- Suggestion: in `fold_rows`' `Flat` arm, split `view.validity.is_null()` into a separate loop that passes
  `Some(..)` unconditionally. Document in `VectorReader::iter` that `for_each`/`fold` is the fast form. Or make
  `next()` cheaper by caching a "flat, no validity" flag in `VectorReaderIter`.

### 1.3 Per-callback wrapper costs (small)

These show up in window cases, where DuckDB calls update, combine and finalize with a few rows at a time.

- `VectorCollection::get_vector_at_mut::<T>` (`crates/duckdb-neo/src/data_chunk.rs`, `wrap`): 1.8% self in
  `top3/running/100000`, 1.0% (`UncheckedString`) in `longest/running/100000`. C++ `Vector::GetView` is 1.3% in
  `longest/running`.
- `ListWriter::set_entry` (`crates/duckdb-neo/src/vector/list_writer.rs`): 1.6% in `top3/running/100000`, for an
  entry check against the child length per row. C++ writes `list_entry_t` directly.
- `VectorWriter<String>::write_bytes`: 0.7 to 1.0% in the string window cases, about the same as C++
  `Arena::AddStringUnsafe` plus the store.

### 1.4 Growing a LIST child zero-fills it first

`Vector<List<T>>::set_child_size` (`crates/duckdb-neo/src/types/list.rs`) initializes new elements like `set_size`.
The uninitialized form is `unsafe`. The C++ `Vector::SetSize` only reserves. In `sorted_list/groups_quarter/1000000`,
`__memset_avx512_unaligned_erms` is 2.4% in Rust against 1.8% in C++. A safe "append these slices to the child"
helper would avoid the double write: for example `ListWriter::push_slice(row, &[T]) -> Result<()>`, which grows the
child itself.

## 2. API gaps and awkwardness

- **Appending to a LIST result.** Finalize's contract is "state `i` goes to row `result_offset + i`". DuckDB's own
  list aggregates therefore append at `ListVector::GetListSize(result)`.
  - There is no public accessor for a LIST vector's child length. `Vector::list_child_len` is `pub(crate)`; the
    family's `list_child_len` helper goes through `children().get_untyped_vector_at(0)?.len()`.
  - `ListWriter` always starts at child element 0, so `set_elements` cannot append after existing elements. The
    family writes `elements().write_slice(offset, ..)` and then `set_entry`.
  - Every shape exercised here (grouped, ungrouped, running, sliding, partitioned and RANGE windows) finalizes with
    `result_offset = 0` into an empty child. So this is about the contract, not a failure today.
  - Suggestion: a `list_writer_appending()` that starts at the current child length, plus a public `child_len()`.
- **Boilerplate.** Every `AggregateCallbacks` implementation has to write `bind` (returning `Ok(())`) and `init`
  (returning `Default::default()`). Default methods for `BindData = ()` and `StateItem: Default` would remove
  twelve methods from this family. The C++ API has no state helpers at all (raw `void **`), so the C++ variant
  defines `StateSize`/`StateInit`/`StateDestroy` templates of its own.
- **Destroy-callback parity.** duckdb-neo registers a destroy callback only when `StateItem` needs drop. The C++
  variant mirrors that with `if constexpr (!std::is_trivially_destructible_v<STATE>)`, so neither side registers
  one for `vs_top3`.
- **`UncheckedString`.** It is a safe type whose docs say invalid UTF-8 is undefined behaviour. That is fine for
  DuckDB VARCHAR, which is always valid. The checked `String` marker runs `from_utf8` on every row.

## 3. Pitfalls in the C++ API (fixed in the C++ variant)

- `UpdateInput::GetRowCount()`, `GetStateCount()`, `GetStates()`, `GetArg()` and `GetResult()` are out-of-line calls
  into the C API. A loop condition `i < input.GetRowCount()` cost about half of `top3/ungrouped/1000000` before it
  was hoisted. duckdb-neo reads them once per callback in its trampolines, so a Rust user cannot make this mistake.
- `Vector::AssignString`, which the header recommends, fetches the logical type and the heap and validates UTF-8 on
  every call. The C++ variant fetches `GetHeap()` once and uses `Arena::AddStringUnsafe`.
- `Vector::SetNull` checks the row against the vector's size, which windows set only after finalize. The C++
  variant writes NULLs through a lazily fetched `ValidityMask` instead.

## 4. Where duckdb-neo is faster than the C++ API

- **Per-state init:** in `sorted_list/groups_quarter/1000000`, where each of the 250,000 groups gets a state, the
  `duckdb::cxx` `AggregateFunction::SetInitCallback` trampoline is 9.0% self. Its samples sit after the call into
  the exception guard. duckdb-neo's `init_callback` is 1.8% (plus `CV2AggregateInit` 3.0% in libduckdb).
  Together with §5, this makes the many-group cases 0.71 to 0.85.
- Everything else Rust wins here comes from the standard library, not from the API (§5).

## 5. Differences that are not about either API

1. **Hash sets.** Rust `HashSet<i64>` is SwissTable with SipHash-1-3. C++ `std::unordered_set<int64_t>` uses
   identity hashing with one heap node per element.
   - `distinct_count/low_card/1000000` (100 distinct values, so every insert hits) is 3.71. Rust spends 31.8% in
     `RandomState::hash_one` and 16.2% in `DefaultHasher::write`; C++ spends 59% in `_Hashtable::_M_insert`.
   - `distinct_count/high_card` is 0.50 to 0.70, `filter_clause` 0.41 to 0.63 and `threads_4` 0.51 to 0.55. Here
     C++ allocates a node per distinct value: `malloc` 33%, `_int_free` 17% and `unlink_chunk` 9%, against Rust's
     open addressing (`HashMap::insert` 54%, `reserve_rehash` 9%).
   - A Rust user who cared would swap in a faster hasher. That is not a duckdb-neo question.
2. **Small-string optimization.** libstdc++'s `std::string` stores up to 15 bytes inline; Rust's `String` always
   allocates.
   - The short keys of `concat_sorted/sliding_9` (1.15 to 1.18) and `longest/running` (1.08 to 1.11) are copied into
     fresh states for every window frame. Rust: `malloc` 18.4%, `_int_free` 17.9%, `realloc` 8.5%, `_int_malloc` 7.6%
     in `concat_sorted/sliding_9`. C++: `malloc` 7.9%, `_int_free` 5.0%.
   - In `longest/running`, the allocator functions are about 18% of Rust against about 2% of C++.
   - `concat_sorted/nulls_10` at 1M rows (1.12) has the same cause.
3. **Sort and select.** Rust's `sort_unstable` (ipnsort) and `select_nth_unstable` beat libstdc++'s `std::sort`
   (introsort) and `std::nth_element` (introselect) on random `i64`/`f64`.
   - `sorted_list/ungrouped` is 0.26 to 0.29: `quicksort` plus `small_sort_network` 60% against `__introsort_loop`
     plus `__final_insertion_sort` 82%.
   - `median/ungrouped` is 0.37 to 0.62, and `median/sliding_257` is 0.48 to 0.61.
   - `vs_median` compares with `f64::total_cmp` in Rust and `operator<` in C++. They agree on the NaN-free inputs
     of the cases.
