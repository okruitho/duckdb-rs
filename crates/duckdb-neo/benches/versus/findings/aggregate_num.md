# Findings: `aggregate_num`

Functions: `vs_sum`, `vs_avg`, `vs_count_if`, `vs_spread`, `vs_wavg`, `vs_sum_scaled`. All six have
plain states (see `../rust/aggregate_num.rs`). The safe duckdb-neo API can express every one of them
without workarounds. Before/after ratios are Rust median / C++ median: "before" is the central round-1
baseline, "after" is 3 repetitions on CPU 5 of a busy machine, so a few percent is noise. Profiles are flat
`perf` profiles from `neo_versus.py perf` (self time).

## What changed

Rust, every `update`:

- The argument is read through the typed accessor `data.get_vector_at_mut::<T>(0)` instead of
  `get_untyped_vector_at_mut(0)` followed by `cast::<T>()`, so the logical type is validated once instead
  of twice. `vs_wavg` still needs `get_vector_array_mut` and `cast`, because it borrows two arguments.
- The row loops are `reader.iter().enumerate().for_each(..)` (and `CombinedVectorIterator::enumerate().for_each`)
  instead of `for` loops. `VectorReaderIter::fold` resolves the storage kind once, outside the loop
  (`vector/reader.rs`, `fold_rows`), while `next()` matches on it for every row.

C++: no change in this round. Round 1 had already moved `GetRowCount()`/`GetStateCount()` out of the
loop condition. The remaining C++ costs are in `duckdb::cxx` itself (see below).

| Case | Rows | Before | After | What changed |
| --- | ---: | ---: | ---: | --- |
| `mixed/four_aggregates` | 2,000,000 | 1.12 slower | 1.05 same | `for_each`; an A/B on the same machine with only the loop form reverted gave 1.13 slower |
| `wavg/group_1024_nulls_10` | 2,000,000 | 1.08 | 1.05 | `for_each` |
| `sum/eight_calls` | 2,000,000 | 1.05 | 0.98 | `for_each`, typed accessor |
| `sum/group_1024` | 2,000,000 | 1.05 | 0.99 | `for_each`, typed accessor |
| `window/sliding_2`, `sliding_9` | 100,000 | 1.06, 1.07 | 1.02, 1.04 | both |
| `window/running` | 100,000 | 2.10 slower | 1.98 slower | both; the time is in the wrapper (gap 1) |
| `window/centered_9` | 100,000 | 1.17 slower | 1.17 to 1.23 slower | none measurable (gap 2) |
| all other cases | 100,000 | 0.87 to 1.05 | 0.87 to 1.06 | unchanged within noise |

## Remaining gaps attributable to duckdb-neo

### 1. Fixed cost per update/finalize call: `window/running` 1.98 to 2.12 slower

A running window without `ORDER BY` streams, and DuckDB calls update and finalize once per row. Ratios:
1.49 to 1.56 at 2,048 rows, 1.98 to 2.10 at 100,000, 2.12 at 2,000,000. Profile at 100,000 rows:

```
rust (after)                                                   cpp
19.8%  VectorCollection::get_vector_at_mut::<i64>              32.9%  libduckdb Vector::Reference (engine)
14.3%  libduckdb CV2AggregateUpdate (engine)                   24.8%  libduckdb CV2AggregateUpdate (engine)
14.0%  libduckdb Vector::Reference (engine)                     8.3%  cxx::Vector::GetView
11.5%  aggregate::finalize_callback::<Sum>                      5.8%  cxx::FinalizeInput::GetStateCount
 9.7%  VectorReader<f64>::new (code-folded; reads i64 here)     4.3%  cxx::FinalizeInput::GetResultOffset
 4.7%  Sum::finalize                                            4.1%  SumFinalize
 3.8%  libduckdb duckdb_v2_vector_get_data_mutable              1.5%  std::string::_M_replace
 3.0%  Sum::update                                              1.3%  libduckdb duckdb_v2_error_info_set_text
 1.9%  readable::<i64>
 1.4%  libduckdb duckdb_v2_vector_get_logical_type
 1.0%  libduckdb duckdb_v2_logical_type_get_id
```

About 55% of the Rust time is wrapper code that runs once per callback. The two engine symbols take the
same absolute time in both languages. Where it comes from:

- `Vector::from_handle` (`crates/duckdb-neo/src/vector/mod.rs:188`) runs for the argument in every update
  (inside `get_vector_at_mut`) and for the result in every finalize (`aggregate/mod.rs:344`). Each run:
  - makes five C calls: logical type, vector type, size, child count, then `validate_as` reads the type id;
  - destroys the logical type again on drop;
  - calls the out-of-line `StorageKind::try_from` (`builder_helpers.rs:324`, a jump table returning
    `Result<StorageKind, Error>` through memory).

  In `get_vector_at_mut` and `finalize_callback`, 62% and 74% of the samples land on the instructions right
  after the `try_from` return, which looks like a store-forwarding stall on the returned value. C++
  `GetArg` + `GetView` makes two C calls.
- `VectorReader::new` (`vector/reader.rs`, through `Vector::view`, `vector/mod.rs:405`) copies the
  32-byte view that `duckdb_v2_vector_get_view` returned with 16-byte loads. 94% of its samples sit on that
  copy.
- `finalize_callback` (`aggregate/mod.rs:327`) calls `set_size` on the result vector in every window
  finalize. That resizes and zero-fills the new rows (`init_rows`, `memset`) before the user overwrites
  them.

Suggestions:

- Make `StorageKind::try_from` and `LogicalTypeID::try_from` `#[inline]`. Better, give the hot path an
  infallible conversion that maps unknown values to `Other`.
- In `Vector::from_handle`, skip `duckdb_v2_vector_get_child_count` for non-nested type ids. Have
  `validate_as` reuse the type id that construction already read rather than calling
  `duckdb_v2_logical_type_get_id` again.
- In `finalize_callback`, use `set_size_uninit` for the rows the finalizer must write anyway; the
  `finalize` contract already requires writing every one of them.
- A typed one-argument fast path, e.g. `VectorCollection::reader_at::<T>(index) -> Result<VectorReader<T>>`,
  could build the reader straight from the handle without materializing a full `Vector`. It would serve
  update callbacks that only read.

### 2. Combined reading of two non-flat arguments: `window/centered_9` 1.17 to 1.23 slower

`vs_wavg` over a centered 9-row frame: the segment tree calls update on a few dictionary-sliced rows at a
time. The user loop costs `Wavg::update` 16.7% in Rust against `WavgUpdate` 11.9% in C++. Wrapper code
adds about 8% more, spread over:

| Symbol | Self |
| --- | ---: |
| `VectorReader::new` | 1.7% |
| `get_vector_array_mut::<2>` | 1.5% |
| `wrap` | 1.3% |
| `readable` | 1.2% |
| the `Vec` collect in `collect_array` (`data_chunk.rs:32`) | 1.1% |
| `malloc`/`free` | about 3% |

The loop itself:

- C++ (`clang`): the view gives each argument a selection pointer or none. clang unswitches the loop on
  "no validity mask", so the hot loop is about 20 instructions: one selection load per argument, then the
  accumulation.
- Rust: `CombinedVectorIterator` only specializes the all-flat case (`vector/combined.rs`,
  `fold_indexed`). For dictionary inputs it calls `VectorReader::get_unchecked` per reader per row, which
  matches on `ReaderStorage` and checks validity every time. In the annotation, the two per-row storage
  branches take 7% of the function each, and the loop is several times longer than clang's.

Suggestions:

- Let `CombinedVectorIterator` resolve each reader's storage once: normalize to a
  `(data, Option<&[u32]> selection, Option<&[u64]> validity)` view, and branch once on "every reader has
  no validity mask".
- Have `get_vector_array_mut` return its array without the intermediate `Vec` (`collect_array` collects
  into a `Vec` and converts).

### 3. Smaller per-row and per-call costs (about 5 to 8% in segment-tree windows)

`window/sliding_257` is 1.06 to 1.10 and `sliding_9` 1.04 to 1.07. User code: Rust `Sum::update` 9.6% +
`Sum::combine` 9.9%, against C++ `SumUpdate` 9.2% + `SumCombine` 8.0%. The rest of the Rust binary,
beyond those two, is about 3.5% (C++: about 2%); the whole binary is 23.1% of samples against 19.1%.

- Combine: the Rust loop bounds-checks `source[index]` on every row and adds the two fields with two
  scalar read-modify-writes. clang unrolls the C++ loop by two and adds both fields with one `paddq`. This
  is compiler-level (the `States` index check), worth about 2% of the case. A `States::zip_mut(&source)`
  (or an `iter_mut` on the target view when DuckDB guarantees distinct combine targets) would drop the
  check.

## API gaps and awkwardness (still true)

- **`bind` is mandatory.** `AggregateCallbacks::bind` has no default, so five functions carry an empty
  `bind`, and every one registers a bind callback plus a `set_bind_data(())`. The C++ API registers bind
  only for `vs_sum_scaled`. This costs only at prepare time; a default `bind` for `BindData = ()` would
  remove the boilerplate.
- **Bind data is an `Option`.** It arrives as `Option<&BindData>` even though `bind` always sets it.
  `vs_sum_scaled` maps `None` to an error by hand; C++ `GetBindData<T>()` throws by itself.
- **Callbacks cannot be shared across `BindData` types.** `vs_sum_scaled` repeats `vs_sum`'s combine and
  finalize because the trait ties them to `BindData`. C++ reuses `SumCombine`/`SumFinalize`.
- **Two arguments mean an allocation and a cast.** Reading two arguments needs `get_vector_array_mut`,
  which allocates a `Vec` per call and returns `Vector<Unknown>`, so each argument also needs `cast`. There
  is no typed `get_vectors_mut::<(f64, f64)>([0, 1])`.
- **Flattening `OTHER` needs a helper.** No accessor narrows to `T` and also flattens `StorageKind::Other`,
  hence the `readable` helper. C++ needs the same (`GetView` throws on `OTHER`).
- **The loop form matters.** A `for` loop over a reader calls `next()`, which matches on the storage kind
  every row; `for_each`/`fold` resolve it once. The difference is 10% on `mixed/four_aggregates` at
  2,000,000 rows. Nothing in the API tells users this. `next()` could specialize via a storage-kind
  enum-of-iterators, or the docs should recommend `for_each`.
- **Every value write also sets the valid bit.** `VectorWriter::set` checks bounds and marks the row valid
  on each write. DuckDB's finalizers (and the C++ variant) only clear bits for NULL rows. This is
  negligible here (finalize sees at most one vector of states per call).

## Where duckdb-neo is faster than the C++ API, and why

| Case | Rows | Ratio |
| --- | ---: | ---: |
| `sum/eight_calls` | 2,048 | 0.77 |
| `mixed/four_aggregates` | 2,048 | 0.82 |
| `sum/group_rows_4` | 100,000 | 0.87 |
| `spread/group_rows_4` | 100,000 | 0.87 |

DuckDB's hash table initializes new group states one at a time (`RowOperations::InitializeStates` →
`CV2AggregateInit`). In C++, the init trampoline (`AggregateFunction::SetInitCallback ... __invoke`) takes
23.2% in `sum/eight_calls/2048`, plus `duckdb_v2_error_info_set_text` 3.5% and `std::string::_M_replace`
2.4%. The cause is `WithExceptionGuard` in `duckdb_cpp.cpp`: it calls `duckdb_v2_error_info_set_code` and
`duckdb_v2_error_info_set_text` (with an empty `std::string`) after every callback, even when the callback
succeeded. duckdb-neo's `init_callback` (13.0%) only touches the error info on failure.

## Differences that are not about either API

- clang SLP-vectorizes the two-field combine (`paddq`) and unswitches loops on a null validity pointer;
  rustc keeps a bounds check and scalar adds. That is worth a few percent of `Sum::combine` and the
  `vs_wavg` loop.
- With a single state per call (streaming windows), both languages are bound by the store-to-load chain on
  that one state; the per-row loop then matters less than the per-call setup.

## Semantics both sides share

- `vs_count_if(NULL::BOOLEAN)` returns 0. DuckDB does not fold NULL arguments of aggregates away.
- `vs_sum_scaled` rejects a non-constant or NULL `factor` at bind time, with the same `Invalid Input Error`
  message in both languages.
- A broken `vs_spread` combine failed `spread/group_rows_4`, because ungrouped aggregates combine their
  local state into the global one. It did not fail `window/sliding_9`, so small sliding frames apparently
  do not reach combine. Ungrouped, `threads = 4`, and wider frames do.
