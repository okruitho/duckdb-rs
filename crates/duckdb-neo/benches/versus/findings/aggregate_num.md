# Findings: `aggregate_num`

Functions: `vs_sum`, `vs_avg`, `vs_count_if`, `vs_spread`, `vs_wavg`, `vs_sum_scaled` (plain states, see
`../rust/aggregate_num.rs`). Neither side needed a workaround for correctness; the safe duckdb-neo API
expresses all six functions.

## duckdb-neo

1. **Fixed cost per update/finalize call is high (streaming running windows).** `window/running`
   (`OVER (ROWS UNBOUNDED PRECEDING)` without `ORDER BY`) streams, and DuckDB calls update and finalize
   once per row. In the smoke run Rust was about 1.8x slower than C++ there (and only there); a flat profile
   at 100,000 rows (`perf record -F 2000`, `--mode loop`, 3 s) shows where the Rust time goes:

   ```
   rust                                                       cpp
   16.3%  libduckdb  CV2AggregateUpdate                       32.1%  libduckdb  Vector::Reference
   14.2%  libduckdb  Vector::Reference                        31.1%  libduckdb  CV2AggregateUpdate
   12.3%  versus     aggregate::finalize_callback::<Sum>       9.4%  versus_cpp cxx::Vector::GetView
   10.9%  versus     VectorReader<f64>::new (folded, i64)      5.1%  versus_cpp FinalizeInput::GetResultOffset
    6.7%  versus     Sum::update                               3.5%  versus_cpp FinalizeInput::GetStateCount
    6.4%  versus     VectorCollection::get_untyped_vector_at_mut
    5.8%  versus     VectorCollection::wrap::<Unknown>
    5.2%  versus     aggregate_num::readable::<i64>
    5.1%  versus     Sum::finalize
    4.7%  libduckdb  duckdb_v2_vector_get_view
    4.4%  libduckdb  duckdb_v2_vector_get_data_mutable
   ```

   About 55% of the Rust profile is wrapper code that runs once per callback: building the
   `VectorCollection` and wrapping the argument (`get_untyped_vector_at_mut`, `wrap`), `cast` plus the
   storage-kind check (`readable`), `VectorReader::new`, and on the finalize side `Vector::from_handle`,
   `len`/`set_size`, `cast::<i64>()`, and `writer()` (`finalize_callback`, `Sum::finalize`,
   `get_data_mutable`). The C++ side does `GetArg` + `GetView` and `GetResult` + `GetDataMutable` and
   nothing else. Sliding frames (segment tree) and grouped aggregation call each callback on whole vectors
   and do not show the gap. The per-call setup is what to trim: avoid re-deriving logical-type details and
   re-checking sizes in `finalize_callback` on every call, and make the argument wrapper and reader cheap
   when there is a single row.

2. **`bind` is mandatory and always registered.** `AggregateCallbacks::bind` has no default, so every
   aggregate (five of the six here, which need no bind data) writes an empty `bind` and gets a bind
   callback plus a `set_bind_data(())` per bind. The C++ API registers a bind callback only for
   `vs_sum_scaled`. This costs only at prepare or with `--unprepared`, but it is boilerplate: a default
   `bind` returning `Default::default()` for `BindData = ()` would remove it.

3. **Bind data arrives as `Option<&BindData>` even when `bind` always sets it.** `vs_sum_scaled` has to turn
   `None` into an error in `update` (`factor(bind_data)?`); C++'s `GetBindData<ScaleFactor>()` throws by
   itself. Not a cost, but the `Option` is noise for the common case.

4. **Callbacks cannot be shared between functions with different `BindData`.** `vs_sum_scaled` has the
   same state, combine, and finalize as `vs_sum`, and C++ reuses `SumCombine`/`SumFinalize` directly. In
   Rust the trait impl ties them to `BindData`, so `SumScaled` repeats `combine` and `finalize` (or would
   need free generic helpers).

5. **Every value write also marks the row valid.** `VectorWriter::set` (`set_unchecked`) writes the value
   and then `mark_valid_unchecked`, and checks bounds per row. DuckDB's own finalizers (and the C++
   variant) only clear bits for NULL rows, because the result vector starts all-valid. Cheap for
   aggregates (finalize sees at most one vector of states per call), but it is a per-row cost shared by
   every writer.

6. **Typed argument access needs a helper.** There is no accessor on `VectorCollection` that narrows to `T`
   and flattens `StorageKind::Other` in one step, so each update starts with
   `readable::<T>(data.get_untyped_vector_at_mut(i)?)` (copied from `benches/support`). C++ needs the same
   (`GetView` throws on `OTHER`), so it is parity, but both APIs could offer it.

## C++ API

1. **Loop bounds hidden behind out-of-line calls.** The natural loop
   `for (idx_t i = 0; i < input.GetRowCount(); i++)`, as written in the C++ API's own tests, calls into
   `duckdb_cpp.cpp` (and the C API) on every iteration, because the getters are not inline. With it the C++
   variant was up to 2x slower than Rust on grouped and ungrouped cases in the smoke run
   (`mixed/four_aggregates`); after hoisting `const auto count = input.GetRowCount();` the cases a partial
   rerun reached (`sum/*`) were at parity. The variant here hoists every count, but the tests teach the slow
   form.

2. **No public writer for aggregate results.** NULL results must go through `GetValidityMutable()` because
   the size-checked `Vector::SetNull` throws while windows finalize past the result's current size. The
   header's own lazy mask writer (`detail::VectorWriter`) is internal, so the variant has its own
   `ResultNulls`. Also, `FinalizeInput::GetResultOffset` and `GetStateCount` are out-of-line calls that show
   up at 5% and 3.5% in the running-window profile.

3. **No typed state access.** States are `void **`; every callback casts with `static_cast<STATE *>`, and
   init needs placement `new`. Correct but untyped; duckdb-neo's `States<T>` is the safer shape.

## Semantics both sides share

- `vs_count_if(NULL::BOOLEAN)` returns 0 (DuckDB does not fold NULL arguments of aggregates away).
- `vs_sum_scaled` rejects a non-constant or NULL `factor` at bind with the same `Invalid Input Error`
  message in both languages.
- A broken `vs_spread` combine (max instead of min) failed `spread/group_rows_4` (ungrouped aggregates
  combine their local state into the global one) but not `window/sliding_9`, so small sliding frames
  apparently do not reach combine; ungrouped, `threads = 4`, and wider frames do.
