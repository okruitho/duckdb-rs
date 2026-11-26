# Findings: `table`

Functions: `vs_series`, `vs_grid` (projection pushdown), `vs_words`, `vs_repeat_row` (constant vectors),
`vs_series_step` (named parameter), `vs_series_filter` (filter pushdown), `vs_series_par` (parallel scan).
Both APIs support all of these, so nothing was skipped. Timings from the shared machine are not used here.

## duckdb-neo

### Every output column needs `set_size`, and safe `set_size` zero-fills

- **What the C++ side does:** it writes each projected column straight into the chunk's buffer
  (`GetDataMutable<T>()`, which is `GetCapacity()` rows long), then calls `SetSize(rows)` once, on vector 0.
  The engine passes that count on to the other vectors (`SetChildCardinality` in `CV2TableExecBatch`).
- **What Rust has to do:** `VectorWriter` takes its length from the vector, so each output vector needs its
  own `vector.set_size(rows)` before `writer()`. Each of those calls makes one `duckdb_v2_vector_set_size`
  call, refreshes the buffers, and runs `init_rows`, which zeroes the new rows (8 bytes per BIGINT row,
  16 bytes per VARCHAR slot). The rows are then written again right away. `set_size_uninit` skips the
  zeroing but is `unsafe`, and the existing `benches/support/dash_split.rs` uses it, which the versus rules
  forbid.
- **Cost:** one extra FFI call plus a memset per column per chunk. For `vs_grid` `SELECT *` that is six
  calls and about 2048 × (8+8+8+16+1+4) bytes zeroed per chunk.
- **Wanted:** a safe way to fill a whole fresh output column without the zeroing. For example, a writer
  that takes `rows` and a closure or iterator and writes every row itself, or a table-function helper that
  sizes all projected vectors at once with uninitialised rows, made safe by the writer filling them.

### `Vector::make_constant` consumes the `Value`

- `make_constant(self, value: Value, count)` takes the `Value` by value, and `Value` has no `Clone`.
  `duckdb_v2_vector_make_constant` only borrows the handle, and the engine copies it
  (`ConstantVector::Reference`).
- **C++:** `vs_repeat_row` keeps the two bind-time `Value`s in its bind data and passes `const Value &` in
  every chunk.
- **Rust:** it keeps `Option<String>` / `Option<i64>` and rebuilds a `Value` for each projected column in
  every chunk (`ToValue::value(context)`). That is a value create plus a destroy through FFI per column per
  chunk, and for VARCHAR a string copy too.
- **Wanted:** `make_constant(self, value: &Value, count)`, or `Clone for Value`.

### Filter pushdown needs interior mutability in bind data

- `TableFilterPushdownCallbacks::pushdown_filter` receives `&Self::BindData`. Recording the accepted bounds
  therefore needs atomics (`fetch_max`/`fetch_min`) or a mutex.
- **C++:** `FilterPushdownInput::GetBindData<T>()` returns `T &`, so it uses plain `int64_t` fields.
- The cost is small (atomics at optimise time, one load per bound at `init_local`), and it is documented,
  but it is extra ceremony that only the Rust side has.

### Per-exec chunk wrapping

- neo's `exec_callback` wraps the output chunk with `DataChunk::borrowed`. That calls
  `data_chunk_get_vector_count` plus `data_chunk_get_vector` for every output vector on each exec call,
  before any user code runs.
- C++ fetches a vector only when the code asks for it (`chunk.GetVector(i)`).
- This is part of what is measured. It matters most for wide projections with few rows per call.

## Both APIs / engine

### `BETWEEN` bounds are not inspectable

- DuckDB rewrites `i >= a AND i < b` into a `COMPARE_BETWEEN` node before filter pushdown.
- Neither `duckdb_neo::expression::Expression` nor `duckdb::cxx::Expression` exposes whether the lower or
  upper bound is inclusive, so a function cannot safely claim it.
- `vs_series_filter` therefore claims only single comparisons (`<`, `<=`, `>`, `>=`, `=` against a BIGINT
  constant). A range filter is left to DuckDB, and the cases avoid it.

### Engine bug: common subplan elimination merges calls whose accepted filters differ

- **Reproducer (same on both sides):**
  ```sql
  SELECT (SELECT count(*) FROM vs_series_filter(100) WHERE i = 99)
       + (SELECT count(*) FROM vs_series_filter(100) WHERE i >= 50)
  ```
  It returns `2` instead of `51`.
- **Cause:** `CommonSubplanOptimizer` compares `LogicalGet`s by their serialized form. For a C API table
  function without serialization callbacks, that form is only the call parameters. Once each call's
  filter has been accepted by the pushdown callback, the two plans look identical and are merged. The second
  subquery then reuses the first one's rows.
- The cases avoid it by giving every edge call a distinct `n`.
- Any v2 C API table function that accepts filters can be hit by this. It should be reported to DuckDB.

## C++ API notes (they shaped the C++ variant)

- `Vector::AssignString` looks up the logical type, the heap, the vector type and the data pointer again
  on every call: about 5 C API calls per string. The C++ variant instead takes `GetHeap()` and
  `GetDataMutable<varchar_t>()` once per chunk and stores `heap.AddStringUnsafe(text)`.
- The generated text is ASCII, which matches what Rust's `&str` guarantees, so neither side validates
  UTF-8.
- `DataChunk::GetCapacity` is marked unstable but is the only way to size a batch. Both sides call it once
  per exec call.
