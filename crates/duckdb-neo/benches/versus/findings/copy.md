# Findings: `copy`

Functions: `vs_lines` (text, COPY TO and FROM) and `vs_fixed` (binary BIGINT/DOUBLE, COPY TO and FROM). The formats
are described at the top of `rust/copy.rs`. Both sides use the same structure: COPY TO encodes a batch into a byte
buffer in the batch callback and writes it in flush, with DuckDB's default batching of one batch per sunk chunk.
COPY FROM reads `vs_lines` through a 1 MiB buffer and `vs_fixed` with one positional read per output chunk, splits
fields first and then fills the output column by column.

These come from writing the code, not from profiles. The timings were taken on a busy machine and are a smoke
test only.

## duckdb-neo

### 1. `Vector::set_size` zeroes rows that are about to be written (COPY FROM, every column, every chunk)

- Wanted: size each output vector to the batch's row count and write every row, as the C++ side does with
  `Vector::SetSize(rows)` and then `GetDataMutable<T>()` / `AssignString`.
- Had to write: `vector.set_size(rows)?` and then `vector.writer()?`. The writer's length is the vector's length,
  so the size must be set first, and the safe `set_size` initializes every new row (`init_rows`) to a zero value.
  The only way around it is `unsafe { set_size_uninit }`, which `benches/support/base56.rs` uses and the versus
  rules forbid.
- Cost: an extra pass over `rows` slots per column per chunk (8 bytes per BIGINT/DOUBLE, 16 per VARCHAR), on top
  of the decode. It matters most for `vs_fixed` reads, whose per-value work is small.
- Possible fix: a safe way to size a vector that the caller fills completely, for example a writer that takes the
  row count and checks on drop (or in a `finish`) that every row was written or nulled, or a
  `set_size_with(rows, |writer| ...)` that zeroes only the rows the closure did not write.

### 2. COPY TO file state needs a `Mutex` because `flush` only gets `&InitData`

- `CopyToFunctionCallbacks::InitData` must be `Send + Sync`, and `flush`/`finalize` receive `&Self::InitData`.
  `file::File` is `Send` but not `Sync`, so the open file lives in a `Mutex<File>` and every flush and the finalize
  take the lock.
- The C++ API hands out `GetInitData<T>()` as `T &`, and documents that flush never runs concurrently for the same
  file, so the C++ side keeps a plain `FileHandle`.
- Cost: one uncontended lock per batch (per 2,048 rows with the default batching). Small, but it is also a
  usability gap: any mutable per-file writer state (a buffer, a row counter) needs interior mutability.
- Possible fix: give flush and finalize exclusive access to a per-file state that batch cannot see, e.g. a second
  associated type that only flush and finalize receive as `&mut`.

### 3. The batch callback copies every column type of the bind data, per batch

- `batch_to_callback` builds the `ColumnDataCollection` with `bind_data.logical_types.clone()`, which calls
  `duckdb_v2_logical_type_copy` once per column (and destroys each copy later) for every batch. The C++ side keeps
  its `std::vector<LogicalType>` in the bind data and passes it by reference to the scan chunk's constructor.
- Cost: `columns` type copies and destroys per 2,048 rows, growing with the column count. Not measured yet.
- Possible fix: share the types (`Arc<[LogicalType]>` or a borrowed slice) instead of cloning them.

### 4. The wrapper keeps each batch's input collection alive until flush

- `CopyFunctionBatchData` stores the collection next to the batch data so that `flush` can scan it again. A format
  that encodes in `batch` (both here) has no use for it, but its memory stays pinned until the batch is flushed.
  The C++ side takes the batch with `TakeBatch()` and drops it at the end of the batch callback.
- Cost: memory held longer; with the default one-batch-per-chunk batching and one thread this is a chunk at a
  time, so no time cost is expected here.

### 5. Always-registered callbacks

- `CopyFunctionBuilder` registers `batch_size` (which sets no target by default) and the COPY FROM global-init
  callback whether or not the type overrides them. DuckDB then runs one extra callback per bind/execution. The
  batching is unaffected (a callback that sets no target leaves `BATCH_SIZE` unset, the same as no callback). This
  is a per-statement cost and shows only in the 1-row and empty-file cases, if at all.

### 6. Reading VARCHAR without per-row UTF-8 validation takes `UncheckedString`

- `VectorReader<String>::get` validates UTF-8 on every read (`std::str::from_utf8(..).expect(..)`); the C++ view
  reads `varchar_t` bytes directly. To do the same work the Rust side reads with `UncheckedString`, like
  `benches/support/base56.rs`. That marker is usable from safe code although its documentation says invalid UTF-8
  is undefined behaviour. DuckDB guarantees valid VARCHAR data, so the choice is fine here. Still, the checked type
  costs a validation per value, and the unchecked one moves a safety obligation into safe code without an `unsafe`
  marker.

### 7. Errors from callbacks are prefixed twice

- A file that cannot be opened in a COPY FROM callback fails with
  `IO Error: IO Error: Cannot open file ...` through duckdb-neo and `IO Error: Cannot open file ...` through the
  C++ API. The error type is added again when the callback's error is handed back to DuckDB.

## C++ API notes

- `FileHandle` is documented as "only usable while the `FileSystem` it came from is still valid", and
  `Context::GetFileSystem()` is borrowed for the callback's duration, which would rule out keeping a file open from
  init to finalize (or from local init across exec calls). The C implementation (`CV2File`) owns the engine handle
  independently of the file-system handle, and duckdb-neo documents that context files may outlive the callback, so
  both sides keep the handle open across callbacks.
- A malformed UTF-8 field fails in `Vector::AssignString` with the engine's "Input is not valid UTF-8" message. The
  Rust side reports the field (`vs_lines: invalid VARCHAR field ...`). The condition is the same.

## Not duckdb-neo: standard library differences in `vs_lines`

- DOUBLE is written with 17 significant digits (`{:.16e}` and `std::to_chars(..., scientific, 16)`, the exponent
  rewritten to Rust's spelling on the C++ side). The shortest round-trip forms cannot be used, because Rust's
  `{:e}` and libstdc++'s `to_chars` break ties differently (e.g. `sqrt(6506) * 1e10` = 806597793203.03125 is
  `8.065977932030313e11` in Rust and `8.065977932030312e11` in C++), which makes the files differ.
- Parsing DOUBLE uses `str::parse::<f64>` after `std::str::from_utf8` in Rust and `std::from_chars` in C++. Rust
  also rejects a leading `+` explicitly to match `from_chars`. Out-of-range input such as `1e400` still differs
  (Rust parses infinity, `from_chars` fails), and so does `nan(...)` (only `from_chars` accepts it). BIGINT is
  formatted and parsed by the same hand-written loop on both sides, so no library difference enters there.
- Any gap in the DOUBLE-heavy `vs_lines` cases (`lines_to/double`, `lines_to/wide*`, `lines_from/wide_nulls_10`,
  `lines_from/builtin_csv`) therefore includes `core::fmt` against Ryu printf and `core` float parsing against
  `from_chars`, which are not duckdb-neo costs. Profile before attributing them.
