# Findings: `existing`

The Rust side registers the tuned modules of `benches/support` unchanged (`rust/existing.rs` includes
`support/mod.rs`); the C++ side ports them (`cpp/existing.cpp`). No timings appear here: round 1 ran on a busy
machine, and the central comparison has not run yet.

## `unsafe` in the included modules

The runner crate denies `unsafe_code`. `rust/existing.rs` allows it on the included `support` module only. That
module contains six `unsafe` blocks:

| Site | What it does | Safe alternative | Cost of the alternative |
| --- | --- | --- | --- |
| `merge_struct.rs:54` | `set_size_uninit(rows)` on the STRUCT output, then `writer.fill_validity(true)` | `output.set_size(rows)`, with no validity call (a fresh result vector is already all valid) | `set_size` runs `init_rows`: it zero-fills every field's data (`i`, `s`, the MAP list entries, `d`), and for the UNION field it zeroes the tag and the first member and NULLs the second (one validity fetch). That is about five memsets of up to 2,048 × 16 bytes plus a few C API calls per chunk, all overwritten right away. The safe API also cannot mark STRUCT rows valid in bulk: `fill_validity` is `unsafe`, and the STRUCT writer has no safe per-row "valid" setter. |
| `dash_split.rs:132` | `set_size_uninit(parts.len())` on the `idx` output | `vector.set_size(parts.len())` | A zero-fill of 8 bytes per row, overwritten by the loop that follows. |
| `dash_split.rs:141` | `set_size_uninit(parts.len())` on the `part` output | `vector.set_size(parts.len())` | A zero-fill of 16 bytes per row (empty strings), overwritten by the loop that follows. |
| `base56.rs:92` (`push_digits`) | Writes digits through a raw pointer into the `Vec` spare capacity, then `set_len` | `out.resize(start + count, 0)` and indexed writes, or digits in a `[u8; 6]` stack buffer and `extend_from_slice` | This is plain Rust, not duckdb-neo API. The C++ port uses the `resize` form (`std::string::resize`, then indexed writes), so the two variants differ only in this zero-fill of at most 6 bytes per block. |
| `base56.rs:471` (`decode_column`) | `set_size_uninit(rows)` on each fixed-width COPY FROM output | `vector.set_size(rows)` | A zero-fill of 1 to 8 bytes per row and column. |
| `base56.rs:496` (`decode_strings`) | `set_size_uninit(rows)` on each VARCHAR COPY FROM output | `vector.set_size(rows)` | A zero-fill of 16 bytes per row. |

The C++ API has no equivalent hazard. `Vector::SetSize` does not initialize rows, and the port writes or NULLs
every row after it, as the Rust code does. So `set_size_uninit` is what keeps the Rust variants doing the same work
as the C++ ones. Possible improvement: a safe duckdb-neo writer that sizes the vector and requires every row to be
written (e.g. a fill-by-iterator or `write_all_rows` API) would remove four of the five duckdb-neo `unsafe` sites
without the zero-fill. A safe bulk "mark all valid" for STRUCT parents would remove the fifth.

I did not measure the safe alternatives. Their cost is the extra memsets listed above; measure them on a quiet
machine before deciding.

## Differences in what the two wrappers register

These are fixed per-query costs. They could show at `rows = 1`:

- **Aggregates (`row_hash`, `int_sum`).** duckdb-neo always registers an aggregate bind callback, so every bind
  of `row_hash`/`int_sum` runs one more callback and allocates bind data (`()` here). The C++ port registers no
  bind callback; the C++ API does not need one. duckdb-neo registers no destroy callback for these states,
  because `HashState`/`SumState` do not need drop. That matches the C++ port, and DuckDB's streaming window path
  stays available to both.
- **COPY TO (`base56`).** duckdb-neo always registers a `batch_size` callback, which returns `None` here. The C
  API then sets `desired_batch_size`, and the binder calls the callback on every `COPY ... TO` bind. It returns
  no target, so batching stays the same as without it (one batch per chunk). The C++ port registers none. The
  cost is one callback round trip per statement.
- **Scalar (`merge_struct`) and table (`dash_split`).** Neither wrapper registers optional callbacks the other
  lacks: bind and init are opt-in in duckdb-neo's scalar builder, and the table function sets the same
  callbacks on both sides.

## C++ API notes (the peer, not duckdb-neo)

- The C++ API's row writers (`detail::VectorWriter`) are internal, and `ScalarExecutor` cannot express MAP,
  UNION or strings. The port therefore writes through raw `GetDataMutable` pointers and a small `OutputValidity`
  helper, which fetches the mask on the first NULL. duckdb-neo's typed writers (`map_writer`, `union_writer`,
  `set_option`) express this directly and safely.
- Aggregate states are untyped `void **` in the C++ API. The port initializes them with placement new. duckdb-neo's
  `States<'_, T>` is typed.
- `ColumnDataCollection::Scan` needs a caller-created `DataChunk` of the batch's types. The port keeps the
  `LogicalType`s in its bind data for that. duckdb-neo's `to_scan` creates the chunk itself.

## Notes on the cases

- Every `merge_struct` check reads the function output from a `MATERIALIZED` CTE. Otherwise DuckDB merges the
  check's outer `WHERE` into the case's own filter and evaluates the function in a different chunk layout. I found
  this when a deliberately broken C++ string copy (one row, position 4 of the chunk) went unnoticed by
  `merge_struct/filtered`.
- Only manual runs can cover the error paths, because a failing setup or check statement fails the runner. I ran
  each one on both runners, and they produce the same messages:
  - an invalid length, digit or overflowing block;
  - a wrong field width or field count;
  - an invalid BOOLEAN byte;
  - invalid UTF-8;
  - a header that does not match the target columns;
  - `does not support TINYINT` (COPY TO and COPY FROM);
  - a negative `dash_split` `n`.
  The UTF-8 message text differs: Rust's `from_utf8` error versus DuckDB's "Input is not valid UTF-8". Also checked
  manually: a final line without `\n` is still a row, and `PARTITION_BY` uses the `b56` file extension.
