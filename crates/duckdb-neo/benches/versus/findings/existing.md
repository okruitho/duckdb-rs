# Findings: `existing`

The Rust side registers the tuned modules of `benches/support` unchanged (`rust/existing.rs` includes
`support/mod.rs`). Those modules are shared with `benches/functions.rs` and were not edited. The C++ side
(`cpp/existing.cpp`) ports them with the same algorithms. Both sides read `VARCHAR` without validating it
(`UncheckedString` in Rust, `varchar_t` views in C++). Only the bytes `COPY ... FROM` reads from a file are
validated, on both sides.

Measured with `run` (3 repetitions, rust/cpp/cpp/rust, CPU 10), with other agents active on other cores. Ratio =
Rust median / C++ median.

## Changes in this round, all on the C++ side

1. **Row and state counts hoisted out of loop conditions.** `UpdateInput::GetRowCount`,
   `GetStateCount`, `GetColumnCount` and `DataChunk::GetRowCount` are out-of-line calls into the C API
   (`duckdb_v2_aggregate_function_update_get_row_count` -> `DataChunk::DeriveSize`). In
   `int_sum/grouped/1000000` the loop condition alone took 20.0% (`UpdateInput::GetRowCount`) + 4.0%
   (`duckdb_v2_aggregate_function_update_get_row_count`) + 4.1% (`DataChunk::DeriveSize`).
2. **Inlined strings copied as they are.** `merge_struct` now copies a `varchar_t` of at most 12 bytes as it is;
   only longer strings go through `Arena::AddStringUnsafe`. `AddStringUnsafe` builds every string through the
   `blob_t(data, size)` constructor: a `memset`, then a libc `memcpy` into a temporary, which is then reloaded.
   That cost `merge_struct/nulls_0/1000000` 16.3% in `__memmove_avx512_unaligned_erms` plus stalls inside
   `MergeStructExec`.
3. **`base56` encoder writes digits with `push_back`.** It formats the digits into a stack buffer and appends them
   with `push_back`, which inlines. It no longer uses `std::string::resize` (out-of-line
   `_M_replace_aux` 7.8%, `memset` 2.4%, `resize` 2.3%) or the per-field `reserve` (out-of-line, 2.3%). Do not use
   `append(ptr, n)` for a few bytes either: it is out of line too, and it made `__memmove_avx512_unaligned_erms`
   take 42% of `base56/to_nulls_0`.

| Case | Ratio before | Ratio after | C++ before -> after | Changed |
| --- | --- | --- | --- | --- |
| `int_sum/ungrouped/1000000` | 0.71 | 1.01 | 3.70 -> 2.64 ms | 1 |
| `int_sum/grouped/1000000` | 0.63 | 1.02 | 5.78 -> 3.70 ms | 1 |
| `int_sum/groups_4/1000000` | 0.71 | 1.02 | 7.11 -> 5.07 ms | 1 |
| `int_sum/groups_quarter/1000000` | 0.74 | 0.88 | 17.2 -> 15.6 ms | 1 |
| `int_sum/filter/1000000` | 0.86 | 1.01 | 4.70 -> 4.00 ms | 1 |
| `int_sum/running/1000000` | 1.33 | 1.62 | 96.2 -> 78.4 ms | 1 |
| `int_sum/sliding/1000000` | 0.84 | 1.02 | 140 -> 117 ms | 1 |
| `int_sum/sliding_1000/1000000` | 0.69 | 1.11 | 180 -> 114 ms | 1 |
| `row_hash/ungrouped/1000000` | 0.92 | 0.98 | 16.8 -> 16.1 ms | 1 |
| `row_hash/groups_quarter/1000000` | 0.87 | 0.93 | 30.8 -> 30.6 ms | 1 |
| `row_hash/sliding/1000000` | 1.01 | 1.08 | 280 -> 264 ms | 1 |
| `merge_struct/project/1000000` | 0.84 | 1.19 | 107 -> 75.2 ms | 2 |
| `merge_struct/nulls_0/1000000` | 0.74 | 1.21 | 102 -> 62.1 ms | 2 |
| `merge_struct/filtered/1000000` | 0.88 | 1.17 | 87.6 -> 66.0 ms | 2 |
| `merge_struct/constant_args/1000000` | 1.07 | 1.21 | 82.8 -> 73.7 ms | 2 |
| `base56/to/1000000`, `to_nulls_0` | 1.00 | 1.00 to 1.04 | unchanged wall time | 3 (see "Not about either API") |

Unchanged cases at 1,000,000 rows:

| Case | Ratio |
| --- | --- |
| `dash_split/scan` | 0.87 |
| `dash_split/part_only` | 0.90 |
| `dash_split/filter_output` | 0.85 |
| `dash_split/count` | 1.47 |
| `dash_split/limit` | 1.05 |
| `base56/from` | 0.90 |
| `base56/from_nulls_0` | 0.82 |
| `row_hash/running` | 2.69 |
| `merge_struct/nulls_100` | 1.30 |

At 1 row, every case is within 5% except `merge_struct/*` at 1.07 to 1.13. The baseline's 1.32 for
`base56/to_nulls_0/1` was noise: it measures 1.00 now.

## Remaining gaps attributable to duckdb-neo, most important first

### 1. Fixed cost per callback of wrapping argument and result vectors

**Affected cases:**
- `row_hash/running`: 2.69 at 1,000,000 rows (394 vs 147 ns per row) and 2.76 at 100,000.
- `int_sum/running`: 1.62 at 1,000,000 rows.

DuckDB's streaming window calls update and finalize once per row, so per-callback cost is everything here.

**C API calls per row**, counted with gdb breakpoints on every `duckdb_v2_*` symbol (the difference between a
100-row and a 200-row run, divided by 100):

| Case | duckdb-neo | C++ API | C++ API without its own error-info calls (see "Faster") |
| --- | --- | --- | --- |
| `row_hash/running` (3 arguments) | 45 | 22 | 18 |
| `int_sum/running` (1 argument) | 29 | 16 | 12 |

What the C++ variant calls for each argument: `update_get_arg`, `vector_get_vector_type` and `vector_get_view`.

What duckdb-neo calls for each argument: `update_get_arg`, `vector_get_logical_type`, `vector_get_vector_type`,
`vector_get_size`, `vector_get_child_count`, `logical_type_get_id` (the cast in `readable`), `vector_get_view` and
`logical_type_destroy`. That is 8 calls instead of 3.

Further per-call overhead in duckdb-neo:
- `update_callback` calls `get_user_data`, `get_bind_data` and `get_arg_count` on every update.
- `finalize_callback` wraps the result vector with the same 6 to 7 calls, and calls `vector_set_size` (through the
  safe `set_size`, hence `init_rows`), because the window sizes the result only afterwards.

**Profile of `row_hash/running/100000`:**

| duckdb-neo symbol | Self % |
| --- | --- |
| `__memmove_avx512_unaligned_erms` | 12.5 |
| `VectorCollection::wrap` (`Vector::from_handle`) | 11.9 |
| `VectorReader<UncheckedString>::new` (the `get_view` call and its reload) | 11.4 |
| `RowHash::update` (user code) | 9.4 |
| `[usize; 3]::try_map` | 9.0 |
| `CV2AggregateUpdate` (engine side) | 7.2 |
| `Vector::Reference` (engine side) | 6.4 |
| `readable::<f64/UncheckedString/bool>` | 7.2 (sum of three) |
| `finalize_callback` | 4.8 |
| `Vec` `from_iter` | 2.6 |

| C++ symbol | Self % |
| --- | --- |
| `cxx::Vector::GetView` (a store-forwarding stall, see below) | 33.2 |
| `CV2AggregateUpdate` | 25.5 |
| `Vector::Reference` | 17.8 |
| `RowHashUpdate` (user code) | 5.3 |

A DWARF call graph places the `memmove` in `get_vector_array_mut` -> `collect_array` (3.6% + 1.9%), in moving the
`(VectorReader, ...)` tuple into `CombinedVectorIterator::new` (4.0%), and in `try_process` (3.7%): each `Vector` is
a large struct (logical type, cached view, children box, ...), and it is moved several times.

**Responsible sources:**
- `crates/duckdb-neo/src/vector/mod.rs:188` `Vector::from_handle` fetches the logical type, size and child count
  eagerly and validates the type on every wrap. The `LogicalType` it creates is destroyed again on drop.
- `crates/duckdb-neo/src/data_chunk.rs:32` `collect_array` collects through a heap `Vec`, then converts it to an
  array.
- `crates/duckdb-neo/src/aggregate/mod.rs:250` `update_callback` and `:327` `finalize_callback`.

**Suggestions:**
- In `from_handle`, fetch only the vector type and the view. Take `len` from `duckdb_v2_vector_view::count`, and
  fetch the logical type and child count lazily (`details`/`OnceCell`), only for nested types or when asked.
- Validate argument types once, not per callback: the parameter types are fixed by the registered signature, so
  one check at bind (or the first call), cached in the function info, makes the per-call `get_logical_type` /
  `get_id` / `destroy` triple unnecessary. A typed accessor such as
  `VectorCollection::arg::<T>(index) -> Result<Vector<'_, T>>` could use the cached result.
- Build the array in `get_vector_array_mut` without a `Vec`, e.g. check `results.iter().any(Result::is_err)` first,
  then `results.map(Result::unwrap)`.
- In `finalize_callback`, size the result without `init_rows`. Every state's row is written by `finalize` anyway,
  which could be a documented requirement of the callback.

**No safe usage in `benches/support` avoids this.** `get_vector_at`, `get_vector_array_mut` and
`get_untyped_vector_at` all go through `from_handle`.

### 2. Per-exec type validation and child wrapping in `merge_struct`

**Affected cases:** `merge_struct/*/1` at 1.07 to 1.13, a fixed 23 to 29 µs per query.

**C API calls per exec call** (one chunk): 270 for duckdb-neo versus 60 for the C++ API. duckdb-neo's calls
break down as:

| Count | Calls |
| --- | --- |
| 32 each | `vector_get_logical_type`, `logical_type_destroy`, `vector_get_vector_type`, `vector_get_size`, `vector_get_child_count` |
| 30 | `logical_type_get_id` |
| 22 | `vector_get_view` |
| 21 | `vector_get_child` |
| 6 | `logical_type_get_param` |
| 6 | `value_destroy` |
| 4 | `value_get_utinyint` |
| 4 | `value_is_null` |

The `get_param` and `Value` calls are the `DECIMAL(18, 3)` width and scale checks: `Decimal::validate` ->
`DecimalSignature::from_logical_type` (`crates/duckdb-neo/src/types/decimal.rs:255, :380`) reads both parameters
as `Value`s, on every exec, for the argument and for the output field. Every nested argument (MAP, UNION and their
children) and every output field is wrapped recursively through `Vector::from_handle`.

**Suggestions:**
- Validate once per bound call, as in gap 1.
- Cache the `TypeDetails` of a parameter: the v2 C API has no dedicated DECIMAL width/scale getter (the C++ API's
  `GetDecimalWidth` goes through the parameters too), so reading them once matters.
- Wrap children lazily.

### 3. No safe way to copy a VARCHAR slot from an input to an output vector

**Affected cases:**
- `merge_struct/project`, `nulls_0`, `filtered` and `constant_args`: 1.17 to 1.22 at 1,000,000 rows.
- The C++ side is 20 to 30% *slower* when it uses `Arena::AddStringUnsafe` for every string (round 1 ratios
  0.74 to 0.88).

The C++ port copies a `varchar_t` of at most 12 bytes as it is (a 16-byte copy). duckdb-neo's safe writer only
takes `&str`, so `copy_column` re-encodes every string: `VectorWriter<String>::write_bytes` is 9.4% of
`merge_struct/nulls_0/1000000`, the largest duckdb-neo symbol. It is fast (`DuckDBBytes::write_parts` with
`copy_short`), but it is still a length check, zeroing and a byte copy per row.

`VectorWriter::copy_from` copies whole vectors, but only for `PlainNative` element types and only when the reader
and writer have the same element type, so VARCHAR (`UncheckedString` -> `String`) cannot use it.

**Suggestion:** add a safe slot copy, e.g.

```rust
VectorWriter<String>::copy_row(&mut self, row: usize, source: &VectorReader<'_, UncheckedString>, source_row: usize) -> Result<()>
```

or a `copy_from` that accepts an `UncheckedString` reader into a `String` writer. Inlined slots would be copied as
they are, and longer strings into this vector's arena. (`crates/duckdb-neo/src/vector/writer.rs:526`
`copy_from` is the place to extend.)

`benches/support/merge_struct.rs` `copy_column` could also use the existing `copy_from` for the `INTEGER` and
`DECIMAL` fields. That is a bulk copy of flat inputs; it was not prototyped.

### 4. Per-row NULL writes through `UnionWriter::set_null`

**Affected case:** `merge_struct/nulls_100` at 1.30 (1,000,000 rows: 28.3 vs 21.8 ms, about 6.5 ns per row).

**Profile:**
- duckdb-neo: `UnionWriter<(i32, String)>::set_null` 8.8%, `(i32, String) as UnionWriters>::set_null` 5.35%, plus
  `copy_column` 3.3 to 3.4% each for the plain fields. Every row nulls 2 members, the tag and the parent through
  separate bounds-checked writer calls (`crates/duckdb-neo/src/vector/union_writer.rs:392`).
- C++: all of it is inlined into `MergeStructExec` (24.6%) as plain bit clears on masks fetched once.

**Suggestion:** make `set_null` on union/struct writers inline-friendly. Fetch all masks once, as `allocated_validity`
already caches them, and clear the bits without the per-member dispatch. Alternatively, add a bulk "copy validity
from a reader" for union parents.

### 5. `dash_split/count` is caused by the Rust standard library, not duckdb-neo

**Affected cases:**
- `dash_split/count`: 1.47 at 1,000,000 rows.
- The rest of `dash_split` would be faster still.

`support/dash_split.rs:111` splits with `input[start..].find('-')`. `str::find(char)` runs `CharSearcher`, which
calls libc `memcmp` for every candidate byte: `__memcmp_evex_movbe` is 58.4% of `dash_split/count/1000000` and
26.4% of `dash_split/scan/1000000` (DWARF call graph: `CharSearcher::next_match` -> `equal_same_length` ->
`memcmp`). The C++ port uses `std::string_view::find(char)` -> `memchr`.

**Prototype, not committed:** `memchr::memchr(b'-', &input.as_bytes()[start..])`, with `memchr` already a
dev-dependency used by `base56`.

| Case at 100,000 / 1,000,000 rows | Ratio before | Ratio with the prototype |
| --- | --- | --- |
| `dash_split/count` | 1.41 / 1.47 | 1.07 / 1.07 |
| `dash_split/scan` | 0.85 / 0.87 | 0.75 / 0.74 |
| `dash_split/part_only` | 0.87 / 0.90 | 0.74 / 0.76 |
| `dash_split/filter_output` | 0.88 / 0.85 | 0.78 / 0.77 |

Recommended for `benches/support`. The output does not change, since `-` is ASCII.

### 6. Small: `int_sum/sliding_1000` at 1.11

The C API calls per row differ by under 0.3, and they are per chunk, not per row. The difference is in the
combine loop: `IntSum::combine` is 14.5% in Rust versus `IntSumCombine` 8.9% in C++. The annotation puts it on the
load of the source state, which is a cache miss in both. Not explained yet.

## Where duckdb-neo is faster than the C++ API, and why

- **`dash_split/scan`, `part_only` and `filter_output` (0.85 to 0.90).** C++ `Arena::AddStringUnsafe` ->
  `AddBlob` -> `blob_t(data, size)` stalls on store forwarding.
  - An inline part is `memset` plus a variable-size libc `memcpy` into a stack temporary, which is then reloaded:
    the annotation of `DashSplitExec` shows 25.2% and 21.1% on the reload.
  - A long part reads its prefix back from the heap copy just written: 41.3% on the load after
    `Arena::Allocate` + `memcpy`.
  - duckdb-neo's `DuckDBBytes::write_parts` (`crates/duckdb-neo/src/bytes.rs:109`) builds the slot in place, copies
    short strings with `copy_short`, and takes the prefix from the source "for the reason above".

  The C++ port keeps `AddStringUnsafe` for these substrings. The C++ API has no other public way to build a string
  from bytes, short of writing the union fields of `blob_t` by hand.
- **`base56/from` (0.82 to 0.90).**
  - The C++ port validates decoded strings through `Arena::AddString`, which uses DuckDB's `Utf8Proc::Analyze`
    (8.1%). Rust's `std::str::from_utf8` does not show among the top symbols.
  - C++ `ReadDigits` is not inlined (5.7%).
- **`int_sum/groups_quarter` (0.88) and `row_hash/groups_quarter` (0.93).** Every C++ API trampoline runs
  `WithExceptionGuard` (`duckdb_cpp.cpp:437`). On success too, it calls `duckdb_v2_error_info_set_code` and
  `duckdb_v2_error_info_set_text` and builds an empty `std::string`: 2 extra C calls per callback. The init
  trampoline (`AggregateFunction::SetInitCallback ... __invoke`) is 13.6% of the C++ profile, versus 5.8% for
  duckdb-neo's `init_callback`. Scalar `ExecInput` also constructs a `Context` eagerly
  (`context_get_factory`, `context_get_config`: 2 more calls per exec).
- **Store forwarding in `cxx::Vector::GetView`.** It reads the C struct back as 16-byte loads right after the C
  API wrote it field by field: 99.9% of `GetView`'s samples are on that load, and `GetView` is 33.2% of
  `row_hash/running`. duckdb-neo's `writable_buffers` (`crates/duckdb-neo/src/vector/writer.rs`) documents and
  avoids the same stall on its write path. Its `VectorReader::new` still copies the whole view (11.4%).

## API gaps and awkwardness

- **`unsafe` in `benches/support`.** Five calls to `set_size_uninit`, in `merge_struct`, `dash_split` (2) and
  `base56` decode (2), plus raw `Vec` writes in `base56::push_digits`. The included module is
  `#[allow(unsafe_code)]`.
  - The safe `set_size` zero-fills every exposed row (`init_rows`), and the rows are overwritten right after.
  - There is no safe bulk "mark all rows valid" for STRUCT parents (`fill_validity` is `unsafe`).
  - A safe `push_digits` would format into a `[u8; 6]` stack buffer and call `extend_from_slice`, the form the
    C++ port now uses.
  - Suggestion: a safe writer that sizes the vector and requires every row to be written (fill-by-iterator).
- **Writing nested output (C++ API).** `ScalarExecutor` cannot express MAP, UNION or strings, and its row writers
  are in `detail::`. The C++ port writes through `GetDataMutable` pointers and a small `OutputValidity` helper.
  duckdb-neo's typed `map_writer`, `union_writer` and `set_option` express this directly and safely.
- **Aggregate states (C++ API).** They are untyped `void **`, so the port uses placement new. duckdb-neo has typed
  `States<'_, T>`.
- **Extra callbacks registered by duckdb-neo.** It always registers an aggregate bind callback and a COPY TO
  `batch_size` callback, which return nothing here; the C++ port registers neither. At 1 row this does not show
  in `row_hash`/`int_sum`/`base56` (all within 5%).

## Not about either API

- **`base56/to`.** Wall time is the same on both sides at every size (377 ms at 1,000,000 rows) with large
  sample spread. Encoding is not what dominates it: loop-mode iteration counts of the same case vary 3-fold
  between runs. The encoders differ only by small amounts in the profiles. Treat these ratios as I/O, not as a
  wrapper result.
- **libstdc++ `std::string` versus Rust `Vec`.** `resize`, `append` and `reserve` are out-of-line calls, while
  `Vec::reserve` / `extend_from_slice` inline. The C++ encoder needed `push_back` to match.
- **`memchr`.** The Rust `memchr` crate versus libc `__memchr_evex` are equivalent in these profiles.
- **The engine dominates several cases on both sides.** `PerfectAggregateHashTable::AddChunk`,
  `FindOrCreateGroupsInternal` and the window segment tree are the largest symbols in the grouped and windowed
  cases. They are identical on both sides and hide wrapper differences below about 5%.
