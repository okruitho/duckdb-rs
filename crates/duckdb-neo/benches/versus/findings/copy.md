# Findings: `copy`

Functions: `vs_lines` (text, COPY TO and COPY FROM) and `vs_fixed` (binary BIGINT/DOUBLE, COPY TO and COPY FROM).
The formats are described at the top of `rust/copy.rs`. Both sides share one structure:
- COPY TO encodes a batch into a byte buffer in the batch callback and writes it in flush, with DuckDB's default
  batching of one batch per sunk chunk.
- COPY FROM reads `vs_lines` through a 1 MiB buffer and `vs_fixed` with one positional read per output chunk. It
  splits the fields first, then fills the output column by column.

All numbers are medians of `run` (3 repetitions) on CPU 9, with other agents building on other cores. Ratios are
Rust / C++.

## Case change: timed COPY TO writes to `/dev/null`

The round-1 COPY TO cases timed writing a fresh file of megabytes into `{tmp}`. Their wall time was mostly file
system writeback, invisible to `perf`, and varied from run to run. On `lines_to/wide/100000`, Rust sampled 29% more
CPU cycles per iteration than C++, yet the timed ratio came out between 0.74 and 1.06. Now the timed statement
writes the same rows to `/dev/null`, and the setup still writes them to `{tmp}`, where the check reads them. The two
`*_to/empty` cases keep a real file, so the fixed cost of creating one is still measured. The baseline COPY TO
ratios are therefore not comparable. The "before" column below re-measures the round-1 code with the new cases.

## Before / after

| Case (rows) | before | after | what changed |
| --- | --- | --- | --- |
| `fixed_to/wide` (100k / 1M) | 0.29 / 0.30 | 0.94 / 0.94 | both: encode column-major into a buffer sized up front, NULLs left zero; C++: `std::vector<char>` instead of `std::string` |
| `fixed_to/bigint` (1M) | 0.62 | 0.97 | same |
| `fixed_to/wide_nulls_50` (1M) | 0.65 | 1.00 | same |
| `fixed_to/filtered_threads_4` (1M) | 0.65 | 0.97 | same |
| `lines_to/varchar_escapes` (1M) | 0.69 | 0.94 | C++: `std::vector<char>` buffer (`std::string` appends go through out-of-line libstdc++ `_M_append`) |
| `lines_to/sep` (1M) | 0.82 | 0.89 | same |
| `lines_to/filtered` (1M) | 0.82 | 0.95 | same |
| `lines_to/bigint` (1M) | 0.81 | 0.86 | same; the rest is codegen, see "not about either API" |
| `lines_to/varchar_long` (1M) | 0.64 | 0.75 | same |
| `lines_from/bigint` (1M) | 1.34 | 1.10 | Rust: the field split iterates the line's bytes instead of indexing (bounds checks) |
| `lines_from/builtin_csv` (1M) | 1.18 | 1.18 | Rust split as above; the rest is float parsing, see below |
| `lines_from/varchar_escapes` (1M) | 0.64 | 1.08 | C++: write through `Vector::GetHeap()` + `Arena::AddString` instead of `AssignString`; both: return the field itself when it has no `\` |
| `lines_from/sep` (1M) | 0.73 | 0.86 | same |
| `lines_from/varchar_long` (2,048 / 1M) | 0.93 / 0.93 | 0.86 / 0.90 | both: no unescape copy for fields without `\` (Rust 2.53 → 1.24 ms at 2,048) |
| `lines_from/huge_line` (2,048) | 0.91 | 1.25 | C++ heap writes; per-byte unescape of a line with an escape every other byte, see gap 3 |
| `fixed_from/bigint` (1M) | 1.11 | 1.14 | unchanged (see gap 1) |
| `fixed_from/wide_nulls_10` (1M) | 1.08 | 1.08 | unchanged |
| `lines_to/double` (1M) | 1.50 | 1.58 | unchanged; Rust float formatting, see below |
| `lines_to/wide` (1M) | 1.48 | 1.64 | the C++ side got faster; Rust float formatting |

The `huge_line` and `varchar_*` rows were measured after the last change (the no-`\` fast path). The other rows
come from the full run just before it, which did not touch their code paths. 1-row cases are all within 1% (fixed
cost of about 300 µs for an empty file and 600 µs for a 1-row file, all in DuckDB and the file system).

Tried and dropped:
- **Integer formatting.** Two variants replaced the stack buffer and its variable-length copy (a `memmove` call):
  pushing the digits and reversing them, and writing them into the output after counting them. Both were 10–25%
  slower on both sides.
- **Bulk COPY FROM writes for `vs_fixed`.** Decoding into a scratch `Vec<T>` and calling `VectorWriter::write_slice`
  instead of per-row `set` gained 4% at 0% NULL and lost 10% at 10% NULL.

## Remaining gaps attributable to duckdb-neo

### 1. Per-row `VectorWriter::set`, plus the zeroing `set_size`, on fixed-width COPY FROM output: 8–14%

- **Cases:**
  - `fixed_from/bigint`: 1.08 at 100k, 1.14 at 1M.
  - `fixed_from/wide_nulls_10`: 1.08 at 1M.
  - Part of `lines_from/bigint` (1.10) and `lines_from/wide_nulls_10` (1.14).
- **Profile, `fixed_from/bigint/2000000`:**
  - The engine (append, delete from `TRUNCATE`) dominates both runners equally.
  - Rust `VsFixed::exec` takes 9.7% self over 202 iterations; C++ `FixedFromExec` takes 6.6% over 217.
  - Inside the Rust exec, about 4.3 of the 9.7 points are `writer.rs` lines: `set`'s bounds check, the typed write,
    and `mark_valid_unchecked`'s check of an existing mask.
  - `__memset_avx512_unaligned_erms` adds 1.0%: `Vector::set_size` zeroing the rows that are about to be written.
  - The C++ loop is a plain store, `data[row] = value`, with the mask fetched on the first NULL.
- **Source:**
  - `crates/duckdb-neo/src/vector/writer.rs`: `VectorWriter::set` / `set_unchecked` / `mark_valid_unchecked`.
  - `crates/duckdb-neo/src/vector/mod.rs`: `Vector::set_size` → `init_rows`. Only `unsafe set_size_uninit` skips it.
- **Suggestion:** a safe bulk writer for a column being filled completely, so the per-row checks are hoisted out of
  the loop. One possible signature:
  `fn fill_with(&mut self, rows: usize, f: impl FnMut(usize) -> Option<T>) -> Result<()>`. It would size the vector
  without zeroing, write every slot, and touch the mask only for `None`. Writing all rows is the closure's contract,
  so no uninitialized slot is ever observable.

### 2. `flush` only gets `&InitData`, so the file needs a `Mutex`

- **API constraint:** `CopyToFunctionCallbacks::InitData` is `Send + Sync`, and `flush` and `finalize` receive
  `&Self::InitData`. `file::File` is `Send` but not `Sync`, so it lives in a `Mutex<File>`, locked once per batch.
- **C++:** the API hands out `T &` and documents that flush never runs concurrently for the same file.
- **Cost:** not visible in any profile (one uncontended lock per 2,048 rows). It is a usability gap: any mutable
  per-file writer state needs interior mutability.
- **Source:** `crates/duckdb-neo/src/copy_function/mod.rs`, `flush_to_callback` / `CopyToFunctionCallbacks::flush`.
- **Suggestion:** a second associated type, available only to `flush` and `finalize`, passed as `&mut`.

### 3. Copying one string into a VARCHAR output slot

- In `lines_from/huge_line` (every 1,024th line is a 2 MB string with an escape every other byte), Rust is 1.25–1.30
  behind C++.
- The profile is dominated by the per-byte unescape loop on both sides: `Vec::push` against `std::vector::push_back`.
  The Rust exec spends about 19% of its self time in `alloc/vec/mod.rs:1038/1044` (push), against about 21% for
  `stl_vector.h:1283/1288` in C++, over fewer iterations. It also spends 10% in the `zip` iterator of the field
  split.
- This looks like codegen of user loops more than duckdb-neo. One duckdb-neo part is that `VectorWriter::<String>::set`
  takes `&str`, so the bytes are validated with `std::str::from_utf8` before the writer copies them into the arena.
  That matches C++ `Arena::AddString` (validate, then copy), so it is not a gap in itself.
- Not investigated further.

### 4. Per-batch type clones and retained input (no measurable cost)

- **Type clones:** `batch_to_callback` clones the bind's `Vec<LogicalType>` for every batch, one
  `duckdb_v2_logical_type_copy` per column. `CopyFunctionBatchData` also keeps each batch's `ColumnDataCollection`
  alive until flush, even for formats that encode in `batch`.
- **Measurements:** `fixed_to/wide` (ten columns, 49 batches per 100k rows) runs at 0.94, so neither cost shows.
  Both come from `crates/duckdb-neo/src/copy_function/mod.rs`.

### 5. Errors from callbacks are prefixed twice

A file that cannot be opened in a COPY FROM callback fails with `IO Error: IO Error: Cannot open file ...` through
duckdb-neo, against `IO Error: Cannot open file ...` through the C++ API.

## API notes and awkwardness

- **Reading VARCHAR without validation:** this takes `UncheckedString`, which COPY TO uses. DuckDB guarantees valid
  UTF-8, and the C++ view reads bytes unvalidated as well. The checked `String` reader runs `std::str::from_utf8`
  on every value. `UncheckedString` is a safe type whose documentation calls invalid UTF-8 undefined behaviour.
- **No dynamic set of mutable output vectors:** `DataChunk::get_vector_array_mut::<N>` needs a compile-time column
  count, so a reader cannot hold one writer per column for a runtime column list. COPY FROM therefore splits fields
  into ranges first and fills one column at a time on both sides, to keep the algorithm the same.
- **C++ `FileHandle` lifetime:** the handle is documented as usable only while its `FileSystem` is valid, and a
  callback's `FileSystem` is borrowed for the callback's duration. The C layer (`CV2File`) keeps the handle
  independent, so both sides keep files open across callbacks.

## Where duckdb-neo or Rust is faster than the C++ API

- **Inline strings:** short VARCHAR writes go through `DuckDBBytes::write_parts` (`src/bytes.rs`), which copies at
  most 12 bytes with `copy_short` and no library call. The C++ `blob_t(const char *, uint32_t)` constructor, used by
  `Arena::AddBlob`, does a `memset` and a variable-length `memcpy` (a `memmove` call). On `lines_from/sep/1000000`,
  C++ spends 16.5% in `__memmove_avx512_unaligned_erms` against 2.4% for Rust. This is the main reason for
  `lines_from/sep` at 0.86.
- **UTF-8 validation:** the C++ `ValidateUTF8` runs DuckDB's `Utf8Proc::Analyze` (4.3% on `lines_from/sep`), while
  Rust's `std::str::from_utf8` is cheaper.
- **COPY TO VARCHAR reads:** `VectorReader<UncheckedString>` reads are as cheap as the C++ view
  (`lines_to/varchar_*` at 0.94–1.01).

## Differences that are not about either API

- **DOUBLE formatting:** `vs_lines` writes DOUBLE with 17 significant digits. The shortest round-trip forms of the two
  standard libraries break ties differently, so the files would differ.
  - Rust's `{:.16e}` runs `core::fmt`'s exact mode: Grisu with a Dragon4 bignum fallback. On `lines_to/wide/100000`,
    `grisu::format_exact_opt` takes 13.5%, `dragon::format_exact` 13.3%, `Big32x40::mul_pow2` 11.7% and
    `float_to_exponential_common_exact` 7.7%.
  - libstdc++'s `std::to_chars(..., scientific, 16)` is Ryu printf, a small share of the C++ profile.
  - So `lines_to/double` (1.58), `lines_to/wide` (1.64) and `lines_to/wide_nulls_50` (1.33) measure the standard
    libraries, not duckdb-neo.
- **DOUBLE parsing:** Rust uses `str::parse::<f64>` after `from_utf8` (`dec2flt` about 16%, `from_utf8` 5% on
  `lines_from/builtin_csv`); C++ uses `std::from_chars` (23%). Most of `builtin_csv`'s 1.18 is the same field-split
  and per-row write cost as `lines_from/bigint`.
- **Integer formatting codegen:** rustc unrolls the 20-iteration digit loop of `push_i64` completely, while clang
  keeps a loop (`copy.cpp`, 42% of the C++ batch). That is why `lines_to/bigint` is at 0.86.
- **Escape-dense text:** the per-byte split and unescape loops compile differently (gap 3).
- **Malformed input:** a few such inputs parse differently, which the cases never contain. `1e400` parses to infinity
  in Rust but fails in `from_chars`, and only `from_chars` accepts `nan(...)`.
