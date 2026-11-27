# Findings: `scalar_str`

Eight string scalar functions (`vs_len`, `vs_upper_ascii`, `vs_wrap`, `vs_repeat`, `vs_first_word`,
`vs_count_char`, `vs_has_prefix_any`, `vs_null_if_empty`), written once with duckdb-neo's safe API and once with
the C++ API (`duckdb::cxx`). Ratios are Rust median / C++ median from `run` (3 repetitions, CPU 3, machine shared
with other agents, so differences of a few percent are noise). Profiles are flat `perf` profiles of `--mode loop`
(`perf scalar_str/<case> --rows N --lang both`) and include the fixture and table setup.

## What changed and its effect

Rust:
- Inputs are read as `UncheckedString` instead of `String`. All eight functions only need the bytes, DuckDB
  guarantees valid UTF-8 in VARCHAR, and the C++ side reads `varchar_t` without validating.
- `vs_first_word` searches for the space with `bytes().position(..)` instead of `str::split_once(' ')`. The char
  searcher behind `split_once` (`CharSearcher::next_match` 14.2%, `memchr_aligned` 15.5%, `memcmp` 5.7%) cost
  more than the whole C++ search.
- `vs_repeat` checks the 4 GiB limit before building, as the C++ side now does.

C++:
- `vs_upper_ascii`, `vs_wrap` and `vs_repeat` build their result in place with a `BuildString` helper: a local
  12-byte buffer for inline results, else one `Arena::Allocate` that the bytes are written into. Before, upper and
  repeat filled a `std::string` and copied it with `AddStringUnsafe` (with an out-of-line
  `basic_string::_M_replace` per row), and wrap copied with `std::copy`.
- `vs_has_prefix_any` compares with `text.substr(0, n) == prefix` instead of `text.compare(0, n, prefix) == 0`.
- `vs_first_word` uses `std::find`, the same byte loop as Rust, instead of `string_view::find` (a libc `memchr`
  call per row, on mostly short strings).

| Case / rows | Before | After | What changed |
| --- | --- | --- | --- |
| `len/long/100000` | 30.56 | 1.06 | Rust: `UncheckedString` (was 72% `core::str::from_utf8`) |
| `len/in_where/100000` | 2.74 | 1.13 | Rust: `UncheckedString` |
| `len/short/100000` | 1.59 | 1.12 | Rust: `UncheckedString` |
| `wrap/column_affixes/100000` | 2.13 | 1.08 | Rust: `UncheckedString`; C++: `BuildString` |
| `wrap/constant_affixes/100000` | 1.24 | 0.73 | both, as above |
| `has_prefix_any/nulls_10/100000` | 1.41 | 0.99 | Rust: `UncheckedString` |
| `has_prefix_any/in_where/1000000` | 0.95 | 0.78 | Rust: `UncheckedString`; C++: `substr ==` |
| `count_char/column/100000` | 1.21 | 0.93 | Rust: `UncheckedString` |
| `first_word/mixed_nulls_10/100000` | 1.38 | 1.01 | Rust: `UncheckedString`, byte search; C++: `std::find` |
| `combined/aggregate_argument/100000` | 1.47 | 1.02 | Rust: `UncheckedString` |
| `combined/nested/1000000` | 1.39 | 0.99 | all of the above |
| `combined/group_by_dictionary/1000000` | 1.02 | 0.68 | Rust: byte search (see "faster" below) |
| `upper_ascii/short/100000` | 0.63 | 0.77 | C++: in place instead of a `std::string` |
| `upper_ascii/medium/100000` | 1.06 | 1.43 | C++: in place (see gap 1) |
| `upper_ascii/non_ascii/100000` | 1.15 | 1.41 | C++: in place (see gap 1) |
| `upper_ascii/filter_90pct/1000000` | 0.95 | 1.20 | C++: in place (see gap 1) |
| `repeat/long_output/100000` | 0.84 | 1.01 | C++: in place (see gap 2) |

Unchanged and close to 1 after the change: `count_char/constant_long`, `null_if_empty/mixed`,
`has_prefix_any/unprepared`, `wrap/constant_input`, `upper_ascii/long` at 100,000 rows and more (dominated by
copying 1 KiB per row on both sides), `combined/one_row` (1.04 to 1.05, the fixed per-query cost).

## Remaining gaps attributable to duckdb-neo

### 1. Writing a non-inlined string stalls on store forwarding (`DuckDBBytes::write_parts`)

Cases: `upper_ascii/medium` 1.12 / 1.43 / 1.39 (2,048 / 100,000 / 1,000,000 rows), `upper_ascii/non_ascii` 1.13 /
1.41 / 1.34, `upper_ascii/filter_90pct` 1.23 / 1.20, `upper_ascii/long/2048` 1.21, and part of
`repeat/long_output/1000000` 1.09 and `combined/many_calls/1000000` 1.10. Every result longer than 12 bytes goes
through this path.

Profile of `upper_ascii/medium/1000000`:
- Rust: `VectorWriter<String>::write_bytes` 23.64% self. `perf annotate` puts 99.25% of it on one load/store
  pair right after the prefix copy:
  `callq *%rbp` (memcpy of the 4-byte prefix into a stack local), `movq 0x2c(%rsp),%rax`,
  `movq %rax,0x4(%r14)` (99.25%). The 8-byte load spans the 4 prefix bytes the memcpy just stored and the
  pointer stored separately, so it cannot be forwarded and waits for the stores to retire.
- C++: `UpperAsciiExec` 5.99%, `duckdb_v2_arena_allocate` 5.32%; the `varchar_t` constructor copies the 4-byte
  prefix with an inlined constant-size `memcpy`.

Source: `crates/duckdb-neo/src/bytes.rs`, `DuckDBBytes::write_parts`, the pointer branch: it builds a
`DuckDBV2BytesPointer` local, fills `prefix` with `copy_parts(parts, pointer.prefix.as_mut_ptr(), 4)` (a
variable-length `copy_nonoverlapping` per part, i.e. a `memcpy` call), and then `slot.write(Self { .. })` copies
the local into the slot. The inline branch already avoids exactly this (its doc comment explains the stall).

Suggestion: in the pointer branch, build the prefix as a value in registers, e.g.
`let prefix: [u8; 4] = first_part[..4].try_into()` when the first part holds at least 4 bytes (the common case;
fall back to `copy_parts` otherwise), or write `length`, `prefix` and `ptr` straight into the slot field by field,
as the inline branch does. `copy_short` already handles the tiny copies without a call.

### 2. No safe way to build a string in place

Cases: `repeat/long_output` 1.06 / 1.01 / 1.09 (was 0.84 to 0.88 while C++ also went through a buffer) and the
share of gap 1 that is the extra copy in `vs_upper_ascii`.

The C++ side reserves the result bytes in the vector's heap (`Arena::Allocate`) and writes the upper-cased or
repeated bytes there once. duckdb-neo's `Arena::allocate` returns `*mut u8`, so filling it needs `unsafe`; the
safe code writes into a reused `String` and then copies it with `VectorWriter::set`. `set_parts` avoids the
copy for concatenations (`vs_wrap`), but not for computed bytes: a `vs_repeat` built as `set_parts` over a
`Vec<&str>` of `n` copies measured the same as the buffer (14.9 ms at 100,000 rows; `Vec::resize` 7% plus the
parts loops), so the buffer stayed.

Profile of `repeat/long_output/100000`: Rust closure 23.17% with `memmove` 29.58%; C++ lambda 10.77% plus the
exec function 9.90% with `memmove` 36.28%.

Source: `crates/duckdb-neo/src/arena.rs` (`Arena::allocate`), `crates/duckdb-neo/src/vector/writer.rs`
(`VectorWriter::<String>::set`, `set_parts`).

Suggestion: a safe in-place writer for byte types, e.g.
`VectorWriter::<BlobValue>::set_with(&mut self, row: usize, len: usize, fill: impl FnOnce(&mut [u8])) -> Result<()>`
(inline or one arena allocation, the prefix taken after `fill`), and for VARCHAR either
`set_str_with(row, len, |buf: &mut [u8]| ..)` that validates UTF-8 afterwards, or an ASCII-transform helper. With
it both `vs_upper_ascii` and `vs_repeat` would write their bytes once, like the C++ side.

### 3. Per-row cost of the typed reader and writer on fixed-width output

Cases: `len/short` 1.05 / 1.12 / 1.07, `len/nulls_50` 1.11 / 1.10, `len/in_where/100000` 1.13,
`null_if_empty/all_null/1000000` 1.11. Small in absolute terms (about 1.7 µs per 2,048-row chunk) but
consistent.

Profile of `len/short/1000000`: Rust closure (`TypedCallback<i64, ..>::exec`) 6.39% self against C++ `LenExec`
1.48%. The Rust loop body is about 20 instructions per row, not vectorized: the reader's selection tested per
row (the `for` loop uses `next`, which re-checks the storage kind every row; only `fold` hoists it), the validity
test, the `Option` test on the string pointer, the writer's bounds check against `len` reloaded from the stack,
the store, and the writer's `validity` pointer reloaded from the stack after each store (42.79% of the function's
samples sit on that reload: the raw-pointer store may alias the writer, so its fields stay in memory). The C++
`ScalarExecutor` checks `AllValid()` once and then runs `data[i] = fun(view.Data<varchar_t>()[SelAt(i)])`.

A `fold` with a `Result` accumulator (which takes the reader's hoisted path) measured no better (len/short/100000
1.10) and reads worse, so the code keeps the `for` loop.

Source: `crates/duckdb-neo/src/vector/reader.rs` (`VectorReaderIter::next` / `get_unchecked` per-row dispatch,
`fold_rows` only for `fold`), `crates/duckdb-neo/src/vector/writer.rs` (`set` bounds check,
`mark_valid_unchecked` reading `self.validity`).

Suggestion: an all-valid fast path in `MicroScalar` (or a `VectorWriter::fill_from(iter)` / `map_from(reader, f)`
that hoists the storage kind, skips the per-row validity work when the input has no mask, and keeps the writer's
pointers in locals).

### 4. `String` readers validate UTF-8 on every read (fixed here by `UncheckedString`)

`ReadableNativeElement for String` (`crates/duckdb-neo/src/types/string.rs`) runs `std::str::from_utf8` per row:
72.30% of `len/long/100000` (ratio 30.56) and 18.43% of `wrap/column_affixes/100000` (ratio 2.13) before the
switch. `UncheckedString` removes it and is what these functions use now. Its safety contract ("invalid UTF-8
causes undefined behavior") is an `unsafe` precondition behind a safe marker type; since DuckDB guarantees valid
UTF-8 in VARCHAR, the plain `String` reader could skip the validation too, or the docs could steer users to
`UncheckedString` for reads. A VARCHAR vector cannot be read as `BlobValue` (`&[u8]`), which would be the obvious
safe alternative for byte-oriented functions.

## API gaps and awkwardness

- No safe in-place string building (gap 2).
- `MicroScalar::create` covers one argument only. The two- and three-argument functions use
  `MicroScalar::with_signature` with an explicit `SignatureBuilder` and a return `LogicalType`, which needs a
  connection at registration (`.return_type(String::logical_type(conn)?)`), plus a `CombinedVectorIterator` and a
  `match` over the tuple of `Option`s. The C++ side writes `ScalarExecutor::Execute<R, A, B, C>(input, fn)`.
- Bind data needs the full trait route: `vs_has_prefix_any` implements `ScalarCallbacks` and
  `ScalarBindCallbacks` and casts and sizes its output itself, because `MicroScalar` has no bind hook. The C++
  variant adds one bind callback to the same executor pattern.
- On the plus side, `VectorWriter::<String>::set_parts` writes `prefix || s || suffix` into one inline slot or
  arena allocation; the C++ API has no multi-part helper, so `vs_wrap` hand-rolls it with `Arena::Allocate`.

## Where duckdb-neo is faster than the C++ API

- Short results: `upper_ascii/short` 0.77 / 0.83, `wrap/constant_affixes` 0.73 / 0.74,
  `combined/group_by_dictionary` 0.71 / 0.68, `repeat/short` 0.90. duckdb-neo writes inline strings straight
  into the result slot with fixed-size overlapping copies (`copy_short` in `crates/duckdb-neo/src/bytes.rs`). The
  C++ API returns a `varchar_t` by value: its constructor zeroes 16 bytes, `memcpy`s the bytes (a libc call for
  a variable length), and the result is reloaded into the vector, which stalls on store forwarding. In
  `group_by_dictionary/1000000` the C++ `FirstWordExec` lambda is 41.09% self with 47.50% of it on
  `addq $0x30,%rsp` right after `movq (%rsp),%rax` that reloads the just-built `varchar_t` (the stalled load),
  against 20.18% + 7.92% (`write_bytes`) on the Rust side. The mirror image of gap 1: duckdb-neo's inline path
  avoids the stall, the C++ API's does not, and the reverse holds for non-inlined strings.
- `vs_wrap` in C++ also pays three `memcpy` calls for tiny parts (`__memmove_avx512_unaligned_erms` 44.84% of
  `wrap/constant_affixes/1000000`, against 10.00% in Rust).

## Differences that are not about either API

- `has_prefix_any/in_where` 0.80 / 0.78: both sides call `bcmp` once per candidate prefix
  (`__memcmp_evex_movbe` 34.69% in C++, 24.06% in Rust), with the same algorithm; the difference is in how each
  compiler lays out the prefix loop (clang unrolls `std::any_of` by four), not in the APIs.
- `vs_first_word`: Rust's `str::split_once(' ')` and C++'s `string_view::find(' ')` are both library searches
  that cost more than a byte loop on short strings (Rust's verifies each match with `memcmp`, C++'s calls libc
  `memchr`); both sides now use a plain loop (`bytes().position`, `std::find`).
- `vs_upper_ascii` uses the same branch-free byte transform on both sides; a branchy `if (c >= 'a' && c <= 'z')`
  in C++ mispredicted on hex input and made the C++ side three times slower.

## Notes on the C++ side

- `Value::Get<varchar_t>().view()` on a temporary dangles for strings of at most 12 bytes, which live inside the
  `varchar_t` itself; the `vs_has_prefix_any` bind keeps the `varchar_t` alive. duckdb-neo's
  `resolved_value::<String>()` returns an owned `String`.
- `ScalarExecutor`'s documentation says string functions drive the vectors directly, but `varchar_t` works as an
  argument and result type when the callable places its results in `result.GetHeap()`; every function here uses
  the executor that way.
