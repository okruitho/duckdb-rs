# Findings: `cast`

User-defined casts between builtin types and two custom types: `vs_cents` (an alias of BIGINT, a count of cents) and
`vs_hex` (an alias of VARCHAR, uppercase hex like `to_hex`). The seven casts are `DOUBLE -> vs_cents`,
`vs_cents -> DOUBLE` (also implicit, cost 1), `VARCHAR -> vs_cents`, `vs_cents -> VARCHAR`, `BIGINT -> vs_hex`,
`vs_hex -> BIGINT` and `vs_cents -> vs_hex`. Parsing and formatting are the same hand-written digit loops in both
languages, so no ratio compares `std::fmt` with `printf`.

Ratios are Rust median / C++ median; above 1 means duckdb-neo is slower. "Before" is the central round-1 baseline.
"After" was measured on a dedicated CPU with other agents busy on other cores (3 repetitions), so differences under
about 5% are noise.

## What changed

- **C++:**
  - `ExecInput::GetRowCount()` is hoisted out of the loop. It was the loop condition, and each call is an
    out-of-line wrapper plus a C API call.
  - String output goes through one `Vector::GetHeap()` per callback, with `Arena::AddStringUnsafe` written straight
    into the `varchar_t` slots. Before, every row called `Vector::AssignString`. The formatted text is ASCII, so it
    needs no UTF-8 validation, and the Rust writer does not validate a `&str` either.
  - Typed input pointers are hoisted out of the loops.
- **Rust:** `VARCHAR -> vs_cents` and `vs_hex -> BIGINT` read their input as `types::UncheckedString` instead of
  `String`. DuckDB guarantees valid UTF-8 in VARCHAR, the parsers only look at bytes, and the C++ side reads bytes
  unvalidated too.
- **Tried and dropped (Rust):** a `fold` over `VectorReaderIter::enumerate()` with a `Result` accumulator in place of
  the `for` loop, for `vs_cents -> DOUBLE`. `fold` resolves the reader's storage kind once, but the ratio did not
  move (1.18 to 1.14, within noise), so I kept the plain loop.

| Case (rows) | Before | After | What changed |
| --- | --- | --- | --- |
| `cents_to_double/project` (2M) | 0.71 | 1.18 slower | C++ hoisted `GetRowCount`; the loop now vectorizes |
| `cents_to_double/implicit_argument` (2M) | 0.68 | 1.11 slower | same |
| `cents_to_double/implicit_compare` (2M) | 0.66 | 1.03 | same; the comparison dominates |
| `double_to_cents/project` (2M) | 0.71 | 1.11 slower | same |
| `double_to_cents/try_50` (2M) | 0.68 | 1.04 | same |
| `varchar_to_cents/nulls_100` (2M) | 0.79 | 1.06 | same |
| `varchar_to_cents/try_50` (2M) | 1.10 slower | 1.01 | Rust `UncheckedString` |
| `varchar_to_cents/*` (2M, other cases) | 1.00 to 1.05 | 0.97 to 1.02 | Rust `UncheckedString`; C++ `GetRowCount` |
| `hex_to_bigint/project` (2M) | 1.17 slower | 1.00 | Rust `UncheckedString`; C++ `GetRowCount` |
| `hex_to_bigint/try_1` (2M) | 1.12 slower | 1.01 | same |
| `cents_to_varchar/short` (2M) | 0.41 | 0.66 faster | C++ one heap and `AddStringUnsafe` instead of `AssignString` |
| `cents_to_varchar/long` (2M) | 0.76 | 1.06 | same |
| `cents_to_varchar/nulls_50` (2M) | 0.48 | 0.74 faster | same |
| `bigint_to_hex/short` (2M) | 0.26 | 0.61 faster | same |
| `bigint_to_hex/long` (2M) | 0.57 | 0.95 | same |
| `cents_to_hex/project` (2M) | 0.41 | 0.77 faster | same |
| `chained` (2M) | 0.49 | 0.72 faster | both sides' changes |
| `many_casts` (2M) | 0.48 | 0.78 faster | both sides' changes |
| `insert/double` (100k) | 0.84 | 1.00 | C++ `GetRowCount` |

At 2,048 rows every case is now between 0.91 and 1.04 ("same"), because the fixed per-query cost of about 130 µs
dominates. The remaining `faster` results (short strings) are explained below under "Places where duckdb-neo is
faster".

## Remaining gaps attributable to duckdb-neo

### 1. Numeric per-row casts do not vectorize (1.11 to 1.18)

- **Cases:**
  - `cents_to_double/project` 1.18 (2M) and 1.22 (100k)
  - `cents_to_double/implicit_argument` 1.11 (2M)
  - `double_to_cents/project` 1.11 (2M) and 1.12 (100k)
- **Profile:** `cents_to_double/project` at 2M rows.
  - Rust: `CentsToDouble::exec` has 25.9% self time.
  - C++: `CentsToDoubleExec` has 16.4% self time.
  - Engine symbols are the same on both sides, e.g. `duckdb::CopyVectorBuffer` at 8.6% (Rust) and 9.8% (C++).
- **Annotation:**
  - Rust: the loop does one scalar `divsd` per row. Samples sit on the instruction after it, `movq 0x48(%rsp),%rax`
    (50.8%): the output pointer is reloaded from the stack every row. There is also a per-row `cmpq $-0x1,%rax`
    (21.2%), which is the `Option` of the reader's per-row physical-index lookup.
  - C++: clang moves the loop-invariant `validity == nullptr` / `sel == nullptr` tests of `VectorView::IsValid` out of
    the loop. It then packs two rows per store (`unpcklpd` and `movupd %xmm2,(%rbx,%rdx,8)`, 72.3%).
- **Responsible code:**
  - `VectorReaderIter::next` → `VectorReader::get_unchecked` → `physical_index_unchecked` matches the storage kind
    per row (`crates/duckdb-neo/src/vector/reader.rs`). The `fold` override (`fold_rows`) hoists that match, but a
    plain `for` loop does not use it.
  - `VectorWriter::set` / `set_option` (`crates/duckdb-neo/src/vector/writer.rs`) do a bounds check, return a
    `Result`, and call `mark_valid_unchecked` per row.
  - Together these keep LLVM from unswitching and vectorizing the loop. Switching to `fold` alone did not help,
    because the writer side stays per-row and fallible.
- **Suggestion:** a safe bulk element-wise API that resolves storage and validity once and works on slices in the
  common flat case. For example:
  - `VectorWriter::<O>::map_from<I>(&mut self, input: &VectorReader<'_, I>, f: impl FnMut(I::Ref<'_>) -> O::Write) -> Result<()>`
    copies input NULLs, and runs a `zip` over `&[I]` / `&mut [O]` when the input is flat (with or without a validity
    mask). LLVM can vectorize that loop.
  - A cast-friendly `try_map_from(.., f: impl FnMut(I::Ref<'_>) -> Option<O::Write>)` that writes NULL for `None`.
    That is exactly the TRY-mode contract.
  - Alternatively, safe slice accessors for `PlainNative` types: `VectorReader::flat_values() -> Option<&[T]>` and
    `VectorWriter::values_mut() -> &mut [T]` for a flat output.

### 2. TRY mode pre-fills the output with NULL (small)

- **Cases:** `double_to_cents/try_50` 1.04 (2M), `varchar_to_cents/try_*` 1.01 to 1.03.
- **Responsible code:** `cast::exec_callback` (`crates/duckdb-neo/src/cast/mod.rs`) runs `output.writer()?.fill_null()`
  before the user callback when the mode is `Try`. Every converted row then sets its validity bit again. The C++
  callback marks only the failed rows.
- It is a whole-mask pass per callback plus a writer build and buffer refresh: cheap next to parsing, but visible on
  cheap numeric casts.
- **Suggestion:** have the writer set validity only for the rows the callback did not reach. For example, track the
  highest written row, or fill only on the error/panic path, which is the only time unwritten rows are possible.

### 3. Per-row NULL writes cost slightly more than in C++ (small)

- **Case:** `varchar_to_cents/nulls_100` 1.06 (2M), 1.10 (100k), status "same".
- **Profile:** all-NULL input at 2M rows. Rust `VarcharToCents::exec` has 17.3% self time, C++ `VarcharToCentsExec`
  13.7%.
  - Rust: the loop runs the reader's per-row storage and validity test, then `VectorWriter::set_null` (bounds check
    plus `mark_null_unchecked`).
  - C++: one `btq` and one `andq` per row on a validity mask taken once with `GetValidityMutable()`.
- **Suggestion:** the same bulk API as in gap 1 would copy the input mask instead of writing NULL row by row.

### 4. `String` input re-validates UTF-8 (fixed here with `UncheckedString`)

- **Before:** with `String` input, `hex_to_bigint/project` was 1.17. `core::str::converts::from_utf8` had 16.9% self
  time in the Rust profile, against nothing comparable in C++.
- **Responsible code:** `ReadableNativeElement for String::get` (`crates/duckdb-neo/src/types/string.rs`) calls
  `std::str::from_utf8(..).expect(..)` on every row read.
- **After:** `types::UncheckedString` (safe, public) brings it to 1.00. But its name suggests the caller is taking a
  risk, so a first-pass user reaches for `String`.
- **Suggestion:** read VARCHAR unvalidated in `String` as well, since DuckDB guarantees valid UTF-8. Failing that,
  document `UncheckedString` as the normal way to read VARCHAR.

## API gaps and awkwardness

- **duckdb-neo: no closure form for casts.** Scalars have `MicroScalar::create(name, closure)`. A cast needs a struct
  plus a `CastFunctionCallbacks` impl with two associated types and a five-argument `exec`, about 15 lines per cast
  against a plain function pointer in C++.
- **duckdb-neo: many C API calls before user code runs.** `cast::exec_callback` fetches user data, row count, input
  and output handles (each through `Vector::from_handle`) and the mode, and calls `set_size` when the length differs,
  all before user code. The C++ `ExecInput` fetches these on demand. The one-row (`varchar_to_cents/constant`,
  `one_row`) cases show no measurable effect (1.01 to 1.02), so this is not a priority.
- **Both: registering a custom type does not return its `LogicalType`.** You register it, then build it again with
  `with_alias` / `WithAlias` for the cast. The two APIs are at parity here.
- **C++: `ExecInput::GetRowCount()` is an out-of-line call into the C API.** Used as the loop condition, as the
  official cast test does, it cost up to 30% of the cast. Before the fix, in `cents_to_double/project`:
  `GetRowCount` had 14.2% self time and `duckdb_v2_cast_function_exec_get_row_count` 8.5%. It could be cached in
  `ExecInput`, or the docs could say to hoist it.
- **C++: `Vector::AssignString` does a type lookup per call.** Before the fix, in `cents_to_varchar/short`:
  `Arena::AddStringUnsafe` 27.1%, `AssignString` 6.2%, `duckdb_v2_vector_get_logical_type` 4.0%,
  `duckdb_v2_validate_utf8` 2.6%, `VectorStringBuffer::GetHeap` 1.9%, `duckdb_v2_logical_type_destroy` 1.3%. That is a
  type lookup, UTF-8 validation and heap lookup per row. The docs suggest `AssignString` "for a straightforward
  fill", which is what a first-pass user writes.
- **Neither API has a cast executor.** `ScalarExecutor` has no counterpart for casts, so both sides write hand loops.

## Places where duckdb-neo is faster than the C++ API

- **Short (inlined) string output: 0.61 to 0.77.** Cases: `bigint_to_hex/short`, `cents_to_varchar/short` and
  `nulls_50`, `cents_to_hex/project`, `chained`, `many_casts`.
  - C++: `Arena::AddStringUnsafe` → `AddBlob` → the `blob_t(const char *, uint32_t)` constructor in
    `duckdb_cpp.hpp` does `memset` plus a runtime-size `memcpy`. Clang emits that as a libc call:
    `__memmove_avx512_unaligned_erms` has 36.8% self time in C++ `bigint_to_hex/short`.
  - Rust: `VectorWriter::<String>::write_bytes` inlines the copy of up to 12 bytes as two overlapping 8-byte moves.
  - The public C++ API has no route that avoids the constructor's copy.
- **Long strings (heap-allocated) are at parity (0.95 to 1.06).** Both sides spend the time in
  `StringHeap::CreateEmptyStringInHeap` and the copy into the arena.

## Differences that are not about either API

- Both formatting casts build the text byte by byte (`String::push` / `std::string::push_back`) and then copy it as
  whole words. That is a store-forwarding stall on both sides. In Rust, the instruction after the 8-byte load in
  `write_bytes` has 45.7% of that symbol's samples. It inflates the cost of short string output for both languages
  alike.
- The numeric gap in gap 1 shows as vectorization in C++ only because clang unswitches the loop on loop-invariant
  pointer tests. It is still the duckdb-neo reader/writer shape that blocks the equivalent in Rust.
- DuckDB constant-folds casts of literals at bind time, so `varchar_to_cents/constant` uses a single-row
  `CROSS JOIN` to deliver a CONSTANT vector at run time. A strict cast failure is not benchmarked; the checks detect
  it as `try(CAST(..) IS NULL) IS NULL`.
