# Findings: `scalar_num`

Numeric scalar functions (`vs_add_one`, `vs_clamp`, `vs_is_even`, `vs_coalesce_zero`, `vs_checked_div`, `vs_mix`,
`vs_strict_sqrt`, `vs_scale`) written once with duckdb-neo's safe API and once with the C++ API (`duckdb::cxx`,
`ScalarExecutor`). Ratios are Rust median / C++ median; above 1 means duckdb-neo is slower. Numbers come from
`run` (3 repetitions) on CPU 2 while other agents built on other cores, so differences under about 5% are noise.

## Summary

Almost every gap in this family has the same cause: the per-row cost of duckdb-neo's safe read and write path
(`VectorReaderIter::next`, `VectorWriter::set_option`). Fixed costs per query and per call are small: about 5% at
2,048 rows and below. Per-row costs are where the gap lies. The C++ executor compiles its all-valid loop to
vectorized code; no safe duckdb-neo loop shape does. Replacing the safe loop with `unsafe` slices (an experiment,
not kept) closes `add_one/project` from 1.32 to 1.08. A safe bulk API would therefore remove most of the gap
(gap 1).

## Changes and before/after

The C++ side is unchanged: `ScalarExecutor` is already the fastest public route, and no C++ variant did avoidable
work. On the Rust side only `vs_add_one` changed: its `for` loop became `fold` with a `Result` accumulator.
`fold` is the one iteration form the readers specialize, resolving the storage kind once per batch instead of per
row.

`for` vs `fold` in `vs_add_one`, same session, 3 repetitions:

| Case | rows | ratio, `for` | ratio, `fold` | Rust time, `for` → `fold` |
| --- | ---: | ---: | ---: | --- |
| add_one/nested_8 | 100,000 | 1.99 | 1.64 | 1.39 ms → 1.12 ms |
| add_one/nested_8 | 2,000,000 | 1.85 | 1.57 | 29.3 ms → 23.0 ms |
| add_one/nulls_10 | 100,000 | 1.31 | 1.22 | 341 µs → 310 µs |
| add_one/nulls_100 | 100,000 | 1.13 | 1.07 | 333 µs → 312 µs |
| add_one/sum | 100,000 | 1.26 | 1.20 | 386 µs → 365 µs |
| add_one/sum | 2,000,000 | 1.37 | 1.30 | 5.15 ms → 4.42 ms |
| add_one/project | 100,000 | 1.35 | 1.34 | unchanged |
| add_one/calls_16 | 2,000,000 | 1.15 | 1.29 | within noise (±7 to 11%) |

`is_even/where_nested` calls `vs_add_one` and benefits too. The round-1 baseline had 1.96 / 1.99 for `nested_8`
and 1.54 for `sum` at 2M rows.

Tried and dropped, because nothing helped or something regressed (quick runs, ratios at 100,000 / 2,000,000 rows):

| Variant (Rust, unless noted) | add_one/project | add_one/nested_8 | is_even/where | scale/const_factor | strict_sqrt/project |
| --- | --- | --- | --- | --- | --- |
| `for` (round 1, kept for all but `vs_add_one`) | 1.32 / 1.25 | 2.02 / 1.89 | 1.17 / 1.21 | 1.26 / 1.11 | 1.00 / 0.94 |
| `fold` with a `Result` accumulator, every function | 1.27 / 1.20 | 1.69 / 1.62 | 2.36 / 2.66 | 2.81 / 2.43 | 2.13 / 1.92 |
| `for_each`, first error kept in an `Option<Error>` | 1.55 / 1.37 | 2.75 / 2.54 | 1.31 / 1.41 | 1.40 / 1.33 | 1.08 / 1.08 |
| `for row in 0..writer.len().min(reader.len())` with `reader.get(row)?` | 1.33 / 1.14 | 1.75 / 1.67 | 1.17 / 1.18 | | |
| collect into a `Vec`, `write_slice`, then `set_null` the NULL rows | 1.33 / 1.30 | 2.03 / 1.84 | | | |
| `unsafe` `as_slice`/`as_slice_mut` when flat and all valid (not allowed, reference only) | **1.08 / 1.04** | 2.00 / 2.15 (has NULLs) | | | |
| `vs_scale`: NULL factor hoisted out of the loop (`fill_null` / `SetAllInvalid`), both sides | | | | Rust 376 → 399 µs, C++ 292 → 296 µs | |

The same `fold` that speeds up `vs_add_one` makes `vs_is_even`, `vs_scale`, `vs_strict_sqrt` and `vs_clamp` about
2x slower. In those functions the `Result<(), Error>` accumulator is kept on the stack. In `is_even`, 99.8%
of the function's samples sit on a single `movups 0x18(%rsp), %xmm1`, the reload of that accumulator in every
iteration. A user cannot predict which way a function will go, which is itself a finding (gap 3).

Case added: `mix/nulls_0`, `vs_mix` without NULLs, so C++ takes its vectorized all-valid loop. It shows the gap
that the NULL-containing `mix/project` hides: 1.37 / 1.24 vs 1.08 / 1.04. I also fixed `all/one_row`'s
`SMALLINT` column (`i % 1000`) so the case works when `--rows` overrides the row count.

## Remaining gaps attributable to duckdb-neo

### 1. No safe bulk or vectorizable path for plain numeric vectors (largest)

- Cases: `add_one/project` 1.34 / 1.26, `mix/nulls_0` 1.37 / 1.24, `coalesce_zero/nulls_0` 1.18 / 1.15,
  `add_one/sum` 1.20 / 1.30, `add_one/calls_16` and `scale/calls_16` around 1.3 (100,000 / 2,000,000 rows).
- Profiles at 2M rows, self time of the user function (iterations in 3 s in parentheses):
  - `add_one/project`: Rust `TypedCallback<i64, …register_add_one…>::exec` 21.7% (313) vs C++
    `AddOneExec` 8.6% (349).
  - `mix/nulls_0`: Rust `…register_mix…::exec` 36.9% (193) vs C++ `MixExec` 26.9% (255).
  - `add_one/sum`: Rust 48.9% (502) vs C++ 35.8% (688).
- C++ annotation, `AddOneExec`: `movdqu`/`psubq`/`movdqu`, two `i64` lanes per instruction, four rows per iteration.
- Rust annotation, `add_one/project` (no NULLs): one scalar row per iteration. Each row pays:
  - the storage-kind branch and an input validity-pointer test;
  - the writer's bounds check (`cmpq %rdx,%r15`, 25.6% of the function's samples);
  - the store;
  - reloads of the writer's data and validity pointers from the stack after every store (`movq 0x10(%rsp)`, 28.4%).
    The store goes through a raw pointer the compiler cannot separate from the writer's own fields.
- Responsible code:
  - `VectorWriter::set` / `set_option` (`crates/duckdb-neo/src/vector/writer.rs:308`, `:328`): a bounds check and a
    `Result` per row, and an early exit that blocks vectorization.
  - `set_unchecked` → `mark_valid_unchecked` (`writer.rs:342`, `:444`): a validity test per row.
  - Slice access exists only as `unsafe` (`VectorReader::as_slice`, `crates/duckdb-neo/src/vector/reader.rs:190`;
    `VectorWriter::as_slice_mut`, `writer.rs:154`). Using them closes the gap to 1.08 / 1.04.
- Suggestions, either of which would do:
  - Safe slices where they are sound:
    - `impl<T: PlainNative> VectorReader<'_, T> { fn valid_values(&self) -> Option<&[T]> }` returns `Some` only
      for flat storage without a validity mask. Every slot is then initialized.
    - `impl<T: PlainNative> VectorWriter<'_, T> { fn values_mut(&mut self) -> &mut [T] }` marks the rows valid.
      `set_size` already zero-initializes new rows.

    User code becomes
    `for (out, v) in writer.values_mut().iter_mut().zip(input) { *out = v.wrapping_add(1) }`.
  - A bulk mapping API that owns the loop, like DuckDB's `UnaryExecutor`:
    `fn map_from<I: PlainNative>(&mut self, input: &VectorReader<'_, I>, f: impl FnMut(I) -> T) -> Result<()>`,
    with `try_map_from` (fallible `f`) and `map_option_from` (`f` returns `Option<T>`). It checks bounds once,
    copies or propagates validity a word at a time (as `copy_from` already does, `writer.rs:526`), and runs a
    branch-free slice loop for flat, all-valid inputs.

    On top of it: `MicroScalar::unary/binary/ternary` constructors matching `ScalarExecutor::Execute` (see the API
    gaps below).

### 2. The per-row NULL path costs about twice the C++ one

- Cases: `add_one/nested_8` 1.64 / 1.57 even after `fold` (1.99 / 1.85 with `for`). Also `add_one/nulls_10`,
  `is_even/where` and `is_even/where_nested` at 1.15 to 1.30, and `clamp/columns` at about 1.3. Each has 10%
  NULLs, so both sides run a scalar loop.
- Profile, `nested_8` at 2M rows: Rust `…register_add_one…::exec` 61.6% (81 iterations) vs C++ `AddOneExec` 45.4%
  (147).
- Per row, C++ tests one input validity bit and stores. Rust also tests the validity pointer and dispatches on the
  storage kind (with `for`). It checks the writer's bounds and reloads the writer's fields. And once the output has
  a mask, it tests the output bit on every valid row in `mark_valid_unchecked` (`btq %r15,%rsi`, 11.6% of the
  function).
- C++'s `ScalarExecutor` instead calls `ResetValidity(count)` once and only sets the bits of NULL rows.
- Suggestion: the bulk API of gap 1 handles NULL inputs the same way, resetting the output validity once and
  writing value rows without touching the mask. Within the current API, `VectorWriter` could reset the mask once
  when it is created over freshly sized output, so that `set` no longer needs to test bits that are already valid.

### 3. Fallible per-row loops cannot use the readers' specialized iteration

- Duckdb-neo's readers override `fold` (`reader.rs:290` `fold_rows`; `combined.rs` `fold_indexed`), which hoists
  the storage dispatch.
- Writer calls return `Result`. The natural loops (`for` with `?`, or `try_for_each`) go through `next()`, because
  `Iterator::try_fold` cannot be overridden on stable Rust.
- Folding with a `Result` accumulator works for `vs_add_one`, but is 2x slower for four other functions (table
  above). Clippy also rejects it (`manual_try_fold`) and suggests `try_fold`, which loses the specialization, so
  `vs_add_one` needs an `#[expect]`.
- Suggestion: make the common loop infallible so that `for_each` (which uses `fold`) applies. Options:
  - a writer row iterator zipped with the reader, `for (slot, value) in writer.slots().zip(reader.iter())`;
  - a bounds-proven `set` on a row index handed out by the writer;
  - the bulk API of gap 1.

### 4. `CombinedVectorIterator` leaves its fast path when any argument is constant

- Cases: `clamp/const_bounds` 1.31 / 1.30 (baseline), against `clamp/columns` at 1.16 with three flat columns.
- Profile, 2M rows: Rust `…register_clamp…::exec` 39.7% vs C++ `ClampExec` 27.1%.
- `CombinedVectorIterator::new` takes the `read_flat_unchecked` path only when every reader is FLAT
  (`crates/duckdb-neo/src/vector/combined.rs`, `all_flat`). With literal bounds, all three readers go through
  `read_unchecked`, a storage dispatch per reader per row. C++ maps every argument through `SelAt`.
- Suggestion: in `fold_indexed`, read the CONSTANT readers once before the loop and keep a flat path for the rest,
  or let `VectorReaderTuple` dispatch per reader outside the loop.

### 5. Small fixed cost per call

- `all/one_row`, 1 row (8 calls): 1.07.
- Rust binary self time is 2.3% vs 0.7% for the C++ binary, roughly 6 µs per query, spread over
  `VectorCollection::wrap`, `get_vector_at_mut`, `LogicalTypeID::try_from`, `StorageKind::try_from` and
  `LogicalType::drop` (each 0.05 to 0.15%).
- At 2,048 rows the ratios are 1.00 to 1.07, within noise.
- Suggestion: none urgent. Fetching the output's logical type once per registration rather than per call would
  remove part of it.

## API gaps and awkwardness (still true)

- **No typed N-argument executor.** C++ writes `ScalarExecutor::Execute<double, double, double, double>(input, fn)`
  and gets NULL propagation plus the all-valid fast path. duckdb-neo's `MicroScalar::create` is unary only. Two or
  more arguments need:
  - `MicroScalar::with_signature`;
  - `CombinedVectorIterator` over `input.get_vector_at::<T>(i)?.reader()?`;
  - a `match` on the tuple of `Option`s;
  - `set_option` per row (`vs_clamp`, `vs_checked_div`, `vs_mix`).

  Suggested:
  `MicroScalar::binary::<A, B, O>(name, |a: A, b: B| -> O)`, and so on, built on the bulk API of gap 1.
- **Bind data needs the full callback trait.** `vs_scale` is a struct implementing `ScalarCallbacks` and
  `ScalarBindCallbacks`. Its `exec` takes six parameters, casts and sizes the output by hand, and unwraps
  `Option<&BindData>` although `with_bind` guarantees it. C++ adds `SetBindCallback` to the same function and keeps
  `ScalarExecutor`.
- **Neither API reads SEQUENCE vectors.** `SELECT vs_add_one(i) FROM range(10) r(i)` fails on both sides:
  - duckdb-neo: "OTHER vectors must be flattened before reading";
  - C++: "`duckdb_v2_vector_get_view`: vector is FSST / SEQUENCE / SHREDDED".

  Neither `MicroScalar` nor `ScalarExecutor` flattens. The cases build tables first.
- **`set_option` takes `Option<&T>`.** A computed value needs `.as_ref()` on a temporary.
- **`Error::internal` invalidates the whole instance**, which its name does not suggest. `Error::invalid_input` is
  the right choice for a query error such as `vs_strict_sqrt`'s.
- `UncheckedString` does not apply: no function in this family reads VARCHAR. The `add_one/dictionary` case gets
  its dictionary vectors from `length(s)` on a dictionary-compressed VARCHAR column.

## Where duckdb-neo is not slower

- `strict_sqrt/project` is at 0.94 to 1.08, and `add_one/order_by`, `clamp/const_all`, `add_one/where_1pct`,
  `add_one/literal_unprepared` and `add_one/dictionary` are within 5 to 10%.
- In all of them, engine work (sort, cross product, filter, `length`, planning) or `sqrt` latency dominates, so the
  per-row wrapper cost is hidden. No case showed duckdb-neo clearly faster.

## Differences not about either API

- `vs_clamp`: the Rust side spells out C++'s `std::min(std::max(v, lo), hi)` with comparisons, because
  `f64::max`/`f64::min` ignore NaN. C++ compiles it to `maxsd`/`minsd`. The algorithm is the same; code generation
  for the comparison may differ slightly.
- `vs_strict_sqrt` is bound by `sqrtsd` latency on both sides.
- The C++ runner is built with clang `-O3` for baseline x86-64 (SSE2 only, hence two lanes). The Rust runner uses
  the `bench` profile, also baseline x86-64. Neither uses AVX, so the vectorization gap is not an instruction-set
  difference.
