# Findings: `cast`

Round 1 (straightforward code on both sides). Timings from the shared machine are not used as evidence here; the
items below come from reading both APIs while writing the same casts, and are to be confirmed with `perf`.

## Design

Two custom types, both registered as an alias of a builtin type in both APIs (`CustomType` + `with_alias` /
`WithAlias`): `vs_cents` (BIGINT, a count of cents) and `vs_hex` (VARCHAR, uppercase hex like `to_hex`). Seven casts:
`DOUBLE -> vs_cents`, `vs_cents -> DOUBLE` (implicit, cost 1), `VARCHAR -> vs_cents`, `vs_cents -> VARCHAR`,
`BIGINT -> vs_hex`, `vs_hex -> BIGINT`, `vs_cents -> vs_hex` (custom to custom). Parsing and formatting are hand-rolled
digit loops on both sides, so the comparison does not measure `std::fmt` against `printf`.

## duckdb-neo (safe Rust)

1. **`String` input re-validates UTF-8 on every row.** `ReadableNativeElement for String::get` runs
   `std::str::from_utf8(..).expect(..)` per row read. The C++ view hands out `varchar_t` bytes without checking. For
   `VARCHAR -> vs_cents` and `vs_hex -> BIGINT` this is a per-row scan of the input that the C++ side does not do.
   DuckDB already guarantees valid UTF-8 in VARCHAR vectors. `types::UncheckedString` avoids the check and is accepted
   as a cast `InputType` (its `TYPE_ID` is VARCHAR). But the name suggests the caller is taking a risk, so a first-pass
   user picks `String`. Wanted: a validation-free `&str` (or `&[u8]`) read of VARCHAR input as the default path.
2. **TRY mode pre-fills the output with NULL.** `cast::exec_callback` calls `output.writer()?.fill_null()` before the
   user callback whenever the mode is `Try`. Every converted row then sets its validity bit again through
   `writer.set`. The C++ API leaves the output alone, and its callback marks only the failed rows invalid. So every
   `TRY_CAST` call pays one extra pass over the validity mask, plus a writer construction and buffer refresh. It buys
   "rows left unwritten read as NULL", including after a panic or error. Seen in the `try_0` / `try_1` / `try_50`
   cases.
3. **The exec wrapper does several C API calls per callback before user code runs**: user data, row count, input
   handle + `Vector::from_handle`, mode, output handle + `Vector::from_handle`, `set_size` when the length differs,
   then `reader()` / `writer()` setup. C++ `ExecInput` fetches these lazily, and a callback that ignores the mode or
   user data never asks for them. This matters for the constant-input case (`varchar_to_cents/constant`, where each
   callback converts one row) and for `one_row`.
4. **Ergonomics: no closure form for casts.** Scalars have `MicroScalar::create(name, closure)`. A cast needs a struct
   plus a `CastFunctionCallbacks` impl with two associated types and a five-argument `exec`, about 15 lines of
   boilerplate per cast against a plain function pointer in C++. Seven casts made `rust/cast.rs` noticeably longer
   than `cpp/cast.cpp` for the same logic.
5. **Ergonomics: custom type then alias type, twice.** Registering a custom type does not give back its
   `LogicalType`. You call `CustomType::new(name, base).register(conn)` and then `base.with_alias(conn, name)` to get
   the type for `CastFunctionBuilder`. The C++ API has the same two-step shape, so this is parity, not a gap. A
   `CustomType::register` that returns the aliased `LogicalType` would remove a source of name mismatches.

## C++ API (`duckdb::cxx`)

1. **`Vector::AssignString` validates UTF-8 of every written string.** The Rust writer takes `&str` and does not need
   to. `AssignStringUnsafe` skips the check, but the docs steer users to `AssignString`. This costs C++ time in the
   formatting casts (`cents_to_varchar/*`, `bigint_to_hex/*`, `cents_to_hex/*`). It is a C++-side handicap to keep in
   mind when reading those ratios.
2. **No cast executor.** `ScalarExecutor` has no counterpart for casts, so the C++ callbacks are hand loops over
   `VectorView` + `GetValidityMutable` + `GetDataMutable`, the same as the official `test_cpp_api_cast_function.cpp`.
   The Rust side is also a hand loop (reader iterator + writer), so the two are at parity.

## Notes on DuckDB behaviour the cases rely on

- An alias type has no implicit casts unless one is registered (`cast_rules.cpp`: differing aliases give -1). So
  `vs_cents < DOUBLE` and `abs(vs_cents)` bind only through the registered `vs_cents -> DOUBLE` implicit cast.
  Explicit casts between an alias and its base type (`vs_cents <-> BIGINT`, `VARCHAR <-> vs_hex`) use DuckDB's
  builtin casts, and the checks use them to read the representation.
- A cast of a literal (`'12.34'::vs_cents`) is constant-folded once at bind time. The constant case therefore uses
  `CROSS JOIN (SELECT '12.34' AS s)`, whose single-row side reaches the callback as a CONSTANT vector of one row.
- A failing strict cast cannot be benchmarked. The checks detect it as `try(CAST(x AS T) IS NULL) IS NULL`: `try`
  turns a raised error into NULL, while a cast that quietly returned NULL would make the inner `IS NULL` true.
