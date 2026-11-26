# Findings: `scalar_str`

Measured with `run --quick` on the shared machine (smoke-test quality; ratios are Rust / C++). Profiles from
`perf scalar_str/<case> --rows 100000`, which include the fixture and setup.

## 1. Every `String` / `&str` read validates UTF-8

`ReadableNativeElement for String` (`src/types/string.rs`) runs `std::str::from_utf8` on every row it reads, so
the cost grows with the string length. The C++ API reads `varchar_t` bytes without validating them, and DuckDB
already guarantees that VARCHAR holds valid UTF-8.

| Case | Rust | C++ | Ratio |
| --- | --- | --- | --- |
| `len/long/100000` (200 to 2,000 bytes) | 17.6 ms | 573 µs | 30.7 |
| `len/in_where/100000` (mixed) | 1.93 ms | 707 µs | 2.73 |
| `wrap/column_affixes/100000` (three VARCHAR reads per row) | 8.39 ms | 3.85 ms | 2.18 |
| `len/short/1000000` | 11.4 ms | 7.72 ms | 1.48 |
| `has_prefix_any/nulls_10/1000000` | 17.5 ms | 12.8 ms | 1.37 |
| `count_char/column/1000000` | 21.3 ms | 16.4 ms | 1.30 |
| `first_word/mixed_nulls_10/100000` | 3.38 ms | 2.61 ms | 1.29 |

Profile of `len/long/100000` (Rust): `72.30% core::str::converts::from_utf8`, then `6.47%` for the function's own
closure. In `wrap/column_affixes/100000` it is `18.43% from_utf8` beside `30.47%` for the closure. The C++
profile of `len/long` shows no per-byte work at all (`LenExec` at 2.16%).

- Wanted: read a VARCHAR row as `&str` or `&[u8]` without per-read validation, e.g. `vs_len` as
  `value.len()` over the stored bytes.
- Had to write: `Vector<'_, String>` readers, which validate. The only reader that skips validation is the
  `UncheckedString` marker, whose safety contract ("every non-NULL row must be valid UTF-8, otherwise undefined
  behavior") is an `unsafe` precondition behind a safe type, so these functions do not use it. Reading a VARCHAR
  vector as `BlobValue` (`&[u8]`) is rejected by the type check.
- Options: trust DuckDB's VARCHAR invariant in the `String` reader (as `UncheckedString` does), or add a safe
  byte view of VARCHAR rows for functions that do not need `&str`.

## 2. No safe way to build a string result in place

`vs_upper_ascii` and `vs_repeat` compute a result of known length. The C++ API can reserve the bytes in the
result's heap (`Arena::Allocate`) and place them with `Vector::SetString`. On duckdb-neo `Arena::allocate`
returns `*mut u8`, so filling it needs `unsafe`; the safe route is to build the value in a reused `String` and
copy it with `VectorWriter::set`. Both variants here use the buffer-and-copy pattern so that they stay comparable,
but only the C++ API can drop the copy without leaving its safe surface.

- Wanted: something like `writer.set_with(row, len, |bytes: &mut [u8]| ...)` for byte types, and a UTF-8-checked
  or ASCII-preserving variant for VARCHAR.

## 3. Smaller awkwardness

- `MicroScalar::create` only covers unary functions. The two- and three-argument functions go through
  `MicroScalar::with_signature`, which needs an explicit `SignatureBuilder` and a `LogicalType` for the return
  type, i.e. a connection at registration (`.return_type(String::logical_type(conn)?)`); `MicroScalar::create`
  infers it from the Rust type.
- Bind data needs the full trait route: `vs_has_prefix_any` implements `ScalarCallbacks` and
  `ScalarBindCallbacks` and casts and sizes the output itself (about 40 lines), because `MicroScalar` has no bind
  hook. The C++ variant adds one bind callback to the same exec pattern.
- On the positive side, `VectorWriter::<String>::set_parts` writes `prefix || s || suffix` into one inline slot
  or arena allocation; the C++ API has no multi-part helper, so its `vs_wrap` hand-rolls the same thing with
  `Arena::Allocate`.
