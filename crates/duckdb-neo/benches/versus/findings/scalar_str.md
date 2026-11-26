# Findings: `scalar_str`

Evidence comes from flat profiles (`perf scalar_str/<case> --rows 100000`, fixture and setup included) taken on a
busy shared machine; the smoke-test timings only pointed at where to look and are not conclusions.

## 1. Every `String` / `&str` read validates UTF-8

`ReadableNativeElement for String` (`src/types/string.rs`) runs `std::str::from_utf8` on every row it reads, so
the cost grows with the string length. The C++ API reads `varchar_t` bytes without validating them, and DuckDB
already guarantees that VARCHAR holds valid UTF-8.

- `len/long/100000` (200 to 2,000 bytes per row), Rust: `72.30% core::str::converts::from_utf8`, then `6.47%`
  for the function's own closure. The C++ profile of the same case has no per-byte work (`LenExec` at 2.16%;
  the scan dominates).
- `wrap/column_affixes/100000` (three VARCHAR reads per row), Rust: `18.43% from_utf8` beside `30.47%` for the
  closure and `4.92% <String as ReadableNativeElement>::get`.
- The smoke runs showed the same direction wherever VARCHAR is read: long reads most, then the multi-argument
  functions (`vs_wrap`, `vs_count_char`) and the short-string read-only cases (`vs_len`, `vs_has_prefix_any`,
  `vs_first_word`).

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

## 4. Notes on the C++ side

- `Value::Get<varchar_t>().view()` on a temporary dangles for strings of at most 12 bytes, which live inside the
  `varchar_t` itself: the first `vs_has_prefix_any` bind split freed memory for short prefix lists and the `check`
  caught it. duckdb-neo's `resolved_value::<String>()` returns an owned `String`, so it has no such trap.
- `ScalarExecutor`'s documentation says string functions drive the vectors directly, but `varchar_t` works as an
  argument and result type when the callable copies results into `result.GetHeap()`; every function here uses
  the executor that way.
- The first C++ `vs_upper_ascii` used `if (byte >= 'a' && byte <= 'z') byte -= 32;`, which profiled as a
  mispredicting scalar loop (47% self time in the lambda on hex input) while Rust's `make_ascii_uppercase` is
  branch-free. The C++ loop now uses the same branch-free form, which clang vectorizes.
