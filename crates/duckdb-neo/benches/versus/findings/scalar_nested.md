# scalar_nested findings

Functions: `vs_list_sum`, `vs_list_scale`, `vs_range_list`, `vs_split`, `vs_list_join`, `vs_point_norm`,
`vs_make_point`, `vs_dot3`, `vs_map_sum`, `vs_list_of_points` (semantics in the header of `rust/scalar_nested.rs`).
Ratios are Rust median / C++ median. The before numbers come from the round 1 baseline. The after numbers come from
`run --rows 2048,100000` (3 repetitions, CPU 4) while other agents were busy on other cores, so changes under about
5% are noise. VARCHAR is read as `duckdb_neo::types::UncheckedString` on the Rust side, because C++ reads the bytes
without validating them either.

## What changed

| Case (rows) | before | after | change |
| --- | ---: | ---: | --- |
| `point_norm/project` (100k) | 1.45 | 1.16 | Rust: untyped `Struct` plus `f64` field readers in one `CombinedVectorIterator`, instead of the derived reader |
| `point_norm/nulls_50` (100k) | 1.43 | 1.16 | same |
| `point_norm/filtered` (100k) | 1.24 | 1.06 | same |
| `nested/norm_of_make_point` (100k) | 1.84 | 1.44 | same, plus the `make_point` change below |
| `make_point/project` (100k) | 1.71 | 1.54 | Rust: write the derived writer's field writers (`writer.x.set`) instead of `writer.set(row, &Point)` |
| `list_of_points/project` (100k) | 2.18 | 1.56 | Rust: write the element fields (`points.x.set`) instead of `PointWriter::set` |
| `list_of_points/long` (2048) | 1.90 | 1.36 | same |
| `list_sum/no_nulls` (100k) | 1.33 | 1.07 | Rust: `map(\|v\| v.copied().unwrap_or(0)).fold(0, i64::wrapping_add)` instead of `flatten().fold` |
| `map_sum/project` (100k) | 1.26 | 1.12 | Rust: read the map's `values()` column by the entry's range instead of `Entries::values()` |
| `map_sum/long` (2048) | 1.16 | 1.08 | same |
| `split/project` (100k) | 0.80 | 0.94 | C++: one `GetHeap` and data pointer per call plus `Arena::AddStringUnsafe`, instead of `AssignStringUnsafe` per piece. Rust: byte search instead of `str::split(char)` |
| `split/long_strings` (2048) | 0.65 | 0.68 | both sides got faster: Rust 30.5 to 25.0 ms, C++ 46.8 to 36.8 ms |
| `nested/split_join` (100k) | 0.73 | 0.88 | C++ string writes as above, in `vs_split` and `vs_list_join` |
| `list_join/project` (100k) | 0.81 | 0.86 | C++ string writes as above |

I tried these and dropped them as no gain:
- **Hoisting `writer.elements()` out of the inner loop** (`range_list`, `list_scale`): no change.
- **Explicit `values.get(index)?` loops for `list_sum`**: worse when there are NULLs (1.14 to 1.36).
- **A per-row scratch `Vec` plus `VectorWriter::write_slice`** (`range_list`, `list_of_points`): long lists improved (`range_list/long` 1.37 to 1.15), but short lists got worse (`range_list/project` 1.38 to 1.57). The per-call checks of `write_slice` cost more than they save on lists of about 8 elements.

Unchanged and still open: `nested/scale_range_sum` (1.72 at 100k), `range_list/*` (1.3 to 1.4), `list_scale/*` (1.2 to 1.26),
`dot3/*` (1.2 to 1.3) and `many_calls` (1.39), which combines the others.

## Remaining gaps attributable to duckdb-neo (most important first)

### 1. List writers pay per element and per row for checks that C++ does not make

Cases: `nested/scale_range_sum` 1.72, `range_list/project` 1.42, `range_list/long` 1.29, `list_scale/*` 1.21 to 1.26,
`list_of_points/project` 1.56.

Profiles at 100k rows:
- `range_list/project` (Rust): `ListWriter<..>::set_entry` 10.8% self. It is out of line and shared by all element types (`ListWriter<String>::set_entry` in the symbol table). The closure is another 22.9%. In C++, `RangeListExec` is 19.1% and stores `list_entry_t` directly.
- `nested/scale_range_sum` (Rust): `set_entry` 10.4%.
- `set_entry` annotated: `entry.offset + entry.length` is overflow-checked, compared with the child's length through a pointer (`cmpq 0x18(%r9)`), the running offset is updated, the validity bit is tested, and `Result<(), Error>` is written through memory.
- Element writes (`VectorWriter::set`/`set_option`, through `ListWriter::elements()`) do a bounds check and a validity-bit test per element. They also reload the data pointer and length from the stack for every store, because the writer lives in memory behind `&mut`. The C++ variants write through a raw `int64_t *`, and clang vectorizes the `vs_range_list` fill.

Sources:
- `crates/duckdb-neo/src/vector/list_writer.rs` (`set_entry`, `elements`)
- `crates/duckdb-neo/src/vector/writer.rs` (`set`, `mark_valid_unchecked`)

Suggestions:
- A row-at-a-time append API that checks capacity once per row, e.g. `ListWriter::push_row(&mut self, row: usize, len: usize) -> Result<&mut [T]>` for plain native `T`, or `fill_row(row, len, impl FnMut(usize) -> Option<T>)`. That is one check per row and a slice the compiler can vectorize.
- `#[inline]` on `set_entry`.
- More generally, a safe `VectorWriter::slice_mut(range) -> Result<&mut [T]>` for `PlainNative` types. Every bit pattern is valid, validity stays as it is, so it needs no `unsafe`; today only `as_slice_mut` exists and it is `unsafe`.

### 2. Nested element iteration is not specialized per storage kind

Cases: `list_sum/project` 1.14, `list_sum/all_null` 1.14, `dot3/project` 1.31, `dot3/filtered` 1.22. This also adds to `list_scale`.

Profiles:
- `list_sum/no_nulls` (100k) before the workaround: `Map<Range<usize>, Elements<VectorReader<i64>>::iter::{closure}>::fold` 32.5% self. The hot loop re-tests the reader's storage tag for every element (`cmpb $0x2,%r10b` 16.4%, `cmpl $0x1,%r8d` 15.9%). In C++, `ListSumExec` (28.1%) runs a vectorized `paddq` loop.
- `dot3/project` (100k): the per-element pattern is the same (`cmpb $0x2,0xc(%rsp)` 5.3%, `cmpl $0x1` 6.6%). Rust closure 56.0% against C++ `Dot3Exec` 50.2%, on a longer total.

Source: `crates/duckdb-neo/src/vector/nested_reader.rs:123`. `Elements::iter` maps each index through `RowReader::get_unchecked`. For flat vectors, `VectorReaderIter::fold` (`reader.rs:295`, `fold_rows`) resolves the storage kind once, outside the loop, but `Elements` does not use it.

The branchless `value.copied().unwrap_or(0)` form recovered `list_sum/no_nulls` (1.33 to 1.07), but that only works for sums.

Suggestion: give `Elements::iter` a named iterator type whose `fold` delegates to a new `RowReader::fold_range(&self, rows: Range<usize>, init, f)`. Its default goes row by row; `VectorReader` overrides it with `fold_rows`. `ArrayReader` rows then get the same benefit.

### 3. Derived STRUCT types are validated against DuckDB on every callback, and validation allocates

Cases (before the workaround): `point_norm/*` 1.24 to 1.45. Still present in `make_point/project` 1.54 and `list_of_points/project` 1.56, where the output type is derived.

Profiles:
- `point_norm/project` (100k, derived reader), Rust: `__strcmp_evex` 3.82%, `StorageCompatibility::FromString` 0.48%, malloc / _int_free / _int_malloc 6.74% / 8.65% / 3.85%.
- Same case, C++: 5.50% / 5.82% / 2.73% and no strcmp.
- `make_point/project` (Rust, derived output writer): `__strcmp_evex` 3.00%.

gdb backtrace for each call:

`StorageCompatibility::FromString` ← `Value::TYPE` ← `duckdb_v2_logical_type_get_param` ← `LogicalType::get_param` (`src/logical_type.rs:161`) ← `derive_support::validate_field` (`src/types/structs.rs:112`) ← derived `validate` ← `Vector::from_handle` (`src/vector/mod.rs:188`, `validate_as` at 507) ← `VectorCollection::get_vector_at_mut` ← `TypedCallback::exec`.

`get_param` makes DuckDB build a `Value` of every field's type just to read the field name, once per field per chunk per argument.

Switching the reader to the untyped `Struct` plus `get_vector_at::<f64>(i)` removed the strcmp and most of the allocation share. That took `point_norm/project` from 1.45 to 1.30; adding `CombinedVectorIterator` brought it to 1.16.

The output side has no cheap equivalent. Writing an untyped `Struct` result needs `children_mut()` and a second pass for the NULL rows, because the struct writer and the field writers cannot be borrowed together.

Suggestions:
- Read field names without materializing a `Value`, for example a `duckdb_v2_logical_type_get_param_name`-style call, or the name from `get_param` with the value left unread.
- Better still, validate once: the scalar function's signature fixes the argument and result types. `MicroScalar`/`ScalarFunctionBuilder` could validate at bind or on the first call, or cache the last validated `LogicalType` per argument slot and skip validation when the handle compares equal.

### 4. `Result<(), Error>` is 32 bytes, so every out-of-line writer call returns through memory

Profile, `list_of_points/project` (100k) with the derived `PointWriter::set_valid` per element: in the closure, `cmpq $-0x1,0x10(%rsp)` right after `callq PointWriter::set_valid` holds 22.7% of the samples, and the following `jne` 11.8%. That is a store-to-load round trip of the `Result` discriminant.

Source: `crates/duckdb-neo/src/error.rs:76` (`Error { code, message: String }`).

Removing the `set_valid` call (the sized elements start valid) is part of the 2.18 to 1.56 gain. The same pattern shows up in `set_entry` (gap 1).

Suggestions:
- Box the error payload, so that `Result<()>` fits in a register.
- Mark the derive-generated `set_valid`/`set`/`set_null` `#[inline]` (`crates/duckdb-loadable-macros/src/neo/struct_vector.rs`).

### 5. Sizing output zero-fills rows that are overwritten right away

Profile: `__memset_avx512_unaligned_erms` 2.3% in `nested/scale_range_sum`, 1.6% in `range_list/project` and 3.0% in `list_of_points/project`, all Rust. It is absent from the C++ profiles.

Sources:
- `src/vector/init.rs:55` (`init_rows`, `write_bytes(0)`), reached from `Vector::<List<T>>::set_child_size` (`src/types/list.rs:190`)
- `MicroScalar`'s `output.set_size(row_count)` (`src/scalar/micro.rs:115`)

Avoiding the fill needs `set_child_size_uninit`, which is `unsafe`. Suggestion: the append API from gap 1 can size the child as it goes without a fill.

### 6. `CombinedVectorIterator` accepts only `VectorReader`s

Derived struct readers, `ListReader`, `ArrayReader` and `MapReader` cannot join it, so they lose its all-flat fast path. Multi-argument functions over nested types (`vs_list_scale`, `vs_list_join`, `vs_dot3`) zip row iterators by hand. `vs_point_norm` only gets the fast path by dropping the derive (gap 3).

Suggestion: implement `VectorReaderTuple` for `RowReader`s, or add a `RowReader::fold_range` (gap 2) that a combined iterator can call.

### 7. `Entries::values()` reads the key of every entry

`MapReader`'s `Entries::iter`/`values()` (`src/vector/nested_reader.rs:261`) decodes each key and asserts that it is non-NULL even when only the values are used. Reading `maps.values()` by `entries.entry()` range took `map_sum/project` from 1.26 to 1.12, at the cost of a bounds-checked `get` per value.

Suggestion: a values-only iterator on `Entries` that skips the key reader.

## API gaps and awkwardness

- **Computed list elements need the columnar route.** `ListWriter::set_elements` takes `Option<&Value>`, so values computed on the fly go through `elements().set(start + i, ..)` plus `set_entry(row, ListIndex::new(start as u64, len as u64))`. Output of unknown length (`vs_split`) needs a counting pass first, because the child cannot grow while the writer is borrowed. C++ mirrors the two passes to keep the algorithms the same.
- **Writing derived struct fields directly relies on rows starting valid.** `vs_make_point` and `vs_list_of_points` now write `writer.x`/`writer.y` and skip `set_valid`, which relies on sized rows being valid (documented for `set_size`/`set_child_size`) and on the engine resetting result validity between chunks. The C++ `ScalarExecutor` makes the same assumption. A cheap and obvious way to write a whole valid struct row would avoid the subtlety.
- **`String` validates UTF-8 on every read.** `ReadableNativeElement for String` runs `from_utf8` per row. `UncheckedString` is the safe way to read without it, but its name suggests otherwise.
- **The untyped `Struct` route is the fast read path for structs (gap 3).** It loses the field-name check and the typed row.
- **No safe `&mut [T]`, no `memchr`.** There is no safe mutable slice over plain native rows (gap 1). `ListIndex`'s fields are private, so ranges are read through `offset()`/`length()`.

## Where duckdb-neo is faster than the C++ API

- **String splitting** (`split/sep_column` 0.56, `split/long_strings` 0.68, `split/project` 0.94). After the C++ variant stopped calling `AssignStringUnsafe` per piece (it was 57.8% self in C++ `split/sep_column`, doing `GetHeap`, `CheckWriteRange` and `GetDataMutable` per call), most of C++ `SplitExec`'s time sits in the `blob_t(const char *, uint32_t)` constructor from `duckdb_cpp.hpp`. For an inlined string it `memset`s the 16-byte value, calls `memcpy` with a variable length into it, and the result is then loaded back. That costs 52.1% of `SplitExec`'s samples on the load right after the `memcpy` call, a store-forwarding stall. Rust's `VectorWriter::write_bytes` (5.4%) builds the inline string without that round trip. The fix belongs in the C++ header: build short strings in a register or a local before storing.
- **Joining strings** (`list_join/*` 0.86). C++ appends to a `std::string` (`basic_string::_M_append` 10.7%, out of line in libstdc++). Rust's `String::push_str` inlines to `memcpy`.

## Differences that are not about either API

- **Rust's `str::split(char)` calls `memcmp` for every match.** `CharSearcher::next_match` `memchr`s the last UTF-8 byte, then compares the encoded char with `memcmp`: 21.2% `__memcmp_evex_movbe` in `split/sep_column` before the change. The Rust variant now searches for the separator byte (`bytes().position`), which is sound because the separator is a single ASCII byte. C++ uses `std::string_view::find`, which is `memchr`.
- **`list_join` compares `std::string` with Rust `String`.** That is a standard-library comparison (above), not an API one.
- **Clang vectorizes the C++ list loops.** Rust's equivalents are blocked by gaps 1 and 2, not by the compiler.
