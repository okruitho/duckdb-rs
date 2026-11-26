# scalar_nested findings

Functions: `vs_list_sum`, `vs_list_scale`, `vs_range_list`, `vs_split`, `vs_list_join`, `vs_point_norm`,
`vs_make_point`, `vs_dot3`, `vs_map_sum`, `vs_list_of_points` (see the header of `rust/scalar_nested.rs`).
Round 1 is correctness only. Timings from the busy machine are not conclusions; the points below are leads to
profile.

## duckdb-neo

1. **Sizing a LIST child zero-fills elements that are overwritten right away.** The only safe way to make room for
   output elements is `Vector::<List<T>>::set_child_size`, which calls `set_size` on the child and then `init_rows`
   (`src/vector/init.rs`): a `write_bytes(0)` over every new element, recursively for struct fields. Every LIST-output
   function here (`vs_list_scale`, `vs_range_list`, `vs_split`, `vs_list_of_points`) then writes every element it
   sized. The C++ peer calls `Vector::SetSize` on the child, which does not initialize. Skipping the fill needs
   `set_child_size_uninit`, which is `unsafe`. Wanted: a safe way to size the child and fill it, e.g. a `ListWriter`
   that grows the child as elements are pushed, or a closure-based fill that proves every element is written.
   In the smoke run the list-building cases (`range_list/*`, `nested/scale_range_sum`, `list_scale/*`) were the ones
   where Rust trailed most; profile these first.

2. **`MicroScalar` zero-fills the output vector too.** `TypedCallback::exec` (`src/scalar/micro.rs`) calls
   `output.set_size(row_count)`, which runs the same `init_rows` over every output row (for LIST output, the
   `list_entry` array) before the closure overwrites it. This affects every `MicroScalar` function in every family,
   not just this one. The C++ API leaves the result as the engine hands it over.

3. **No way to write a LIST in one pass.** The child has to be sized before `list_writer()` is taken, and the writer
   holds the vector's exclusive borrow, so the child cannot grow while rows are written. `vs_split` therefore counts
   separators in a first pass and splits in a second. The C++ API could grow the child with `SetSize` between rows
   (re-fetching the data pointer). To keep the algorithms the same, the C++ version also makes two passes.

4. **Computed list elements need the columnar API.** `ListWriter::set_elements` takes an iterator of
   `Option<&Value>`, so it fits values that already exist somewhere (a `List<T>`, input rows), not values computed on
   the fly (`x * k`, `0..n`, `Point { .. }`). The functions instead write `writer.elements().set(start + i, &v)` and
   then `writer.set_entry(row, ListIndex::new(start as u64, len as u64))`. That works, but needs `u64` casts and a
   bounds check per element plus one per row. Something like `set_elements_with(row, len, |i| value)` or an
   iterator of owned values would be the natural form.

5. **`Entries::values()` reads every key.** `MapReader`'s `Entries::iter`/`values()` decode the key of each entry as
   well, with an `expect("MAP keys must not be NULL")` validity check, even when only the values are used.
   `vs_map_sum` calls `entries.values()` because that is the obvious API. The C++ version reads only the values
   child. The columnar route (`maps.values()` plus `entry()` ranges) avoids the key reads but is less discoverable. A
   values-only iterator on `Entries` would fix it.

6. **`String` validates UTF-8 on every read.** `ReadableNativeElement for String` runs `std::str::from_utf8` (and
   panics on failure) for each row. The C++ `varchar_t::view()` does no validation, so the Rust side reads `VARCHAR`
   as `UncheckedString` to do the same work, as `benches/support` does. A first-time user would pick `String` and
   pay for the check, and `UncheckedString` sounds unsafe even though it is a safe type that relies on DuckDB's
   guarantee. Writes of `&str` do not validate, which matches the C++ side's `AssignStringUnsafe`.

7. **`CombinedVectorIterator` only combines `VectorReader`s.** It does not accept `RowReader`s (`ListReader`,
   `ArrayReader`, derived struct readers), so multi-argument functions with a nested argument (`vs_list_scale`,
   `vs_list_join`, `vs_dot3`) zip two row iterators by hand and lose its all-flat fast path. Only `vs_make_point`
   (two `DOUBLE`s) uses it.

8. **Struct derive works well.** `#[derive(AsDuckDB)] #[duckdb(reader, writer)] struct Point { x: f64, y: f64 }`
   gives the signature type, a row reader (`PointRef { x: Option<&f64>, y: Option<&f64> }`) and a row writer that
   composes into `List<Point>`. No gap here. Each field comes back as `Option<&f64>`, so reading needs `*point.x?`.

## C++ API (`duckdb::cxx`)

1. **`ScalarExecutor` stops at primitives and `std::tuple`.** LIST, ARRAY, MAP and VARCHAR functions drive
   `VectorView`s by hand: list entries, the child's selection and validity, and `idx * 3` for ARRAY. For NULL
   output the family needs its own `NullWriter` (validity fetched lazily and reset with `SetAllValid`), because
   `detail::VectorWriter` is internal and `Vector::SetNull` is size-checked against a result whose size the engine
   only sets after the callback. STRUCT read and write through `std::tuple<double, double>` is concise.
2. **Nothing documents that a LIST child's validity starts all-valid.** The C++ list writers only touch the child's
   mask when they write a NULL element. They rely on the engine resetting the result's child mask between chunks
   (it does; the checks cover multi-chunk inputs).
3. **`AssignStringUnsafe` looks up the heap on every call** (as its docs say). `vs_split` writes one string per
   element this way; the alternative is `GetHeap` plus `SetString`. In the smoke run `split/*` was the one group
   where C++ trailed Rust clearly, so this may be the C++ side's cost. Profile it before reading that comparison as
   a duckdb-neo win.
