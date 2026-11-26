# Findings: `scalar_num`

Numeric scalar functions written once with duckdb-neo's safe API and once with the C++ API (`duckdb::cxx`). Round 1
covers correctness and a smoke run only; the timings below come from a busy machine and are not evidence of a gap.

## API gaps and awkward spots

### No typed multi-argument executor with NULL propagation (duckdb-neo)

- C++: `ScalarExecutor::Execute<double, double, double, double>(input, fn)` reads every argument, propagates
  NULLs, and has an all-valid fast path: when no argument has a validity mask, it calls `fn` per row with no
  per-row validity test and never touches the result mask.
- duckdb-neo: `MicroScalar::create` covers one argument only. Two or more arguments need
  `MicroScalar::with_signature`, a `CombinedVectorIterator` over one reader per argument, a `match` on the tuple of
  `Option`s and `VectorWriter::set_option` per row (`vs_clamp`, `vs_checked_div`, `vs_mix`). Every row pays an
  `Option` test per argument and `set_option` branches per row, also when no argument holds a NULL. Whether the
  optimizer removes this is for the profiling round; the API offers no all-valid path to opt into.
- `CombinedVectorIterator` takes its fast path only when every reader is FLAT. A CONSTANT argument (`vs_clamp(x,
  0.25, 0.75)`) moves every reader onto the per-row selection path; C++ maps rows through `SelAt` in every case.
- What would help: a `MicroScalar` constructor for N typed arguments with NULL propagation, e.g.
  `MicroScalar::create3::<f64, f64, f64, f64>("vs_clamp", |v, lo, hi| ...)`, with an all-valid fast path.

### Bind data forces the full callback trait (duckdb-neo)

- C++: `vs_scale` adds `SetBindCallback(ScaleBind)` to the same `ScalarFunction` and keeps `ScalarExecutor`
  (its vector-level form reads only argument 0).
- duckdb-neo: `MicroScalar` has no bind. `vs_scale` is a struct implementing `ScalarCallbacks` and
  `ScalarBindCallbacks`, registered with `ScalarFunctionBuilder::new(...).with_bind()`. Its `exec` takes six
  parameters, casts the output with `output.cast::<i64>()`, sizes it with `set_size(vectors.row_count()?)` (which
  `MicroScalar` otherwise does), and unwraps `Option<&BindData>` although `with_bind` guarantees it. Not a cost,
  but about twice the code of the C++ variant.

### Neither API reads SEQUENCE vectors (both)

`SELECT vs_add_one(i) FROM range(10) r(i)` fails on both sides: `range()` hands the function a SEQUENCE
(`OTHER`) vector, and both typed readers reject it.

- duckdb-neo: `Invalid Input Error: OTHER vectors must be flattened before reading`. `MicroScalar::create` passes
  the vector as is; the closure would have to call `Vector::flatten` itself.
- C++: `duckdb_v2_vector_get_view: vector is FSST / SEQUENCE / SHREDDED — call duckdb_v2_vector_flatten first`.
  `ScalarExecutor` does not flatten either.

A plain query over `range()` is common, so the helpers on both sides (or the C API, before calling a scalar
function) should flatten. The cases avoid this by building tables first.

### Minor (duckdb-neo)

- `VectorWriter::set_option` takes `Option<&T>`, so a computed value needs `.as_ref()` on a temporary
  (`writer.set_option(row, result.as_ref())`). The C++ executor takes the value, or a `std::optional<T>`.
- Errors: `Error::invalid_input` fails only the query (`vs_strict_sqrt`), but `Error::internal` invalidates the
  whole instance. The name does not suggest that, and a function author can easily pick it for "cannot happen"
  branches such as missing bind data.

## Equivalences worth knowing

- `vs_clamp` spells out `std::min(std::max(v, lo), hi)` in Rust (`if v < lo { lo } else { v }` ...), because
  `f64::max`/`f64::min` ignore NaN while `std::max`/`std::min` do not. The two sides agree for NaN too.
- NULL handling: `FunctionProperty::HasSpecialNullHandling(true)` matches `SetNullHandling(SPECIAL)`.
- Bind-time constants: `BindArgument::resolved_value::<i64>()` (`None` when not constant, `Some(None)` for NULL)
  matches `TryGetConstantArgument` plus `Value::IsNull`/`Get<int64_t>`.
- A literal-only call (`SELECT vs_add_one(41)`) is folded while planning by running the function once, so the
  `literal_unprepared` case measures bind plus one execution per query on both sides.
- The error path of `vs_strict_sqrt` is checked with `TRY`, which turns each failing row into NULL. That proves
  negative input does not produce a value, but `TRY` cannot tell an error from a function that returns NULL.
- `vs_scale` with a column `factor` fails at bind on both sides (`vs_scale: factor must be a constant`); this was
  checked by hand, as a bind error is not catchable with `TRY` in a case's `check`.
