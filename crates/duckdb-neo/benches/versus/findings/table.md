# Findings: `table`

## Functions

| Function | Notes |
| --- | --- |
| `vs_series` | single-threaded scan |
| `vs_grid` | six columns, projection pushdown |
| `vs_words` | one VARCHAR column; `len` picks inline (≤ 12 bytes) or heap strings |
| `vs_repeat_row` | constant vectors, projection pushdown |
| `vs_series_step` | named-only parameter `step` |
| `vs_series_filter` | filter pushdown |
| `vs_series_par` | parallel scan |

- Both APIs support all of these, so nothing was skipped.
- The family only writes VARCHAR, it never reads VARCHAR vectors, so `UncheckedString` is not needed. The one string input,
  `vs_repeat_row`'s `s`, is a bind-time constant read through `Value::get::<String>()`. The C++ side keeps the `Value` itself.
- Ratios are the Rust median divided by the C++ median (above 1 means duckdb-neo is slower).
- "Before" is the round 1 central baseline. "After" is `run` (3 repetitions) on a dedicated CPU, with other agents
  building on other cores, so differences under about 5% are noise.

## What changed in this round

| Case (rows 2048 / 100k / 2M) | Before | After | Change |
| --- | --- | --- | --- |
| `words/huge_1000` (2048 / 100k) | 5.37 / 6.47 | 1.04 / 1.10 | both: the `len` dots of padding are built once at bind; each word copies a prefix of them |
| `words/long_40` | 1.29 / 1.39 / 1.44 | 1.22 / 1.14 / 1.16 | same |
| `words/where_like` | 1.10 / 1.18 / 1.19 | 1.07 / 1.14 / 1.19 | same |
| `words/inline_8` | 0.57 / 0.48 / 0.57 | 0.64 / 0.59 / 0.66 | same, plus C++ formats digits into a `char` array instead of appending to `std::string` |
| `words/truncated_3` | 0.75 / 0.64 / 0.72 | 0.76 / 0.64 / 0.66 | same |
| `grid/all_columns` | 0.68 / 0.64 / 0.66 | 0.79 / 0.77 / 0.81 | C++ builds `tag` in a `char[24]` and hands a `string_view` to `AddStringUnsafe` |
| `grid/subset_reordered` | 0.64 / 0.59 / 0.63 | 0.75 / 0.73 / 0.79 | same |
| `grid/where` | 0.67 / 0.62 / 0.69 | 0.78 / 0.75 / 0.81 | same |
| `grid/group_by` | 0.85 / 0.75 / 0.77 | 0.91 / 0.84 / 0.89 | same |

Unchanged code, re-measured (100k / 2M):

| Case | After |
| --- | --- |
| `series/count` | 1.28 / 1.49 |
| `grid/count` | 1.29 / 1.55 |
| `repeat_row/count` | 1.27 / 1.83 |
| `repeat_row/all_columns` | 0.97 / 1.04 |

### What the `huge_1000` / `long_40` gap was

- The round 1 gap was neither API: Rust padded with `word.extend(std::iter::repeat_n('.', n))`, which pushes one `char` at a time.
- C++ used `std::string::assign(n, '.')`, which is one `memset`.
- The Rust profile showed `<String as Extend<char>>::extend::<RepeatN<char>>` at 4% at `len = 40`; at `len = 1000` it dominated.
- Copying a prefix of a prepared padding string (`push_str` / `std::string::assign(padding, 0, n)`) is the same work on both sides.

Kept the same on both sides on purpose:

- The data is written per row. `vs_series` could be a `make_sequence` vector (`Vector::make_sequence`, C++ `Vector::MakeSequence`, both safe and public), but that would no longer measure writing values.
- Every exec call writes into the output chunk. Nothing is cached across callbacks.

## Remaining gaps attributable to duckdb-neo, most important first

### 1. Every vector access re-wraps the vector

**Cases:**

| Case | 100k / 2M | 20M |
| --- | --- | --- |
| `repeat_row/count` | 1.27 / 1.83 | 1.81 (baseline) |
| `series/count` | 1.28 / 1.49 | — |
| `grid/count` | 1.29 / 1.55 | — |

- These are the cases where per-row work is cheapest. The residual after excluding gap 2 (about 1.28–1.32 at 2M, see below) is mostly this.
- There are about 977 chunks for 2M rows. Each Rust exec call costs about 100–180 ns more than the C++ one.

**Profile** `table/repeat_row/count/20000000`, self time:

| Rust | C++ |
| --- | --- |
| `VectorCollection::get_vector_at_mut::<String>` 10.5% | `RepeatRowExec` 2.5% |
| `Vector<String>::make_constant` 5.8% | the `SetExecCallback` trampoline 2.7% |
| `table_function::exec_callback::<RepeatRow>` 5.5% | |
| `value::create` 4.1% | |
| `DataChunk::wrap` 3.5% | |
| `RepeatRow::exec` 1.7% | |
| `duckdb_v2_value_create_varchar` 1.0% | |
| `duckdb_v2_vector_get_logical_type` 0.5% | |
| `duckdb_v2_logical_type_destroy` 0.35% | |

- In C++ the wrapper cost stays below the engine's own `VectorCacheEntry::ResetFromCache` and `ConstantVector::Reference`.
- Rust ran 682 iterations in 3 s, C++ 1246.

**Responsible code:**
- `crates/duckdb-neo/src/vector/mod.rs:188` `Vector::from_handle` runs on every `DataChunk::get_vector_at_mut` / `get_untyped_vector_at_mut` (`src/data_chunk.rs:247`, `:221`, via `VectorCollection::wrap` at `:132`). Each run makes five or more C API calls and creates and destroys a `LogicalType`:
  - `vector_get_logical_type`, which allocates a type copy that is dropped again;
  - `vector_get_vector_type`;
  - `vector_get_size`;
  - `vector_get_child_count`;
  - `get_child` for each child.
- `crates/duckdb-neo/src/vector/mod.rs:645` `make_constant` calls `from_handle` again after `duckdb_v2_vector_make_constant`, plus `constant_set_valid`.
- `crates/duckdb-neo/src/data_chunk.rs:466` `DataChunk::wrap` fetches `get_vector_count` and every vector handle before `exec` runs, even for columns the code never touches.

C++ does one `GetVector` (one call) per used column and passes the `const Value &` to `MakeConstant` (one call).

**Suggestions:**
- Validate `T` against a type id instead of a full `LogicalType` copy. `duckdb_v2_vector_get_logical_type` is the only allocation, and an id comparison is enough for non-nested `T`.
- Fetch size, kind and children lazily, or cache the wrapped output vectors per output chunk handle. The chunk lent to a table function's `exec` is the same object on every call, so the projected types could be validated once.
- `make_constant` already knows the new kind (constant) and length, so it can update `self` rather than re-wrapping.
- Fetch vector handles on first access in `DataChunk::borrowed`.

### 2. Safe `set_size` zero-fills every output column

**Cases:** `series/count` 1.49, `grid/count` 1.55, `series/sum` 1.13, `series/threads4` 1.12 (2M, baseline for the last two).

**Evidence:**
- Profile `table/series/count/2000000`:
  - Rust: `__memset_avx512_unaligned_erms` 11.8%, next to `fill_series` 42.8%.
  - C++: no memset; `FillSeries` 41.9%.
  - The fill loops compile to the same 2×128-bit `paddq`/`movdqu` loop on both sides.
- Replacing `set_size` by `unsafe set_size_uninit` in a local, uncommitted experiment changed the ratios:

  | Case | 100k | 2M |
  | --- | --- | --- |
  | `series/count` | 1.28 → 1.18 | 1.49 → 1.28 |
  | `grid/count` | 1.29 → 1.16 | 1.55 → 1.32 |

  So the zero-fill is about half of the gap. The other half is gap 1.

**Responsible code:**
- `crates/duckdb-neo/src/vector/mod.rs:602` `set_size` calls `init_rows` (`src/vector/init.rs:55`), which zeroes `rows × size_of` bytes. The writer immediately overwrites them.
- `VectorWriter` takes its length from the vector, so every projected column needs its own `set_size` before `writer()`.
- C++ writes into `GetDataMutable<T>()`, which has room for the chunk's capacity, and calls `SetSize` once on vector 0. The engine propagates that size to the other vectors.

**Suggestion:** a safe whole-column fill that makes the zeroing unnecessary. Possible signatures:
- `Vector::<T>::fill_with(&mut self, rows: usize, f: impl FnMut(usize) -> T) -> Result<()>` for `PlainNative` `T`;
- `Vector::<T>::write_iter(&mut self, values: impl ExactSizeIterator<Item = T>)`.

Either sizes the vector uninitialized internally and writes every row before returning. A `VectorWriter` built with a length larger than `len()`, which sizes the vector when dropped, would also do.

### 3. Heap strings: store-forwarding stall while writing the pointer form

**Cases:**

| Case | 2048 / 100k / 2M |
| --- | --- |
| `words/long_40` | 1.22 / 1.14 / 1.16 |
| `words/where_like` (16 bytes) | 1.07 / 1.14 / 1.19 |

`words/huge_1000` is about 1.04–1.10, because copying the 1000 bytes dominates there.

**Evidence** (profile `table/words/long_40/100000`):

| Rust | C++ |
| --- | --- |
| `VectorWriter<String>::write_bytes` 15.1% | nothing comparable in `WordsExec` (0.9%) |
| `__memmove` 32.9% | `__memmove` 37.5% |
| `StringHeap::CreateEmptyStringInHeap` 18.5% | `StringHeap::CreateEmptyStringInHeap` 23.0% |

- 99.75% of `write_bytes`' samples fall on `movq %rax, 0x4(%r14)`.
- Just before it, a PLT `memcpy` call writes the prefix into a stack temporary at `0x2c(%rsp)`, the pointer is stored next to it, and an 8-byte load at `0x2c(%rsp)` spans both stores. That load cannot be forwarded.

**Responsible code:**
- `crates/duckdb-neo/src/bytes.rs:131-146` (`DuckDBBytes::write_parts`, pointer branch) builds a `DuckDBV2BytesPointer` local, copies the prefix with `copy_parts(parts, prefix, 4)`, then copies the struct into the slot.
  - That `copy_parts` has a variable length (an out-of-line `memcpy`), although the prefix of a non-inlined string is always exactly 4 bytes.
- The C++ `blob_t(const char *, uint32_t)` constructor copies the prefix with a constant-size `memcpy(prefix, data, 4)`, which compiles to a single 4-byte move.

**Suggestion:**
- For the single-part case (`[part]`, as for the inline branch's `copy_short`), write length, a fixed `[u8; 4]` prefix and pointer straight into `slot`, with no stack temporary.
- For several parts, gather the first 4 bytes into a `[u8; 4]` before the store.

### 4. `make_constant` consumes its `Value`

- `Vector::make_constant(self, value: Value, count)` takes ownership, and `Value` has no `Clone`. `duckdb_v2_vector_make_constant` only borrows the handle, because the engine copies the value.
- So `vs_repeat_row` rebuilds a `Value` from `Option<String>` / `Option<i64>` for every column of every chunk. That shows up as `value::create` 4.1% and `duckdb_v2_value_create_varchar` 1.0% in the `repeat_row/count` profile above.
- C++ keeps the bind-time `Value`s in its bind data.
- **Suggestion:** `make_constant(self, value: &Value, count: usize)`, or `impl Clone for Value` (via `duckdb_v2_value_copy` or an equivalent).

## API gaps and awkwardness

**duckdb-neo:**
- **No safe uninitialized sizing.** `set_size_uninit` is `unsafe`, which is why gap 2 exists. `benches/support/dash_split.rs` uses it, but the versus rules forbid it.
- **Filter pushdown needs interior mutability.** `TableFilterPushdownCallbacks::pushdown_filter` gets `&Self::BindData`, so recording accepted bounds needs atomics (`fetch_max` / `fetch_min`). C++ gets `T &` from `FilterPushdownInput::GetBindData`. The cost is negligible, but only Rust has the ceremony.
- **The per-exec `DataChunk` wrapper.** It is built for every exec call and dropped again (`drop_glue::<VectorCollection>` and `drop_glue::<DataChunk>` together about 0.8% in `series/count`). It is part of gap 1.

**Both APIs:**
- **`BETWEEN` bounds are not inspectable.** DuckDB turns `i >= a AND i < b` into `COMPARE_BETWEEN` before pushdown, and neither `Expression` API says whether the bounds are inclusive. So `vs_series_filter` claims only single comparisons, and the cases use those.
- **Engine bug: common subplan elimination merges calls whose accepted filters differ.**
  - Reproducer (same result on both sides):

    ```sql
    SELECT (SELECT count(*) FROM vs_series_filter(100) WHERE i = 99)
         + (SELECT count(*) FROM vs_series_filter(100) WHERE i >= 50)
    ```

    It returns `2` instead of `51`.
  - `CommonSubplanOptimizer` compares `LogicalGet`s by their serialization. For a C API table function that is the call parameters only, so the filters a pushdown callback accepted are invisible to it.
  - The cases avoid this by using distinct arguments. It should be reported to DuckDB.

## Where duckdb-neo is faster than the C++ API

### Inline strings (≤ 12 bytes)

**Cases:**

| Case | 2048 / 100k / 2M |
| --- | --- |
| `words/inline_8` | 0.64 / 0.59 / 0.66 |
| `words/truncated_3` | 0.76 / 0.64 / 0.66 |
| `grid/all_columns` | 0.79 / 0.77 / 0.81 |
| `grid/subset_reordered` | 0.75 / 0.73 / 0.79 |
| `grid/where` | 0.78 / 0.75 / 0.81 |

The `grid` cases include the `tag` column, 5 to 11 bytes.

**Evidence** (profile `table/words/inline_8/100000`):

| Rust | C++ |
| --- | --- |
| `write_bytes` 81%, mostly a stall on the user's freshly written `String` | `__memmove` 46.9% |
| no libc call | `WordsExec` 35.9%, with 78% of it on `movl (%rsp), %eax` right after the `memcpy` call |

Iterations in 3 s: Rust 1944, C++ 1116.

**Why:**
- In C++ the only public ways to produce an inline `varchar_t` are `Arena::AddString*`, `AddBlob` or the `blob_t` constructors.
  - These `memset` a stack temporary, copy the bytes with a variable-length libc `memcpy`, and return by value.
  - The 4-byte length reload then hits a store-forwarding stall.
  - `Vector::AssignString` is worse: it looks up the type, heap and vector kind per call (about 5 C API calls per string), which is why the C++ variant uses `GetHeap()` / `GetDataMutable<varchar_t>()` once per chunk.
- duckdb-neo writes inline strings directly into the output slot (`crates/duckdb-neo/src/bytes.rs:115-127`, `copy_short`).

### Wide projections

With six output columns (`grid/*`), Rust's per-row writer loops beat C++ even though Rust pays gaps 1 and 2 per column. The strings dominate.

## Differences that are not about either API

- **Building small strings.**
  - Rust pushes ASCII `char`s one by one into a reused `String`, which stays inline.
  - `std::string::append` and `assign` call out-of-line libc `memmove` for every piece. That is why the C++ variant formats digits into a `char` array (`FormatDecimal`, `FormatTag`).
  - Both sides still read back bytes they have just written byte-wise, so both show store-forwarding stalls at the first copy.
- **Padding.** `extend(repeat_n(c, n))` is a per-`char` loop in Rust, while `std::string::assign(n, c)` is a `memset`. That was the whole round 1 `huge_1000` gap of 5–6×. It is fixed now by copying a prefix of a prepared padding string on both sides.
- **Parallel scans.** `series_par/*` with `threads = 4` varies by ±10% between runs, because the instance's other threads share the machine. Treat its ratios as noise unless re-measured.
