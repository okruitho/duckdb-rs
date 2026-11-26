# duckdb-neo versus the C++ API

The same user-defined functions written twice, once on duckdb-neo and once on DuckDB's C++ API over the v2
C API (`tools/cpp` in DuckDB), run through the same queries against the same `libduckdb.so`. The goal is to
find where duckdb-neo costs more than its C++ peer: per row, per vector, per callback, and per query.

This file is the specification. When it and the code disagree, fix whichever is wrong and say so.

## Layout

| Path | Purpose |
| --- | --- |
| `benches/versus.rs` | Rust runner (`[[bench]] name = "versus"`, `harness = false`). |
| `benches/versus/rust/mod.rs` | Declares one module per family and calls each `register`. |
| `benches/versus/rust/<family>.rs` | The family's functions on duckdb-neo. |
| `benches/versus/cpp/CMakeLists.txt`, `runner.cpp`, `registry.hpp` | C++ runner; families self-register. |
| `benches/versus/cpp/<family>.cpp` | The family's functions on the C++ API. |
| `benches/versus/cases/<family>.toml` | The family's cases: setup, timed query, check. Read by the driver only. |
| `scripts/neo_versus.py` | Driver: builds both runners, runs them alternately, compares, profiles. |

A family owns exactly three files: `rust/<family>.rs`, `cpp/<family>.cpp` and `cases/<family>.toml`.
Nothing else needs editing to add functions or cases to a family:

- `rust/<family>.rs` defines `pub fn register(conn: &Connection) -> Result<()>`. `rust/mod.rs` already
  declares and calls every family of the suite (unimplemented ones are empty stubs).
- `cpp/<family>.cpp` includes `registry.hpp` and defines its registration as
  `VERSUS_FAMILY(<family>) { ... }`, with `conn` (a `duckdb::cxx::Connection &`) in scope. CMake globs
  `cpp/*.cpp`, so a new file is picked up by the next `build`.
- `cases/<family>.toml` is found by its file name; the file stem is the family name.

## Rules for function code

Both variants must be what a careful user of the public API would write.

- **Rust:** only duckdb-neo's safe public API. No `unsafe` (the runner crate sets `#![deny(unsafe_code)]`),
  no `duckdb_neo::ffi`, no `duckdb_neo::raw`, no `libduckdb_sys`.
- **C++:** only `duckdb_cpp.hpp` (`duckdb::cxx`). No `duckdb_v2.h` include and no `duckdb_v2_*` call.
- Same name, signature, NULL handling, error conditions and results in both languages. Every function name
  starts with `vs_`.
- Same algorithm on both sides. Do not make one side smarter than the other.
- When the safe API cannot express something, or forces a slow pattern, do not work around it: write the
  best safe version and record the gap in the family's findings (see "Findings").

## Runner protocol

Both runners take the same arguments and print one JSON object on one line to stdout. Diagnostics go to
stderr. A non-zero exit status is a failure.

```
runner --mode <check|bench|loop> [--threads N] (--setup SQL)* --check SQL --query SQL [--reset SQL]
       [--unprepared] [--warmup-ms N] [--samples N] [--min-sample-ms N] [--max-seconds N]
```

Steps, identical in both languages:

1. Fixture: environment; instance with the option `threads` (default `1`); an in-memory database attached
   as the default; one connection; every family registered on that connection.
2. Run each `--setup` statement in order and drain it.
3. Run `--check` and read row 0, column 0, which the case must return as `VARCHAR`. `NULL` is the string
   `NULL`. With `--mode check`, print `{"check": "<value>"}` and exit. Each `--setup` and `--check` string
   holds exactly one statement.
4. Prepare `--query` once, the way `benches/functions.rs` does (`conn.prepare(sql, false)`), using the same
   v2 C API calls in both languages. With `--unprepared`, each iteration runs the SQL text through the
   connection's query call instead.
5. One iteration: run `--reset` if given (untimed), read the monotonic clock, execute, fetch every chunk
   until the result is exhausted without reading values (only each chunk's row count), destroy the result,
   read the clock.
6. Warm up: iterate until `--warmup-ms` (default 200) has passed, at least 3 iterations.
7. Calibrate: `iters = clamp(ceil(min_sample_ns / median warm-up iteration), 1, 1_000_000)` with
   `--min-sample-ms` default 2, where the median of `n` values is the `n / 2`-th smallest (from 0). With
   `--reset`, `iters = 1`.
8. Sample: up to `--samples` (default 30) samples of `iters` iterations each, stopping once
   `--max-seconds` (default 3) of sampling has passed with at least 5 samples taken. A sample's value is its
   timed nanoseconds divided by `iters`.
9. Print `{"check": "...", "iters": N, "result_rows": N, "samples_ns": [..]}`. `result_rows` is the number
   of rows the last iteration returned.

`--mode loop` skips warm-up and sampling and iterates for `--max-seconds` (at least once), for
`perf record`; it prints the same object with the number of iterations as `iters` and no samples.

Per iteration both runners make the same v2 C API calls:

| Step | Rust (duckdb-neo) | C++ (`duckdb::cxx`) | v2 C API |
| --- | --- | --- | --- |
| prepare, once | `conn.prepare(sql, false)` | `ParseSQL`, `Next` twice, `Prepare(statement, false)` | `parse_sql`, `statement_iterator_next` ×2, `prepared_statement_create` |
| execute | `statement.execute(Parameters::None)` | `PreparedStatement::Execute()` | `prepared_statement_execute` |
| unprepared | `conn.query(sql, Parameters::None)` | `Connection::Execute(sql)` | `parse_sql`, `statement_iterator_next` ×2, `statement_execute` |
| per chunk | `next_chunk()`, `row_count()`, drop | `FetchChunk()`, `GetRowCount()`, destructor | `result_fetch_chunk`, `data_chunk_get_size`, `data_chunk_destroy` |
| end | drop the result | destructor | `result_destroy` |

duckdb-neo's `DataChunk` also calls `data_chunk_get_vector_count` and `data_chunk_get_vector` per column
for every chunk; that is part of what is measured. Both results are the lazy streaming kind. The fixtures
differ in one call: the C++ API has no startup option for `threads`, so its runner sets it right after
creating the instance (`Config::SetOption`, GLOBAL) where duckdb-neo passes it to the instance builder.

## Cases

`cases/<family>.toml`:

```toml
[[case]]
name = "add_one/project"             # reported as <family>/<name>/<rows>
rows = [1, 64, 2048, 100_000, 2_000_000]
setup = ["CREATE TABLE t AS SELECT i::BIGINT AS v FROM range({rows}) r(i)"]
query = "SELECT vs_add_one(v) FROM t"
check = "SELECT count(*)::VARCHAR FROM t WHERE vs_add_one(v) IS DISTINCT FROM v + 1"
expect = "0"                         # optional; both languages must always agree with each other
# reset = "TRUNCATE target"          # optional, untimed, before every iteration
# unprepared = true                  # optional
# threads = 4                        # optional
```

Placeholders: `{rows}` and `{tmp}` (a fresh empty directory per runner process, removed afterwards). Write
every SQL string so that it also works at `rows = 1`. `setup` may also be a single string. Unknown keys are
an error.

A good `check` compares the function with an independent computation in builtin SQL and returns the number
of mismatching rows (`expect = "0"`). Where no builtin equivalent exists, return a digest, such as
`sum(hash(...))::VARCHAR`, and add a golden value in `expect` for at least one row count through
`expect_at = { 2048 = "..." }`, which overrides `expect` at that row count. The driver fails a job when the
languages disagree on `check` or `result_rows`, or `expect` differs.

Vary the usage, not only the row count: each family should cover most of

- sizes from one row through one vector (2,048) to millions, so both fixed and per-row costs show;
- plain projection, a `WHERE` before the function (selection vectors and dictionaries reach the function),
  the function inside `WHERE`, constant arguments, nested calls, many calls in one query;
- NULL rates of 0%, about 10%, about 50%, and all-NULL;
- for aggregates: ungrouped, few groups, many groups, `FILTER`, running windows, sliding frames from tiny to
  thousands of rows, partitioned windows;
- for table and copy functions: one row, projection of some columns, `LIMIT`, filters on the output.

Keep each job's default run under a few seconds; trim the largest row count where a query is slow.

## Driver

```sh
python3 scripts/neo_versus.py build                      # both runners
python3 scripts/neo_versus.py check [--filter F]         # every case at its two smallest row counts
python3 scripts/neo_versus.py check --all-rows [--filter F]   # every row count; takes no lock, safe in parallel
python3 scripts/neo_versus.py run [--filter F] [--rows 1,2048] [--repetitions 3] [--json out.json]
python3 scripts/neo_versus.py perf <family/case> --rows N [--lang rust|cpp|both]
```

`check`, `run` and `perf` build first unless given `--no-build`, and take `--lto` for the LTO runners.

- `build` compiles the Rust runner with `cargo bench -p duckdb-neo --bench versus --no-run` and the C++
  runner with CMake into `target/neo-versus/`. The C++ API sources come from `git show
  <ref>:tools/cpp/...` of a DuckDB checkout (`--duckdb-repo`, default `~/duckdb`; `--duckdb-ref`, default
  `origin/v2.0-cyanoptera`, whose `duckdb_v2.h` matches the downloaded library). Both runners load the
  `libduckdb.so` that `libduckdb-sys` downloaded; `build` fails unless its `duckdb_v2.h` is the ref's and
  `ldd` resolves both runners to it. Rust builds with `CARGO_PROFILE_BENCH_DEBUG=line-tables-only` and C++
  with `-O3 -DNDEBUG -g` (clang++), so `perf` can symbolize. `--lto` builds both with link-time optimization
  (`CARGO_PROFILE_BENCH_LTO=fat`, one codegen unit; `-flto` with lld) into separate directories: the default
  Rust runner builds in the workspace `target/`, the LTO one in `target/neo-versus/lto/cargo/`, each C++
  runner in `target/neo-versus/<default|lto>/cpp/`, and each pair loads the library its cargo build
  downloaded.
- `check` runs each case at its two smallest row counts in both languages, as a bench run with no warm-up
  and one sample, so the query's errors and `result_rows` surface too.
- `run` executes each job as `rust, cpp, cpp, rust` per repetition, pinned to one CPU (`--cpu`, default 2)
  under an exclusive lock on `/tmp/neo-versus.lock`, so concurrent agents do not measure at the same time.
  It reports the median per language, the ratio `rust / cpp`, and a status: `slower` or `faster` when the
  ratio leaves `1 ± --threshold` (default 0.10) and every Rust sample median is on the same side of every
  C++ one, `inconclusive` when they overlap, else `same`. `--rows` replaces each case's row counts,
  `--samples` and `--max-seconds` go to the runners, `--quick` means 10 samples and one repetition, and
  `--markdown` appends the table to a file.
- `perf` records a flat profile (`perf record -F`, no call graph) of `--mode loop` per language for
  `--max-seconds` (default 3), keeps it in `target/neo-versus/perf/`, and prints the top 40 symbols by self
  time with their DSO. The profile includes the fixture and setup, which are small next to 3 seconds.
- Builds run under `taskset -c 4-31` so a measurement on CPU 2 keeps its core.

## Findings

Each family keeps `benches/versus/findings/<family>.md`: where the safe duckdb-neo API was slower, more
awkward, or unable to express what the C++ variant does, with the profile lines or measurements that show
it. These drive the improvements to duckdb-neo itself.
