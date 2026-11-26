#!/usr/bin/env python3
"""Compare user-defined functions on duckdb-neo with the same functions on DuckDB's C++ API.

`build` compiles both runners, `check` verifies every case in both languages, `run` benchmarks
them alternately on one CPU, and `perf` profiles one job. See
crates/duckdb-neo/benches/versus/README.md.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
VERSUS_DIR = WORKSPACE_DIR / "crates/duckdb-neo/benches/versus"
CASES_DIR = VERSUS_DIR / "cases"
CPP_SOURCE_DIR = VERSUS_DIR / "cpp"
OUT_DIR = WORKSPACE_DIR / "target/neo-versus"
MEASUREMENT_CPUS = range(2, 12)
CPP_API_FILES = ["duckdb_cpp.hpp", "duckdb_cpp.cpp", "duckdb_cpp_extension.hpp"]
LANGS = ["rust", "cpp"]
CASE_KEYS = {
    "name",
    "rows",
    "setup",
    "query",
    "check",
    "expect",
    "expect_at",
    "reset",
    "unprepared",
    "threads",
}


class VersusError(Exception):
    pass


@dataclass
class Case:
    family: str
    name: str
    rows: list[int]
    setup: list[str]
    query: str
    check: str
    expect: str | None = None
    expect_at: dict[int, str] = field(default_factory=dict)
    reset: str | None = None
    unprepared: bool = False
    threads: int | None = None

    @property
    def id(self) -> str:
        return f"{self.family}/{self.name}"

    def expected(self, rows: int) -> str | None:
        return self.expect_at.get(rows, self.expect)


@dataclass
class Job:
    case: Case
    rows: int

    @property
    def name(self) -> str:
        return f"{self.case.id}/{self.rows}"


@dataclass
class Runners:
    variant: str
    rust: Path
    cpp: Path
    lib_dir: Path

    def binary(self, lang: str) -> Path:
        return self.rust if lang == "rust" else self.cpp

    def env(self) -> dict[str, str]:
        path = os.environ.get("LD_LIBRARY_PATH")
        library_path = f"{self.lib_dir}:{path}" if path else str(self.lib_dir)
        return dict(os.environ, LD_LIBRARY_PATH=library_path)


def log(message: str) -> None:
    print(f"==> {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------
# Cases


def load_cases() -> list[Case]:
    cases = []
    for path in sorted(CASES_DIR.glob("*.toml")):
        family = path.stem
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as error:
            raise VersusError(f"{path}: {error}") from error
        unknown_top = set(data) - {"case"}
        if unknown_top:
            raise VersusError(f"{path}: unknown top-level keys {sorted(unknown_top)}")
        for index, raw in enumerate(data.get("case", [])):
            where = f"{path.name} case {raw.get('name', index)!r}"
            unknown = set(raw) - CASE_KEYS
            if unknown:
                raise VersusError(f"{where}: unknown keys {sorted(unknown)}")
            for key in ("name", "rows", "query", "check"):
                if key not in raw:
                    raise VersusError(f"{where}: missing {key!r}")
            setup = raw.get("setup", [])
            if isinstance(setup, str):
                setup = [setup]
            rows = raw["rows"]
            if not rows or not all(isinstance(r, int) and r >= 0 for r in rows):
                raise VersusError(f"{where}: rows must be a non-empty list of counts")
            expect_at = {int(k): str(v) for k, v in raw.get("expect_at", {}).items()}
            cases.append(
                Case(
                    family=family,
                    name=raw["name"],
                    rows=sorted(rows),
                    setup=list(setup),
                    query=raw["query"],
                    check=raw["check"],
                    expect=None if raw.get("expect") is None else str(raw["expect"]),
                    expect_at=expect_at,
                    reset=raw.get("reset"),
                    unprepared=bool(raw.get("unprepared", False)),
                    threads=raw.get("threads"),
                )
            )
    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise VersusError(f"duplicate case names: {', '.join(duplicates)}")
    return cases


def parse_rows(text: str | None) -> list[int] | None:
    if not text:
        return None
    try:
        return [int(part.strip().replace("_", "")) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise VersusError(f"--rows expects comma-separated counts, got {text!r}") from error


def select_jobs(cases: list[Case], filters: list[str], rows: list[int] | None) -> list[Job]:
    jobs = []
    for case in cases:
        if filters and not any(f in case.id for f in filters):
            continue
        for count in rows if rows is not None else case.rows:
            jobs.append(Job(case, count))
    if not jobs:
        raise VersusError("no case matches the filters")
    return jobs


def substitute(sql: str, rows: int, tmp: str) -> str:
    return sql.replace("{rows}", str(rows)).replace("{tmp}", tmp)


def runner_args(job: Job, mode: str, tmp: str, options: argparse.Namespace | None) -> list[str]:
    case = job.case
    args = ["--mode", mode]
    if case.threads is not None:
        args += ["--threads", str(case.threads)]
    for sql in case.setup:
        args += ["--setup", substitute(sql, job.rows, tmp)]
    args += ["--check", substitute(case.check, job.rows, tmp)]
    args += ["--query", substitute(case.query, job.rows, tmp)]
    if case.reset:
        args += ["--reset", substitute(case.reset, job.rows, tmp)]
    if case.unprepared:
        args.append("--unprepared")
    for flag in ("warmup_ms", "samples", "min_sample_ms", "max_seconds"):
        value = getattr(options, flag, None)
        if value is not None:
            args += ["--" + flag.replace("_", "-"), str(value)]
    return args


# ---------------------------------------------------------------------------------------------
# Building


def build_cpus() -> list[str]:
    cpus = [cpu for cpu in sorted(os.sched_getaffinity(0)) if cpu > MEASUREMENT_CPUS[-1]]
    if not cpus or shutil.which("taskset") is None:
        return []
    return ["taskset", "-c", ",".join(map(str, cpus))]


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if result.returncode != 0:
        raise VersusError(f"git {' '.join(args)} in {repo} failed: {result.stderr.decode().strip()}")
    return result.stdout


def write_if_changed(path: Path, content: bytes) -> None:
    if not path.exists() or path.read_bytes() != content:
        path.write_bytes(content)


def extract_cpp_api(repo: Path, ref: str) -> tuple[Path, bytes]:
    """Extract the C++ API of `ref` and return its directory and the ref's duckdb_v2.h."""
    sha = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    directory = OUT_DIR / "duckdb-cpp" / sha
    directory.mkdir(parents=True, exist_ok=True)
    for name in CPP_API_FILES:
        write_if_changed(directory / name, git(repo, "show", f"{sha}:tools/cpp/{name}"))
    return directory, git(repo, "show", f"{sha}:src/include/duckdb_v2.h")


def variant_dir(variant: str) -> Path:
    return OUT_DIR / variant


def cargo_target_dir(variant: str) -> Path:
    return WORKSPACE_DIR / "target" if variant == "default" else variant_dir(variant) / "cargo"


def build_rust(variant: str) -> Path:
    env = dict(
        os.environ,
        CARGO_TARGET_DIR=str(cargo_target_dir(variant)),
        CARGO_TERM_COLOR="never",
        CARGO_PROFILE_BENCH_DEBUG="line-tables-only",
    )
    if variant == "lto":
        env.update(CARGO_PROFILE_BENCH_LTO="fat", CARGO_PROFILE_BENCH_CODEGEN_UNITS="1")
    command = [
        *build_cpus(),
        "cargo",
        "bench",
        "-p",
        "duckdb-neo",
        "--bench",
        "versus",
        "--no-run",
        "--message-format=json-render-diagnostics",
    ]
    log(f"building the Rust runner ({variant})")
    result = subprocess.run(
        command, cwd=WORKSPACE_DIR, env=env, stdout=subprocess.PIPE, text=True, check=False
    )
    if result.returncode != 0:
        raise VersusError(f"building the Rust runner ({variant}) failed")
    executable = None
    for line in result.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            message.get("reason") == "compiler-artifact"
            and message.get("target", {}).get("name") == "versus"
            and message.get("executable")
        ):
            executable = Path(message["executable"])
    if executable is None:
        raise VersusError("cargo did not report the versus executable")
    return executable


def find_lib_dir(variant: str, header: bytes) -> Path:
    """The download directory of libduckdb-sys whose duckdb_v2.h matches the C++ API's ref."""
    host = next(
        line.split(":", 1)[1].strip()
        for line in subprocess.run(
            ["rustc", "-vV"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
        if line.startswith("host:")
    )
    root = cargo_target_dir(variant) / "duckdb-download" / host
    candidates = sorted(path.parent for path in root.glob("*/libduckdb.so"))
    if not candidates:
        raise VersusError(f"no libduckdb.so under {root}; did the Rust build download one?")
    for candidate in candidates:
        if (candidate / "duckdb_v2.h").is_file() and (candidate / "duckdb_v2.h").read_bytes() == header:
            return candidate
    raise VersusError(
        f"no libduckdb under {root} ships the duckdb_v2.h of the C++ API's ref; "
        "pick a --duckdb-ref that matches the downloaded library"
    )


def build_cpp(variant: str, api_dir: Path, lib_dir: Path) -> Path:
    build_dir = variant_dir(variant) / "cpp"
    generator = ["-G", "Ninja"] if shutil.which("ninja") else []
    cxx = shutil.which("clang++")
    if cxx is None:
        raise VersusError("clang++ is required for the C++ runner")
    configure = [
        "cmake",
        "-S",
        str(CPP_SOURCE_DIR),
        "-B",
        str(build_dir),
        *generator,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_COMPILER={cxx}",
        f"-DDUCKDB_CPP_DIR={api_dir}",
        f"-DDUCKDB_LIB_DIR={lib_dir}",
        f"-DVERSUS_LTO={'ON' if variant == 'lto' else 'OFF'}",
    ]
    log(f"building the C++ runner ({variant})")
    if subprocess.run(configure, stdout=subprocess.DEVNULL, check=False).returncode != 0:
        raise VersusError(f"configuring the C++ runner ({variant}) failed")
    build_command = [*build_cpus(), "cmake", "--build", str(build_dir)]
    if subprocess.run(build_command, stdout=sys.stderr, check=False).returncode != 0:
        raise VersusError(f"building the C++ runner ({variant}) failed")
    return build_dir / "versus_cpp"


def loaded_libduckdb(binary: Path, env: dict[str, str]) -> Path:
    result = subprocess.run(["ldd", str(binary)], env=env, capture_output=True, text=True, check=False)
    for line in result.stdout.splitlines():
        name, _, location = line.strip().partition(" => ")
        if name.startswith("libduckdb"):
            path = location.split(" (")[0].strip()
            if path == "not found" or not path:
                break
            return Path(path).resolve()
    raise VersusError(f"{binary} does not resolve libduckdb:\n{result.stdout}{result.stderr}")


def manifest_path(variant: str) -> Path:
    return variant_dir(variant) / "runners.json"


def build(variant: str, repo: Path, ref: str) -> Runners:
    api_dir, header = extract_cpp_api(repo, ref)
    rust = build_rust(variant)
    lib_dir = find_lib_dir(variant, header)
    cpp = build_cpp(variant, api_dir, lib_dir)
    runners = Runners(variant, rust, cpp, lib_dir)
    verify_runners(runners)
    manifest_path(variant).write_text(
        json.dumps({"rust": str(rust), "cpp": str(cpp), "lib_dir": str(lib_dir)}, indent=2) + "\n"
    )
    return runners


def verify_runners(runners: Runners) -> None:
    expected = (runners.lib_dir / "libduckdb.so").resolve()
    for lang in LANGS:
        binary = runners.binary(lang)
        if not binary.is_file():
            raise VersusError(f"the {lang} runner {binary} does not exist; run `build` first")
        loaded = loaded_libduckdb(binary, runners.env())
        if loaded != expected:
            raise VersusError(f"the {lang} runner loads {loaded}, not {expected}")


def get_runners(args: argparse.Namespace) -> Runners:
    variant = "lto" if args.lto else "default"
    if not args.no_build:
        return build(variant, args.duckdb_repo, args.duckdb_ref)
    path = manifest_path(variant)
    if not path.is_file():
        raise VersusError(f"no {variant} runners built yet; run `build{' --lto' if args.lto else ''}` first")
    data = json.loads(path.read_text())
    runners = Runners(variant, Path(data["rust"]), Path(data["cpp"]), Path(data["lib_dir"]))
    verify_runners(runners)
    return runners


# ---------------------------------------------------------------------------------------------
# Running


@contextlib.contextmanager
def measurement_lock(cpu: int):
    path = Path(f"/tmp/neo-versus.cpu{cpu}.lock")
    with open(path, "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"waiting for {path}, another measurement is running on CPU {cpu}")
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_runner(
    runners: Runners,
    lang: str,
    job: Job,
    mode: str,
    options: argparse.Namespace | None,
    prefix: list[str] | None = None,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="neo-versus-") as tmp:
        command = [*(prefix or []), str(runners.binary(lang)), *runner_args(job, mode, tmp, options)]
        result = subprocess.run(command, env=runners.env(), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise VersusError(
            f"{job.name}: the {lang} runner failed (exit code {result.returncode}):\n{result.stderr.strip()}"
        )
    lines = result.stdout.strip().splitlines()
    try:
        output = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise VersusError(f"{job.name}: the {lang} runner printed no JSON line:\n{result.stdout}") from error
    return output


def verify_check(job: Job, outputs: dict[str, list[dict[str, object]]]) -> str:
    """Fail unless every run of both languages agrees on `check` and `result_rows`, and with `expect`."""
    checks = {(lang, out["check"]) for lang, runs in outputs.items() for out in runs}
    values = {value for _, value in checks}
    if len(values) != 1:
        detail = ", ".join(f"{lang}={value!r}" for lang, value in sorted(checks))
        raise VersusError(f"{job.name}: the check differs between runs: {detail}")
    value = values.pop()
    expected = job.case.expected(job.rows)
    if expected is not None and value != expected:
        raise VersusError(f"{job.name}: check returned {value!r}, expected {expected!r}")
    rows = {(lang, out["result_rows"]) for lang, runs in outputs.items() for out in runs if "result_rows" in out}
    if len({count for _, count in rows}) > 1:
        detail = ", ".join(f"{lang}={count}" for lang, count in sorted(rows))
        raise VersusError(f"{job.name}: result_rows differs: {detail}")
    return value


@dataclass
class Result:
    job: Job
    check: str
    rows: int
    iters: dict[str, list[int]]
    medians: dict[str, list[float]]
    status: str = ""
    ratio: float = 0.0

    def median(self, lang: str) -> float:
        return statistics.median(self.medians[lang])

    def classify(self, threshold: float) -> None:
        rust, cpp = self.medians["rust"], self.medians["cpp"]
        self.ratio = self.median("rust") / self.median("cpp")
        if self.ratio > 1 + threshold:
            self.status = "slower" if min(rust) > max(cpp) else "inconclusive"
        elif self.ratio < 1 - threshold:
            self.status = "faster" if max(rust) < min(cpp) else "inconclusive"
        else:
            self.status = "same"


def bench_job(runners: Runners, job: Job, args: argparse.Namespace) -> Result:
    prefix = ["taskset", "-c", str(args.cpu)]
    outputs: dict[str, list[dict[str, object]]] = {lang: [] for lang in LANGS}
    with measurement_lock(args.cpu):
        for _ in range(args.repetitions):
            for lang in ["rust", "cpp", "cpp", "rust"]:
                outputs[lang].append(run_runner(runners, lang, job, "bench", args, prefix))
    check = verify_check(job, outputs)
    medians = {lang: [statistics.median(out["samples_ns"]) for out in runs] for lang, runs in outputs.items()}
    iters = {lang: [out["iters"] for out in runs] for lang, runs in outputs.items()}
    result = Result(job, check, outputs["rust"][0]["result_rows"], iters, medians)
    result.classify(args.threshold)
    return result


def format_ns(ns: float) -> str:
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.3g} {unit}"
    return f"{ns:.3g} ns"


def spread(values: list[float]) -> str:
    if len(values) < 2:
        return ""
    return f" ±{(max(values) - min(values)) / 2 / statistics.median(values):.1%}"


def machine_description() -> str:
    load = ", ".join(f"{value:.2f}" for value in os.getloadavg())
    return f"{platform.system()} {platform.machine()}, {os.cpu_count()} cores, load average {load}"


def table(results: list[Result], markdown: bool) -> list[str]:
    header = ["Job", "Rust", "C++", "Rust/C++", "Status"]
    rows = [
        [
            result.job.name,
            format_ns(result.median("rust")) + spread(result.medians["rust"]),
            format_ns(result.median("cpp")) + spread(result.medians["cpp"]),
            f"{result.ratio:.2f}",
            result.status,
        ]
        for result in results
    ]
    if markdown:
        lines = ["| " + " | ".join(header) + " |", "| --- | ---: | ---: | ---: | --- |"]
        return lines + ["| `" + row[0] + "` | " + " | ".join(row[1:]) + " |" for row in rows]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    align = ["<", ">", ">", ">", "<"]
    return [
        "  ".join(f"{cell:{align[i]}{widths[i]}}" for i, cell in enumerate(row)).rstrip()
        for row in [header, ["-" * width for width in widths], *rows]
    ]


def command_build(args: argparse.Namespace) -> int:
    for variant in ["lto"] if args.lto else ["default"]:
        runners = build(variant, args.duckdb_repo, args.duckdb_ref)
        print(f"{variant}: rust {runners.rust}")
        print(f"{variant}: cpp  {runners.cpp}")
        print(f"{variant}: both load {runners.lib_dir / 'libduckdb.so'}")
    return 0


# `check` also runs the query a few times, so its errors and row counts surface before `run`.
CHECK_PASS = argparse.Namespace(warmup_ms=0, samples=1, min_sample_ms=0, max_seconds=0)


def command_check(args: argparse.Namespace) -> int:
    runners = get_runners(args)
    cases = [case for case in load_cases() if not args.filter or any(f in case.id for f in args.filter)]
    if not cases:
        raise VersusError("no case matches the filters")
    failures = []
    count = 0
    for case in cases:
        for rows in case.rows if args.all_rows else case.rows[:2]:
            job = Job(case, rows)
            count += 1
            try:
                outputs = {lang: [run_runner(runners, lang, job, "bench", CHECK_PASS)] for lang in LANGS}
                value = verify_check(job, outputs)
                print(f"ok    {job.name}: check {value!r}, {outputs['rust'][0]['result_rows']} result rows")
            except VersusError as error:
                failures.append(str(error))
                print(f"FAIL  {error}")
    if failures:
        print(f"{len(failures)} of {count} jobs failed", file=sys.stderr)
        return 1
    print(f"all {count} jobs agree")
    return 0


def command_run(args: argparse.Namespace) -> int:
    jobs = select_jobs(load_cases(), args.filter, parse_rows(args.rows))
    runners = get_runners(args)
    machine = machine_description()
    log(f"{len(jobs)} jobs, {args.repetitions} repetition(s), CPU {args.cpu}, {runners.variant} build, {machine}")
    results = []
    for job in jobs:
        result = bench_job(runners, job, args)
        log(f"{job.name}: rust {format_ns(result.median('rust'))}, cpp {format_ns(result.median('cpp'))}")
        results.append(result)
    print("\n".join(table(results, markdown=False)))
    if args.markdown:
        lines = [
            f"## duckdb-neo versus C++ ({runners.variant} build)",
            "",
            f"- {args.repetitions} repetition(s) of rust, cpp, cpp, rust on CPU {args.cpu}; {machine} before the runs",
            f"- `slower`/`faster` beyond ±{args.threshold:.0%} only when every Rust run median is on the same side "
            "of every C++ one",
            "",
            *table(results, markdown=True),
            "",
        ]
        with open(args.markdown, "a", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")
    if args.json:
        payload = {
            "variant": runners.variant,
            "machine": machine,
            "cpu": args.cpu,
            "repetitions": args.repetitions,
            "threshold": args.threshold,
            "results": [
                {
                    "name": result.job.name,
                    "status": result.status,
                    "ratio": result.ratio,
                    "check": result.check,
                    "result_rows": result.rows,
                    "rust_median_ns": result.median("rust"),
                    "cpp_median_ns": result.median("cpp"),
                    "rust_ns": result.medians["rust"],
                    "cpp_ns": result.medians["cpp"],
                    "rust_iters": result.iters["rust"],
                    "cpp_iters": result.iters["cpp"],
                }
                for result in results
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def perf_frequency() -> int:
    try:
        limit = int(Path("/proc/sys/kernel/perf_event_max_sample_rate").read_text())
    except (OSError, ValueError):
        limit = 20000
    return max(1000, min(20000, limit))


def command_perf(args: argparse.Namespace) -> int:
    if shutil.which("perf") is None:
        raise VersusError("perf is not installed")
    cases = {case.id: case for case in load_cases()}
    case_id = args.job.rsplit("/", 1)[0] if args.job not in cases else args.job
    if case_id not in cases:
        raise VersusError(f"unknown case {args.job!r}; cases are {', '.join(sorted(cases))}")
    case = cases[case_id]
    rows = parse_rows(args.rows)
    if rows is None and case_id != args.job:
        rows = parse_rows(args.job.rsplit("/", 1)[1])
    if not rows or len(rows) != 1:
        raise VersusError("perf profiles one row count; pass --rows N")
    job = Job(case, rows[0])
    runners = get_runners(args)
    perf_dir = OUT_DIR / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)
    langs = LANGS if args.lang == "both" else [args.lang]
    outputs: dict[str, list[dict[str, object]]] = {}
    for lang in langs:
        data = perf_dir / f"{job.name.replace('/', '-')}-{runners.variant}-{lang}.data"
        prefix = [
            "perf",
            "record",
            "-q",
            "-F",
            str(perf_frequency()),
            "-o",
            str(data),
            "--",
            "taskset",
            "-c",
            str(args.cpu),
        ]
        log(f"{job.name}: recording {lang} for {args.max_seconds} s")
        with measurement_lock(args.cpu):
            outputs[lang] = [run_runner(runners, lang, job, "loop", args, prefix)]
        report = subprocess.run(
            [
                "perf",
                "report",
                "--stdio",
                "-i",
                str(data),
                "--no-children",
                "--fields",
                "overhead,dso,sym",
                "--field-separator",
                "\t",
                "--percent-limit",
                "0.1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if report.returncode != 0:
            raise VersusError(f"perf report failed for {data}:\n{report.stderr.strip()}")
        symbols = []
        for line in report.stdout.splitlines():
            fields = [part.strip() for part in line.split("\t")]
            if len(fields) == 3 and not line.startswith("#"):
                overhead, dso, symbol = fields
                symbols.append(f"{overhead:>7}  {dso:<24}  {symbol[:160]}")
        print(f"\n## {job.name} ({lang}, {runners.variant} build, {outputs[lang][0]['iters']} iterations)")
        print(f"perf data: {data}")
        print("\n".join(symbols[: args.top]))
    verify_check(job, outputs)
    return 0


# ---------------------------------------------------------------------------------------------
# Arguments


def parse_args(argv: list[str]) -> argparse.Namespace:
    build_options = argparse.ArgumentParser(add_help=False)
    build_options.add_argument(
        "--duckdb-repo",
        type=Path,
        default=Path.home() / "duckdb",
        help="DuckDB checkout to take tools/cpp from (default ~/duckdb)",
    )
    build_options.add_argument(
        "--duckdb-ref",
        default="origin/v2.0-cyanoptera",
        help="ref whose duckdb_v2.h matches the downloaded library (default origin/v2.0-cyanoptera)",
    )
    build_options.add_argument("--lto", action="store_true", help="link-time optimized runners")

    runner_options = argparse.ArgumentParser(add_help=False)
    runner_options.add_argument("--no-build", action="store_true", help="use the runners of the last build")
    runner_options.add_argument(
        "--filter", action="append", default=[], help="substring of family/case (repeatable)"
    )

    default_cpu = int(os.environ.get("NEO_VERSUS_CPU", "2"))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", parents=[build_options], help="build both runners")
    check = commands.add_parser(
        "check", parents=[build_options, runner_options], help="check every case at its two smallest row counts"
    )
    check.add_argument("--all-rows", action="store_true", help="check every row count of each case")

    run = commands.add_parser("run", parents=[build_options, runner_options], help="benchmark the cases")
    run.add_argument("--rows", help="comma-separated row counts, replacing each case's own")
    run.add_argument("--repetitions", type=int, help="rust, cpp, cpp, rust rounds per job (default 3, 1 with --quick)")
    run.add_argument("--samples", type=int, help="samples per runner (runner default 30, 10 with --quick)")
    run.add_argument("--max-seconds", type=float, help="sampling time limit per runner (runner default 3)")
    run.add_argument("--threshold", type=float, default=0.10, help="ratio change that counts (default 0.10)")
    run.add_argument("--cpu", type=int, default=default_cpu, help="CPU to pin the runners to (default $NEO_VERSUS_CPU or 2)")
    run.add_argument("--quick", action="store_true", help="10 samples and one repetition")
    run.add_argument("--json", help="write the results as JSON to this file")
    run.add_argument("--markdown", help="append a markdown report to this file")

    perf = commands.add_parser("perf", parents=[build_options], help="profile one job")
    perf.add_argument("job", help="family/case, or family/case/rows")
    perf.add_argument("--rows", help="row count to profile")
    perf.add_argument("--lang", choices=["rust", "cpp", "both"], default="both")
    perf.add_argument("--max-seconds", type=float, default=3.0, help="seconds of --mode loop (default 3)")
    perf.add_argument("--cpu", type=int, default=default_cpu, help="CPU to pin the runner to (default $NEO_VERSUS_CPU or 2)")
    perf.add_argument("--top", type=int, default=40, help="symbols to print (default 40)")
    perf.add_argument("--no-build", action="store_true", help="use the runners of the last build")

    args = parser.parse_args(argv)
    args.duckdb_repo = args.duckdb_repo.expanduser()
    if args.command == "run":
        if args.quick and args.samples is None:
            args.samples = 10
        if args.repetitions is None:
            args.repetitions = 1 if args.quick else 3
        if args.repetitions < 1:
            parser.error("--repetitions must be at least 1")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    commands = {"build": command_build, "check": command_check, "run": command_run, "perf": command_perf}
    try:
        return commands[args.command](args)
    except VersusError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
