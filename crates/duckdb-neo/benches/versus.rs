//! Rust runner of the versus benchmarks: runs one case against the functions in `versus/rust`.
//!
//! The arguments, the timing and the JSON line it prints are shared with `versus/cpp/runner.cpp`;
//! see `versus/README.md`. Run it through `scripts/neo_versus.py`.

#![deny(unsafe_code)]

#[path = "versus/rust/mod.rs"]
mod families;

use std::{process::ExitCode, time::Instant};

use duckdb_neo::{
    Error, Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    instance::{AttachBuilder, InstanceBuilder},
    statement::PreparedStatement,
    types::UncheckedString,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Check,
    Bench,
    Loop,
}

#[derive(Debug)]
struct Options {
    mode: Mode,
    threads: String,
    setup: Vec<String>,
    check: String,
    query: String,
    reset: Option<String>,
    unprepared: bool,
    warmup_ms: u64,
    samples: usize,
    min_sample_ms: u64,
    max_seconds: f64,
}

fn parse_number<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T> {
    value
        .parse()
        .map_err(|_| Error::invalid_input(format!("{flag} expects a number, got {value:?}")))
}

fn parse_options(mut args: impl Iterator<Item = String>) -> Result<Options> {
    let mut options = Options {
        mode: Mode::Bench,
        threads: "1".to_owned(),
        setup: Vec::new(),
        check: String::new(),
        query: String::new(),
        reset: None,
        unprepared: false,
        warmup_ms: 200,
        samples: 30,
        min_sample_ms: 2,
        max_seconds: 3.0,
    };
    let mut mode = None;
    while let Some(flag) = args.next() {
        if flag == "--unprepared" {
            options.unprepared = true;
            continue;
        }
        // `cargo bench` passes `--bench` to every harness-less benchmark.
        if flag == "--bench" {
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| Error::invalid_input(format!("{flag} expects a value")))?;
        match flag.as_str() {
            "--mode" => {
                mode = Some(match value.as_str() {
                    "check" => Mode::Check,
                    "bench" => Mode::Bench,
                    "loop" => Mode::Loop,
                    _ => return Err(Error::invalid_input(format!("unknown mode {value:?}"))),
                })
            }
            "--threads" => options.threads = value,
            "--setup" => options.setup.push(value),
            "--check" => options.check = value,
            "--query" => options.query = value,
            "--reset" => options.reset = Some(value),
            "--warmup-ms" => options.warmup_ms = parse_number(&flag, &value)?,
            "--samples" => options.samples = parse_number(&flag, &value)?,
            "--min-sample-ms" => options.min_sample_ms = parse_number(&flag, &value)?,
            "--max-seconds" => options.max_seconds = parse_number(&flag, &value)?,
            _ => return Err(Error::invalid_input(format!("unknown argument {flag:?}"))),
        }
    }
    options.mode = mode.ok_or_else(|| Error::invalid_input("--mode is required"))?;
    if options.check.is_empty() || options.query.is_empty() {
        return Err(Error::invalid_input("--check and --query are required"));
    }
    Ok(options)
}

fn read_check(conn: &Connection, sql: &str) -> Result<String> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result
        .next_chunk()?
        .ok_or_else(|| Error::invalid_input("the check query returned no rows"))?;
    let value = chunk
        .get_vector_at::<UncheckedString>(0)?
        .reader()?
        .get(0)?
        .map(str::to_owned);
    Ok(value.unwrap_or_else(|| "NULL".to_owned()))
}

enum Query<'a> {
    Prepared(PreparedStatement<'a>),
    Unprepared(&'a str),
}

struct Iteration {
    ns: u64,
    rows: usize,
}

fn iterate(conn: &Connection, query: &Query<'_>, reset: Option<&str>) -> Result<Iteration> {
    if let Some(reset) = reset {
        conn.query(reset, Parameters::None)?.drain()?;
    }
    let start = Instant::now();
    let mut result = match query {
        Query::Prepared(statement) => statement.execute(Parameters::None)?,
        Query::Unprepared(sql) => conn.query(*sql, Parameters::None)?,
    };
    let mut rows = 0;
    while let Some(chunk) = result.next_chunk()? {
        rows += chunk.row_count()?;
    }
    drop(result);
    let ns = start.elapsed().as_nanos() as u64;
    Ok(Iteration { ns, rows })
}

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn run(options: &Options) -> Result<String> {
    let environment = Environment::new()?;
    let instance = InstanceBuilder::new()
        .option("threads", options.threads.as_str())
        .build(&environment)?;
    instance.attach(AttachBuilder::new(StorageLocation::InMemory).make_default())?;
    let conn = instance.connect()?;
    families::register_all(&conn)?;

    for sql in &options.setup {
        conn.query(sql.as_str(), Parameters::None)?.drain()?;
    }
    let check = read_check(&conn, &options.check)?;
    if options.mode == Mode::Check {
        return Ok(format!("{{\"check\": {}}}", json_string(&check)));
    }

    let query = if options.unprepared {
        Query::Unprepared(&options.query)
    } else {
        Query::Prepared(conn.prepare(options.query.as_str(), false)?)
    };
    let reset = options.reset.as_deref();
    let max_ns = (options.max_seconds * 1e9) as u128;

    let mut iters = 0u64;
    let mut last_rows = 0;
    let mut samples_ns = Vec::new();
    if options.mode == Mode::Loop {
        let start = Instant::now();
        while iters == 0 || start.elapsed().as_nanos() < max_ns {
            last_rows = iterate(&conn, &query, reset)?.rows;
            iters += 1;
        }
    } else {
        let warmup_ns = u128::from(options.warmup_ms) * 1_000_000;
        let mut warmup = Vec::new();
        let start = Instant::now();
        while warmup.len() < 3 || start.elapsed().as_nanos() < warmup_ns {
            warmup.push(iterate(&conn, &query, reset)?.ns);
        }
        let typical_ns = median(&mut warmup).max(1);
        let min_sample_ns = options.min_sample_ms * 1_000_000;
        iters = if reset.is_some() {
            1
        } else {
            min_sample_ns.div_ceil(typical_ns).clamp(1, 1_000_000)
        };

        let start = Instant::now();
        while samples_ns.len() < options.samples {
            let mut sample_ns = 0;
            for _ in 0..iters {
                let iteration = iterate(&conn, &query, reset)?;
                sample_ns += iteration.ns;
                last_rows = iteration.rows;
            }
            samples_ns.push(format!("{:.1}", sample_ns as f64 / iters as f64));
            if samples_ns.len() >= 5 && start.elapsed().as_nanos() >= max_ns {
                break;
            }
        }
    }
    Ok(format!(
        "{{\"check\": {}, \"iters\": {iters}, \"result_rows\": {last_rows}, \"samples_ns\": [{}]}}",
        json_string(&check),
        samples_ns.join(", ")
    ))
}

fn main() -> ExitCode {
    match parse_options(std::env::args().skip(1)).and_then(|options| run(&options)) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("versus (rust): {error}");
            ExitCode::FAILURE
        }
    }
}
