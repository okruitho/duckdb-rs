use std::sync::Arc;

use super::*;
use crate::{
    Error, Parameters,
    enums::FunctionProperty,
    environment::{Environment, StorageLocation},
    error::DuckDBError,
    instance::{AttachBuilder, InstanceBuilder},
    types::List,
};

#[test]
fn inferred_signature_and_sized_batches() -> Result<()> {
    let offset = Arc::new(7_i64);
    let weak = Arc::downgrade(&offset);
    let function = MicroScalar::create(
        "micro_add",
        move |ctx, input: Vector<'_, i32>, mut output: Vector<'_, i64>| {
            assert_eq!(input.logical_type(), &i32::logical_type(ctx)?);
            assert_eq!(output.logical_type(), &i64::logical_type(ctx)?);
            assert_eq!(output.len(), input.len());
            let mut writer = output.writer()?;
            for (row, value) in input.reader()?.iter().enumerate() {
                writer.set_option(row, value.map(|value| i64::from(*value) + *offset).as_ref())?;
            }
            Ok(())
        },
    );

    {
        let env = Environment::new()?;
        let conn = env.open(StorageLocation::InMemory)?.connect()?;
        function.register(&conn)?;
        assert!(weak.upgrade().is_some());

        let chunk = conn
            .query("SELECT micro_add(input := 35)", Parameters::None)?
            .next_chunk()?
            .unwrap();
        assert_eq!(chunk.get_vector_at::<i64>(0)?.reader()?.get(0)?, Some(&42));

        let mut result = conn.query(
            "SELECT micro_add(CASE WHEN i % 5 = 0 THEN NULL ELSE i::INTEGER END) \
             FROM range(10000) t(i) WHERE i % 3 <> 0 ORDER BY i",
            Parameters::None,
        )?;
        let mut actual = Vec::new();
        while let Some(chunk) = result.next_chunk()? {
            actual.extend(chunk.get_vector_at::<i64>(0)?.reader()?.iter().map(|v| v.copied()));
        }
        let expected: Vec<_> = (0..10000)
            .filter(|i| i % 3 != 0)
            .map(|i| (i % 5 != 0).then_some(i + 7))
            .collect();
        assert_eq!(actual, expected);
    }
    assert!(weak.upgrade().is_none());
    Ok(())
}

#[test]
fn nested_input_and_string_output() -> Result<()> {
    let env = Environment::new()?;
    let conn = env.open(StorageLocation::InMemory)?.connect()?;
    MicroScalar::create(
        "micro_list_length",
        |_ctx, input: Vector<'_, List<i32>>, mut output: Vector<'_, String>| {
            let mut writer = output.writer()?;
            for (row, value) in input.reader()?.iter().enumerate() {
                let text = value.map(|list| format!("length: {}", list.length()));
                writer.set_option(row, text.as_deref())?;
            }
            Ok(())
        },
    )
    .register(&conn)?;

    let chunk = conn
        .query(
            "SELECT micro_list_length(v) \
             FROM (VALUES ([1, NULL, 3]), ([]::INTEGER[]), (NULL::INTEGER[])) t(v)",
            Parameters::None,
        )?
        .next_chunk()?
        .unwrap();
    assert_eq!(
        chunk.get_vector_at::<String>(0)?.reader()?.iter().collect::<Vec<_>>(),
        [Some("length: 3"), Some("length: 0"), None]
    );
    Ok(())
}

#[test]
fn properties_control_null_handling() -> Result<()> {
    let env = Environment::new()?;
    let conn = env.open(StorageLocation::InMemory)?.connect()?;
    for (name, special) in [("micro_default_nulls", false), ("micro_special_nulls", true)] {
        MicroScalar::create(name, |_ctx, input: Vector<'_, i32>, mut output: Vector<'_, i32>| {
            let mut writer = output.writer()?;
            for (row, value) in input.reader()?.iter().enumerate() {
                writer.set(row, &value.copied().unwrap_or(42))?;
            }
            Ok(())
        })
        .set_property(FunctionProperty::HasSpecialNullHandling(special))
        .register(&conn)?;
    }
    let chunk = conn
        .query(
            "SELECT micro_default_nulls(NULL::INTEGER), micro_special_nulls(NULL::INTEGER)",
            Parameters::None,
        )?
        .next_chunk()?
        .unwrap();
    assert_eq!(chunk.get_vector_at::<i32>(0)?.reader()?.get(0)?, None);
    assert_eq!(chunk.get_vector_at::<i32>(1)?.reader()?.get(0)?, Some(&42));
    Ok(())
}

#[test]
fn callback_errors_propagate() -> Result<()> {
    let env = Environment::new()?;
    let conn = env.open(StorageLocation::InMemory)?.connect()?;
    MicroScalar::create(
        "micro_error",
        |_ctx, _input: Vector<'_, i32>, _output: Vector<'_, i32>| Err(Error::invalid_input("micro failed")),
    )
    .register(&conn)?;

    let error = conn
        .query("SELECT micro_error(1)", Parameters::None)
        .and_then(|mut result| result.drain())
        .expect_err("callback error must reach the caller");
    assert_eq!(error.code, DuckDBError::InputInvalid);
    assert!(error.message.contains("micro failed"), "{error}");
    conn.query("SELECT 42", Parameters::None)?.drain()?;
    Ok(())
}

#[test]
fn callback_panics_are_caught() -> Result<()> {
    let env = Environment::new()?;
    let instance = InstanceBuilder::new().option("threads", "1").build(&env)?;
    instance.attach(AttachBuilder::new(StorageLocation::InMemory).make_default())?;
    let conn = instance.connect()?;
    MicroScalar::create(
        "micro_panic",
        |_ctx, _input: Vector<'_, i32>, _output: Vector<'_, i32>| panic!("micro panicked"),
    )
    .register(&conn)?;

    let error = conn
        .query("SELECT micro_panic(1)", Parameters::None)
        .and_then(|mut result| result.drain())
        .expect_err("panic must reach the caller as an error");
    assert!(error.message.contains("micro panicked"), "{error}");
    Ok(())
}

#[test]
fn explicit_signature_sizes_variadic_batches() -> Result<()> {
    let env = Environment::new()?;
    let conn = env.open(StorageLocation::InMemory)?.connect()?;
    let base = Arc::new(42_i64);
    MicroScalar::with_signature(
        "micro_sum",
        SignatureBuilder::new()
            .param(FunctionParameter::vararg::<i32>("values"))
            .return_type(i64::logical_type(&conn)?),
        move |ctx, input, mut output: Vector<'_, i64>| {
            assert_eq!(output.logical_type(), &i64::logical_type(ctx)?);
            assert_eq!(output.len(), input.row_count()?);
            let mut sums = vec![*base; output.len()];
            for column in 0..input.col_count() {
                for (sum, value) in sums
                    .iter_mut()
                    .zip(input.get_vector_at::<i32>(column)?.reader()?.iter())
                {
                    *sum += i64::from(value.copied().unwrap_or(0));
                }
            }
            let mut writer = output.writer()?;
            for (row, sum) in sums.into_iter().enumerate() {
                writer.set(row, &sum)?;
            }
            Ok(())
        },
    )
    .register(&conn)?;

    let chunk = conn
        .query("SELECT micro_sum(), micro_sum(1, 2, 3)", Parameters::None)?
        .next_chunk()?
        .unwrap();
    assert_eq!(chunk.get_vector_at::<i64>(0)?.reader()?.get(0)?, Some(&42));
    assert_eq!(chunk.get_vector_at::<i64>(1)?.reader()?.get(0)?, Some(&48));

    let mut result = conn.query(
        "SELECT micro_sum(i::INTEGER, 3) FROM range(5000) t(i) ORDER BY i",
        Parameters::None,
    )?;
    let mut actual = Vec::new();
    while let Some(chunk) = result.next_chunk()? {
        actual.extend(
            chunk
                .get_vector_at::<i64>(0)?
                .reader()?
                .iter()
                .map(|value| value.copied()),
        );
    }
    assert_eq!(actual, (45..5045).map(Some).collect::<Vec<_>>());
    Ok(())
}

#[test]
fn explicit_signature_checks_output_type() -> Result<()> {
    let env = Environment::new()?;
    let conn = env.open(StorageLocation::InMemory)?.connect()?;
    MicroScalar::with_signature(
        "micro_mismatched",
        SignatureBuilder::new().return_type(String::logical_type(&conn)?),
        |_ctx, _input, _output: Vector<'_, i32>| panic!("mismatched output must not reach the closure"),
    )
    .register(&conn)?;

    let error = conn
        .query("SELECT micro_mismatched()", Parameters::None)
        .and_then(|mut result| result.drain())
        .expect_err("output must match the declared logical type");
    assert!(error.message.contains("Vector logical type mismatch"), "{error}");
    conn.query("SELECT 42", Parameters::None)?.drain()?;
    Ok(())
}

#[test]
fn captures_are_dropped_without_registration() -> Result<()> {
    let env = Environment::new()?;
    let conn = env.open(StorageLocation::InMemory)?.connect()?;
    for register in [false, true] {
        let captured = Arc::new(42_i32);
        let weak = Arc::downgrade(&captured);
        let function = MicroScalar::create("", move |_ctx, input: Vector<'_, i32>, mut output: Vector<'_, i32>| {
            let mut writer = output.writer()?;
            for row in 0..input.len() {
                writer.set(row, &*captured)?;
            }
            Ok(())
        });
        if register {
            let error = function.register(&conn).expect_err("empty name must be rejected");
            assert!(error.message.contains("Function name cannot be empty"), "{error}");
        } else {
            drop(function);
        }
        assert!(weak.upgrade().is_none());
    }
    Ok(())
}

fn count_mismatches(conn: &crate::connection::Connection, sql: &str) -> Result<Option<i64>> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.unwrap();
    Ok(chunk.get_vector_at::<i64>(0)?.reader()?.get(0)?.copied())
}

#[test]
fn row_executors_match_sql_over_every_layout() -> Result<()> {
    let env = Environment::new()?;
    let conn = env.open(StorageLocation::InMemory)?.connect()?;
    MicroScalar::unary("row_add", |value: i64| value.wrapping_add(1))
        .set_property(FunctionProperty::HasSpecialNullHandling(true))
        .register(&conn)?;
    MicroScalar::unary("row_coalesce", |value: Option<i64>| value.unwrap_or(-1))
        .set_property(FunctionProperty::HasSpecialNullHandling(true))
        .register(&conn)?;
    MicroScalar::unary("row_is_odd", |value: i64| (value % 2 != 0).then_some(true)).register(&conn)?;
    MicroScalar::binary("row_div", |a: i64, b: i64| a.checked_div(b))
        .set_property(FunctionProperty::HasSpecialNullHandling(true))
        .register(&conn)?;
    MicroScalar::binary("row_pick", |a: Option<i64>, b: Option<i64>| a.or(b))
        .set_property(FunctionProperty::HasSpecialNullHandling(true))
        .register(&conn)?;
    MicroScalar::ternary("row_clamp", |v: i64, lo: Option<i64>, hi: i64| {
        v.max(lo.unwrap_or(i64::MIN)).min(hi)
    })
    .set_property(FunctionProperty::HasSpecialNullHandling(true))
    .register(&conn)?;
    MicroScalar::quaternary("row_mix", |a: i32, b: i64, c: Option<f64>, d: i16| {
        c.map(|c| f64::from(a) * c + b as f64 - f64::from(d))
    })
    .set_property(FunctionProperty::HasSpecialNullHandling(true))
    .register(&conn)?;
    conn.query(
        "CREATE TABLE t AS SELECT CASE WHEN i % 7 = 3 OR i BETWEEN 128 AND 191 THEN NULL ELSE i - 2500 END AS v, \
         CASE WHEN i % 11 = 5 THEN NULL ELSE i % 13 - 6 END AS w FROM range(5000) r(i)",
        Parameters::None,
    )?
    .drain()?;
    let (min, max) = (i64::MIN, i64::MAX);
    let checks = |a: &str, b: &str| {
        [
            format!("row_add({a}) IS DISTINCT FROM CASE WHEN {a} = {max} THEN {min} ELSE {a} + 1 END"),
            format!("row_coalesce({a}) IS DISTINCT FROM coalesce({a}, -1)"),
            format!("row_is_odd({a}) IS DISTINCT FROM CASE WHEN {a} % 2 <> 0 THEN true END"),
            format!(
                "row_div({a}, {b}) IS DISTINCT FROM \
                 CASE WHEN {b} <> 0 AND NOT ({a} = {min} AND {b} = -1) THEN {a} // {b} END"
            ),
            format!("row_pick({a}, {b}) IS DISTINCT FROM coalesce({a}, {b})"),
            format!(
                "row_clamp({a}, {b}, 100) IS DISTINCT FROM \
                 CASE WHEN {a} IS NOT NULL THEN least(greatest({a}, coalesce({b}, {a})), 100) END"
            ),
            format!(
                "row_mix(({b} % 1000)::INTEGER, {a}, {a}::DOUBLE / 4, ({b} % 100)::SMALLINT) IS DISTINCT FROM \
                 ({b} % 1000)::DOUBLE * ({a}::DOUBLE / 4) + {a}::DOUBLE - ({b} % 100)::DOUBLE"
            ),
        ]
    };
    for (from, a, b) in [
        (
            "test_vector_types(NULL::BIGINT, NULL::BIGINT, all_flat := false)",
            "test_vector",
            "test_vector2",
        ),
        ("t", "v", "w"),
        ("t WHERE v IS NULL OR v % 3 <> 1", "v", "w"),
        ("t", "v", "3"),
        ("t", "42", "w"),
        ("t CROSS JOIN (SELECT NULL::BIGINT AS n)", "v", "n"),
        ("t CROSS JOIN (SELECT NULL::BIGINT AS n)", "n", "w"),
        ("range(3000)", "range", "(range % 5)"),
        ("t WHERE v > 10000", "v", "w"),
    ] {
        for check in checks(a, b) {
            let sql = format!("SELECT count(*) FILTER (WHERE {check}) FROM {from}");
            assert_eq!(count_mismatches(&conn, &sql)?, Some(0), "{sql}");
        }
    }
    Ok(())
}

#[test]
fn row_executors_infer_signatures() -> Result<()> {
    let env = Environment::new()?;
    let conn = env.open(StorageLocation::InMemory)?.connect()?;
    MicroScalar::quaternary("row_types", |a: i32, _b: Option<bool>, _c: f32, _d: u8| {
        Some(i64::from(a))
    })
    .register(&conn)?;
    let mut result = conn.query(
        "SELECT parameters::VARCHAR || ' ' || parameter_types::VARCHAR || ' ' || return_type \
         FROM duckdb_functions() WHERE function_name = 'row_types'",
        Parameters::None,
    )?;
    let chunk = result.next_chunk()?.unwrap();
    assert_eq!(
        chunk.get_vector_at::<String>(0)?.reader()?.get(0)?,
        Some("[arg0, arg1, arg2, arg3] [INTEGER, BOOLEAN, FLOAT, UTINYINT] BIGINT")
    );
    Ok(())
}

#[test]
fn row_executor_panics_become_errors() -> Result<()> {
    let env = Environment::new()?;
    let instance = InstanceBuilder::new().option("threads", "1").build(&env)?;
    instance.attach(AttachBuilder::new(StorageLocation::InMemory).make_default())?;
    let conn = instance.connect()?;
    MicroScalar::binary("row_panic", |a: i64, b: i64| {
        assert!(a != 3000, "row function panicked");
        a + b
    })
    .register(&conn)?;
    let error = conn
        .query("SELECT row_panic(range, 1) FROM range(5000)", Parameters::None)
        .and_then(|mut result| result.drain())
        .expect_err("panic must reach the caller as an error");
    assert!(error.message.contains("row function panicked"), "{error}");
    Ok(())
}
