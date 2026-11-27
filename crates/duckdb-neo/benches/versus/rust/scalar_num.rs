//! Numeric scalar functions; the C++ twin is `../cpp/scalar_num.cpp`. NULL in any argument gives NULL unless noted.
//!
//! - `vs_add_one(BIGINT) -> BIGINT`: wrapping `v + 1`.
//! - `vs_clamp(v DOUBLE, lo DOUBLE, hi DOUBLE) -> DOUBLE`: `min(max(v, lo), hi)`.
//! - `vs_is_even(BIGINT) -> BOOLEAN`: `v % 2 = 0`.
//! - `vs_coalesce_zero(BIGINT) -> BIGINT`: NULL becomes 0, through special NULL handling.
//! - `vs_checked_div(a BIGINT, b BIGINT) -> BIGINT`: `a / b` truncated, NULL for `b = 0` or overflow.
//! - `vs_mix(a INTEGER, b BIGINT, c DOUBLE, d SMALLINT) -> DOUBLE`: `a * c + b - d` in double.
//! - `vs_strict_sqrt(DOUBLE) -> DOUBLE`: `sqrt(v)`, an error for negative `v`.
//! - `vs_scale(v BIGINT, factor BIGINT) -> BIGINT`: wrapping `v * factor`; `factor` is a constant read at bind.

use duckdb_neo::{
    DuckDBType, Error, Result,
    bind_arguments::BindArguments,
    connection::{Connection, Context},
    data_chunk::VectorCollection,
    enums::FunctionProperty,
    scalar::{MicroScalar, ScalarBindCallbacks, ScalarCallbacks, ScalarFunctionBuilder},
    signature::{FunctionParameter, SignatureBuilder},
    vector::{Unknown, Vector},
};

fn register_add_one(conn: &Connection) -> Result<()> {
    MicroScalar::unary("vs_add_one", |value: i64| value.wrapping_add(1)).register(conn)
}

// Spelled like C++'s `std::min(std::max(v, lo), hi)`, which differs from `f64::max`/`f64::min` on NaN.
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    let low = if v < lo { lo } else { v };
    if hi < low { hi } else { low }
}

fn register_clamp(conn: &Connection) -> Result<()> {
    MicroScalar::ternary("vs_clamp", |v: f64, lo: f64, hi: f64| clamp(v, lo, hi)).register(conn)
}

fn register_is_even(conn: &Connection) -> Result<()> {
    MicroScalar::unary("vs_is_even", |value: i64| value % 2 == 0).register(conn)
}

fn register_coalesce_zero(conn: &Connection) -> Result<()> {
    MicroScalar::unary("vs_coalesce_zero", |value: Option<i64>| value.unwrap_or(0))
        .set_property(FunctionProperty::HasSpecialNullHandling(true))
        .register(conn)
}

fn register_checked_div(conn: &Connection) -> Result<()> {
    MicroScalar::binary("vs_checked_div", |a: i64, b: i64| a.checked_div(b)).register(conn)
}

fn register_mix(conn: &Connection) -> Result<()> {
    MicroScalar::quaternary("vs_mix", |a: i32, b: i64, c: f64, d: i16| {
        f64::from(a) * c + b as f64 - f64::from(d)
    })
    .register(conn)
}

fn register_strict_sqrt(conn: &Connection) -> Result<()> {
    MicroScalar::create(
        "vs_strict_sqrt",
        |_context, input: Vector<'_, f64>, mut output: Vector<'_, f64>| {
            output.writer()?.try_map_from(&input.reader()?, |value| {
                if *value < 0.0 {
                    return Err(Error::invalid_input("vs_strict_sqrt: negative input"));
                }
                Ok(value.sqrt())
            })
        },
    )
    .register(conn)
}

struct Scale;

impl ScalarBindCallbacks for Scale {
    fn bind(&self, _context: &Context, arguments: BindArguments<'_>) -> Result<Self::BindData> {
        arguments
            .arg(1)?
            .resolved_value::<i64>()?
            .ok_or_else(|| Error::invalid_input("vs_scale: factor must be a constant"))
    }
}

impl ScalarCallbacks for Scale {
    // The constant factor; `None` when it is NULL.
    type BindData = Option<i64>;
    type InitData = ();

    fn exec(
        &self,
        _context: &Context,
        bind_data: Option<&Self::BindData>,
        _init_data: Option<&mut Self::InitData>,
        vectors: &mut VectorCollection,
        output: Vector<'_, Unknown>,
    ) -> Result<()> {
        let factor = *bind_data.ok_or_else(|| Error::invalid_input("vs_scale: missing bind data"))?;
        let mut output = output.cast::<i64>()?;
        output.set_size(vectors.row_count()?)?;
        output
            .writer()?
            .map_option_from(&vectors.get_vector_at::<i64>(0)?.reader()?, |value| {
                match (value, factor) {
                    (Some(value), Some(factor)) => Some(value.wrapping_mul(factor)),
                    _ => None,
                }
            })
    }
}

fn register_scale(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<i64>("v"))
        .param(FunctionParameter::new::<i64>("factor"))
        .return_type(i64::logical_type(conn)?);
    ScalarFunctionBuilder::new("vs_scale", signature, Scale)
        .with_bind()
        .register(conn)
}

pub fn register(conn: &Connection) -> Result<()> {
    register_add_one(conn)?;
    register_clamp(conn)?;
    register_is_even(conn)?;
    register_coalesce_zero(conn)?;
    register_checked_div(conn)?;
    register_mix(conn)?;
    register_strict_sqrt(conn)?;
    register_scale(conn)
}
