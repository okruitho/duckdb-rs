//! Casts between builtin types and two custom types, `vs_cents` (a BIGINT count of cents) and `vs_hex` (a VARCHAR of
//! uppercase hex digits without leading zeros, like `to_hex`). NULL converts to NULL; a row that does not convert
//! raises an error, or is NULL under `TRY_CAST`.
//!
//! - `DOUBLE -> vs_cents`: `round(x * 100)`, half away from zero; fails on NaN, infinities and BIGINT overflow.
//! - `vs_cents -> DOUBLE`: `cents / 100`; also implicit (cost 1), so `vs_cents` binds to DOUBLE parameters.
//! - `VARCHAR -> vs_cents`: parses `[-]digits[.d[d]]`; fails on anything else and on overflow.
//! - `vs_cents -> VARCHAR`: formats `[-]units.cc`.
//! - `BIGINT -> vs_hex`: the two's complement bits in hex.
//! - `vs_hex -> BIGINT`: parses 1 to 16 hex digits of either case; fails on anything else.
//! - `vs_cents -> vs_hex`: the cent count in hex.

use duckdb_neo::{
    DuckDBType, Error, Result,
    cast::{CastFunctionBuilder, CastFunctionCallbacks, CastMode},
    connection::{Connection, Context},
    custom_type::CustomType,
    logical_type::LogicalType,
    types::UncheckedString,
    vector::Vector,
};

const CENTS_TYPE: &str = "vs_cents";
const HEX_TYPE: &str = "vs_hex";
const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";

fn conversion_error(value: &str, target: &str) -> Error {
    Error::invalid_input(format!("could not convert '{value}' to {target}"))
}

fn double_to_cents(value: f64) -> Option<i64> {
    let scaled = (value * 100.0).round();
    // The bounds are -2^63 and 2^63; NaN is outside every range.
    if (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&scaled) {
        Some(scaled as i64)
    } else {
        None
    }
}

fn parse_cents(text: &str) -> Option<i64> {
    let bytes = text.as_bytes();
    let negative = bytes.first() == Some(&b'-');
    let mut position = usize::from(negative);
    let units_start = position;
    let mut units: u64 = 0;
    while let Some(&byte) = bytes.get(position).filter(|byte| byte.is_ascii_digit()) {
        units = units.checked_mul(10)?.checked_add(u64::from(byte - b'0'))?;
        position += 1;
    }
    if position == units_start {
        return None;
    }
    let mut fraction: u64 = 0;
    if position < bytes.len() {
        if bytes[position] != b'.' {
            return None;
        }
        let digits = &bytes[position + 1..];
        if digits.is_empty() || digits.len() > 2 || !digits.iter().all(u8::is_ascii_digit) {
            return None;
        }
        for &digit in digits {
            fraction = fraction * 10 + u64::from(digit - b'0');
        }
        if digits.len() == 1 {
            fraction *= 10;
        }
    }
    let magnitude = units.checked_mul(100)?.checked_add(fraction)?;
    let limit = if negative { 1 << 63 } else { i64::MAX as u64 };
    if magnitude > limit {
        return None;
    }
    Some(if negative {
        (magnitude as i64).wrapping_neg()
    } else {
        magnitude as i64
    })
}

fn push_decimal(mut value: u64, out: &mut String) {
    let mut digits = [0u8; 20];
    let mut len = 0;
    loop {
        digits[len] = b'0' + (value % 10) as u8;
        len += 1;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    for &digit in digits[..len].iter().rev() {
        out.push(char::from(digit));
    }
}

fn format_cents(cents: i64, out: &mut String) {
    out.clear();
    if cents < 0 {
        out.push('-');
    }
    let magnitude = cents.unsigned_abs();
    push_decimal(magnitude / 100, out);
    out.push('.');
    let fraction = magnitude % 100;
    out.push(char::from(b'0' + (fraction / 10) as u8));
    out.push(char::from(b'0' + (fraction % 10) as u8));
}

fn format_hex(value: i64, out: &mut String) {
    out.clear();
    let bits = value as u64;
    let mut shift = 60;
    while shift > 0 && (bits >> shift) == 0 {
        shift -= 4;
    }
    loop {
        out.push(char::from(HEX_DIGITS[((bits >> shift) & 0xF) as usize]));
        if shift == 0 {
            break;
        }
        shift -= 4;
    }
}

fn parse_hex(text: &str) -> Option<i64> {
    let bytes = text.as_bytes();
    if bytes.is_empty() || bytes.len() > 16 {
        return None;
    }
    let mut bits: u64 = 0;
    for &byte in bytes {
        let digit = match byte {
            b'0'..=b'9' => byte - b'0',
            b'A'..=b'F' => byte - b'A' + 10,
            b'a'..=b'f' => byte - b'a' + 10,
            _ => return None,
        };
        bits = (bits << 4) | u64::from(digit);
    }
    Some(bits as i64)
}

struct DoubleToCents;

impl CastFunctionCallbacks for DoubleToCents {
    type InputType = f64;
    type OutputType = i64;

    fn exec(
        &self,
        _context: &Context,
        mode: CastMode,
        input: Vector<'_, f64>,
        mut output: Vector<'_, i64>,
    ) -> Result<()> {
        let mut writer = output.writer()?;
        for (row, value) in input.reader()?.iter().enumerate() {
            let Some(&value) = value else {
                writer.set_null(row)?;
                continue;
            };
            match double_to_cents(value) {
                Some(cents) => writer.set(row, &cents)?,
                None if mode == CastMode::Try => writer.set_null(row)?,
                None => return Err(Error::invalid_input("DOUBLE value out of vs_cents range")),
            }
        }
        Ok(())
    }
}

struct CentsToDouble;

impl CastFunctionCallbacks for CentsToDouble {
    type InputType = i64;
    type OutputType = f64;

    fn exec(
        &self,
        _context: &Context,
        _mode: CastMode,
        input: Vector<'_, i64>,
        mut output: Vector<'_, f64>,
    ) -> Result<()> {
        let mut writer = output.writer()?;
        for (row, cents) in input.reader()?.iter().enumerate() {
            writer.set_option(row, cents.map(|&cents| cents as f64 / 100.0).as_ref())?;
        }
        Ok(())
    }
}

// DuckDB guarantees valid UTF-8 in VARCHAR and the parsers only look at bytes, so the input is not re-validated.
struct VarcharToCents;

impl CastFunctionCallbacks for VarcharToCents {
    type InputType = UncheckedString;
    type OutputType = i64;

    fn exec(
        &self,
        _context: &Context,
        mode: CastMode,
        input: Vector<'_, UncheckedString>,
        mut output: Vector<'_, i64>,
    ) -> Result<()> {
        let mut writer = output.writer()?;
        for (row, text) in input.reader()?.iter().enumerate() {
            let Some(text) = text else {
                writer.set_null(row)?;
                continue;
            };
            match parse_cents(text) {
                Some(cents) => writer.set(row, &cents)?,
                None if mode == CastMode::Try => writer.set_null(row)?,
                None => return Err(conversion_error(text, CENTS_TYPE)),
            }
        }
        Ok(())
    }
}

struct CentsToVarchar;

impl CastFunctionCallbacks for CentsToVarchar {
    type InputType = i64;
    type OutputType = String;

    fn exec(
        &self,
        _context: &Context,
        _mode: CastMode,
        input: Vector<'_, i64>,
        mut output: Vector<'_, String>,
    ) -> Result<()> {
        let mut writer = output.writer()?;
        let mut text = String::new();
        for (row, cents) in input.reader()?.iter().enumerate() {
            let Some(&cents) = cents else {
                writer.set_null(row)?;
                continue;
            };
            format_cents(cents, &mut text);
            writer.set(row, &text)?;
        }
        Ok(())
    }
}

/// Formats a BIGINT-backed input as `vs_hex`; serves both `BIGINT -> vs_hex` and `vs_cents -> vs_hex`.
struct IntegerToHex;

impl CastFunctionCallbacks for IntegerToHex {
    type InputType = i64;
    type OutputType = String;

    fn exec(
        &self,
        _context: &Context,
        _mode: CastMode,
        input: Vector<'_, i64>,
        mut output: Vector<'_, String>,
    ) -> Result<()> {
        let mut writer = output.writer()?;
        let mut text = String::new();
        for (row, value) in input.reader()?.iter().enumerate() {
            let Some(&value) = value else {
                writer.set_null(row)?;
                continue;
            };
            format_hex(value, &mut text);
            writer.set(row, &text)?;
        }
        Ok(())
    }
}

struct HexToBigint;

impl CastFunctionCallbacks for HexToBigint {
    type InputType = UncheckedString;
    type OutputType = i64;

    fn exec(
        &self,
        _context: &Context,
        mode: CastMode,
        input: Vector<'_, UncheckedString>,
        mut output: Vector<'_, i64>,
    ) -> Result<()> {
        let mut writer = output.writer()?;
        for (row, text) in input.reader()?.iter().enumerate() {
            let Some(text) = text else {
                writer.set_null(row)?;
                continue;
            };
            match parse_hex(text) {
                Some(value) => writer.set(row, &value)?,
                None if mode == CastMode::Try => writer.set_null(row)?,
                None => return Err(conversion_error(text, "BIGINT")),
            }
        }
        Ok(())
    }
}

pub fn register(conn: &Connection) -> Result<()> {
    let bigint = i64::logical_type(conn)?;
    let double = f64::logical_type(conn)?;
    let varchar = String::logical_type(conn)?;
    CustomType::new(CENTS_TYPE, i64::logical_type(conn)?).register(conn)?;
    CustomType::new(HEX_TYPE, String::logical_type(conn)?).register(conn)?;
    let cents: LogicalType = bigint.with_alias(conn, CENTS_TYPE)?;
    let hex: LogicalType = varchar.with_alias(conn, HEX_TYPE)?;

    CastFunctionBuilder::new(double.clone(), cents.clone(), DoubleToCents).register(conn)?;
    CastFunctionBuilder::new(cents.clone(), double, CentsToDouble)
        .implicit_cast_cost(1)
        .register(conn)?;
    CastFunctionBuilder::new(varchar.clone(), cents.clone(), VarcharToCents).register(conn)?;
    CastFunctionBuilder::new(cents.clone(), varchar, CentsToVarchar).register(conn)?;
    CastFunctionBuilder::new(bigint.clone(), hex.clone(), IntegerToHex).register(conn)?;
    CastFunctionBuilder::new(hex.clone(), bigint, HexToBigint).register(conn)?;
    CastFunctionBuilder::new(cents, hex, IntegerToHex).register(conn)
}
