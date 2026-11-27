//! String scalar functions; `../cpp/scalar_str.cpp` is the C++ side.
//!
//! - `vs_len(s)`: byte length of `s`.
//! - `vs_upper_ascii(s)`: ASCII letters uppercased, other bytes unchanged.
//! - `vs_wrap(s, prefix, suffix)`: `prefix || s || suffix`.
//! - `vs_repeat(s, n)`: `s` repeated `n` times; NULL for negative `n`.
//! - `vs_first_word(s)`: the prefix of `s` up to its first space.
//! - `vs_count_char(s, c)`: occurrences of the byte `c` in `s`; an error unless `c` is one byte.
//! - `vs_has_prefix_any(s, prefixes)`: whether `s` starts with any entry of the constant comma-separated
//!   `prefixes`, which is split once at bind time.
//! - `vs_null_if_empty(s)`: NULL for '' and NULL, otherwise `s`.
//!
//! Inputs are read as `UncheckedString`: DuckDB guarantees valid UTF-8 in VARCHAR, so reads skip validation.

use duckdb_neo::{
    DuckDBType, Error, Result,
    bind_arguments::BindArguments,
    connection::{Connection, Context},
    data_chunk::VectorCollection,
    enums::FunctionProperty,
    scalar::{MicroScalar, ScalarBindCallbacks, ScalarCallbacks, ScalarFunctionBuilder},
    signature::{FunctionParameter, SignatureBuilder},
    types::UncheckedString,
    vector::{CombinedVectorIterator, Unknown, Vector},
};

fn register_len(conn: &Connection) -> Result<()> {
    MicroScalar::create(
        "vs_len",
        |_context, input: Vector<'_, UncheckedString>, mut output: Vector<'_, i64>| {
            let mut writer = output.writer()?;
            for (row, value) in input.reader()?.iter().enumerate() {
                writer.set_option(row, value.map(|value| value.len() as i64).as_ref())?;
            }
            Ok(())
        },
    )
    .register(conn)
}

fn register_upper_ascii(conn: &Connection) -> Result<()> {
    MicroScalar::create(
        "vs_upper_ascii",
        |_context, input: Vector<'_, UncheckedString>, mut output: Vector<'_, String>| {
            let mut writer = output.writer()?;
            for (row, value) in input.reader()?.iter().enumerate() {
                match value {
                    Some(value) => writer.set_with(row, value.len(), |buffer| {
                        buffer.push_str(value)?;
                        buffer.as_mut_str().make_ascii_uppercase();
                        Ok(())
                    })?,
                    None => writer.set_null(row)?,
                }
            }
            Ok(())
        },
    )
    .register(conn)
}

fn register_wrap(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<String>("s"))
        .param(FunctionParameter::new::<String>("prefix"))
        .param(FunctionParameter::new::<String>("suffix"))
        .return_type(String::logical_type(conn)?);
    MicroScalar::with_signature(
        "vs_wrap",
        signature,
        |_context, input: &mut VectorCollection, mut output: Vector<'_, String>| {
            let strings = input.get_vector_at::<UncheckedString>(0)?;
            let prefixes = input.get_vector_at::<UncheckedString>(1)?;
            let suffixes = input.get_vector_at::<UncheckedString>(2)?;
            let rows = CombinedVectorIterator::new((strings.reader()?, prefixes.reader()?, suffixes.reader()?))?;
            let mut writer = output.writer()?;
            for (row, values) in rows.enumerate() {
                match values {
                    (Some(value), Some(prefix), Some(suffix)) => {
                        writer.set_parts(row, Some(&[prefix, value, suffix]))?;
                    }
                    _ => writer.set_null(row)?,
                }
            }
            Ok(())
        },
    )
    .register(conn)
}

fn register_repeat(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<String>("s"))
        .param(FunctionParameter::new::<i32>("n"))
        .return_type(String::logical_type(conn)?);
    MicroScalar::with_signature(
        "vs_repeat",
        signature,
        |_context, input: &mut VectorCollection, mut output: Vector<'_, String>| {
            let strings = input.get_vector_at::<UncheckedString>(0)?;
            let counts = input.get_vector_at::<i32>(1)?;
            let rows = CombinedVectorIterator::new((strings.reader()?, counts.reader()?))?;
            let mut writer = output.writer()?;
            for (row, values) in rows.enumerate() {
                match values {
                    (Some(value), Some(&count)) if count >= 0 => {
                        let len = value.len() * count as usize;
                        if len > u32::MAX as usize {
                            return Err(Error::invalid_input("vs_repeat: the result exceeds 4 GiB"));
                        }
                        writer.set_with(row, len, |buffer| {
                            for _ in 0..count {
                                buffer.push_str(value)?;
                            }
                            Ok(())
                        })?;
                    }
                    _ => writer.set_null(row)?,
                }
            }
            Ok(())
        },
    )
    .register(conn)
}

fn register_first_word(conn: &Connection) -> Result<()> {
    MicroScalar::create(
        "vs_first_word",
        |_context, input: Vector<'_, UncheckedString>, mut output: Vector<'_, String>| {
            let mut writer = output.writer()?;
            for (row, value) in input.reader()?.iter().enumerate() {
                // A byte search: `str::split_once(' ')` runs a char searcher that verifies each match.
                let word =
                    value.map(|value| &value[..value.bytes().position(|byte| byte == b' ').unwrap_or(value.len())]);
                writer.set_option(row, word)?;
            }
            Ok(())
        },
    )
    .register(conn)
}

fn register_count_char(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<String>("s"))
        .param(FunctionParameter::new::<String>("c"))
        .return_type(i64::logical_type(conn)?);
    MicroScalar::with_signature(
        "vs_count_char",
        signature,
        |_context, input: &mut VectorCollection, mut output: Vector<'_, i64>| {
            let strings = input.get_vector_at::<UncheckedString>(0)?;
            let needles = input.get_vector_at::<UncheckedString>(1)?;
            let rows = CombinedVectorIterator::new((strings.reader()?, needles.reader()?))?;
            let mut writer = output.writer()?;
            for (row, values) in rows.enumerate() {
                match values {
                    (Some(value), Some(needle)) => {
                        let &[needle] = needle.as_bytes() else {
                            return Err(Error::invalid_input("vs_count_char: c must be exactly one byte"));
                        };
                        let count = value.bytes().filter(|&byte| byte == needle).count();
                        writer.set(row, &(count as i64))?;
                    }
                    _ => writer.set_null(row)?,
                }
            }
            Ok(())
        },
    )
    .register(conn)
}

struct HasPrefixAny;

impl ScalarCallbacks for HasPrefixAny {
    /// The entries of `prefixes`.
    type BindData = Vec<String>;
    type InitData = ();

    fn exec(
        &self,
        _context: &Context,
        bind_data: Option<&Self::BindData>,
        _init_data: Option<&mut Self::InitData>,
        input: &mut VectorCollection,
        output: Vector<'_, Unknown>,
    ) -> Result<()> {
        let prefixes = bind_data.expect("with_bind sets the prefixes");
        let mut output = output.cast::<bool>()?;
        output.set_size(input.row_count()?)?;
        let strings = input.get_vector_at::<UncheckedString>(0)?;
        let mut writer = output.writer()?;
        for (row, value) in strings.reader()?.iter().enumerate() {
            let matched = value.map(|value| prefixes.iter().any(|prefix| value.starts_with(prefix.as_str())));
            writer.set_option(row, matched.as_ref())?;
        }
        Ok(())
    }
}

impl ScalarBindCallbacks for HasPrefixAny {
    fn bind(&self, _context: &Context, arguments: BindArguments<'_>) -> Result<Self::BindData> {
        let prefixes = arguments
            .arg(1)?
            .resolved_value::<String>()?
            .flatten()
            .ok_or_else(|| Error::invalid_input("vs_has_prefix_any: prefixes must be a non-NULL constant"))?;
        Ok(prefixes.split(',').map(str::to_owned).collect())
    }
}

fn register_has_prefix_any(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<String>("s"))
        .param(FunctionParameter::new::<String>("prefixes"))
        .return_type(bool::logical_type(conn)?);
    ScalarFunctionBuilder::new("vs_has_prefix_any", signature, HasPrefixAny)
        .with_bind()
        .register(conn)
}

fn register_null_if_empty(conn: &Connection) -> Result<()> {
    MicroScalar::create(
        "vs_null_if_empty",
        |_context, input: Vector<'_, UncheckedString>, mut output: Vector<'_, String>| {
            let mut writer = output.writer()?;
            for (row, value) in input.reader()?.iter().enumerate() {
                writer.set_option(row, value.filter(|value| !value.is_empty()))?;
            }
            Ok(())
        },
    )
    .set_property(FunctionProperty::HasSpecialNullHandling(true))
    .register(conn)
}

pub fn register(conn: &Connection) -> Result<()> {
    register_len(conn)?;
    register_upper_ascii(conn)?;
    register_wrap(conn)?;
    register_repeat(conn)?;
    register_first_word(conn)?;
    register_count_char(conn)?;
    register_has_prefix_any(conn)?;
    register_null_if_empty(conn)
}
