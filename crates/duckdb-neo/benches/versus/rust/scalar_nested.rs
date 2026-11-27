//! Scalar functions over LIST, STRUCT, ARRAY and MAP; every function returns NULL when an argument is NULL.
//!
//! - `vs_list_sum(BIGINT[]) -> BIGINT`: wrapping sum of the non-NULL elements, 0 for an empty list.
//! - `vs_list_scale(l BIGINT[], k BIGINT) -> BIGINT[]`: each element times `k` (wrapping), NULL elements stay NULL.
//! - `vs_range_list(n INTEGER) -> INTEGER[]`: `[0, 1, .., n - 1]`, empty for `n <= 0`.
//! - `vs_split(s VARCHAR, sep VARCHAR) -> VARCHAR[]`: `s` split on `sep`, which must be a single byte.
//! - `vs_list_join(l VARCHAR[], sep VARCHAR) -> VARCHAR`: the non-NULL elements joined by `sep`.
//! - `vs_point_norm(STRUCT(x DOUBLE, y DOUBLE)) -> DOUBLE`: `sqrt(x * x + y * y)`, NULL if a field is NULL.
//! - `vs_make_point(x DOUBLE, y DOUBLE) -> STRUCT(x DOUBLE, y DOUBLE)`.
//! - `vs_dot3(a DOUBLE[3], b DOUBLE[3]) -> DOUBLE`: dot product, NULL if an element is NULL.
//! - `vs_map_sum(MAP(VARCHAR, INTEGER)) -> BIGINT`: wrapping sum of the non-NULL values, 0 for an empty map.
//! - `vs_list_of_points(n INTEGER) -> STRUCT(x DOUBLE, y DOUBLE)[]`: the points `(i, i * i)` for `i` in `0..n`.

use duckdb_neo::{
    AsDuckDB, DuckDBType, Error, Result,
    connection::Connection,
    scalar::MicroScalar,
    signature::{FunctionParameter, SignatureBuilder},
    types::{Array, List, ListIndex, Map, Struct, UncheckedString},
    vector::{CombinedVectorIterator, RowReader, RowWriter, Vector},
};

#[derive(AsDuckDB)]
#[duckdb(writer)]
struct Point {
    x: f64,
    y: f64,
}

type Vec3 = Array<f64, 3>;

fn register_list_sum(conn: &Connection) -> Result<()> {
    MicroScalar::create(
        "vs_list_sum",
        |_context, input: Vector<'_, List<i64>>, mut output: Vector<'_, i64>| {
            let lists = input.list_reader()?;
            let mut writer = output.writer()?;
            for (row, list) in lists.iter().enumerate() {
                let sum = list.map(|elements| {
                    elements
                        .iter()
                        .map(|value| value.copied().unwrap_or(0))
                        .fold(0_i64, i64::wrapping_add)
                });
                writer.set_option(row, sum.as_ref())?;
            }
            Ok(())
        },
    )
    .register(conn)
}

fn register_list_scale(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<List<i64>>("l"))
        .param(FunctionParameter::new::<i64>("k"))
        .return_type(List::<i64>::logical_type(conn)?);
    MicroScalar::with_signature(
        "vs_list_scale",
        signature,
        |_context, input, mut output: Vector<'_, List<i64>>| {
            let [lists, factors] = input.get_vector_array_mut([0, 1])?;
            let (lists, factors) = (lists.cast::<List<i64>>()?, factors.cast::<i64>()?);
            let (lists, factors) = (lists.list_reader()?, factors.reader()?);
            let elements = lists.elements().len();
            output.list_appender(|scaled| {
                scaled.reserve(elements)?;
                for (row, (list, factor)) in lists.iter().zip(factors.iter()).enumerate() {
                    let (Some(list), Some(&factor)) = (list, factor) else {
                        scaled.set_null(row)?;
                        continue;
                    };
                    scaled.set_elements(
                        row,
                        list.iter().map(|value| value.map(|value| value.wrapping_mul(factor))),
                    )?;
                }
                Ok(())
            })
        },
    )
    .register(conn)
}

fn register_range_list(conn: &Connection) -> Result<()> {
    MicroScalar::create(
        "vs_range_list",
        |_context, input: Vector<'_, i32>, mut output: Vector<'_, List<i32>>| {
            let counts = input.reader()?;
            let elements = counts.iter().map(|n| n.map_or(0, |&n| n.max(0) as usize)).sum();
            output.list_appender(|lists| {
                lists.reserve(elements)?;
                for (row, n) in counts.iter().enumerate() {
                    match n {
                        Some(&n) => lists.set_with(row, n.max(0) as usize, |value| value as i32)?,
                        None => lists.set_null(row)?,
                    }
                }
                Ok(())
            })
        },
    )
    .register(conn)
}

fn separator_byte(separator: &str) -> Result<u8> {
    match separator.as_bytes() {
        [byte] => Ok(*byte),
        _ => Err(Error::invalid_input("vs_split: the separator must be a single byte")),
    }
}

fn register_split(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<String>("s"))
        .param(FunctionParameter::new::<String>("sep"))
        .return_type(List::<String>::logical_type(conn)?);
    MicroScalar::with_signature(
        "vs_split",
        signature,
        |_context, input, mut output: Vector<'_, List<String>>| {
            let [strings, separators] = input.get_vector_array_mut([0, 1])?;
            let (strings, separators) = (
                strings.cast::<UncheckedString>()?,
                separators.cast::<UncheckedString>()?,
            );
            let (strings, separators) = (strings.reader()?, separators.reader()?);
            output.list_appender(|lists| {
                for (row, (string, separator)) in strings.iter().zip(separators.iter()).enumerate() {
                    let (Some(string), Some(separator)) = (string, separator) else {
                        lists.set_null(row)?;
                        continue;
                    };
                    let separator = separator_byte(separator)?;
                    let mut rest = Some(string);
                    // `str::split(char)` calls `memcmp` per match; cutting at an ASCII byte keeps char boundaries.
                    let parts = std::iter::from_fn(|| {
                        let text = rest?;
                        Some(Some(match text.bytes().position(|byte| byte == separator) {
                            Some(end) => {
                                rest = Some(&text[end + 1..]);
                                &text[..end]
                            }
                            None => {
                                rest = None;
                                text
                            }
                        }))
                    });
                    lists.set_elements(row, parts)?;
                }
                Ok(())
            })
        },
    )
    .register(conn)
}

fn register_list_join(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<List<String>>("l"))
        .param(FunctionParameter::new::<String>("sep"))
        .return_type(String::logical_type(conn)?);
    MicroScalar::with_signature(
        "vs_list_join",
        signature,
        |_context, input, mut output: Vector<'_, String>| {
            let [lists, separators] = input.get_vector_array_mut([0, 1])?;
            let (lists, separators) = (
                lists.cast::<List<UncheckedString>>()?,
                separators.cast::<UncheckedString>()?,
            );
            let (lists, separators) = (lists.list_reader()?, separators.reader()?);
            let mut writer = output.writer()?;
            let mut joined = String::new();
            for (row, (list, separator)) in lists.iter().zip(separators.iter()).enumerate() {
                let (Some(list), Some(separator)) = (list, separator) else {
                    writer.set_null(row)?;
                    continue;
                };
                joined.clear();
                for (index, value) in list.iter().flatten().enumerate() {
                    if index > 0 {
                        joined.push_str(separator);
                    }
                    joined.push_str(value);
                }
                writer.set(row, &joined)?;
            }
            Ok(())
        },
    )
    .register(conn)
}

fn register_point_norm(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<Point>("input"))
        .return_type(f64::logical_type(conn)?);
    MicroScalar::with_signature(
        "vs_point_norm",
        signature,
        |_context, input, mut output: Vector<'_, f64>| {
            // The untyped STRUCT skips the derived type's per-call field validation, and its rows combine with
            // the field readers in one `CombinedVectorIterator`.
            let points = input.get_vector_at_mut::<Struct>(0)?;
            let fields = points
                .children()
                .ok_or_else(|| Error::invalid_input("vs_point_norm: the point has no fields"))?;
            let (xs, ys) = (fields.get_vector_at::<f64>(0)?, fields.get_vector_at::<f64>(1)?);
            let rows = CombinedVectorIterator::new((points.reader()?, xs.reader()?, ys.reader()?))?;
            let mut writer = output.writer()?;
            for (row, (point, x, y)) in rows.enumerate() {
                let norm = point.and(x.zip(y)).map(|(x, y)| (x * x + y * y).sqrt());
                writer.set_option(row, norm.as_ref())?;
            }
            Ok(())
        },
    )
    .register(conn)
}

fn register_make_point(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<f64>("x"))
        .param(FunctionParameter::new::<f64>("y"))
        .return_type(Point::logical_type(conn)?);
    MicroScalar::with_signature(
        "vs_make_point",
        signature,
        |_context, input, mut output: Vector<'_, Point>| {
            let [xs, ys] = input.get_vector_array_mut([0, 1])?;
            let (xs, ys) = (xs.cast::<f64>()?, ys.cast::<f64>()?);
            let mut writer = output.row_writer()?;
            for (row, (x, y)) in CombinedVectorIterator::new((xs.reader()?, ys.reader()?))?.enumerate() {
                match (x, y) {
                    // Rows start valid, so writing the fields is enough.
                    (Some(x), Some(y)) => {
                        writer.x.set(row, x)?;
                        writer.y.set(row, y)?;
                    }
                    _ => writer.set_null(row)?,
                }
            }
            Ok(())
        },
    )
    .register(conn)
}

fn register_dot3(conn: &Connection) -> Result<()> {
    let signature = SignatureBuilder::new()
        .param(FunctionParameter::new::<Vec3>("a"))
        .param(FunctionParameter::new::<Vec3>("b"))
        .return_type(f64::logical_type(conn)?);
    MicroScalar::with_signature("vs_dot3", signature, |_context, input, mut output: Vector<'_, f64>| {
        let [lefts, rights] = input.get_vector_array_mut([0, 1])?;
        let (lefts, rights) = (lefts.cast::<Vec3>()?, rights.cast::<Vec3>()?);
        let (lefts, rights) = (lefts.array_reader()?, rights.array_reader()?);
        let mut writer = output.writer()?;
        for (row, (left, right)) in lefts.iter().zip(rights.iter()).enumerate() {
            let dot = left.zip(right).and_then(|(left, right)| {
                left.iter()
                    .zip(right.iter())
                    .try_fold(0.0, |sum, (a, b)| Some(sum + a? * b?))
            });
            writer.set_option(row, dot.as_ref())?;
        }
        Ok(())
    })
    .register(conn)
}

fn register_map_sum(conn: &Connection) -> Result<()> {
    MicroScalar::create(
        "vs_map_sum",
        |_context, input: Vector<'_, Map<UncheckedString, i32>>, mut output: Vector<'_, i64>| {
            let maps = input.map_reader()?;
            let mut writer = output.writer()?;
            for (row, entries) in maps.iter().enumerate() {
                let sum = entries.map(|entries| {
                    entries.values().fold(0_i64, |sum, value| match value {
                        Some(value) => sum.wrapping_add(i64::from(*value)),
                        None => sum,
                    })
                });
                writer.set_option(row, sum.as_ref())?;
            }
            Ok(())
        },
    )
    .register(conn)
}

fn register_list_of_points(conn: &Connection) -> Result<()> {
    MicroScalar::create(
        "vs_list_of_points",
        |_context, input: Vector<'_, i32>, mut output: Vector<'_, List<Point>>| {
            let counts = input.reader()?;
            let elements = counts.iter().map(|n| n.map_or(0, |&n| n.max(0) as usize)).sum();
            output.set_child_size(elements)?;
            let mut writer = output.list_writer()?;
            for (row, n) in counts.iter().enumerate() {
                let Some(&n) = n else {
                    writer.set_null(row)?;
                    continue;
                };
                let start = writer.child_offset();
                let len = n.max(0);
                // The sized elements start valid, so writing the fields is enough.
                let points = writer.elements();
                for index in 0..len {
                    let x = f64::from(index);
                    let element = start + index as usize;
                    points.x.set(element, &x)?;
                    points.y.set(element, &(x * x))?;
                }
                writer.set_entry(row, ListIndex::new(start as u64, len as u64))?;
            }
            Ok(())
        },
    )
    .register(conn)
}

pub fn register(conn: &Connection) -> Result<()> {
    register_list_sum(conn)?;
    register_list_scale(conn)?;
    register_range_list(conn)?;
    register_split(conn)?;
    register_list_join(conn)?;
    register_point_norm(conn)?;
    register_make_point(conn)?;
    register_dot3(conn)?;
    register_map_sum(conn)?;
    register_list_of_points(conn)
}
