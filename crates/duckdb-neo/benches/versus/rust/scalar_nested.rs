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
    types::{Array, List, ListIndex, Map, UncheckedString},
    vector::{CombinedVectorIterator, RowReader, RowWriter, Vector},
};

#[derive(AsDuckDB)]
#[duckdb(reader, writer)]
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
                        .flatten()
                        .fold(0_i64, |sum, value| sum.wrapping_add(*value))
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
            let elements = lists
                .iter()
                .zip(factors.iter())
                .map(|row| match row {
                    (Some(list), Some(_)) => list.len(),
                    _ => 0,
                })
                .sum();
            output.set_child_size(elements)?;
            let mut writer = output.list_writer()?;
            for (row, (list, factor)) in lists.iter().zip(factors.iter()).enumerate() {
                let (Some(list), Some(&factor)) = (list, factor) else {
                    writer.set_null(row)?;
                    continue;
                };
                let start = writer.child_offset();
                for (index, value) in list.iter().enumerate() {
                    let scaled = value.map(|value| value.wrapping_mul(factor));
                    writer.elements().set_option(start + index, scaled.as_ref())?;
                }
                writer.set_entry(row, ListIndex::new(start as u64, list.len() as u64))?;
            }
            Ok(())
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
            output.set_child_size(elements)?;
            let mut writer = output.list_writer()?;
            for (row, n) in counts.iter().enumerate() {
                let Some(&n) = n else {
                    writer.set_null(row)?;
                    continue;
                };
                let start = writer.child_offset();
                let len = n.max(0);
                for value in 0..len {
                    writer.elements().set(start + value as usize, &value)?;
                }
                writer.set_entry(row, ListIndex::new(start as u64, len as u64))?;
            }
            Ok(())
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
            let mut elements = 0;
            for (string, separator) in strings.iter().zip(separators.iter()) {
                if let (Some(string), Some(separator)) = (string, separator) {
                    let separator = separator_byte(separator)?;
                    elements += string.bytes().filter(|&byte| byte == separator).count() + 1;
                }
            }
            output.set_child_size(elements)?;
            let mut writer = output.list_writer()?;
            for (row, (string, separator)) in strings.iter().zip(separators.iter()).enumerate() {
                let (Some(string), Some(separator)) = (string, separator) else {
                    writer.set_null(row)?;
                    continue;
                };
                // A single byte of valid UTF-8 is ASCII, so it is also a `char`.
                let separator = char::from(separator_byte(separator)?);
                let start = writer.child_offset();
                let mut len = 0;
                for part in string.split(separator) {
                    writer.elements().set(start + len, part)?;
                    len += 1;
                }
                writer.set_entry(row, ListIndex::new(start as u64, len as u64))?;
            }
            Ok(())
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
    MicroScalar::create(
        "vs_point_norm",
        |_context, input: Vector<'_, Point>, mut output: Vector<'_, f64>| {
            let points = input.row_reader()?;
            let mut writer = output.writer()?;
            for (row, point) in points.iter().enumerate() {
                let norm = point
                    .and_then(|point| Some((*point.x?, *point.y?)))
                    .map(|(x, y)| (x * x + y * y).sqrt());
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
                    (Some(&x), Some(&y)) => writer.set(row, &Point { x, y })?,
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
                    entries
                        .values()
                        .flatten()
                        .fold(0_i64, |sum, value| sum.wrapping_add(i64::from(*value)))
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
                for index in 0..len {
                    let x = f64::from(index);
                    writer.elements().set(start + index as usize, &Point { x, y: x * x })?;
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
