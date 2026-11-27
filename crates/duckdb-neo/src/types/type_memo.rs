//! Per-thread memos of what was read from recently validated logical types.

use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    thread::LocalKey,
};

use crate::{Result, enums::LogicalTypeID, error::check_api_call, ffi, logical_type::LogicalType};

/// Entries per memo; a lookup costs up to one equality test per entry, so this stays small.
const MEMO_CAPACITY: usize = 16;

/// A full memo admits one in this many new types, so cycling through more types than it holds
/// costs little more than reading them without a memo.
const FULL_MEMO_ADMISSION: u32 = 16;

/// Values read from recently seen logical types, found again by type equality instead of re-reading parameters.
pub(crate) struct TypeMemo<V> {
    entries: RefCell<VecDeque<(LogicalType, V)>>,
    refused: Cell<u32>,
}

impl<V> TypeMemo<V> {
    pub(crate) const fn new() -> Self {
        Self {
            entries: RefCell::new(VecDeque::new()),
            refused: Cell::new(0),
        }
    }

    fn admits_another(&self) -> bool {
        let Ok(entries) = self.entries.try_borrow() else {
            return false;
        };
        if entries.len() < MEMO_CAPACITY {
            return true;
        }
        let refused = self.refused.get().wrapping_add(1);
        self.refused.set(refused);
        refused.is_multiple_of(FULL_MEMO_ADMISSION)
    }
}

/// Return `read` of the first remembered value it accepts whose type equals `logical_type`, and move
/// that entry to the front.
pub(crate) fn memo_find<V: 'static, R>(
    memo: &'static LocalKey<TypeMemo<V>>,
    logical_type: &LogicalType,
    read: impl Fn(&V) -> Option<R>,
) -> Result<Option<R>> {
    memo.try_with(|memo| {
        let Ok(mut entries) = memo.entries.try_borrow_mut() else {
            return Ok(None);
        };
        for index in 0..entries.len() {
            let (remembered, value) = &entries[index];
            if let Some(found) = read(value)
                && check_api_call!(
                    ffi::duckdb_v2_logical_type_is_equal,
                    remembered.handle,
                    logical_type.handle,
                    RET
                )?
            {
                if index > 0
                    && let Some(entry) = entries.remove(index)
                {
                    entries.push_front(entry);
                }
                return Ok(Some(found));
            }
        }
        Ok(None)
    })
    .unwrap_or(Ok(None))
}

/// Remember `value` for `logical_type` if it is a built-in type without an alias: the memo outlives every
/// instance, and catalog types such as an `ENUM` share instance storage.
pub(crate) fn memo_remember<V: 'static>(
    memo: &'static LocalKey<TypeMemo<V>>,
    logical_type: &LogicalType,
    value: V,
) -> Result<()> {
    if !memo.try_with(TypeMemo::admits_another).unwrap_or(false) || !is_builtin(logical_type)? {
        return Ok(());
    }
    let copy = LogicalType::copy_handle(&mut { logical_type.handle })?;
    // A destroyed or busy memo only loses the entry.
    let _ = memo.try_with(|memo| {
        if let Ok(mut entries) = memo.entries.try_borrow_mut() {
            entries.truncate(MEMO_CAPACITY - 1);
            entries.push_front((copy, value));
        }
    });
    Ok(())
}

/// Whether `logical_type` and every type among its parameters is a built-in kind under its canonical name.
fn is_builtin(logical_type: &LogicalType) -> Result<bool> {
    let Some(name) = logical_type.id().ok().and_then(canonical_name) else {
        return Ok(false);
    };
    // DuckDB reports the alias of a user-defined or extension type in place of the canonical name.
    if logical_type.name()? != name {
        return Ok(false);
    }
    for index in 0..logical_type.param_count()? {
        let (_, param) = logical_type.get_param(index)?;
        if param.logical_type()?.id()? == LogicalTypeID::Type && !is_builtin(&param.type_value_get_inner()?)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The name DuckDB gives a type without an alias, for the kinds that never share instance storage.
fn canonical_name(id: LogicalTypeID) -> Option<&'static str> {
    use LogicalTypeID::*;
    Some(match id {
        Boolean => "BOOLEAN",
        TinyInt => "TINYINT",
        SmallInt => "SMALLINT",
        Integer => "INTEGER",
        BigInt => "BIGINT",
        HugeInt => "HUGEINT",
        UTinyInt => "UTINYINT",
        USmallInt => "USMALLINT",
        UInteger => "UINTEGER",
        UBigInt => "UBIGINT",
        UHugeInt => "UHUGEINT",
        Float => "FLOAT",
        Double => "DOUBLE",
        Decimal => "DECIMAL",
        Varchar => "VARCHAR",
        Blob => "BLOB",
        Bit => "BIT",
        BigNum => "BIGNUM",
        Uuid => "UUID",
        Interval => "INTERVAL",
        Date => "DATE",
        Time => "TIME",
        TimeNs => "TIME_NS",
        TimeTz => "TIME WITH TIME ZONE",
        Timestamp => "TIMESTAMP",
        TimestampSec => "TIMESTAMP_S",
        TimestampMs => "TIMESTAMP_MS",
        TimestampNs => "TIMESTAMP_NS",
        TimestampTz => "TIMESTAMP WITH TIME ZONE",
        TimestampTzNs => "TIMESTAMPTZ_NS",
        Struct => "STRUCT",
        Tuple => "TUPLE",
        List => "LIST",
        Map => "MAP",
        Array => "ARRAY",
        Union => "UNION",
        _ => return None,
    })
}

#[cfg(test)]
pub(crate) fn memo_len<V: 'static>(memo: &'static LocalKey<TypeMemo<V>>) -> usize {
    memo.with(|memo| memo.entries.borrow().len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Parameters,
        environment::{Environment, StorageLocation},
    };

    #[test]
    fn builtin_types_are_recognized_by_their_canonical_names() -> Result<()> {
        let env = Environment::new()?;
        let conn = env.open(StorageLocation::InMemory)?.connect()?;
        for text in [
            "BOOLEAN",
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
            "UHUGEINT",
            "FLOAT",
            "DOUBLE",
            "DECIMAL(18, 3)",
            "VARCHAR",
            "BLOB",
            "BIT",
            "BIGNUM",
            "UUID",
            "INTERVAL",
            "DATE",
            "TIME",
            "TIME_NS",
            "TIMETZ",
            "TIMESTAMP",
            "TIMESTAMP_S",
            "TIMESTAMP_MS",
            "TIMESTAMP_NS",
            "TIMESTAMPTZ",
            "TIMESTAMPTZ_NS",
            "STRUCT(a INTEGER, b VARCHAR[])",
            "MAP(VARCHAR, DOUBLE[3])",
            "UNION(num INTEGER, str VARCHAR)",
        ] {
            assert!(is_builtin(&LogicalType::from_text(&conn, text)?)?, "{text}");
        }
        conn.execute("CREATE TYPE mood AS ENUM ('sad', 'happy')", Parameters::None)?;
        conn.execute("CREATE TYPE money AS DECIMAL(18, 3)", Parameters::None)?;
        conn.execute("CREATE TYPE point AS STRUCT(x DOUBLE, y DOUBLE)", Parameters::None)?;
        // A catalog alias of a built-in type binds to the built-in type itself, without the alias.
        for text in ["money", "point", "STRUCT(m money, p point)"] {
            assert!(is_builtin(&LogicalType::from_text(&conn, text)?)?, "{text}");
        }
        for text in [
            "mood",
            "STRUCT(a mood)",
            "STRUCT(xs mood[])",
            "MAP(VARCHAR, mood)",
            "UNION(num INTEGER, m mood)",
            "VARIANT",
            "GEOMETRY",
        ] {
            assert!(!is_builtin(&LogicalType::from_text(&conn, text)?)?, "{text}");
        }
        let aliased = LogicalType::from_text(&conn, "DECIMAL(18, 3)")?.with_alias(&conn, "amount")?;
        assert!(!is_builtin(&aliased)?);
        assert!(!is_builtin(
            &LogicalType::from_text(&conn, "STRUCT(a INTEGER)")?.with_alias(&conn, "row")?
        )?);
        let list = LogicalType::create(&conn, "LIST", Parameters::positional(&[&aliased]))?;
        assert!(!is_builtin(&list)?);
        Ok(())
    }
}
