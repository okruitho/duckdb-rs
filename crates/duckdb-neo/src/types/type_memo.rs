//! Per-thread memos of what was read from recently validated logical types.

use std::{cell::RefCell, collections::VecDeque, thread::LocalKey};

use crate::{Result, error::check_api_call, ffi, logical_type::LogicalType};

const MEMO_CAPACITY: usize = 8;

/// Values read from a few logical types, found again by type equality.
///
/// DuckDB returns type parameters as owned values, allocating and, for child types, serializing them;
/// an equality test reads nothing. Only types that never share instance storage may be remembered.
pub(crate) struct TypeMemo<V> {
    entries: RefCell<VecDeque<(LogicalType, V)>>,
}

impl<V> TypeMemo<V> {
    pub(crate) const fn new() -> Self {
        Self {
            entries: RefCell::new(VecDeque::new()),
        }
    }
}

/// Return `read` of the first remembered value it accepts whose type equals `logical_type`.
pub(crate) fn memo_find<V: 'static, R>(
    memo: &'static LocalKey<TypeMemo<V>>,
    logical_type: &LogicalType,
    read: impl Fn(&V) -> Option<R>,
) -> Result<Option<R>> {
    memo.try_with(|memo| {
        let Ok(entries) = memo.entries.try_borrow() else {
            return Ok(None);
        };
        for (remembered, value) in entries.iter() {
            if let Some(found) = read(value)
                && check_api_call!(
                    ffi::duckdb_v2_logical_type_is_equal,
                    remembered.handle,
                    logical_type.handle,
                    RET
                )?
            {
                return Ok(Some(found));
            }
        }
        Ok(None)
    })
    .unwrap_or(Ok(None))
}

/// Remember `value` for `logical_type`, which must not share instance storage, e.g. through an `ENUM`.
pub(crate) fn memo_remember<V: 'static>(
    memo: &'static LocalKey<TypeMemo<V>>,
    logical_type: &LogicalType,
    value: V,
) -> Result<()> {
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

#[cfg(test)]
pub(crate) fn memo_len<V: 'static>(memo: &'static LocalKey<TypeMemo<V>>) -> usize {
    memo.with(|memo| memo.entries.borrow().len())
}
