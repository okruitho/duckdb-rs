//! Passing strings and bytes through the C API.

use std::marker::PhantomData;
use std::os::raw::c_char;

use crate::{AsRaw, ffi, raw::RawExt};

/// Return the `(pointer, length)` bytes DuckDB handed back; a null pointer reads as empty.
///
/// The caller picks `'a`: tie it to the owning handle, or `'static` for static strings.
pub(crate) fn bytes_from_tuple<'a>((ptr, len): (*const c_char, ffi::idx_t)) -> &'a [u8] {
    if ptr.is_null() || len == 0 {
        return &[];
    }
    // SAFETY: the C API returns `len` bytes that stay valid while their owner is borrowed.
    unsafe { std::slice::from_raw_parts(ptr.cast(), len as usize) }
}

/// An item the C API takes in a pointer array: a handle, or a string with a parallel length.
pub(crate) trait HasPtr {
    type Ptr: Copy;

    fn ptr(&self) -> Self::Ptr;

    /// The byte length for the parallel lengths array; `None` for items without one, such as handles.
    fn length(&self) -> Option<ffi::idx_t> {
        None
    }
}

impl HasPtr for str {
    type Ptr = *const c_char;

    fn ptr(&self) -> Self::Ptr {
        self.as_ptr().cast()
    }

    fn length(&self) -> Option<ffi::idx_t> {
        Some(self.len() as ffi::idx_t)
    }
}

impl HasPtr for &str {
    type Ptr = *const c_char;

    fn ptr(&self) -> Self::Ptr {
        (**self).ptr()
    }

    fn length(&self) -> Option<ffi::idx_t> {
        (**self).length()
    }
}

impl HasPtr for String {
    type Ptr = *const c_char;

    fn ptr(&self) -> Self::Ptr {
        self.as_str().ptr()
    }

    fn length(&self) -> Option<ffi::idx_t> {
        self.as_str().length()
    }
}

impl<T: AsRaw> HasPtr for T {
    type Ptr = T::Raw;

    fn ptr(&self) -> Self::Ptr {
        self.raw()
    }
}

/// Borrow items as the C API's parallel pointer and length arrays; both are null when empty.
///
/// Handles fill only the pointer array, so [`Self::lengths`] stays null for them.
pub(crate) struct DecomposedArrays<'a, T: HasPtr + ?Sized> {
    ptrs: Vec<T::Ptr>,
    child_lengths: Vec<ffi::idx_t>,
    _items: PhantomData<&'a T>,
}

impl<'a, T: HasPtr + ?Sized + 'a> DecomposedArrays<'a, T> {
    pub(crate) fn new(items: impl IntoIterator<Item = &'a T>) -> Self {
        let mut ptrs = Vec::new();
        let mut child_lengths = Vec::new();
        for item in items {
            ptrs.push(item.ptr());
            child_lengths.extend(item.length());
        }
        DecomposedArrays {
            ptrs,
            child_lengths,
            _items: PhantomData,
        }
    }

    pub(crate) fn ptrs(&self) -> *const T::Ptr {
        if self.ptrs.is_empty() {
            std::ptr::null()
        } else {
            self.ptrs.as_ptr()
        }
    }

    pub(crate) fn child_lengths(&self) -> *const ffi::idx_t {
        if self.child_lengths.is_empty() {
            std::ptr::null()
        } else {
            self.child_lengths.as_ptr()
        }
    }

    pub(crate) fn len(&self) -> ffi::idx_t {
        self.ptrs.len() as ffi::idx_t
    }
}

/// Return the bytes of a `duckdb_v2_bytes` read from a vector.
///
/// `value` must come from DuckDB, so its pointer form points to `length` bytes valid while it is borrowed.
pub(crate) fn bytes_view(value: &ffi::duckdb_v2_bytes) -> &[u8] {
    // SAFETY: the inline form holds up to 12 bytes in place; longer values use the pointer form.
    unsafe {
        if value.value.inlined.length <= 12 {
            let len = value.value.inlined.length as usize;
            std::slice::from_raw_parts(value.value.inlined.inlined.as_ptr().cast(), len)
        } else {
            let len = value.value.pointer.length as usize;
            std::slice::from_raw_parts(value.value.pointer.ptr.cast(), len)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DecomposedArrays, bytes_from_tuple};
    use crate::{
        AsRaw, ToValue,
        environment::{Environment, StorageLocation},
    };

    #[test]
    fn null_tuple_is_empty() {
        assert_eq!(bytes_from_tuple((std::ptr::null(), 0)), b"");
        assert_eq!(bytes_from_tuple((std::ptr::null(), 5)), b"");
        assert_eq!(crate::str_from_tuple((std::ptr::null(), 0)), "");
    }

    #[test]
    fn decomposed_arrays_split_strings() {
        let strings = ["ab", "", "cde"];
        let arrays = DecomposedArrays::new(&strings);
        assert_eq!(arrays.len(), 3);
        // SAFETY: both arrays hold `len` entries borrowed from `strings`.
        let (ptrs, lengths) = unsafe {
            (
                std::slice::from_raw_parts(arrays.ptrs(), 3),
                std::slice::from_raw_parts(arrays.child_lengths(), 3),
            )
        };
        assert_eq!(lengths, [2, 0, 3]);
        assert_eq!(ptrs[2], strings[2].as_ptr().cast());

        let empty = DecomposedArrays::new(&[] as &[&str]);
        assert!(empty.ptrs().is_null());
        assert!(empty.child_lengths().is_null());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn decomposed_arrays_take_handles() -> crate::Result<()> {
        let conn = Environment::new()?.open(StorageLocation::InMemory)?.connect()?;
        let values = [1_i32.value(&conn)?, 2_i32.value(&conn)?];
        let arrays = DecomposedArrays::new(&values);
        assert_eq!(arrays.len(), 2);
        assert!(arrays.child_lengths().is_null());
        // SAFETY: the pointer array holds `len` handles borrowed from `values`.
        let ptrs = unsafe { std::slice::from_raw_parts(arrays.ptrs(), 2) };
        // SAFETY: the handles are only compared.
        assert_eq!(ptrs, unsafe { [values[0].get_raw(), values[1].get_raw()] });
        Ok(())
    }
}
