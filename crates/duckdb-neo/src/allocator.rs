use duckdb_loadable_macros::{AsRaw, FromRaw};

use crate::{
    Result,
    connection::{Connection, Context},
    error::{check_api_call, check_api_call_no_err},
    extension::Extension,
    ffi,
    raw::RawExt,
};
use std::ffi::c_void;

trait HasAllocator {
    fn create_allocator(&self) -> Result<ffi::duckdb_v2_allocator_handle>;
}

impl HasAllocator for Extension {
    fn create_allocator(&self) -> Result<ffi::duckdb_v2_allocator_handle> {
        check_api_call!(ffi::duckdb_v2_allocator_create_with_extension, self.raw(), RET)
    }
}

impl HasAllocator for Connection {
    fn create_allocator(&self) -> Result<ffi::duckdb_v2_allocator_handle> {
        check_api_call!(ffi::duckdb_v2_allocator_create_with_connection, self.raw(), RET)
    }
}

impl HasAllocator for Context {
    fn create_allocator(&self) -> Result<ffi::duckdb_v2_allocator_handle> {
        check_api_call!(ffi::duckdb_v2_allocator_create_with_context, self.raw(), RET)
    }
}

#[derive(Debug, AsRaw, FromRaw)]
pub struct Allocator {
    pub(crate) handle: ffi::duckdb_v2_allocator_handle,
}

impl Allocator {
    #[allow(private_bounds)]
    pub fn new<C: HasAllocator>(link: &C) -> Result<Self> {
        link.create_allocator().map(|handle| Allocator { handle })
    }

    /// Allocates `size` uninitialized bytes aligned to `alignment`, which must be a power of two. A `size` of 0 yields
    /// NULL. Release the memory with [`free`](Self::free) on this allocator, passing the same `size` and `alignment`.
    #[inline]
    pub fn alloc(&self, size: usize, alignment: usize) -> Result<*mut u8> {
        check_api_call!(
            ffi::duckdb_v2_allocator_alloc,
            self.raw(),
            size as u64,
            alignment as u64,
            RET
        )
        .map(|x| x.cast())
    }

    /// Like [`alloc`](Self::alloc), but the memory is zeroed.
    #[inline]
    pub fn calloc(&self, size: usize, alignment: usize) -> Result<*mut u8> {
        check_api_call!(
            ffi::duckdb_v2_allocator_alloc_zeroed,
            self.raw(),
            size as u64,
            alignment as u64,
            RET
        )
        .map(|x| x.cast())
    }

    /// # Safety
    /// `ptr` must be NULL or come from this allocator with `size` and `alignment`, and must not be used afterwards.
    #[inline]
    pub unsafe fn free(&self, ptr: *mut u8, size: usize, alignment: usize) -> Result<()> {
        check_api_call_no_err!(
            ffi::duckdb_v2_allocator_dealloc,
            self.raw(),
            ptr.cast::<c_void>(),
            size as u64,
            alignment as u64
        )
    }

    /// On failure `ptr` stays valid.
    ///
    /// # Safety
    /// `ptr` must be NULL with a `size` of 0, or come from this allocator with `size` and `alignment`. On success it
    /// must not be used afterwards.
    #[inline]
    pub unsafe fn realloc(&self, ptr: *mut u8, size: usize, alignment: usize, new_size: usize) -> Result<*mut u8> {
        check_api_call!(
            ffi::duckdb_v2_allocator_realloc,
            self.raw(),
            ptr.cast(),
            size as u64,
            alignment as u64,
            new_size as u64,
            RET
        )
        .map(|x| x.cast())
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        check_api_call_no_err!(ffi::duckdb_v2_allocator_destroy, &mut self.raw()).unwrap()
    }
}

unsafe impl Sync for Allocator {}
unsafe impl Send for Allocator {}

#[cfg(feature = "global_allocator")]
pub mod global_allocator {
    use crate::ffi;
    use std::{alloc::GlobalAlloc, ffi::c_void, ptr::null_mut, sync::OnceLock};

    /// The raw functions of an allocator that is never destroyed.
    struct RawFunctions {
        handle: ffi::duckdb_v2_allocator_handle,
        state: *mut c_void,
        alloc: unsafe extern "C" fn(*mut c_void, u64, u64) -> *mut c_void,
        alloc_zeroed: unsafe extern "C" fn(*mut c_void, u64, u64) -> *mut c_void,
        dealloc: unsafe extern "C" fn(*mut c_void, *mut c_void, u64, u64),
        realloc: unsafe extern "C" fn(*mut c_void, *mut c_void, u64, u64, u64) -> *mut c_void,
    }

    // SAFETY: DuckDB's allocator functions are thread-safe, and their state lives for the rest of the process.
    unsafe impl Send for RawFunctions {}
    unsafe impl Sync for RawFunctions {}

    static RAW_FUNCTIONS: OnceLock<RawFunctions> = OnceLock::new();

    #[derive(Debug)]
    pub struct GlobalAllocator;

    impl GlobalAllocator {
        #[inline]
        fn raw() -> &'static RawFunctions {
            // Rust's runtime allocates before `main`, and any allocation from here would re-enter this allocator.
            RAW_FUNCTIONS.get_or_init(|| {
                let mut handle = null_mut();
                let mut raw = ffi::duckdb_v2_allocator_raw_functions {
                    state: null_mut(),
                    alloc: None,
                    alloc_zeroed: None,
                    dealloc: None,
                    realloc: None,
                };
                // SAFETY: a NULL extension creates an allocator that is attributed later; no error info is requested.
                // The allocator is never destroyed, so its functions stay valid.
                let created = unsafe {
                    ffi::duckdb_v2_allocator_create_with_extension(null_mut(), &mut handle, null_mut())
                        == ffi::DUCKDB_V2_ERROR::DUCKDB_V2_ERROR_NONE
                        && ffi::duckdb_v2_allocator_get_raw_functions(handle, &mut raw, null_mut())
                            == ffi::DUCKDB_V2_ERROR::DUCKDB_V2_ERROR_NONE
                };
                let (true, Some(alloc), Some(alloc_zeroed), Some(dealloc), Some(realloc)) =
                    (created, raw.alloc, raw.alloc_zeroed, raw.dealloc, raw.realloc)
                else {
                    std::process::abort();
                };
                RawFunctions {
                    handle,
                    state: raw.state,
                    alloc,
                    alloc_zeroed,
                    dealloc,
                    realloc,
                }
            })
        }
    }

    /// Attributes the global allocator to `extension` in `duckdb_extension_memory()`, unless an earlier extension
    /// claimed it: in a statically linked binary it serves the whole process, not one extension.
    pub(crate) fn register_extension(extension: ffi::duckdb_v2_extension_handle) {
        // SAFETY: the allocator is never destroyed and `extension` is loading. A later extension is refused with
        // INPUT_INVALID, which leaves the first one in place; no error info is requested, so nothing is allocated.
        unsafe {
            ffi::duckdb_v2_allocator_register_extension(GlobalAllocator::raw().handle, extension, null_mut());
        }
    }

    // GlobalAlloc guarantees a non-zero size and a power-of-two alignment, which is all the raw functions require.
    unsafe impl GlobalAlloc for GlobalAllocator {
        #[inline]
        unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
            let raw = Self::raw();
            // SAFETY: see above.
            unsafe { (raw.alloc)(raw.state, layout.size() as u64, layout.align() as u64).cast() }
        }

        #[inline]
        unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
            let raw = Self::raw();
            // SAFETY: see above.
            unsafe { (raw.alloc_zeroed)(raw.state, layout.size() as u64, layout.align() as u64).cast() }
        }

        #[inline]
        unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
            let raw = Self::raw();
            // SAFETY: `ptr` came from this allocator with `layout`.
            unsafe { (raw.dealloc)(raw.state, ptr.cast(), layout.size() as u64, layout.align() as u64) }
        }

        #[inline]
        unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
            let raw = Self::raw();
            // SAFETY: `ptr` came from this allocator with `layout`, and `new_size` is non-zero.
            unsafe {
                (raw.realloc)(
                    raw.state,
                    ptr.cast(),
                    layout.size() as u64,
                    layout.align() as u64,
                    new_size as u64,
                )
                .cast()
            }
        }
    }

    #[global_allocator]
    pub static ALLOCATOR: GlobalAllocator = GlobalAllocator;
}
