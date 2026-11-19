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

    #[inline]
    pub fn free(&self, ptr: *mut c_void, size: usize, alignment: usize) -> Result<()> {
        check_api_call_no_err!(
            ffi::duckdb_v2_allocator_dealloc,
            self.raw(),
            ptr,
            size as u64,
            alignment as u64
        )
    }

    #[inline]
    pub fn realloc(&self, ptr: *mut c_void, size: usize, alignment: usize, new_size: usize) -> Result<*mut u8> {
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
    use crate::allocator::Allocator;
    use crate::ffi;
    use std::{alloc::GlobalAlloc, sync::OnceLock};

    static ALLOCATOR_INNER: OnceLock<Allocator> = OnceLock::new();

    #[derive(Debug)]
    pub struct GlobalAllocator;

    impl GlobalAllocator {
        fn inner() -> &'static Allocator {
            // Rust's runtime allocates before `main`, and any allocation from here would re-enter this allocator.
            ALLOCATOR_INNER.get_or_init(|| {
                let mut handle = std::ptr::null_mut();
                // SAFETY: a NULL extension creates an allocator that is attributed later; no error info is requested.
                let code = unsafe {
                    ffi::duckdb_v2_allocator_create_with_extension(
                        std::ptr::null_mut(),
                        &mut handle,
                        std::ptr::null_mut(),
                    )
                };
                if code != ffi::DUCKDB_V2_ERROR::DUCKDB_V2_ERROR_NONE {
                    std::process::abort();
                }
                Allocator { handle }
            })
        }
    }

    unsafe impl GlobalAlloc for GlobalAllocator {
        unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
            Self::inner().alloc(layout.size(), layout.align()).unwrap()
        }

        unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
            Self::inner().calloc(layout.size(), layout.align()).unwrap()
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
            Self::inner().free(ptr.cast(), layout.size(), layout.align()).unwrap()
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
            Self::inner()
                .realloc(ptr.cast(), layout.size(), layout.align(), new_size)
                .unwrap()
        }
    }

    #[global_allocator]
    pub static ALLOCATOR: GlobalAllocator = GlobalAllocator;
}
