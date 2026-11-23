//! The allocators of a statically linked extension: those it creates through [`Allocator`], and the Rust global
//! allocator.
//!
//! Registration is process-wide, and the counters behind `duckdb_extension_memory()` are shared by every instance, so
//! each test holds `LOCK` while it allocates.
#![cfg(feature = "build-extension-static")]

use std::sync::{Mutex, MutexGuard, Once, OnceLock};

use duckdb_neo::{
    Parameters,
    allocator::Allocator,
    connection::Connection,
    duckdb_entrypoint_c_api_v2,
    environment::{Environment, StorageLocation},
    error::DuckDBError,
    extension::{Extension, register_static_extension},
    instance::Instance,
    scalar::MicroScalar,
    vector::Vector,
};

const EXTENSION: &str = "neo_static_alloc";
const MIB: usize = 1 << 20;
const ALIGNMENTS: [usize; 7] = [1, 2, 8, 16, 64, 4096, 65536];
const SIZES: [usize; 8] = [1, 7, 15, 16, 17, 100, 4097, MIB];

static LOCK: Mutex<()> = Mutex::new(());
static EXTENSION_ALLOCATOR: OnceLock<Allocator> = OnceLock::new();
static CONTEXT_BLOCKS: Mutex<Vec<ContextBlock>> = Mutex::new(Vec::new());

/// A block allocated by `neo_context_alloc`, together with the allocator to free it with.
struct ContextBlock {
    allocator: Allocator,
    ptr: *mut u8,
    size: usize,
}

// SAFETY: the block is only freed through its own allocator, which is thread-safe.
unsafe impl Send for ContextBlock {}

// Registered first, so the global allocator is attributed to it rather than to `EXTENSION`.
#[duckdb_entrypoint_c_api_v2(extension_name = "neo_static_alloc_host", api_version = "v2.0.0")]
fn host_entrypoint(_extension: &Extension) -> duckdb_neo::Result<()> {
    Ok(())
}

#[duckdb_entrypoint_c_api_v2(extension_name = "neo_static_alloc", api_version = "v2.0.0")]
fn entrypoint(extension: &Extension) -> duckdb_neo::Result<()> {
    // Every instance loads the extension, but the allocator is process-wide: the first one serves them all.
    if EXTENSION_ALLOCATOR.get().is_none() {
        let _ = EXTENSION_ALLOCATOR.set(Allocator::new(extension)?);
    }
    // Allocates `size` bytes from an allocator of the executing database, and keeps them in `CONTEXT_BLOCKS`.
    MicroScalar::create(
        "neo_context_alloc",
        |context, input: Vector<'_, i64>, mut output: Vector<'_, i64>| {
            let mut writer = output.writer()?;
            for (i, size) in input.reader()?.iter().enumerate() {
                let size = *size.expect("size must not be NULL");
                let allocator = Allocator::new(context)?;
                let ptr = allocator.alloc(size as usize, 8)?;
                CONTEXT_BLOCKS.lock().unwrap().push(ContextBlock {
                    allocator,
                    ptr,
                    size: size as usize,
                });
                writer.set_option(i, Some(&size))?;
            }
            Ok(())
        },
    )
    .register(extension)
}

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn open() -> duckdb_neo::Result<Instance> {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // SAFETY: the describe functions come from `#[duckdb_entrypoint_c_api_v2]`.
        unsafe {
            register_static_extension(duckdb_extension_neo_static_alloc_host_describe).unwrap();
            register_static_extension(duckdb_extension_neo_static_alloc_describe).unwrap();
        }
    });
    Environment::new()?.open(StorageLocation::InMemory)
}

fn extension_allocator() -> &'static Allocator {
    EXTENSION_ALLOCATOR
        .get()
        .expect("an instance must be open to have loaded the extension")
}

fn query_i64(connection: &Connection, sql: &str) -> duckdb_neo::Result<i64> {
    let mut result = connection.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("query should return a chunk");
    Ok(*chunk
        .get_vector_at::<i64>(0)?
        .reader()?
        .get(0)?
        .expect("query should return a value"))
}

fn extension_memory(connection: &Connection, name: &str) -> duckdb_neo::Result<i64> {
    query_i64(
        connection,
        &format!("SELECT memory_usage_bytes FROM duckdb_extension_memory() WHERE extension_name = '{name}'"),
    )
}

fn extension_tag_memory(connection: &Connection) -> duckdb_neo::Result<i64> {
    query_i64(
        connection,
        "SELECT memory_usage_bytes FROM duckdb_memory() WHERE tag = 'EXTENSION'",
    )
}

fn fill_pattern(ptr: *mut u8, size: usize) {
    for i in 0..size {
        // SAFETY: the callers pass a block of at least `size` bytes.
        unsafe { ptr.add(i).write((i * 31 + 7) as u8) };
    }
}

fn has_pattern(ptr: *const u8, size: usize) -> bool {
    // SAFETY: the callers pass a block of at least `size` bytes.
    (0..size).all(|i| unsafe { ptr.add(i).read() } == (i * 31 + 7) as u8)
}

fn is_zeroed(ptr: *const u8, size: usize) -> bool {
    // SAFETY: the callers pass a block of at least `size` bytes.
    (0..size).all(|i| unsafe { ptr.add(i).read() } == 0)
}

fn is_aligned(ptr: *const u8, alignment: usize) -> bool {
    (ptr as usize).is_multiple_of(alignment)
}

#[test]
fn test_extension_allocator_round_trips() -> duckdb_neo::Result<()> {
    let _guard = lock();
    let _instance = open()?;
    let allocator = extension_allocator();
    for alignment in ALIGNMENTS {
        for size in SIZES {
            let ptr = allocator.alloc(size, alignment)?;
            assert!(is_aligned(ptr, alignment), "{size} bytes aligned to {alignment}");
            fill_pattern(ptr, size);
            // SAFETY: each block comes from `allocator` with the size and alignment it is passed with.
            unsafe {
                let grown = allocator.realloc(ptr, size, alignment, size * 3)?;
                assert!(is_aligned(grown, alignment));
                assert!(has_pattern(grown, size));
                let shrunk = allocator.realloc(grown, size * 3, alignment, size)?;
                assert!(has_pattern(shrunk, size));
                allocator.free(shrunk, size, alignment)?;

                let zeroed = allocator.calloc(size, alignment)?;
                assert!(is_aligned(zeroed, alignment));
                assert!(is_zeroed(zeroed, size));
                allocator.free(zeroed, size, alignment)?;
            }
        }
    }
    Ok(())
}

#[test]
fn test_extension_allocator_rejects_invalid_requests() -> duckdb_neo::Result<()> {
    let _guard = lock();
    let _instance = open()?;
    let allocator = extension_allocator();
    assert!(allocator.alloc(0, 8)?.is_null());
    for alignment in [0, 3, 24] {
        assert_eq!(
            allocator.alloc(8, alignment).unwrap_err().code,
            DuckDBError::InputInvalid
        );
    }
    assert_eq!(
        allocator.alloc(1 << 62, 8).unwrap_err().code,
        DuckDBError::ResourceOutOfMemory
    );

    let ptr = allocator.alloc(64, 8)?;
    fill_pattern(ptr, 64);
    // SAFETY: `ptr` comes from `allocator` with 64 bytes aligned to 8, and stays valid when realloc fails.
    unsafe {
        assert_eq!(
            allocator.realloc(ptr, 64, 8, 1 << 62).unwrap_err().code,
            DuckDBError::ResourceOutOfMemory
        );
        assert!(has_pattern(ptr, 64));
        allocator.free(ptr, 64, 8)?;
        allocator.free(std::ptr::null_mut(), 0, 8)?;
    }
    Ok(())
}

#[test]
fn test_extension_allocator_is_attributed_to_its_extension() -> duckdb_neo::Result<()> {
    let _guard = lock();
    let connection = open()?.connect()?;
    let allocator = extension_allocator();
    let baseline = extension_memory(&connection, EXTENSION)?;

    let ptr = allocator.alloc(MIB, 8)?;
    assert_eq!(extension_memory(&connection, EXTENSION)?, baseline + MIB as i64);

    // SAFETY: `ptr` comes from `allocator` with `MIB` bytes aligned to 8.
    let ptr = unsafe { allocator.realloc(ptr, MIB, 8, 2 * MIB)? };
    assert_eq!(extension_memory(&connection, EXTENSION)?, baseline + 2 * MIB as i64);

    // The counter is process-wide, and the memory counts towards no database.
    let other = open()?.connect()?;
    assert_eq!(extension_memory(&other, EXTENSION)?, baseline + 2 * MIB as i64);

    // SAFETY: as above, after the realloc.
    unsafe { allocator.free(ptr, 2 * MIB, 8)? };
    assert_eq!(extension_memory(&connection, EXTENSION)?, baseline);
    Ok(())
}

#[test]
fn test_extension_allocator_outlives_every_instance() -> duckdb_neo::Result<()> {
    let _guard = lock();
    let baseline = extension_memory(&open()?.connect()?, EXTENSION)?;
    let allocator = extension_allocator();
    // No instance is open any more.
    let ptr = allocator.alloc(MIB, 8)?;
    fill_pattern(ptr, MIB);
    assert_eq!(extension_memory(&open()?.connect()?, EXTENSION)?, baseline + MIB as i64);
    assert!(has_pattern(ptr, MIB));
    // SAFETY: `ptr` comes from `allocator` with `MIB` bytes aligned to 8.
    unsafe { allocator.free(ptr, MIB, 8)? };
    assert_eq!(extension_memory(&open()?.connect()?, EXTENSION)?, baseline);
    Ok(())
}

#[test]
fn test_connection_allocator_respects_the_memory_limit() -> duckdb_neo::Result<()> {
    let _guard = lock();
    let instance = open()?;
    let connection = instance.connect()?;
    connection.execute("SET memory_limit = '16MiB'", Parameters::None)?;
    let allocator = Allocator::new(&connection)?;
    let baseline = extension_tag_memory(&connection)?;

    assert_eq!(
        allocator.alloc(32 * MIB, 8).unwrap_err().code,
        DuckDBError::ResourceOutOfMemory
    );
    let ptr = allocator.alloc(MIB, 8)?;
    fill_pattern(ptr, MIB);
    assert_eq!(extension_tag_memory(&connection)?, baseline + MIB as i64);
    // SAFETY: `ptr` comes from `allocator` with `MIB` bytes aligned to 8, and stays valid when realloc fails.
    unsafe {
        assert_eq!(
            allocator.realloc(ptr, MIB, 8, 32 * MIB).unwrap_err().code,
            DuckDBError::ResourceOutOfMemory
        );
    }
    assert!(has_pattern(ptr, MIB));
    assert_eq!(extension_tag_memory(&connection)?, baseline + MIB as i64);

    // The allocator keeps the database alive, not the connection.
    drop(connection);
    // SAFETY: as above.
    unsafe { allocator.free(ptr, MIB, 8)? };
    assert_eq!(extension_tag_memory(&instance.connect()?)?, baseline);
    Ok(())
}

#[test]
fn test_context_allocator_charges_its_database_and_outlives_the_query() -> duckdb_neo::Result<()> {
    let _guard = lock();
    let connection = open()?.connect()?;
    let baseline = extension_tag_memory(&connection)?;

    // A column input, so that the call is not folded into a constant.
    let size = query_i64(
        &connection,
        &format!("SELECT neo_context_alloc({MIB} + i) FROM range(1) t(i)"),
    )?;
    assert_eq!(size, MIB as i64);
    assert_eq!(extension_tag_memory(&connection)?, baseline + MIB as i64);

    let blocks = std::mem::take(&mut *CONTEXT_BLOCKS.lock().unwrap());
    assert_eq!(blocks.len(), 1);
    for block in blocks {
        // SAFETY: each block comes from its allocator with `size` bytes aligned to 8.
        unsafe { block.allocator.free(block.ptr, block.size, 8)? };
    }
    assert_eq!(extension_tag_memory(&connection)?, baseline);
    Ok(())
}

#[cfg(feature = "global_allocator")]
mod global_allocator {
    use super::*;
    use std::alloc::{Layout, alloc, alloc_zeroed, dealloc, realloc};

    const HOST_EXTENSION: &str = "neo_static_alloc_host";
    // Rust allocations of DuckDB's own result handling while a test queries the counter.
    const SLACK: i64 = MIB as i64;

    #[test]
    fn test_global_allocator_serves_every_layout() -> duckdb_neo::Result<()> {
        let _guard = lock();
        for alignment in ALIGNMENTS {
            for size in SIZES {
                let layout = Layout::from_size_align(size, alignment).unwrap();
                let grown_layout = Layout::from_size_align(size * 3, alignment).unwrap();
                // SAFETY: every block is freed or resized with the layout it was allocated with.
                unsafe {
                    let ptr = alloc(layout);
                    assert!(!ptr.is_null());
                    assert!(is_aligned(ptr, alignment), "{size} bytes aligned to {alignment}");
                    fill_pattern(ptr, size);
                    let grown = realloc(ptr, layout, size * 3);
                    assert!(is_aligned(grown, alignment));
                    assert!(has_pattern(grown, size));
                    let shrunk = realloc(grown, grown_layout, size);
                    assert!(has_pattern(shrunk, size));
                    dealloc(shrunk, layout);

                    let zeroed = alloc_zeroed(layout);
                    assert!(is_aligned(zeroed, alignment));
                    assert!(is_zeroed(zeroed, size));
                    dealloc(zeroed, layout);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_global_allocator_is_attributed_to_the_first_extension() -> duckdb_neo::Result<()> {
        let _guard = lock();
        let connection = open()?.connect()?;
        let before = extension_memory(&connection, HOST_EXTENSION)?;
        let explicit_before = extension_memory(&connection, EXTENSION)?;

        let block = vec![1_u8; 8 * MIB];
        let zeroed = vec![0_u8; 8 * MIB];
        let during = extension_memory(&connection, HOST_EXTENSION)?;
        assert!(
            (16 * MIB as i64..16 * MIB as i64 + SLACK).contains(&(during - before)),
            "grew by {}",
            during - before
        );
        // A later extension cannot claim it.
        assert_eq!(extension_memory(&connection, EXTENSION)?, explicit_before);

        drop((block, zeroed));
        let after = extension_memory(&connection, HOST_EXTENSION)?;
        assert!(
            (during - after - 16 * MIB as i64).abs() < SLACK,
            "shrank by {}",
            during - after
        );
        Ok(())
    }

    #[test]
    fn test_global_allocator_counts_frees_on_other_threads() -> duckdb_neo::Result<()> {
        let _guard = lock();
        let connection = open()?.connect()?;
        let before = extension_memory(&connection, HOST_EXTENSION)?;

        // More threads than DuckDB keeps per-thread counters for, each freeing blocks another one allocated.
        let (sender, receiver) = std::sync::mpsc::channel::<Vec<u8>>();
        let receiver = std::sync::Arc::new(Mutex::new(receiver));
        let threads: Vec<_> = (0..300)
            .map(|i| {
                let sender = sender.clone();
                let receiver = receiver.clone();
                std::thread::spawn(move || {
                    for round in 0..100 {
                        let strings: Vec<String> = (0..10).map(|j| format!("{i}-{round}-{j}")).collect();
                        sender.send(vec![i as u8; 1000 + round]).unwrap();
                        let block = receiver.lock().unwrap().recv().unwrap();
                        assert!(block.iter().all(|&byte| byte == block[0]));
                        drop(strings);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        drop((sender, receiver));

        let after = extension_memory(&connection, HOST_EXTENSION)?;
        assert!((after - before).abs() < SLACK, "changed by {}", after - before);
        Ok(())
    }
}
