use std::marker::PhantomData;

use crate::{
    Result,
    error::{DuckDBError, Error},
    ffi,
    vector::Arena,
};

/// DuckDB's internal representation of `VARCHAR`-like data, such as bytes and strings.
///
/// Short values are stored inline. Longer values refer to memory owned by the
/// DuckDB arena they were written into, so that arena must remain
/// alive while the bytes are accessed.
///
/// Use [`Arena::add_bytes`] or [`Arena::add_str`] to encode values for a vector's arena.
/// Construction borrows the arena, not the input:
///
/// ```
/// use duckdb_neo::vector::{Arena, DuckDBBytes};
///
/// fn encode(arena: &Arena) -> duckdb_neo::Result<DuckDBBytes<'_>> {
///     DuckDBBytes::new(arena, b"value")
/// }
/// ```
///
/// The encoded value cannot outlive its arena:
///
/// ```compile_fail
/// use duckdb_neo::vector::{Arena, DuckDBBytes};
///
/// fn escape(arena: &Arena) -> duckdb_neo::Result<DuckDBBytes<'static>> {
///     DuckDBBytes::new(arena, b"a value stored in the arena")
/// }
/// ```
///
/// Borrowed bytes prevent resizing the owning vector:
///
/// ```compile_fail,E0499
/// use duckdb_neo::{Result, vector::Vector};
///
/// fn resize(vector: &mut Vector<'_, String>) -> Result<()> {
///     let mut writer = vector.writer()?;
///     let bytes = writer.arena()?.add_str("a value stored in the arena")?;
///     vector.set_size(10)?;
///     assert_eq!(bytes.get_data(), b"a value stored in the arena");
///     Ok(())
/// }
/// ```
///
/// Nor can the writer replace storage while bytes are borrowed:
///
/// ```compile_fail,E0499
/// use duckdb_neo::{Result, vector::Vector};
///
/// fn overwrite(vector: &mut Vector<'_, String>) -> Result<()> {
///     let mut writer = vector.writer()?;
///     let bytes = writer.arena()?.add_str("a value stored in the arena")?;
///     writer.set(0, "replacement")?;
///     assert_eq!(bytes.get_data(), b"a value stored in the arena");
///     Ok(())
/// }
/// ```
#[repr(transparent)]
pub struct DuckDBBytes<'arena> {
    data: ffi::duckdb_v2_bytes,
    _marker: PhantomData<&'arena ()>,
}

impl std::fmt::Debug for DuckDBBytes<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuckDBBytes")
            .field("size", &self.size())
            .finish_non_exhaustive()
    }
}

type DuckDBV2BytesPointer = ffi::duckdb_v2_bytes__bindgen_ty_1__bindgen_ty_1;
type DuckDBV2BytesInlined = ffi::duckdb_v2_bytes__bindgen_ty_1__bindgen_ty_2;
impl<'arena> DuckDBBytes<'arena> {
    /// Encode bytes in DuckDB's internal representation, copying long values into `arena`.
    pub fn new(arena: &'arena Arena, value: &[u8]) -> Result<Self> {
        Self::from_parts(&[value], || Ok(arena))
    }

    /// Encode `parts` concatenated, copying long values into the arena `arena` returns.
    #[inline(always)]
    pub(crate) fn from_parts<P: AsRef<[u8]> + ?Sized>(
        parts: &[&P],
        arena: impl FnOnce() -> Result<&'arena Arena>,
    ) -> Result<Self> {
        let mut bytes = std::mem::MaybeUninit::uninit();
        // SAFETY: the local is valid for one write, which write_parts performs on success.
        unsafe {
            Self::write_parts(bytes.as_mut_ptr(), parts, arena)?;
            Ok(bytes.assume_init())
        }
    }

    /// As [`Self::from_parts`], encoding straight into `slot`.
    ///
    /// Building a value elsewhere and copying it would reload fresh stores with a wider load,
    /// which stalls store forwarding.
    ///
    /// # Safety
    /// `slot` must be valid for writes of one `DuckDBBytes`.
    #[inline(always)]
    pub(crate) unsafe fn write_parts<P: AsRef<[u8]> + ?Sized>(
        slot: *mut Self,
        parts: &[&P],
        arena: impl FnOnce() -> Result<&'arena Arena>,
    ) -> Result<()> {
        let length = total_length(parts)?;
        if length <= ffi::DUCKDB_V2_BYTES_INLINE_LENGTH {
            // SAFETY: the caller guarantees the slot; the parts fit the inline buffer after the length.
            unsafe {
                slot.cast::<[u64; 2]>().write([0, 0]);
                slot.cast::<u32>().write(length);
                let inlined = slot
                    .cast::<u8>()
                    .add(std::mem::offset_of!(DuckDBV2BytesInlined, inlined));
                match parts {
                    [part] => copy_short(part.as_ref(), inlined),
                    _ => copy_parts(parts, inlined, length as usize),
                }
            }
            return Ok(());
        }

        let ptr = arena()?.allocate(length as usize)?;
        // SAFETY: the caller guarantees the slot, and the allocation holds `length` bytes. Each field is
        // stored straight into the slot, the prefix from the parts rather than the fresh arena copy, for
        // the reason above; the parts hold more than the 4 prefix bytes.
        unsafe {
            copy_parts(parts, ptr, length as usize);
            let slot = slot.cast::<DuckDBV2BytesPointer>();
            (&raw mut (*slot).length).write(length);
            let prefix = (&raw mut (*slot).prefix).cast::<u8>();
            match parts {
                [first, ..] if first.as_ref().len() >= 4 => {
                    prefix
                        .cast::<[u8; 4]>()
                        .write_unaligned(first.as_ref().as_ptr().cast::<[u8; 4]>().read_unaligned());
                }
                _ => copy_prefix(parts, prefix),
            }
            (&raw mut (*slot).ptr).write(ptr.cast());
        }
        Ok(())
    }

    /// Encode the bytes `fill` pushes into a buffer of `capacity` bytes, straight into `slot`.
    ///
    /// Short capacities build the value inside the slot, longer ones in one arena allocation; a result
    /// that ends up short is moved inline. If `fill` fails or panics, the slot holds an empty value.
    ///
    /// # Safety
    /// `slot` must be valid for writes of one `DuckDBBytes`.
    #[inline(always)]
    pub(crate) unsafe fn write_with(
        slot: *mut Self,
        capacity: usize,
        arena: impl FnOnce() -> Result<&'arena Arena>,
        fill: impl FnOnce(&mut BytesBuffer<'_>) -> Result<()>,
    ) -> Result<()> {
        if u32::try_from(capacity).is_err() {
            return Err(capacity_exceeds_u32(capacity));
        }
        let inline = capacity <= ffi::DUCKDB_V2_BYTES_INLINE_LENGTH as usize;
        let data = if inline {
            // SAFETY: the caller guarantees the slot; the inline buffer follows the length.
            unsafe {
                slot.cast::<[u64; 2]>().write([0, 0]);
                slot.cast::<u8>()
                    .add(std::mem::offset_of!(DuckDBV2BytesInlined, inlined))
            }
        } else {
            arena()?.allocate(capacity)?
        };
        let guard = EmptyOnDrop(slot);
        let mut buffer = BytesBuffer {
            data,
            len: 0,
            capacity,
            _marker: PhantomData,
        };
        fill(&mut buffer)?;
        // Defensive: the higher-ranked closure cannot exchange its buffer with another row's.
        if buffer.data != data {
            return Err(Error::invalid_input("the buffer was exchanged with another row's"));
        }
        std::mem::forget(guard);
        let len = buffer.len;
        // SAFETY: `buffer` is the one created above (same data), so its first `len <= capacity` bytes at
        // `data` are initialized; inline data already sits in the zeroed slot, after the length.
        unsafe {
            if inline {
                slot.cast::<u32>().write(len as u32);
            } else if len <= ffi::DUCKDB_V2_BYTES_INLINE_LENGTH as usize {
                slot.cast::<[u64; 2]>().write([0, 0]);
                slot.cast::<u32>().write(len as u32);
                let inlined = slot
                    .cast::<u8>()
                    .add(std::mem::offset_of!(DuckDBV2BytesInlined, inlined));
                copy_short(std::slice::from_raw_parts(data, len), inlined);
            } else {
                let slot = slot.cast::<DuckDBV2BytesPointer>();
                (&raw mut (*slot).length).write(len as u32);
                (&raw mut (*slot).prefix)
                    .cast::<[u8; 4]>()
                    .write_unaligned(data.cast::<[u8; 4]>().read_unaligned());
                (&raw mut (*slot).ptr).write(data.cast());
            }
        }
        Ok(())
    }

    /// Return the payload length in bytes.
    pub fn size(&self) -> usize {
        unsafe { self.data.value.inlined.length as usize }
    }

    /// Return the payload as a borrowed byte slice.
    #[inline]
    pub fn get_data(&self) -> &[u8] {
        if unsafe { self.data.value.inlined.length } <= ffi::DUCKDB_V2_BYTES_INLINE_LENGTH {
            unsafe {
                std::slice::from_raw_parts(
                    self.data.value.inlined.inlined.as_ptr().cast(),
                    self.data.value.inlined.length as usize,
                )
            }
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    self.data.value.pointer.ptr.cast(),
                    self.data.value.pointer.length as usize,
                )
            }
        }
    }
}

/// Leaves a slot an empty value unless forgotten, so a failed fill never exposes partial bytes.
struct EmptyOnDrop<'arena>(*mut DuckDBBytes<'arena>);

impl Drop for EmptyOnDrop<'_> {
    fn drop(&mut self) {
        // SAFETY: `write_with`'s caller guarantees the slot for its duration, which covers this guard.
        unsafe { self.0.cast::<[u64; 2]>().write([0, 0]) };
    }
}

/// A fixed-capacity byte buffer that builds one row in place; see
/// [`VectorWriter::<BlobValue>::set_with`](crate::vector::VectorWriter::set_with).
///
/// Only pushed bytes are readable, so the value never contains uninitialized memory.
#[derive(Debug)]
pub struct BytesBuffer<'row> {
    data: *mut u8,
    len: usize,
    capacity: usize,
    _marker: PhantomData<&'row mut [u8]>,
}

impl BytesBuffer<'_> {
    /// Return the number of bytes pushed so far.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether nothing has been pushed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the maximum number of bytes the row can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Append one byte; fails when the buffer is full.
    #[inline]
    pub fn push(&mut self, byte: u8) -> Result<()> {
        self.extend_from_slice(&[byte])
    }

    /// Append bytes; fails, appending nothing, when they exceed the remaining capacity.
    #[inline]
    pub fn extend_from_slice(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.capacity - self.len {
            return Err(capacity_exceeded(self.capacity));
        }
        // SAFETY: the bytes fit the `capacity` bytes at `data` after the `len` already written.
        unsafe {
            let destination = self.data.add(self.len);
            if bytes.len() <= ffi::DUCKDB_V2_BYTES_INLINE_LENGTH as usize {
                copy_short(bytes, destination);
            } else if bytes.len() <= 64 {
                copy_ascending(bytes, destination);
            } else {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
            }
        }
        self.len += bytes.len();
        Ok(())
    }

    /// Borrow the bytes pushed so far.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the first `len` bytes at `data` were written by pushes.
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    /// Mutably borrow the bytes pushed so far.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as in `as_slice`; the buffer exclusively borrows the row's storage.
        unsafe { std::slice::from_raw_parts_mut(self.data, self.len) }
    }
}

/// A fixed-capacity UTF-8 buffer that builds one row in place; see
/// [`VectorWriter::<String>::set_with`](crate::vector::VectorWriter::set_with).
///
/// Only whole `&str`s and `char`s can be pushed, so the value is valid UTF-8 without a check.
///
/// Each buffer is tied to its row, so buffers of nested calls cannot be exchanged:
///
/// ```compile_fail,E0521
/// use duckdb_neo::{Result, vector::VectorWriter};
///
/// fn swap(outer: &mut VectorWriter<'_, String>, inner: &mut VectorWriter<'_, String>) -> Result<()> {
///     outer.set_with(0, 40, |outer_buffer| {
///         inner.set_with(0, 4, |inner_buffer| {
///             std::mem::swap(outer_buffer, inner_buffer);
///             Ok(())
///         })
///     })
/// }
/// ```
#[repr(transparent)]
#[derive(Debug)]
pub struct StrBuffer<'row>(BytesBuffer<'row>);

impl<'row> StrBuffer<'row> {
    /// # Safety
    /// The pushed bytes of `bytes` must be valid UTF-8.
    #[inline(always)]
    pub(crate) unsafe fn from_bytes_mut<'a>(bytes: &'a mut BytesBuffer<'row>) -> &'a mut Self {
        // SAFETY: `StrBuffer` is a transparent wrapper, and the caller guarantees its invariant.
        unsafe { &mut *(bytes as *mut BytesBuffer<'row>).cast::<Self>() }
    }

    /// Return the number of bytes pushed so far.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return whether nothing has been pushed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Return the maximum number of bytes the row can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// Append a character; fails when its UTF-8 encoding exceeds the remaining capacity.
    #[inline]
    pub fn push(&mut self, value: char) -> Result<()> {
        self.push_str(value.encode_utf8(&mut [0; 4]))
    }

    /// Append a string; fails, appending nothing, when it exceeds the remaining capacity.
    #[inline]
    pub fn push_str(&mut self, value: &str) -> Result<()> {
        self.0.extend_from_slice(value.as_bytes())
    }

    /// Borrow the text pushed so far.
    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: only whole strings were pushed, and `as_mut_str` keeps the text valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(self.0.as_slice()) }
    }

    /// Mutably borrow the text pushed so far, e.g. for [`str::make_ascii_uppercase`].
    #[inline]
    pub fn as_mut_str(&mut self) -> &mut str {
        // SAFETY: as in `as_str`; safe `&mut str` methods preserve UTF-8.
        unsafe { std::str::from_utf8_unchecked_mut(self.0.as_mut_slice()) }
    }
}

#[cold]
fn capacity_exceeded(capacity: usize) -> Error {
    Error {
        code: DuckDBError::InputOutOfRange,
        message: format!("the value exceeds the buffer capacity of {capacity} bytes"),
    }
}

#[cold]
fn capacity_exceeds_u32(capacity: usize) -> Error {
    Error {
        code: DuckDBError::InputOutOfRange,
        message: format!("capacity {capacity} exceeds the maximum of {} bytes", u32::MAX),
    }
}

#[inline(always)]
fn total_length<P: AsRef<[u8]> + ?Sized>(parts: &[&P]) -> Result<u32> {
    parts
        .iter()
        .try_fold(0u32, |length, part| {
            length.checked_add(u32::try_from(part.as_ref().len()).ok()?)
        })
        .ok_or_else(|| Error {
            code: DuckDBError::InputOutOfRange,
            message: format!("combined byte length exceeds the maximum of {} bytes", u32::MAX),
        })
}

/// Copy at most 12 bytes with a few overlapping word copies instead of a `memcpy` call.
///
/// # Safety
/// `destination` must be valid for writes of `value.len()` bytes.
#[inline(always)]
unsafe fn copy_short(value: &[u8], destination: *mut u8) {
    let (source, len) = (value.as_ptr(), value.len());
    // SAFETY: every access stays within the first `len` bytes of both buffers.
    unsafe {
        if len >= 8 {
            let tail = len - 8;
            destination
                .cast::<u64>()
                .write_unaligned(source.cast::<u64>().read_unaligned());
            destination
                .add(tail)
                .cast::<u64>()
                .write_unaligned(source.add(tail).cast::<u64>().read_unaligned());
        } else if len >= 4 {
            let tail = len - 4;
            destination
                .cast::<u32>()
                .write_unaligned(source.cast::<u32>().read_unaligned());
            destination
                .add(tail)
                .cast::<u32>()
                .write_unaligned(source.add(tail).cast::<u32>().read_unaligned());
        } else if len > 0 {
            *destination = *source;
            *destination.add(len / 2) = *source.add(len / 2);
            *destination.add(len - 1) = *source.add(len - 1);
        }
    }
}

/// Copy bytes in ascending chunks that do not overlap, instead of a `memcpy` call.
///
/// Callers read freshly pushed bytes back; loads matching these stores can be forwarded, unlike
/// loads spanning the overlapping stores of `copy_short` or `memcpy`.
///
/// # Safety
/// `destination` must be valid for writes of `value.len()` bytes.
#[inline(always)]
unsafe fn copy_ascending(value: &[u8], destination: *mut u8) {
    let (source, len) = (value.as_ptr(), value.len());
    let mut offset = 0;
    // SAFETY: every access stays within the first `len` bytes of both buffers.
    unsafe {
        while len - offset >= 16 {
            let chunk = source.add(offset).cast::<[u8; 16]>().read_unaligned();
            destination.add(offset).cast::<[u8; 16]>().write_unaligned(chunk);
            offset += 16;
        }
        if len - offset >= 8 {
            let chunk = source.add(offset).cast::<[u8; 8]>().read_unaligned();
            destination.add(offset).cast::<[u8; 8]>().write_unaligned(chunk);
            offset += 8;
        }
        if len - offset >= 4 {
            let chunk = source.add(offset).cast::<[u8; 4]>().read_unaligned();
            destination.add(offset).cast::<[u8; 4]>().write_unaligned(chunk);
            offset += 4;
        }
        if len - offset >= 2 {
            let chunk = source.add(offset).cast::<[u8; 2]>().read_unaligned();
            destination.add(offset).cast::<[u8; 2]>().write_unaligned(chunk);
            offset += 2;
        }
        if offset < len {
            *destination.add(offset) = *source.add(offset);
        }
    }
}

/// Copy the first 4 bytes of the concatenated `parts` byte by byte, without a `memcpy` call.
///
/// # Safety
/// `destination` must be valid for writes of 4 bytes, and `parts` must hold at least as many.
#[inline(always)]
unsafe fn copy_prefix<P: AsRef<[u8]> + ?Sized>(parts: &[&P], destination: *mut u8) {
    let mut offset = 0;
    for part in parts {
        for &byte in part.as_ref() {
            if offset == 4 {
                return;
            }
            // SAFETY: `offset < 4`, within the destination.
            unsafe { destination.add(offset).write(byte) };
            offset += 1;
        }
    }
}

/// # Safety
/// `destination` must be valid for writes of `limit` bytes, and `parts` must hold at least as many.
#[inline(always)]
unsafe fn copy_parts<P: AsRef<[u8]> + ?Sized>(parts: &[&P], destination: *mut u8, limit: usize) {
    let mut offset = 0;
    for part in parts {
        if offset == limit {
            break;
        }
        let part = part.as_ref();
        let take = part.len().min(limit - offset);
        // SAFETY: `offset + take <= limit`, within the destination.
        unsafe { std::ptr::copy_nonoverlapping(part.as_ptr(), destination.add(offset), take) };
        offset += take;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DuckDBType, Parameters,
        connection::Connection,
        data_chunk::DataChunk,
        environment::{Environment, StorageLocation},
        scalar::MicroScalar,
        types::{BlobValue, UncheckedString},
        vector::Vector,
    };

    fn with_arena(test: impl FnOnce(&Arena) -> Result<()>) -> Result<()> {
        let env = Environment::new()?;
        let conn = env.open(StorageLocation::InMemory)?.connect()?;
        let mut chunk = DataChunk::create(&conn, &[String::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<String>(0)?;
        test(vector.writer()?.arena()?)
    }

    #[test]
    fn constructors_copy_their_inputs() -> Result<()> {
        with_arena(|arena| {
            for expected in ["", "héllo", "a longer string with 🦆"] {
                let values = {
                    let input = expected.to_owned();
                    [
                        DuckDBBytes::new(arena, input.as_bytes())?,
                        arena.add_bytes(input.as_bytes())?,
                        arena.add_str(&input)?,
                        arena.add_strings(&["", &input, ""])?,
                    ]
                };
                for value in values {
                    assert_eq!(value.size(), expected.len());
                    assert_eq!(value.get_data(), expected.as_bytes());
                }
            }
            Ok(())
        })
    }

    #[test]
    fn single_parts_of_every_inline_length_round_trip() -> Result<()> {
        with_arena(|arena| {
            let source = b"0123456789abcdef";
            for len in 0..=source.len() {
                let encoded = arena.add_bytes(&source[..len])?;
                assert_eq!(encoded.get_data(), &source[..len], "length {len}");
                // SAFETY: DuckDBBytes is 16 plain bytes.
                let raw = unsafe { std::mem::transmute_copy::<DuckDBBytes<'_>, [u8; 16]>(&encoded) };
                if len <= ffi::DUCKDB_V2_BYTES_INLINE_LENGTH as usize {
                    assert!(raw[4 + len..].iter().all(|byte| *byte == 0), "length {len} padding");
                } else {
                    assert_eq!(raw[..4], (len as u32).to_ne_bytes(), "length {len}");
                    assert_eq!(raw[4..8], source[..4], "length {len} prefix");
                }
            }
            Ok(())
        })
    }

    #[test]
    fn parts_match_concatenation_and_prefix() -> Result<()> {
        with_arena(|arena| {
            let cases: &[&[&[u8]]] = &[
                &[],
                &[b"", b""],
                &[b"a", b"bc"],
                &[b"1234", b"", b"56789012"],
                &[b"a", b"", b"bc", b"defghijklm"],
                &[b"", b"a longer first part", b"", b"suffix"],
                &[b"abcd", b"efghijklm"],
                &[b"ab", b"", b"c", b"defghijklmnop"],
                &[&[0xf0, 0x9f], &[0xa6, 0x86]],
                &[&[0, 255], b"long binary payload", &[128]],
            ];
            for parts in cases {
                let expected = parts.concat();
                let encoded = arena.add_parts(parts)?;
                assert_eq!(encoded.size(), expected.len());
                assert_eq!(encoded.get_data(), expected);
                if expected.len() > ffi::DUCKDB_V2_BYTES_INLINE_LENGTH as usize {
                    let prefix =
                        unsafe { encoded.data.value.pointer.prefix }.map(|byte| u8::from_ne_bytes(byte.to_ne_bytes()));
                    assert_eq!(prefix.as_slice(), &expected[..prefix.len()]);
                }
            }
            Ok(())
        })
    }

    #[test]
    fn parts_copy_their_inputs() -> Result<()> {
        with_arena(|arena| {
            let encoded = {
                let first = String::from("temporary input ");
                let second = String::from("and suffix");
                arena.add_parts(&[first.as_bytes(), second.as_bytes()])?
            };
            assert_eq!(encoded.get_data(), b"temporary input and suffix");
            Ok(())
        })
    }

    #[test]
    fn parts_reject_combined_length_overflow() -> Result<()> {
        with_arena(|arena| {
            let fragment = vec![0u8; 1024 * 1024];
            let parts = vec![fragment.as_slice(); 4096];
            let error = arena.add_parts(&parts).expect_err("4 GiB exceeds the encoded length");
            assert_eq!(error.code, DuckDBError::InputOutOfRange);
            Ok(())
        })
    }

    fn with_string_vector(rows: usize, test: impl FnOnce(&mut Vector<'_, String>) -> Result<()>) -> Result<()> {
        let env = Environment::new()?;
        let conn = env.open(StorageLocation::InMemory)?.connect()?;
        let mut chunk = DataChunk::create(&conn, &[String::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<String>(0)?;
        vector.set_size(rows)?;
        test(&mut vector)
    }

    fn raw_slot(vector: &Vector<'_, String>, row: usize) -> Result<[u8; 16]> {
        let reader = vector.reader()?;
        // SAFETY: the row is in bounds, and a slot is 16 plain bytes.
        Ok(unsafe { reader.as_ptr().add(row).cast::<[u8; 16]>().read() })
    }

    #[test]
    fn set_with_builds_every_length_inline_or_in_the_arena() -> Result<()> {
        let source = "0123456789abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let lengths = 0..=source.len();
        // Each length once with an exact capacity, once with spare capacity beyond the inline limit.
        with_string_vector(2 * lengths.clone().count(), |vector| {
            let mut writer = vector.writer()?;
            for len in lengths.clone() {
                for (row, capacity) in [(2 * len, len), (2 * len + 1, len + 20)] {
                    writer.set_with(row, capacity, |buffer| {
                        assert_eq!((buffer.len(), buffer.capacity()), (0, capacity));
                        buffer.push_str(&source[..len / 2])?;
                        buffer.push_str(&source[len / 2..len])?;
                        buffer.as_mut_str().make_ascii_uppercase();
                        assert_eq!(buffer.len(), len);
                        Ok(())
                    })?;
                }
            }
            let reader = vector.reader()?;
            for len in lengths {
                let expected = source[..len].to_ascii_uppercase();
                for row in [2 * len, 2 * len + 1] {
                    assert_eq!(reader.get(row)?, Some(expected.as_str()), "length {len}");
                    let raw = raw_slot(vector, row)?;
                    assert_eq!(raw[..4], (len as u32).to_ne_bytes());
                    if len <= ffi::DUCKDB_V2_BYTES_INLINE_LENGTH as usize {
                        assert!(raw[4 + len..].iter().all(|byte| *byte == 0), "length {len} padding");
                    } else {
                        assert_eq!(raw[4..8], expected.as_bytes()[..4], "length {len} prefix");
                    }
                }
            }
            Ok(())
        })
    }

    #[test]
    fn set_with_pushes_chars_and_marks_null_rows_valid() -> Result<()> {
        with_string_vector(2, |vector| {
            let mut writer = vector.writer()?;
            writer.set_null(0)?;
            writer.set_with(0, 16, |buffer| {
                for value in ['ä', '🦆', 'x'] {
                    buffer.push(value)?;
                }
                assert_eq!(buffer.as_str(), "ä🦆x");
                Ok(())
            })?;
            writer.set_with(1, 0, |buffer| {
                assert!(buffer.is_empty());
                Ok(())
            })?;
            assert_eq!(vector.reader()?.iter().collect::<Vec<_>>(), [Some("ä🦆x"), Some("")]);
            Ok(())
        })
    }

    #[test]
    fn set_with_failures_leave_an_empty_row_with_its_validity() -> Result<()> {
        with_string_vector(6, |vector| {
            let mut writer = vector.writer()?;
            for row in 0..6 {
                writer.set(row, "a previous value longer than inline")?;
            }
            writer.set_null(1)?;
            writer.set_null(3)?;
            for (row, capacity) in [(0, 8), (1, 8), (2, 40), (3, 40)] {
                let error = writer
                    .set_with(row, capacity, |buffer| {
                        buffer.push_str("partial")?;
                        buffer.push_str(&"x".repeat(capacity))
                    })
                    .expect_err("the second push exceeds the capacity");
                assert_eq!(error.code, DuckDBError::InputOutOfRange);
            }
            let error = writer
                .set_with(4, 4, |buffer| {
                    buffer.push_str("é")?;
                    Err(Error::invalid_input("rejected"))
                })
                .expect_err("the fill fails");
            assert_eq!(error.message, "rejected");
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                writer.set_with(5, 40, |buffer| {
                    buffer.push_str("a partial value beyond the inline limit")?;
                    panic!("fill panicked")
                })
            }));
            assert!(panicked.is_err());
            assert!(writer.set_with(6, 1, |_| Ok(())).is_err());
            assert_eq!(
                writer.set_with(0, u32::MAX as usize + 1, |_| Ok(())).unwrap_err().code,
                DuckDBError::InputOutOfRange
            );
            let values = vector.reader()?.iter().collect::<Vec<_>>();
            assert_eq!(values, [Some(""), None, Some(""), None, Some(""), Some("")]);
            for row in 0..6 {
                assert_eq!(raw_slot(vector, row)?, [0; 16], "row {row}");
            }
            Ok(())
        })
    }

    #[test]
    fn blob_set_with_builds_binary_values() -> Result<()> {
        let env = Environment::new()?;
        let conn = env.open(StorageLocation::InMemory)?.connect()?;
        let mut chunk = DataChunk::create(&conn, &[BlobValue::logical_type(&conn)?], true)?;
        let mut vector = chunk.get_vector_at_mut::<BlobValue>(0)?;
        vector.set_size(2)?;
        let mut writer = vector.writer()?;
        writer.set_with(0, 3, |buffer| {
            buffer.push(0)?;
            buffer.extend_from_slice(&[255, 128])?;
            assert!(buffer.push(1).is_err());
            Ok(())
        })?;
        writer.set_with(1, 64, |buffer| {
            buffer.extend_from_slice(&[0xff; 20])?;
            buffer.as_mut_slice()[19] = 0;
            assert_eq!(buffer.as_slice().len(), 20);
            Ok(())
        })?;
        let mut long = vec![0xff; 20];
        long[19] = 0;
        assert_eq!(
            vector.reader()?.iter().collect::<Vec<_>>(),
            [Some(&[0, 255, 128][..]), Some(long.as_slice())]
        );
        Ok(())
    }

    fn register_upper(conn: &Connection) -> Result<()> {
        MicroScalar::create(
            "upper_in_place",
            |_ctx, input: Vector<'_, UncheckedString>, mut output: Vector<'_, String>| {
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

    #[test]
    fn set_with_in_a_scalar_over_flat_constant_and_dictionary_inputs() -> Result<()> {
        let env = Environment::new()?;
        let conn = env.open(StorageLocation::InMemory)?.connect()?;
        register_upper(&conn)?;
        for query in [
            "SELECT count(*) FILTER (WHERE upper_in_place(s) IS DISTINCT FROM upper(s))::VARCHAR FROM \
             (SELECT CASE WHEN i % 7 = 0 THEN NULL ELSE repeat('ab', i % 23) END AS s FROM range(5000) t(i))",
            "SELECT count(*) FILTER (WHERE upper_in_place('a constant longer than twelve') \
             <> 'A CONSTANT LONGER THAN TWELVE')::VARCHAR FROM range(3000)",
            "SELECT count(*) FILTER (WHERE u IS DISTINCT FROM upper(s))::VARCHAR FROM \
             (SELECT s, upper_in_place(s) AS u FROM (SELECT (['short', NULL, 'a much longer value'])[1 + i % 3] AS s \
             FROM range(5000) t(i)) GROUP BY s)",
        ] {
            let chunk = conn.query(query, Parameters::None)?.next_chunk()?.unwrap();
            assert_eq!(
                chunk.get_vector_at::<String>(0)?.reader()?.get(0)?,
                Some("0"),
                "{query}"
            );
        }
        Ok(())
    }
}
