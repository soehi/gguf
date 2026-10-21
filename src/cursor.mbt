// Bounds-checked little-endian cursor over a byte buffer.
//
// Every read is a `Result`: the cursor checks the remaining length before it
// touches the buffer, so a truncated model file yields `Truncated` with the
// exact offset instead of an out-of-bounds read. Offsets and lengths are
// `Int64` throughout so that a >2 GiB model is representable even on a 32-bit
// index type; only the final `Bytes` access narrows to `Int`.

///|
/// A read position within `buf`.
///
/// `pos` is an absolute byte offset and is always kept within
/// `[0, buf.length()]`. Constructed with `ByteCursor::new` and moved only
/// through the `read_*` methods, which advance it on success.
pub(all) struct ByteCursor {
  buf : Bytes
  mut pos : Int64
} derive(Debug)

///|
/// Create a cursor positioned at the start of `buf`.
pub fn ByteCursor::new(buf : Bytes) -> ByteCursor {
  { buf, pos: 0L }
}

///|
/// Total buffer length, independent of the current position.
pub fn ByteCursor::len(self : ByteCursor) -> Int64 {
  self.buf.length().to_int64()
}

///|
/// Bytes not yet consumed.
pub fn ByteCursor::remaining(self : ByteCursor) -> Int64 {
  self.len() - self.pos
}

///|
/// Whether the cursor has reached the end of the buffer.
pub fn ByteCursor::is_at_end(self : ByteCursor) -> Bool {
  self.pos >= self.len()
}

///|
/// Check that `n` more bytes are available; on failure report the shortfall
/// against the current position.
fn ByteCursor::ensure(self : ByteCursor, n : Int64) -> Result[Unit, GgufError] {
  let remaining = self.remaining()
  if n > remaining {
    Err(Truncated(offset=self.pos, needed=n, remaining=remaining))
  } else {
    Ok(())
  }
}

///|
/// Position the cursor at an absolute offset, which must lie within the buffer.
pub fn ByteCursor::seek(self : ByteCursor, pos : Int64) -> Result[Unit, GgufError] {
  if pos < 0L || pos > self.len() {
    Err(Truncated(offset=pos, needed=0L, remaining=self.len()))
  } else {
    self.pos = pos
    Ok(())
  }
}

///|
/// Advance by `n` bytes after checking they exist.
pub fn ByteCursor::skip(self : ByteCursor, n : Int64) -> Result[Unit, GgufError] {
  match self.ensure(n) {
    Ok(_) => {
      self.pos = self.pos + n
      Ok(())
    }
    Err(e) => Err(e)
  }
}

///|
/// Read `n` bytes and advance past them.
pub fn ByteCursor::read_bytes(
  self : ByteCursor,
  n : Int64,
) -> Result[Bytes, GgufError] {
  match self.ensure(n) {
    Ok(_) => {
      let start = self.pos
      self.pos = self.pos + n
      Ok(self.buf.view(start=start.to_int(), end=(start + n).to_int()).to_owned())
    }
    Err(e) => Err(e)
  }
}

///|
/// Read an unsigned byte (0..255).
pub fn ByteCursor::read_u8(self : ByteCursor) -> Result[Int, GgufError] {
  match self.ensure(1L) {
    Ok(_) => {
      let v = self.buf[self.pos.to_int()].to_int()
      self.pos = self.pos + 1L
      Ok(v)
    }
    Err(e) => Err(e)
  }
}

///|
/// Read a little-endian `u16` as an `Int` in 0..65535.
pub fn ByteCursor::read_u16(self : ByteCursor) -> Result[Int, GgufError] {
  match self.ensure(2L) {
    Ok(_) => {
      let v = self.buf.unsafe_read_uint16_le(self.pos.to_int()).to_int()
      self.pos = self.pos + 2L
      Ok(v)
    }
    Err(e) => Err(e)
  }
}

///|
/// Read a little-endian `u32` as an unsigned bit pattern.
pub fn ByteCursor::read_u32(self : ByteCursor) -> Result[UInt, GgufError] {
  match self.ensure(4L) {
    Ok(_) => {
      let v = self.buf.unsafe_read_uint32_le(self.pos.to_int())
      self.pos = self.pos + 4L
      Ok(v)
    }
    Err(e) => Err(e)
  }
}

///|
/// Read a little-endian `u64`.
pub fn ByteCursor::read_u64(self : ByteCursor) -> Result[UInt64, GgufError] {
  match self.ensure(8L) {
    Ok(_) => {
      let v = self.buf.unsafe_read_uint64_le(self.pos.to_int())
      self.pos = self.pos + 8L
      Ok(v)
    }
    Err(e) => Err(e)
  }
}

///|
/// Read a little-endian signed `i8`, sign-extended.
pub fn ByteCursor::read_i8(self : ByteCursor) -> Result[Int, GgufError] {
  match self.read_u8() {
    Ok(v) => Ok(if v >= 128 { v - 256 } else { v })
    Err(e) => Err(e)
  }
}

///|
/// Read a little-endian signed `i16`, sign-extended.
pub fn ByteCursor::read_i16(self : ByteCursor) -> Result[Int, GgufError] {
  match self.read_u16() {
    Ok(v) => Ok(if v >= 32768 { v - 65536 } else { v })
    Err(e) => Err(e)
  }
}

///|
/// Read a little-endian `f32` as raw IEEE 754 bits.
pub fn ByteCursor::read_f32(self : ByteCursor) -> Result[Float, GgufError] {
  match self.read_u32() {
    Ok(bits) => Ok(Float::reinterpret_from_uint(bits))
    Err(e) => Err(e)
  }
}

///|
/// Read a little-endian `f64`.
pub fn ByteCursor::read_f64(self : ByteCursor) -> Result[Double, GgufError] {
  match self.read_u64() {
    Ok(bits) => Ok(bits.reinterpret_as_double())
    Err(e) => Err(e)
  }
}

///|
/// Read a GGUF string: a `u64` byte length followed by that many UTF-8 bytes.
///
/// Invalid UTF-8 is not fatal — GGUF files in the wild occasionally carry
/// non-UTF-8 chat templates — so the bytes are preserved and the caller sees
/// `(text, raw)` where `text` is a lossy decode.
pub fn ByteCursor::read_string(
  self : ByteCursor,
) -> Result[(String, Bytes), GgufError] {
  match self.read_u64() {
    Err(e) => Err(e)
    Ok(len) => {
      if len > 0x7FFFFFFFUL {
        return Err(CountTooLarge(offset=self.pos - 8L, count=len))
      }
      match self.read_bytes(len.reinterpret_as_int64()) {
        Err(e) => Err(e)
        Ok(raw) => Ok((decode_utf8_lossy(raw), raw))
      }
    }
  }
}

///|
/// Decode UTF-8, substituting U+FFFD for malformed sequences.
///
/// `@encoding/utf8.decode_lossy` is used rather than the raising variant so a
/// bad template or token string degrades to readable text instead of aborting
/// the whole parse.
pub fn decode_utf8_lossy(raw : Bytes) -> String {
  @utf8.decode_lossy(raw.view())
}
