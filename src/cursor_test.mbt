// Unit tests for the bounds-checked little-endian cursor.

///|
test "read_u8 advances and reports unsigned values" {
  let c = @gguf.ByteCursor::new(b"\x00\x7f\xff")
  assert_eq(c.read_u8(), Ok(0))
  assert_eq(c.read_u8(), Ok(127))
  assert_eq(c.read_u8(), Ok(255))
  assert_true(c.is_at_end())
}

///|
test "read_i8 sign-extends two's complement" {
  let c = @gguf.ByteCursor::new(b"\x00\x7f\x80\xff")
  assert_eq(c.read_i8(), Ok(0))
  assert_eq(c.read_i8(), Ok(127))
  assert_eq(c.read_i8(), Ok(-128))
  assert_eq(c.read_i8(), Ok(-1))
}

///|
test "read_u16 is little-endian" {
  let c = @gguf.ByteCursor::new(b"\x34\x12\xff\xff")
  assert_eq(c.read_u16(), Ok(0x1234))
  assert_eq(c.read_u16(), Ok(0xFFFF))
}

///|
test "read_i16 sign-extends two's complement" {
  let c = @gguf.ByteCursor::new(b"\xff\xff\x00\x80\xff\x7f")
  assert_eq(c.read_i16(), Ok(-1))
  assert_eq(c.read_i16(), Ok(-32768))
  assert_eq(c.read_i16(), Ok(32767))
}

///|
test "read_u32 is little-endian" {
  let c = @gguf.ByteCursor::new(b"\x78\x56\x34\x12")
  assert_eq(c.read_u32(), Ok(0x12345678U))
}

///|
test "read_u64 is little-endian" {
  let c = @gguf.ByteCursor::new(b"\x08\x07\x06\x05\x04\x03\x02\x01")
  assert_eq(c.read_u64(), Ok(0x0102030405060708UL))
}

///|
test "read_f32 decodes IEEE 754 bit patterns" {
  let c = @gguf.ByteCursor::new(b"\x00\x00\x80\x3f\x00\x00\x80\xbf")
  assert_eq(c.read_f32(), Ok(1.0))
  assert_eq(c.read_f32(), Ok(-1.0))
}

///|
test "read_f64 decodes IEEE 754 bit patterns" {
  let c = @gguf.ByteCursor::new(b"\x00\x00\x00\x00\x00\x00\xf0\x3f")
  assert_eq(c.read_f64(), Ok(1.0))
}

///|
test "reads past the end fail with the offending offset" {
  let c = @gguf.ByteCursor::new(b"\x01\x02")
  assert_eq(c.read_u16(), Ok(0x0201))
  assert_eq(
    c.read_u32(),
    Err(@gguf.Truncated(offset=2L, needed=4L, remaining=0L)),
  )
  // A failed read must not move the cursor.
  assert_eq(c.remaining(), 0L)
}

///|
test "truncated mid-value reports how much was left" {
  let c = @gguf.ByteCursor::new(b"\x01\x02\x03")
  assert_eq(
    c.read_u64(),
    Err(@gguf.Truncated(offset=0L, needed=8L, remaining=3L)),
  )
}

///|
test "skip and seek are bounds-checked" {
  let c = @gguf.ByteCursor::new(b"\x00\x01\x02\x03")
  assert_eq(c.skip(2L), Ok(()))
  assert_eq(c.read_u8(), Ok(2))
  assert_eq(
    c.skip(99L),
    Err(@gguf.Truncated(offset=3L, needed=99L, remaining=1L)),
  )
  assert_eq(c.seek(1L), Ok(()))
  assert_eq(c.read_u8(), Ok(1))
  assert_true(c.seek(5L) is Err(_))
  assert_true(c.seek(-1L) is Err(_))
}

///|
test "read_bytes copies the requested region" {
  let c = @gguf.ByteCursor::new(b"\xde\xad\xbe\xef")
  assert_eq(c.read_bytes(2L), Ok(b"\xde\xad"))
  assert_eq(c.read_bytes(2L), Ok(b"\xbe\xef"))
  assert_eq(c.read_bytes(1L), Err(@gguf.Truncated(offset=4L, needed=1L, remaining=0L)))
}

///|
test "read_string reads a u64 length then UTF-8 bytes" {
  let c = @gguf.ByteCursor::new(b"\x05\x00\x00\x00\x00\x00\x00\x00hello")
  let (text, raw) = c.read_string().unwrap()
  assert_eq(text, "hello")
  assert_eq(raw, b"hello")
  assert_true(c.is_at_end())
}

///|
test "read_string accepts multi-byte UTF-8" {
  // "中文" is 6 UTF-8 bytes.
  let c = @gguf.ByteCursor::new(b"\x06\x00\x00\x00\x00\x00\x00\x00\xe4\xb8\xad\xe6\x96\x87")
  let (text, raw) = c.read_string().unwrap()
  assert_eq(text, "中文")
  assert_eq(raw.length(), 6)
}

///|
test "read_string keeps going when the bytes are not valid UTF-8" {
  // A lone 0xFF is not a legal UTF-8 sequence; the raw bytes must survive.
  let c = @gguf.ByteCursor::new(b"\x02\x00\x00\x00\x00\x00\x00\x00\xff\xfe")
  let (text, raw) = c.read_string().unwrap()
  assert_eq(raw, b"\xff\xfe")
  // Lossy decoding substitutes U+FFFD, so the replacement char is present.
  assert_true(text.contains("\u{fffd}"))
}

///|
test "read_string with a length beyond the buffer reports truncation" {
  let c = @gguf.ByteCursor::new(b"\xff\x00\x00\x00\x00\x00\x00\x00ab")
  assert_eq(
    c.read_string(),
    Err(@gguf.Truncated(offset=8L, needed=255L, remaining=2L)),
  )
}

///|
test "read_string refuses lengths that cannot be indexed" {
  // 0x80000000 is exactly one past the largest indexable length.
  let c = @gguf.ByteCursor::new(b"\x00\x00\x00\x80\x00\x00\x00\x00")
  match c.read_string() {
    Ok(_) => fail("expected a length error")
    Err(e) =>
      match e {
        @gguf.CountTooLarge(count~, ..) => assert_eq(count, 0x80000000UL)
        _ => fail("expected CountTooLarge, got \{e}")
      }
  }
}

///|
test "an empty buffer reports truncation rather than succeeding" {
  let c = @gguf.ByteCursor::new(b"")
  assert_true(c.is_at_end())
  assert_eq(c.read_u8(), Err(@gguf.Truncated(offset=0L, needed=1L, remaining=0L)))
  assert_eq(c.read_string(), Err(@gguf.Truncated(offset=0L, needed=8L, remaining=0L)))
}

///|
test "len and remaining track the whole buffer" {
  let c = @gguf.ByteCursor::new(b"\x01\x02\x03")
  assert_eq(c.len(), 3L)
  assert_eq(c.remaining(), 3L)
  c.skip(1L) |> ignore
  assert_eq(c.len(), 3L)
  assert_eq(c.remaining(), 2L)
}
