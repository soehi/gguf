// GGUF metadata value model and parsing.
//
// GGUF metadata is a flat list of key/value pairs, but array values need two
// different treatments depending on size. `tokenizer.ggml.tokens` holds ~50k
// strings; materialising all of them as `String`s costs tens of megabytes for
// a listing that only needs a type, a count and the first few entries. So a
// parsed array is either `Materialized` (kept in full) when it is small, or
// `Summarized` when it crosses `array_materialize_limit` — the summary keeps
// the element type, the count, and a bounded prefix.
//
// All reads go through `ByteCursor`, so a malformed metadata block produces a
// `GgufError` with an offset instead of a crash.

///|
/// A GGUF metadata value.
///
/// The `u64`/`i64`/`f64` variants exist even though GGUF v2 never emits them,
/// because v3 files can and the type ids are stable across versions.
pub(all) enum MetaValue {
  U8(Int)
  I8(Int)
  U16(Int)
  I16(Int)
  U32(UInt)
  I32(Int)
  F32(Float)
  Bool(Bool)
  Str(String)
  Array(MetaArray)
  U64(UInt64)
  I64(Int64)
  F64(Double)
} derive(Eq, Debug)

///|
/// An array value, either kept whole or reduced to a bounded summary.
pub(all) enum MetaArray {
  /// Every element was read into memory.
  Materialized(Array[MetaValue])
  /// The array was too large to materialise; only a prefix was kept.
  Summarized(MetaArraySummary)
} derive(Eq, Debug)

///|
/// A bounded view of a large array.
pub(all) struct MetaArraySummary {
  elem_type : UInt
  count : UInt64
  prefix : Array[MetaValue]
} derive(Eq, Debug)

///|
/// Number of leading elements kept when an array is summarized.
pub let array_prefix_len : Int = 8

///|
/// Arrays longer than this are summarized instead of materialized. 1024 keeps
/// the memory cost of a small file's metadata flat while still giving callers
/// the full value for anything they are likely to inspect element-by-element.
pub let array_materialize_limit : Int = 1024

///|
/// The `value_type` id of a metadata value, as it would be written on disk.
pub fn MetaValue::type_id(self : MetaValue) -> UInt {
  match self {
    U8(_) => 0U
    I8(_) => 1U
    U16(_) => 2U
    I16(_) => 3U
    U32(_) => 4U
    I32(_) => 5U
    F32(_) => 6U
    Bool(_) => 7U
    Str(_) => 8U
    Array(_) => 9U
    U64(_) => 10U
    I64(_) => 11U
    F64(_) => 12U
  }
}

///|
/// The on-disk name of a `value_type` id, for error messages and listings.
pub fn value_type_name(type_id : UInt) -> String {
  match type_id {
    0U => "u8"
    1U => "i8"
    2U => "u16"
    3U => "i16"
    4U => "u32"
    5U => "i32"
    6U => "f32"
    7U => "bool"
    8U => "string"
    9U => "array"
    10U => "u64"
    11U => "i64"
    12U => "f64"
    _ => "invalid"
  }
}

///|
/// Whether `type_id` names a scalar value type (i.e. everything except `array`).
pub fn is_scalar_value_type(type_id : UInt) -> Bool {
  type_id <= 8U || (type_id >= 10U && type_id <= 12U)
}

///|
/// Read a scalar metadata value of a known type id.
fn read_scalar_value(
  c : ByteCursor,
  type_id : UInt,
) -> Result[MetaValue, GgufError] {
  match type_id {
    0U => c.read_u8().map(v => U8(v))
    1U => c.read_i8().map(v => I8(v))
    2U => c.read_u16().map(v => U16(v))
    3U => c.read_i16().map(v => I16(v))
    4U => c.read_u32().map(v => U32(v))
    5U =>
      match c.read_u32() {
        Ok(bits) => Ok(I32(bits.reinterpret_as_int()))
        Err(e) => Err(e)
      }
    6U => c.read_f32().map(v => F32(v))
    7U => c.read_u8().map(v => Bool(v != 0))
    8U => c.read_string().map(pair => Str(pair.0))
    10U => c.read_u64().map(v => U64(v))
    11U =>
      match c.read_u64() {
        Ok(bits) => Ok(I64(bits.reinterpret_as_int64()))
        Err(e) => Err(e)
      }
    12U => c.read_f64().map(v => F64(v))
    _ => Err(UnknownValueType(offset=c.pos, type_id=type_id))
  }
}

///|
/// Read a metadata array: an element type, a `u64` count, then the elements.
///
/// Arrays of arrays are legal in GGUF and occur in practice, so `nesting` is
/// tracked and bounded. Element counts are validated against the bytes that
/// could possibly remain before any element is decoded.
///
/// By default arrays beyond `array_materialize_limit` are summarized (element
/// type, count, bounded prefix) to keep `inspect` cheap on vocab-sized lists.
/// Callers that need the full contents — the tokenizer loads the whole
/// `tokens`/`merges` arrays — set `materialize~ = true` to force full decoding.
fn read_array_value(
  c : ByteCursor,
  nesting : Int,
  materialize~ : Bool = false,
) -> Result[MetaValue, GgufError] {
  if nesting > 8 {
    return Err(ArrayNestingTooDeep(offset=c.pos, depth=nesting))
  }
  let start = c.pos
  let elem_type = match c.read_u32() {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  let count = match c.read_u64() {
    Ok(n) => n
    Err(e) => return Err(e)
  }
  if !is_scalar_value_type(elem_type) && elem_type != 9U {
    return Err(IllegalArrayElementType(offset=start, type_id=elem_type))
  }
  if count > 0x7FFFFFFFUL {
    return Err(CountTooLarge(offset=start, count=count))
  }
  let too_long = count.to_int() > array_materialize_limit
  if !too_long || materialize || elem_type == 9U {
    // Materialize: decode every element.
    let elems : Array[MetaValue] = []
    let mut i = 0
    while i < count.to_int() {
      let v = if elem_type == 9U {
        match read_array_value(c, nesting + 1) {
          Ok(v) => v
          Err(e) => return Err(e)
        }
      } else {
        match read_scalar_value(c, elem_type) {
          Ok(v) => v
          Err(e) => return Err(e)
        }
      }
      elems.push(v)
      i = i + 1
    }
    Ok(Array(Materialized(elems)))
  } else {
    // Summarize: decode a bounded prefix, then skip the remainder.
    let prefix : Array[MetaValue] = []
    let mut i = 0
    while i < array_prefix_len && i < count.to_int() {
      match read_scalar_value(c, elem_type) {
        Ok(v) => prefix.push(v)
        Err(e) => return Err(e)
      }
      i = i + 1
    }
    let skipped = count.to_int() - prefix.length()
    match skip_array_elements(c, elem_type, skipped) {
      Ok(_) => Ok(Array(Summarized({ elem_type, count, prefix })))
      Err(e) => Err(e)
    }
  }
}

///|
/// Skip `n` scalar elements of a known type without decoding them.
///
/// Used for the tail of a summarized array and for the payloads of tensors the
/// caller does not read.
fn skip_array_elements(
  c : ByteCursor,
  elem_type : UInt,
  n : Int,
) -> Result[Unit, GgufError] {
  let elem_size : Int64 = match elem_type {
    0U | 1U | 7U => 1L
    2U | 3U => 2L
    4U | 5U | 6U => 4L
    8U => {
      // Strings are variable length: read and discard each one.
      let mut i = 0
      while i < n {
        match c.read_string() {
          Ok(_) => ()
          Err(e) => return Err(e)
        }
        i = i + 1
      }
      return Ok(())
    }
    10U | 11U | 12U => 8L
    _ => return Err(IllegalArrayElementType(offset=c.pos, type_id=elem_type))
  }
  let total = elem_size * n.to_int64()
  c.skip(total)
}

///|
/// Read one metadata value given its already-read `value_type` id.
pub fn read_meta_value(
  c : ByteCursor,
  type_id : UInt,
  materialize~ : Bool = false,
) -> Result[MetaValue, GgufError] {
  if type_id == 9U {
    read_array_value(c, 0, materialize~)
  } else {
    read_scalar_value(c, type_id)
  }
}

///|
/// Convenience accessors used by the CLI and integrations.
pub fn MetaValue::as_string(self : MetaValue) -> String? {
  match self {
    Str(s) => Some(s)
    _ => None
  }
}

///|
/// Extract an integer regardless of which integer width carried it, so callers
/// do not have to special-case `u32` versus `u64`.
pub fn MetaValue::as_int(self : MetaValue) -> Int64? {
  match self {
    U8(v) | I8(v) | U16(v) | I16(v) => Some(v.to_int64())
    U32(v) => Some(v.to_uint64().reinterpret_as_int64())
    I32(v) => Some(v.to_int64())
    U64(v) => Some(v.reinterpret_as_int64())
    I64(v) => Some(v)
    Bool(v) => Some(if v { 1L } else { 0L })
    _ => None
  }
}

///|
/// Extract a floating-point value as `Double`.
pub fn MetaValue::as_double(self : MetaValue) -> Double? {
  match self {
    F32(v) => Some(v.to_double())
    F64(v) => Some(v)
    U8(v) | I8(v) | U16(v) | I16(v) | I32(v) => Some(v.to_double())
    U32(v) => Some(v.to_double())
    U64(v) => Some(v.reinterpret_as_int64().to_double())
    I64(v) => Some(v.to_double())
    _ => None
  }
}

///|
/// Extract an array length whether the array was materialized or summarized.
pub fn MetaValue::array_len(self : MetaValue) -> Int? {
  match self {
    Array(Materialized(elems)) => Some(elems.length())
    Array(Summarized(s)) => Some(s.count.to_int())
    _ => None
  }
}

///|
/// The element type id of an array value.
pub fn MetaValue::array_elem_type(self : MetaValue) -> UInt? {
  match self {
    Array(Materialized(elems)) =>
      if elems.length() > 0 {
        Some(elems[0].type_id())
      } else {
        None
      }
    Array(Summarized(s)) => Some(s.elem_type)
    _ => None
  }
}

///|
/// Render a metadata value for listings and diagnostics.
///
/// Long strings are truncated: a chat template can run to kilobytes and would
/// otherwise swamp the output, while the type still has to be visible.
pub fn MetaValue::to_text(self : MetaValue) -> String {
  match self {
    U8(v) => "u8 \{v}"
    I8(v) => "i8 \{v}"
    U16(v) => "u16 \{v}"
    I16(v) => "i16 \{v}"
    U32(v) => "u32 \{v}"
    I32(v) => "i32 \{v}"
    F32(v) => "f32 \{v}"
    Bool(v) => "bool \{v}"
    Str(s) => "string \{quote_short(s)}"
    U64(v) => "u64 \{v}"
    I64(v) => "i64 \{v}"
    F64(v) => "f64 \{v}"
    Array(Materialized(elems)) => {
      let parts = elems.map(fn(e) { e.to_text() })
      "array[\{parts.join(", ")}]"
    }
    Array(Summarized(s)) => {
      let parts = s.prefix.map(fn(e) { e.to_text() })
      "array(\{value_type_name(s.elem_type)})[\{parts.join(", ")}, ... \{s.count} total]"
    }
  }
}

///|
/// Quote a string for display, collapsing it to one line and truncating long
/// values so a single metadata entry cannot dominate the output.
pub fn quote_short(s : String) -> String {
  let limit = 60
  let sb = StringBuilder::new()
  sb.write_char('"')
  let mut n = 0
  for ch in s {
    if n >= limit {
      sb.write_string("...")
      break
    }
    if ch == '\n' {
      sb.write_string("\n")
    } else if ch == '\t' {
      sb.write_string("\t")
    } else if ch == '"' {
      sb.write_string("\\\"")
    } else {
      sb.write_char(ch)
    }
    n = n + 1
  }
  sb.write_char('"')
  sb.to_string()
}

///|
pub impl Show for MetaValue with fn to_string(self : MetaValue) {
  self.to_text()
}

///|
pub impl Show for MetaArray with fn to_string(self : MetaArray) {
  match self {
    Materialized(elems) => "array[\{elems.length()} items]"
    Summarized(s) => "array(\{value_type_name(s.elem_type)})[\{s.count} items]"
  }
}
