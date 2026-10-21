// GGUF parse errors.
//
// Every variant carries the file offset where the problem was detected so a
// malformed model file produces an actionable message instead of a bare
// "parse failed". Nothing in this package panics or indexes out of bounds on
// hostile input.

///|
/// Error type for GGUF container parsing.
pub(all) suberror GgufError {
  /// Fewer bytes than a valid header, or the file ended mid-structure.
  Truncated(offset~ : Int64, needed~ : Int64, remaining~ : Int64)
  /// The first four bytes were not `GGUF`.
  BadMagic(found~ : UInt)
  /// GGUFv1 is refused outright; v2 and v3 are accepted.
  UnsupportedVersion(version~ : UInt, supported_max~ : UInt)
  /// A metadata value carried a `value_type` outside 0..12.
  UnknownValueType(offset~ : Int64, type_id~ : UInt)
  /// `general.alignment` was present but not a power of two, or zero.
  BadAlignment(value~ : UInt)
  /// `general.alignment` was present with a type other than u32.
  AlignmentWrongType(type_id~ : UInt)
  /// An array nested deeper than the parser tracks; guards against cycles in
  /// crafted files.
  ArrayNestingTooDeep(offset~ : Int64, depth~ : Int)
  /// An array element type was one of the scalar kinds an array may not hold.
  IllegalArrayElementType(offset~ : Int64, type_id~ : UInt)
  /// A declared count times the element size overflows an `Int64`.
  CountOverflow(offset~ : Int64, count~ : UInt64, elem_size~ : Int64)
  /// A declared count or dimension is so large it cannot be indexed.
  CountTooLarge(offset~ : Int64, count~ : UInt64)
  /// `tensor_count` was larger than the remaining file could possibly hold.
  TensorCountImplausible(count~ : UInt64, remaining~ : Int64)
  /// A tensor declared more dimensions than GGUF allows.
  TooManyDims(offset~ : Int64, n_dims~ : UInt, max_dims~ : Int)
  /// The data section starts past the end of the buffer.
  DataSectionOutOfRange(data_start~ : Int64, buffer_len~ : Int64)
  /// A tensor's `offset` plus its byte size runs past the data section.
  TensorOutOfRange(name~ : String, offset~ : Int64, nbytes~ : Int64, data_len~ : Int64)
  /// A tensor's type is understood but this build cannot dequantize it.
  TensorNotSupported(name~ : String, type_id~ : UInt, type_name~ : String)
  /// The tensor package rejected a payload that GGUF-level checks had passed.
  DequantFailed(name~ : String, reason~ : String)
} derive(Eq, Debug)

///|
/// Human-readable rendering of a `GgufError`, always naming the offset where
/// the parse gave up.
pub fn GgufError::message(self : GgufError) -> String {
  match self {
    Truncated(offset~, needed~, remaining~) =>
      "truncated at offset \{offset}: needed \{needed} bytes, \{remaining} remain"
    BadMagic(found~) =>
      "bad magic \"\{magic_to_text(found)}\": expected \"GGUF\""
    UnsupportedVersion(version~, supported_max~) =>
      "unsupported GGUF version \{version}: this build accepts 2..\{supported_max}"
    UnknownValueType(offset~, type_id~) =>
      "unknown metadata value type \{type_id} at offset \{offset}"
    BadAlignment(value~) => "general.alignment = \{value} is not a positive power of two"
    AlignmentWrongType(type_id~) =>
      "general.alignment has type \{type_id}, expected u32 (4)"
    ArrayNestingTooDeep(offset~, depth~) =>
      "array nesting depth \{depth} at offset \{offset} exceeds the limit"
    IllegalArrayElementType(offset~, type_id~) =>
      "array element type \{type_id} at offset \{offset} is not a scalar value type"
    CountOverflow(offset~, count~, elem_size~) =>
      "count \{count} at offset \{offset} overflows when multiplied by \{elem_size} bytes"
    CountTooLarge(offset~, count~) =>
      "count \{count} at offset \{offset} is too large to index"
    TensorCountImplausible(count~, remaining~) =>
      "tensor_count \{count} cannot fit in the remaining \{remaining} bytes"
    TooManyDims(offset~, n_dims~, max_dims~) =>
      "tensor at offset \{offset} declares \{n_dims} dimensions, maximum is \{max_dims}"
    DataSectionOutOfRange(data_start~, buffer_len~) =>
      "data section starts at \{data_start}, past the \{buffer_len}-byte buffer"
    TensorOutOfRange(name~, offset~, nbytes~, data_len~) =>
      "tensor \{name}: bytes [\{offset}, \{offset + nbytes}) fall outside the \{data_len}-byte data section"
    TensorNotSupported(name~, type_id~, type_name~) =>
      "tensor \{name}: type \{type_id} (\{type_name}) cannot be dequantized by this build"
    DequantFailed(name~, reason~) => "tensor \{name}: dequantization failed: \{reason}"
  }
}

///|
/// Render the four magic bytes of a rejected header as printable ASCII, so the
/// error message shows what the file actually started with.
fn magic_to_text(found : UInt) -> String {
  let sb = StringBuilder::new()
  for shift = 0; shift <= 24; shift = shift + 8 {
    let b = ((found >> shift) & 0xFF).reinterpret_as_int()
    sb.write_char(if b >= 32 && b < 127 { b.unsafe_to_char() } else { '.' })
  }
  sb.to_string()
}

///|
pub impl Show for GgufError with fn to_string(self : GgufError) {
  self.message()
}
