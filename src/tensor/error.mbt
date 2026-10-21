// Dequantization errors. Kept separate from `gguf`'s parse errors so that
// `tensor/` stays a standalone package with no dependency on container parsing.

///|
/// Error type for tensor byte-size computation and dequantization.
pub(all) suberror TensorError {
  /// The ggml type id is not in the known type table at all.
  UnknownType(type_id~ : Int)
  /// The type is known by name, but this build cannot dequantize it.
  UnsupportedType(type_id~ : Int, name~ : String)
  /// Element count is negative.
  NegativeLength(n_elems~ : Int64)
  /// Element count is not a whole number of blocks for this type.
  NotBlockAligned(n_elems~ : Int64, block_size~ : Int)
  /// Element count cannot be indexed within a 32-bit buffer.
  LengthOverflow(n_elems~ : Int64)
  /// The source region starting at `offset` does not hold `needed` bytes.
  BufferTooSmall(offset~ : Int64, needed~ : Int64, available~ : Int64)
  /// The destination buffer has no room for `n_elems` values at `out_offset`.
  OutputTooSmall(out_offset~ : Int, n_elems~ : Int64, out_len~ : Int)
} derive(Eq, Debug)

///|
/// Human-readable rendering of a `TensorError`, including the offending
/// offset or type id where one applies.
pub fn TensorError::message(self : TensorError) -> String {
  match self {
    UnknownType(type_id~) => "unknown ggml type id \{type_id}"
    UnsupportedType(type_id~, name~) =>
      "unsupported ggml type \{type_id} (\{name}): dequantization not implemented"
    NegativeLength(n_elems~) => "negative element count \{n_elems}"
    NotBlockAligned(n_elems~, block_size~) =>
      "\{n_elems} elements is not a multiple of block size \{block_size}"
    LengthOverflow(n_elems~) => "element count \{n_elems} exceeds 32-bit index range"
    BufferTooSmall(offset~, needed~, available~) =>
      "buffer too small at offset \{offset}: need \{needed} bytes, have \{available}"
    OutputTooSmall(out_offset~, n_elems~, out_len~) =>
      "output too small at \{out_offset}: need \{n_elems} floats, have \{out_len}"
  }
}

///|
pub impl Show for TensorError with fn to_string(self) {
  self.message()
}
