// ggml storage-type table.
//
// Ids, names, block sizes and per-block byte sizes are transcribed from
// llama.cpp: the `GGML_TYPE_*` enum in `ggml/include/ggml.h`, the
// `type_traits[]` array in `ggml/src/ggml.c`, and the `block_*` structs plus
// `static_assert`s in `ggml/src/ggml-common.h`. Those three files are the
// authority for every number below.

///|
/// A ggml tensor storage type as it appears in a GGUF tensor info.
///
/// `block_size` is the number of elements covered by one on-disk block, or 1
/// for types stored element-by-element. `type_size` is the on-disk byte size of
/// one block, or of one element when `block_size` is 1.
///
/// Types whose ids were removed from the format (`q4_2`, `q4_3`, the repacked
/// `q4_0_4_4` variants, the repacked `iq4_nl` variants) are still listed by
/// name so that a tensor directory parses, but report `block_size` 0 and cannot
/// be sized or dequantized.
pub(all) struct GgmlType {
  id : Int
  name : String
  block_size : Int
  type_size : Int
} derive(Eq, Debug)

///|
/// Look up a ggml type by its on-disk id.
///
/// Unknown ids yield a type with `block_size` 0 whose name records the id, so
/// callers can report "unknown type N" without guessing a layout.
pub fn GgmlType::from_id(id : Int) -> GgmlType {
  match id {
    0 => { id: 0, name: "f32", block_size: 1, type_size: 4 }
    1 => { id: 1, name: "f16", block_size: 1, type_size: 2 }
    2 => { id: 2, name: "q4_0", block_size: 32, type_size: 18 }
    3 => { id: 3, name: "q4_1", block_size: 32, type_size: 20 }
    4 => { id: 4, name: "q4_2 (removed)", block_size: 0, type_size: 0 }
    5 => { id: 5, name: "q4_3 (removed)", block_size: 0, type_size: 0 }
    6 => { id: 6, name: "q5_0", block_size: 32, type_size: 22 }
    7 => { id: 7, name: "q5_1", block_size: 32, type_size: 24 }
    8 => { id: 8, name: "q8_0", block_size: 32, type_size: 34 }
    9 => { id: 9, name: "q8_1", block_size: 32, type_size: 36 }
    10 => { id: 10, name: "q2_K", block_size: 256, type_size: 84 }
    11 => { id: 11, name: "q3_K", block_size: 256, type_size: 110 }
    12 => { id: 12, name: "q4_K", block_size: 256, type_size: 144 }
    13 => { id: 13, name: "q5_K", block_size: 256, type_size: 176 }
    14 => { id: 14, name: "q6_K", block_size: 256, type_size: 210 }
    15 => { id: 15, name: "q8_K", block_size: 256, type_size: 292 }
    16 => { id: 16, name: "iq2_xxs", block_size: 256, type_size: 66 }
    17 => { id: 17, name: "iq2_xs", block_size: 256, type_size: 74 }
    18 => { id: 18, name: "iq3_xxs", block_size: 256, type_size: 98 }
    19 => { id: 19, name: "iq1_s", block_size: 256, type_size: 50 }
    20 => { id: 20, name: "iq4_nl", block_size: 32, type_size: 18 }
    21 => { id: 21, name: "iq3_s", block_size: 256, type_size: 110 }
    22 => { id: 22, name: "iq2_s", block_size: 256, type_size: 82 }
    23 => { id: 23, name: "iq4_xs", block_size: 256, type_size: 136 }
    24 => { id: 24, name: "i8", block_size: 1, type_size: 1 }
    25 => { id: 25, name: "i16", block_size: 1, type_size: 2 }
    26 => { id: 26, name: "i32", block_size: 1, type_size: 4 }
    27 => { id: 27, name: "i64", block_size: 1, type_size: 8 }
    28 => { id: 28, name: "f64", block_size: 1, type_size: 8 }
    29 => { id: 29, name: "iq1_m", block_size: 256, type_size: 56 }
    30 => { id: 30, name: "bf16", block_size: 1, type_size: 2 }
    31 => {
      id: 31,
      name: "q4_0_4_4 (removed)",
      block_size: 0,
      type_size: 0,
    }
    32 => {
      id: 32,
      name: "q4_0_4_8 (removed)",
      block_size: 0,
      type_size: 0,
    }
    33 => {
      id: 33,
      name: "q4_0_8_8 (removed)",
      block_size: 0,
      type_size: 0,
    }
    34 => { id: 34, name: "tq1_0", block_size: 256, type_size: 54 }
    35 => { id: 35, name: "tq2_0", block_size: 256, type_size: 66 }
    36 => {
      id: 36,
      name: "iq4_nl_4_4 (removed)",
      block_size: 0,
      type_size: 0,
    }
    37 => {
      id: 37,
      name: "iq4_nl_4_8 (removed)",
      block_size: 0,
      type_size: 0,
    }
    38 => {
      id: 38,
      name: "iq4_nl_8_8 (removed)",
      block_size: 0,
      type_size: 0,
    }
    39 => { id: 39, name: "mxfp4", block_size: 32, type_size: 17 }
    40 => { id: 40, name: "nvfp4", block_size: 64, type_size: 36 }
    41 => { id: 41, name: "q1_0", block_size: 128, type_size: 18 }
    42 => { id: 42, name: "q2_0", block_size: 64, type_size: 18 }
    _ => { id, name: "unknown", block_size: 0, type_size: 0 }
  }
}

///|
/// Total number of ids in the ggml type enum (`GGML_TYPE_COUNT`).
pub let ggml_type_count : Int = 43

///|
/// Whether the id maps to a real storage layout rather than `unknown` or a
/// removed type.
pub fn GgmlType::is_known(self : GgmlType) -> Bool {
  self.block_size > 0
}

///|
/// Whether the type packs several elements into one block (i.e. is quantized).
pub fn GgmlType::is_quantized(self : GgmlType) -> Bool {
  self.block_size > 1
}

///|
/// Whether this build can dequantize the type.
///
/// Covers f32, f16, q4_0, q5_0, q8_0, q4_K and q6_K. Everything else is parsed
/// and sized correctly but rejected on dequantization with a named error.
pub fn GgmlType::is_dequantizable(self : GgmlType) -> Bool {
  match self.id {
    0 | 1 | 2 | 6 | 8 | 12 | 14 => true
    _ => false
  }
}

///|
/// On-disk byte size of `n_elems` elements of this type.
///
/// Mirrors `ggml_nbytes()` for a contiguous tensor: `n_elems / block_size`
/// blocks times the per-block byte size. Element counts that do not fill whole
/// blocks are rejected rather than rounded, because a real GGUF never contains
/// them and silently truncating would misreport the data section.
pub fn GgmlType::nbytes(self : GgmlType, n_elems : Int64) -> Result[Int64, TensorError] {
  if n_elems < 0L {
    return Err(NegativeLength(n_elems=n_elems))
  }
  if !self.is_known() {
    return Err(UnknownType(type_id=self.id))
  }
  let block_size = self.block_size.to_int64()
  if n_elems % block_size != 0L {
    return Err(NotBlockAligned(n_elems=n_elems, block_size=self.block_size))
  }
  Ok(n_elems / block_size * self.type_size.to_int64())
}
