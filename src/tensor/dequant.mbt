// Dequantization kernels for the M0 type set: f32, f16, q4_0, q5_0, q8_0,
// q4_K, q6_K.
//
// Each kernel below is a line-by-line transcription of the matching
// `dequantize_row_*` function in llama.cpp's `ggml/src/ggml-quants.c`, with the
// byte offsets taken from the `block_*` structs in `ggml/src/ggml-common.h`.
// Byte order is little-endian throughout.

///|
/// Read one byte as an unsigned value in 0..255.
fn byte_at(buf : Bytes, off : Int64) -> Int {
  buf[off.to_int()].to_int()
}

///|
/// Read one byte as a signed two's-complement value in -128..127.
///
/// `Byte` is unsigned in this toolchain, so sign extension is done explicitly.
fn sbyte_at(buf : Bytes, off : Int64) -> Int {
  let u = byte_at(buf, off)
  if u >= 128 {
    u - 256
  } else {
    u
  }
}

///|
/// Read a little-endian `u16` as an `Int` in 0..65535.
fn u16_le(buf : Bytes, off : Int64) -> Int {
  byte_at(buf, off) | (byte_at(buf, off + 1L) << 8)
}

///|
/// Read a little-endian `u32` as an unsigned bit pattern.
fn u32_le(buf : Bytes, off : Int64) -> UInt {
  buf.unsafe_read_uint32_le(off.to_int())
}

///|
/// Decode IEEE 754 half precision to single precision.
///
/// Handles the three ranges the hardware conversion would: zero, subnormal
/// halves (re-normalized into the f32 exponent range) and infinities/NaNs.
fn f16_to_f32(h : Int) -> Float {
  let sign = (h & 0x8000) << 16
  let exp = (h >> 10) & 0x1F
  let mant = h & 0x3FF
  let bits = if exp == 0 {
    if mant == 0 {
      sign
    } else {
      let mut m = mant
      let mut e = 113
      while (m & 0x400) == 0 {
        m = m << 1
        e = e - 1
      }
      sign | (e << 23) | ((m & 0x3FF) << 13)
    }
  } else if exp == 0x1F {
    sign | 0x7F800000 | (mant << 13)
  } else {
    sign | ((exp + 112) << 23) | (mant << 13)
  }
  Float::reinterpret_from_uint(bits.reinterpret_as_uint())
}

///|
/// Decode bf16 (an f32 with the low 16 mantissa bits truncated) to f32.
fn bf16_to_f32(v : Int) -> Float {
  Float::reinterpret_from_uint(v.reinterpret_as_uint() << 16)
}

///|
/// f32: one element per block, stored little-endian as raw IEEE 754 bits.
fn dq_f32(buf : Bytes, base : Int64, out : FixedArray[Float], o : Int) -> Unit {
  out[o] = Float::reinterpret_from_uint(u32_le(buf, base))
}

///|
/// f16: one element per block, little-endian half precision.
fn dq_f16(buf : Bytes, base : Int64, out : FixedArray[Float], o : Int) -> Unit {
  out[o] = f16_to_f32(u16_le(buf, base))
}

///|
/// q4_0: fp16 scale `d`, then 32 signed 4-bit values packed two per byte.
///
/// The low nibble holds element `j` and the high nibble element `j + 16`;
/// `dequantize_row_q4_0` centres both on 8 before scaling.
fn dq_q4_0(buf : Bytes, base : Int64, out : FixedArray[Float], o : Int) -> Unit {
  let d = f16_to_f32(u16_le(buf, base))
  for j = 0; j < 16; j = j + 1 {
    let b = byte_at(buf, base + 2L + j.to_int64())
    out[o + j] = Float::from_int((b & 0xF) - 8) * d
    out[o + j + 16] = Float::from_int((b >> 4) - 8) * d
  }
}

///|
/// q5_0: fp16 scale `d`, 4 bytes of high bits, then 16 nibble-pair bytes.
///
/// `block_q5_0 { ggml_half d; uint8_t qh[4]; uint8_t qs[16]; }` — 22 bytes per
/// 32 elements. Element `j` takes bit `j` of the 32-bit little-endian `qh` word
/// as its fifth bit (`dequantize_row_q5_0`: `xh_0` from bit `j`, `xh_1` from
/// bit `j + 16`), i.e. byte `j/8` at bit `j%8`. Value =
/// `d * ((nibble | fifth_bit << 4) - 16)`, centred on 16 like q4_0's 8.
fn dq_q5_0(buf : Bytes, base : Int64, out : FixedArray[Float], o : Int) -> Unit {
  let d = f16_to_f32(u16_le(buf, base))
  for j = 0; j < 16; j = j + 1 {
    let b = byte_at(buf, base + 6L + j.to_int64())
    let xh = byte_at(buf, base + 2L + (j >> 3).to_int64())
    let yh = byte_at(buf, base + 2L + ((j + 16) >> 3).to_int64())
    out[o + j] = Float::from_int(
        ((b & 0xF) | (((xh >> (j & 7)) & 1) << 4)) - 16,
      ) * d
    out[o + j + 16] = Float::from_int(
        ((b >> 4) | (((yh >> (j & 7)) & 1) << 4)) - 16,
      ) * d
  }
}

///|
/// q8_0: fp16 scale `d`, then 32 signed 8-bit values.
fn dq_q8_0(buf : Bytes, base : Int64, out : FixedArray[Float], o : Int) -> Unit {
  let d = f16_to_f32(u16_le(buf, base))
  for j = 0; j < 32; j = j + 1 {
    out[o + j] = Float::from_int(sbyte_at(buf, base + 2L + j.to_int64())) * d
  }
}

///|
/// Unpack the 6-bit scale and 6-bit min for sub-block `j` of a q4_K/q5_K super-block.
///
/// `get_scale_min_k4` in ggml-quants.c: sub-blocks 0..3 take their low 6 bits
/// from `q[j]` and `q[j+4]`; sub-blocks 4..7 take their low nibbles from
/// `q[j+4]`, and their high 2 bits from the top bits of `q[j-4]`/`q[j]`.
fn scale_min_k4(j : Int, buf : Bytes, q_off : Int64) -> (Int, Int) {
  if j < 4 {
    (
      byte_at(buf, q_off + j.to_int64()) & 63,
      byte_at(buf, q_off + (j + 4).to_int64()) & 63,
    )
  } else {
    let hi = byte_at(buf, q_off + (j + 4).to_int64())
    let carry_scale = byte_at(buf, q_off + (j - 4).to_int64())
    let carry_min = byte_at(buf, q_off + j.to_int64())
    ((hi & 0xF) | ((carry_scale >> 6) << 4), (hi >> 4) | ((carry_min >> 6) << 4))
  }
}

///|
/// q4_K: 256-element super-block.
///
/// Layout: `d` (f16), `dmin` (f16), 12 packed scale/min bytes, then 128 bytes of
/// 4-bit quants. Eight 32-element sub-blocks; `q4_K`'s quants are unsigned and
/// offset by `dmin * m` rather than centred (`dequantize_row_q4_K`).
fn dq_q4_K(buf : Bytes, base : Int64, out : FixedArray[Float], o : Int) -> Unit {
  let d = f16_to_f32(u16_le(buf, base))
  let min = f16_to_f32(u16_le(buf, base + 2L))
  let sc_base = base + 4L
  let q_base = base + 16L
  for group = 0; group < 4; group = group + 1 {
    let (sc_lo, m_lo) = scale_min_k4(group * 2, buf, sc_base)
    let (sc_hi, m_hi) = scale_min_k4(group * 2 + 1, buf, sc_base)
    let d_lo = d * Float::from_int(sc_lo)
    let min_lo = min * Float::from_int(m_lo)
    let d_hi = d * Float::from_int(sc_hi)
    let min_hi = min * Float::from_int(m_hi)
    let q = q_base + (group * 32).to_int64()
    let y = o + group * 64
    for l = 0; l < 32; l = l + 1 {
      let b = byte_at(buf, q + l.to_int64())
      out[y + l] = d_lo * Float::from_int(b & 0xF) - min_lo
      out[y + l + 32] = d_hi * Float::from_int(b >> 4) - min_hi
    }
  }
}

///|
/// q6_K: 256-element super-block.
///
/// Layout: 128 bytes of low nibbles (`ql`), 64 bytes of high 2-bit planes
/// (`qh`), 16 signed 8-bit sub-block scales, then the f16 scale `d`. The
/// 256 elements are processed as two 128-element halves; within each, `qh`
/// supplies bits 4..5 for elements at stride 32 and `scales` are indexed in
/// steps of 2 per 16-element group (`dequantize_row_q6_K`).
fn dq_q6_K(buf : Bytes, base : Int64, out : FixedArray[Float], o : Int) -> Unit {
  let d = f16_to_f32(u16_le(buf, base + 208L))
  let ql_base = base
  let qh_base = base + 128L
  let sc_base = base + 192L
  for half = 0; half < 2; half = half + 1 {
    let ql = ql_base + (half * 64).to_int64()
    let qh = qh_base + (half * 32).to_int64()
    let sc = sc_base + (half * 8).to_int64()
    let y = o + half * 128
    for l = 0; l < 32; l = l + 1 {
      let sc_idx = l / 16
      let qhv = byte_at(buf, qh + l.to_int64())
      let ql_lo = byte_at(buf, ql + l.to_int64())
      let ql_hi = byte_at(buf, ql + (l + 32).to_int64())
      let q1 = ((ql_lo & 0xF) | (((qhv >> 0) & 3) << 4)) - 32
      let q2 = ((ql_hi & 0xF) | (((qhv >> 2) & 3) << 4)) - 32
      let q3 = ((ql_lo >> 4) | (((qhv >> 4) & 3) << 4)) - 32
      let q4 = ((ql_hi >> 4) | (((qhv >> 6) & 3) << 4)) - 32
      out[y + l] = d *
        Float::from_int(sbyte_at(buf, sc + sc_idx.to_int64())) *
        Float::from_int(q1)
      out[y + l + 32] = d *
        Float::from_int(sbyte_at(buf, sc + (sc_idx + 2).to_int64())) *
        Float::from_int(q2)
      out[y + l + 64] = d *
        Float::from_int(sbyte_at(buf, sc + (sc_idx + 4).to_int64())) *
        Float::from_int(q3)
      out[y + l + 96] = d *
        Float::from_int(sbyte_at(buf, sc + (sc_idx + 6).to_int64())) *
        Float::from_int(q4)
    }
  }
}

///|
/// Apply `f` to each of `n_blocks` consecutive blocks.
///
/// Recursion rather than a `for` loop so nothing is mutated, and so the target
/// functions stay one-block-in/one-block-out. Block counts for real models are
/// in the low thousands to hundreds of thousands; at the default stack this is
/// far inside what the native and wasm-gc backends handle.
fn each_block(
  buf : Bytes,
  start : Int64,
  n_blocks : Int64,
  out : FixedArray[Float],
  out_offset : Int,
  block_bytes : Int,
  block_elems : Int,
  f : (Bytes, Int64, FixedArray[Float], Int) -> Unit,
) -> Unit {
  if n_blocks <= 0L {
    return
  }
  f(buf, start, out, out_offset)
  each_block(
    buf,
    start + block_bytes.to_int64(),
    n_blocks - 1L,
    out,
    out_offset + block_elems,
    block_bytes,
    block_elems,
    f,
  )
}

///|
/// Dispatch a block type to its kernel.
///
/// # Panics
/// Never: unknown and unsupported types are reported by `dequant_into` before
/// this is reached, so every arm here is a supported M0 type.
fn dequant_kernel(
  ty : GgmlType,
  buf : Bytes,
  start : Int64,
  n_blocks : Int64,
  out : FixedArray[Float],
  out_offset : Int,
  f : (Bytes, Int64, FixedArray[Float], Int) -> Unit,
) -> Unit {
  each_block(buf, start, n_blocks, out, out_offset, ty.type_size, ty.block_size, f)
}

///|
/// Dequantize `n_elems` elements of type `ty` from `buf[offset:]` into `out`,
/// starting at `out_offset`.
///
/// Reads are bounds-checked against `buf.length()` before the first block is
/// touched; a short or misaligned source region fails with `BufferTooSmall`
/// rather than reading past the end. The caller owns all storage, so no
/// intermediate allocation happens per element.
pub fn dequant_into(
  buf : Bytes,
  offset : Int64,
  ty : GgmlType,
  n_elems : Int64,
  out : FixedArray[Float],
  out_offset : Int,
) -> Result[Unit, TensorError] {
  if n_elems < 0L {
    return Err(NegativeLength(n_elems=n_elems))
  }
  if !ty.is_known() {
    return Err(UnknownType(type_id=ty.id))
  }
  if !ty.is_dequantizable() {
    return Err(UnsupportedType(type_id=ty.id, name=ty.name))
  }
  if n_elems % ty.block_size.to_int64() != 0L {
    return Err(NotBlockAligned(n_elems=n_elems, block_size=ty.block_size))
  }
  if n_elems > 0x7FFFFFFFL {
    return Err(LengthOverflow(n_elems=n_elems))
  }
  let needed = ty.nbytes(n_elems).unwrap()
  let available = buf.length().to_int64()
  if offset < 0L || offset + needed > available {
    return Err(BufferTooSmall(offset=offset, needed=needed, available=available))
  }
  let out_end = out_offset + n_elems.to_int()
  if out_offset < 0 || out_end > out.length() {
    return Err(OutputTooSmall(out_offset=out_offset, n_elems=n_elems, out_len=out.length()))
  }
  let n_blocks = n_elems / ty.block_size.to_int64()
  match ty.id {
    0 => {
      dequant_kernel(ty, buf, offset, n_blocks, out, out_offset, dq_f32)
      Ok(())
    }
    1 => {
      dequant_kernel(ty, buf, offset, n_blocks, out, out_offset, dq_f16)
      Ok(())
    }
    2 => {
      dequant_kernel(ty, buf, offset, n_blocks, out, out_offset, dq_q4_0)
      Ok(())
    }
    6 => {
      dequant_kernel(ty, buf, offset, n_blocks, out, out_offset, dq_q5_0)
      Ok(())
    }
    8 => {
      dequant_kernel(ty, buf, offset, n_blocks, out, out_offset, dq_q8_0)
      Ok(())
    }
    12 => {
      dequant_kernel(ty, buf, offset, n_blocks, out, out_offset, dq_q4_K)
      Ok(())
    }
    14 => {
      dequant_kernel(ty, buf, offset, n_blocks, out, out_offset, dq_q6_K)
      Ok(())
    }
    _ => Err(UnsupportedType(type_id=ty.id, name=ty.name))
  }
}

///|
/// Dequantize into a freshly allocated `Array[Float]`.
///
/// Convenience for callers that want the values and not the buffer — the CLI
/// and the golden tests. The engine will prefer `dequant_into` with a reusable
/// working buffer so that per-token decoding allocates nothing.
pub fn dequant(
  buf : Bytes,
  offset : Int64,
  ty : GgmlType,
  n_elems : Int64,
) -> Result[Array[Float], TensorError] {
  if n_elems < 0L || n_elems > 0x7FFFFFFFL {
    return Err(LengthOverflow(n_elems=n_elems))
  }
  let out : FixedArray[Float] = FixedArray::make(n_elems.to_int(), 0.0)
  match dequant_into(buf, offset, ty, n_elems, out, 0) {
    Ok(_) => Ok(Array::from_fixed_array(out))
    Err(e) => Err(e)
  }
}

///|
/// Dequantize `n_elems` values of an arbitrary type into a new array.
///
/// Same as `dequant`, but takes a raw id so the CLI can pass a directory entry
/// straight through.
pub fn dequant_by_id(
  buf : Bytes,
  offset : Int64,
  type_id : Int,
  n_elems : Int64,
) -> Result[Array[Float], TensorError] {
  dequant(buf, offset, GgmlType::from_id(type_id), n_elems)
}

///|
/// Decode a bf16 value; exposed for the CLI's eventual bf16 model support.
pub fn bf16_to_float(v : Int) -> Float {
  bf16_to_f32(v)
}

///|
/// Decode an f16 half-precision bit pattern; exposed for tests and the CLI.
pub fn half_to_float(bits : Int) -> Float {
  f16_to_f32(bits)
}
