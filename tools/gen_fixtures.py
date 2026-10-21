#!/usr/bin/env python3
"""Generate MoonBit test fixtures for this module's GGUF and tensor packages.

Two kinds of fixture are produced, both as *data-only* `.mbt` files so that
`moon test` runs identically on `native` and `wasm-gc` with no file I/O:

  1. A synthetic GGUF container exercising all 12 metadata value types, nested
     arrays, an array long enough to be summarized, non-default alignment and
     one tensor of every M0 dequantization type.

  2. Golden dequantization vectors: hand-built ggml blocks plus the expected
     floats computed here with numpy. The expected values are emitted as exact
     IEEE-754 f64 bit patterns, so the generated file cannot drift through
     decimal rounding.

The numpy reference implementations deliberately re-derive the layouts from
llama.cpp's `dequantize_row_q*` functions rather than reusing any MoonBit code,
so an agreement between the two is real evidence and not a tautology.

Usage:
    python tools/gen_fixtures.py

Writes:
    src/gen_fixture_test.mbt
    src/tensor/gen_golden_test.mbt
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# This generator lives in the gguf module's own repository, so it writes
# straight into that module's `src/`.
GGUF_OUT = REPO_ROOT / "src" / "gen_fixture_test.mbt"
TENSOR_OUT = REPO_ROOT / "src" / "tensor" / "gen_golden_test.mbt"

# ggml type ids, from GGML_TYPE_* in llama.cpp's ggml.h.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14

# GGUF metadata value-type ids, from gguf.h.
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian


# --------------------------------------------------------------------------
# MoonBit literal rendering
# --------------------------------------------------------------------------


def bytes_literal(data: bytes, per_line: int = 16) -> str:
    r"""Render `data` as a concatenation of MoonBit byte-string literals.

    A byte-string literal cannot span lines in MoonBit, and a top-level `let`
    in a blackbox test file cannot call a helper defined in another test file,
    so the fixture is emitted as `b".." + b".." + ...`. That keeps the file
    self-contained and compiles on both native and wasm-gc.
    """
    chunks = []
    for i in range(0, len(data), per_line):
        chunk = data[i : i + per_line]
        body = "".join("\\x%02x" % b for b in chunk)
        chunks.append('b"%s"' % body)
    if not chunks:
        return 'b""'
    joined = (" +\n  ").join(chunks)
    return "\n  " + joined


def f64_bits(values) -> str:
    """Render floats as an `Array[UInt64]` of exact IEEE-754 bit patterns."""
    items = [f"0x{struct.unpack('<Q', struct.pack('<d', float(v)))[0]:016X}UL" for v in values]
    lines = []
    for i in range(0, len(items), 4):
        lines.append("  " + ", ".join(items[i : i + 4]) + ",")
    return "[\n" + "\n".join(lines) + "\n]"


# --------------------------------------------------------------------------
# GGUF writer
# --------------------------------------------------------------------------


class GgufWriter:
    """Minimal GGUF v3 writer, mirroring llama.cpp's layout."""

    def __init__(self, alignment: int = 32):
        self.alignment = alignment
        self.kv_entries: list[bytes] = []
        self.tensors: list[dict] = []

    # -- primitives --------------------------------------------------------

    @staticmethod
    def _u8(v):    return struct.pack("<B", v)
    @staticmethod
    def _i8(v):    return struct.pack("<b", v)
    @staticmethod
    def _u16(v):   return struct.pack("<H", v)
    @staticmethod
    def _i16(v):   return struct.pack("<h", v)
    @staticmethod
    def _u32(v):   return struct.pack("<I", v)
    @staticmethod
    def _i32(v):   return struct.pack("<i", v)
    @staticmethod
    def _f32(v):   return struct.pack("<f", v)
    @staticmethod
    def _u64(v):   return struct.pack("<Q", v)
    @staticmethod
    def _i64(v):   return struct.pack("<q", v)
    @staticmethod
    def _f64(v):   return struct.pack("<d", v)

    @staticmethod
    def _string(s: str) -> bytes:
        raw = s.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    # -- metadata ----------------------------------------------------------

    def kv(self, key: str, type_id: int, payload: bytes) -> None:
        self.kv_entries.append(self._string(key) + self._u32(type_id) + payload)

    def kv_scalar(self, key: str, value) -> None:
        """Write a scalar with the type tag inferred from the Python value."""
        if isinstance(value, bool):
            self.kv(key, GGUF_TYPE_BOOL, self._u8(1 if value else 0))
        elif isinstance(value, np.uint8) or (isinstance(value, int) and value < 0x100 and key.endswith(".u8")):
            self.kv(key, GGUF_TYPE_UINT8, self._u8(value))
        elif isinstance(value, int):
            self.kv(key, GGUF_TYPE_INT32, self._i32(value))
        elif isinstance(value, float):
            self.kv(key, GGUF_TYPE_FLOAT32, self._f32(value))
        elif isinstance(value, str):
            self.kv(key, GGUF_TYPE_STRING, self._string(value))
        else:
            raise TypeError(f"uninferred metadata type for {key!r}: {type(value)}")

    def kv_array(self, key: str, elem_type: int, encoded_elems: list[bytes]) -> None:
        payload = self._u32(elem_type) + self._u64(len(encoded_elems)) + b"".join(encoded_elems)
        self.kv(key, GGUF_TYPE_ARRAY, payload)

    def kv_array_raw(self, key: str, elem_type: int, count: int, body: bytes) -> None:
        """Write an array whose body is supplied pre-encoded (for nesting)."""
        payload = self._u32(elem_type) + self._u64(count) + body
        self.kv(key, GGUF_TYPE_ARRAY, payload)

    # -- tensors -----------------------------------------------------------

    def tensor(self, name: str, dims: list[int], type_id: int, payload: bytes) -> None:
        self.tensors.append(
            {"name": name, "dims": dims, "type_id": type_id, "payload": payload}
        )

    # -- serialization -----------------------------------------------------

    def build(self) -> tuple[bytes, dict]:
        """Serialize, returning the bytes plus the facts a test should assert."""
        # Tensor payload offsets, each padded to the alignment.
        offset = 0
        facts_offsets = []
        for t in self.tensors:
            t["offset"] = offset
            facts_offsets.append(offset)
            padded = (len(t["payload"]) + self.alignment - 1) // self.alignment * self.alignment
            offset += padded

        header = (
            self._u32(GGUF_MAGIC)
            + self._u32(3)  # version
            + self._u64(len(self.tensors))
            + self._u64(len(self.kv_entries))
        )
        meta = b"".join(self.kv_entries)
        tensor_info = b"".join(self._tensor_info_bytes(t) for t in self.tensors)
        body_pre_pad = header + meta + tensor_info
        pad = (-len(body_pre_pad)) % self.alignment
        prefix = body_pre_pad + b"\x00" * pad

        data = bytearray()
        for i, t in enumerate(self.tensors):
            data += t["payload"]
            if i != len(self.tensors) - 1:
                data += b"\x00" * (t["offset"] + len(t["payload"]) - (t["offset"] + len(t["payload"])))
            # pad each tensor (except the last) up to the alignment
            if i != len(self.tensors) - 1:
                nxt = self.tensors[i + 1]["offset"]
                data += b"\x00" * (nxt - len(data))

        buf = prefix + bytes(data)
        facts = {
            "version": 3,
            "alignment": self.alignment,
            "data_start": len(prefix),
            "tensor_offsets": facts_offsets,
            "tensor_names": [t["name"] for t in self.tensors],
            "tensor_dims": [t["dims"] for t in self.tensors],
            "tensor_type_ids": [t["type_id"] for t in self.tensors],
            "kv_count": len(self.kv_entries),
        }
        return buf, facts

    def _tensor_info_bytes(self, t: dict) -> bytes:
        out = self._string(t["name"]) + self._u32(len(t["dims"]))
        for d in t["dims"]:
            out += self._u64(d)
        out += self._u32(t["type_id"])
        out += self._u64(t["offset"])
        return out


# --------------------------------------------------------------------------
# ggml block builders + numpy reference dequantizers
# --------------------------------------------------------------------------


def f16_bits(x: float) -> int:
    """Round a float to the nearest f16 bit pattern (matches llama.cpp's cast)."""
    return int(np.float16(x).view(np.uint16))


def load_f16(block: bytes, offset: int = 0) -> float:
    """Read an f16 bit pattern and widen it to a Python float.

    `np.float16(bits)` would reinterpret the integer *numerically* rather than
    as a bit pattern, which silently multiplies any scale by ~32768. Decoding
    through a buffer view is what actually reinterprets the bits.
    """
    return float(np.frombuffer(block, dtype="<f2", count=1, offset=offset)[0])


def build_q4_0(values: np.ndarray, d: float) -> bytes:
    """Q4_0 block: f16 d + 16 bytes of nibbles. Low nibble = elem j, high = elem j+16."""
    block = struct.pack("<H", f16_bits(d))
    for j in range(16):
        lo = int(values[j]) + 8
        hi = int(values[j + 16]) + 8
        block += bytes([(lo & 0xF) | ((hi & 0xF) << 4)])
    return block


def dequant_q4_0(block: bytes) -> np.ndarray:
    """Reference from dequantize_row_q4_0 in ggml-quants.c."""
    d = load_f16(block, 0)
    out = np.zeros(32, dtype=np.float64)
    for j in range(16):
        b = block[2 + j]
        out[j] = ((b & 0x0F) - 8) * d
        out[j + 16] = ((b >> 4) - 8) * d
    return out


def build_q5_0(values: np.ndarray, d: float) -> bytes:
    """Q5_0 block: f16 d + 4 high-bit bytes + 16 nibble-pair bytes.

    Element j stores its low 4 bits in qs like q4_0 and its fifth bit as bit j
    of the little-endian 32-bit qh word. Values are centred on 16.
    """
    block = struct.pack("<H", f16_bits(d))
    qh = 0
    for j in range(32):
        if (int(values[j]) + 16) & 0x10:
            qh |= 1 << j
    block += struct.pack("<I", qh)
    for j in range(16):
        lo = (int(values[j]) + 16) & 0x0F
        hi = (int(values[j + 16]) + 16) & 0x0F
        block += bytes([lo | (hi << 4)])
    return block


def dequant_q5_0(block: bytes) -> np.ndarray:
    """Reference from dequantize_row_q5_0 in ggml-quants.c.

    value = d * ((qs nibble | qh bit << 4) - 16), where qh's bit j belongs to
    element j: qh[j/4] >> (j%4) for the byte extraction.
    """
    d = load_f16(block, 0)
    qh = struct.unpack("<I", block[2:6])[0]
    out = np.zeros(32, dtype=np.float64)
    for j in range(16):
        b = block[6 + j]
        xh = (qh >> j) & 1
        yh = (qh >> (j + 16)) & 1
        out[j] = (((b & 0x0F) | (xh << 4)) - 16) * d
        out[j + 16] = (((b >> 4) | (yh << 4)) - 16) * d
    return out


def build_q8_0(values: np.ndarray, d: float) -> bytes:
    """Q8_0 block: f16 d + 32 int8 values."""
    block = struct.pack("<H", f16_bits(d))
    block += struct.pack("<32b", *[int(v) for v in values])
    return block


def dequant_q8_0(block: bytes) -> np.ndarray:
    """Reference from dequantize_row_q8_0 in ggml-quants.c."""
    d = load_f16(block, 0)
    qs = np.frombuffer(block, dtype=np.int8, count=32, offset=2).astype(np.float64)
    return qs * d


def pack_scale_min_k4(scales: list[int], mins: list[int]) -> bytes:
    """Inverse of get_scale_min_k4, for building a q4_K block."""
    q = bytearray(12)
    for j in range(8):
        s, m = scales[j] & 0x3F, mins[j] & 0x3F
        if j < 4:
            q[j] |= s
            q[j + 4] |= m
        else:
            q[j + 4] |= (s & 0x0F) | ((m & 0x0F) << 4)
            q[j - 4] |= ((s >> 4) & 0x03) << 6
            q[j] |= ((m >> 4) & 0x03) << 6
    return bytes(q)


def build_q4_K(quants: np.ndarray, scales: list[int], mins: list[int], d: float, dmin: float) -> bytes:
    """Q4_K super-block: d, dmin, 12 scale bytes, 128 nibble-pair bytes.

    Each 32-element sub-block spans two 32-byte runs of qs: the low nibble of
    qs[g*32+l] carries element g*64+l and its high nibble carries element
    g*64+32+l. Pairing adjacent elements instead (the obvious-but-wrong layout)
    would decode to a permutation of the intended values, which the round-trip
    self-check below catches.
    """
    block = struct.pack("<H", f16_bits(d)) + struct.pack("<H", f16_bits(dmin))
    block += pack_scale_min_k4(scales, mins)
    for g in range(4):
        for l in range(32):
            lo = int(quants[g * 64 + l]) & 0x0F
            hi = int(quants[g * 64 + 32 + l]) & 0x0F
            block += bytes([lo | (hi << 4)])
    return block


def dequant_q4_K(block: bytes) -> np.ndarray:
    """Reference from dequantize_row_q4_K + get_scale_min_k4 in ggml-quants.c.

    The nibble pairs are laid out so that qs[group*32 + l]'s low nibble is
    element group*64 + l and its high nibble is element group*64 + 32 + l.
    """
    d = load_f16(block, 0)
    dmin = load_f16(block, 2)
    q = block[4:16]
    qs = block[16:144]

    def get_scale_min_k4(j: int) -> tuple[int, int]:
        if j < 4:
            return q[j] & 63, q[j + 4] & 63
        return (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )

    out = np.zeros(256, dtype=np.float64)
    idx = 0
    for group in range(4):
        sc0, m0 = get_scale_min_k4(group * 2)
        sc1, m1 = get_scale_min_k4(group * 2 + 1)
        d1, min1 = d * sc0, dmin * m0
        d2, min2 = d * sc1, dmin * m1
        for l in range(32):
            b = qs[group * 32 + l]
            out[idx] = d1 * (b & 0xF) - min1
            out[idx + 32] = d2 * (b >> 4) - min2
            idx += 1
        idx += 32
    return out


def build_q6_K(quants: np.ndarray, scales: list[int], d: float) -> bytes:
    """Q6_K super-block: 128 ql bytes, 64 qh bytes, 16 int8 scales, f16 d.

    Element n in 0..255 is stored as: n = half*128 + plane*32 + l, with the low
    nibble in ql and bits 4..5 in qh. This is the inverse of the loop in
    dequantize_row_q6_K.
    """
    ql = bytearray(128)
    qh = bytearray(64)
    for n in range(256):
        half = n // 128
        r = n % 128
        l = r % 32
        plane = r // 32
        u = int(quants[n]) + 32  # stored unsigned in 0..63
        low = u & 0x0F
        high = (u >> 4) & 0x03
        # ql index within this half; ql word index is 64*half + l (+32 for planes 1,3)
        ql_idx = half * 64 + (32 if plane in (1, 3) else 0) + l
        # low nibble goes in the low half of the byte for planes 0,1 and the
        # high half for planes 2,3.
        target_byte = ql[ql_idx]
        if plane in (0, 1):
            ql[ql_idx] = (target_byte & 0xF0) | low
        else:
            ql[ql_idx] = (target_byte & 0x0F) | (low << 4)
        # qh word index is 32*half + l, bit pair = plane
        qh_idx = half * 32 + l
        qh[qh_idx] = qh[qh_idx] | (high << (2 * plane))
    block = bytes(ql) + bytes(qh)
    block += struct.pack("<16b", *[int(s) for s in scales])
    block += struct.pack("<H", f16_bits(d))
    return block


def dequant_q6_K(block: bytes) -> np.ndarray:
    """Reference from dequantize_row_q6_K in ggml-quants.c."""
    d = load_f16(block, 208)
    ql = block[:128]
    qh = block[128:192]
    sc = np.frombuffer(block[192:208], dtype=np.int8).astype(np.float64)

    out = np.zeros(256, dtype=np.float64)
    for n in range(0, 256, 128):
        for l in range(32):
            si = l // 16
            q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
            out[n + l] = d * sc[si + 0] * q1
            out[n + l + 32] = d * sc[si + 2] * q2
            out[n + l + 64] = d * sc[si + 4] * q3
            out[n + l + 96] = d * sc[si + 6] * q4
        ql = ql[64:]
        qh = qh[32:]
        sc = sc[8:]
    return out


def build_f32(values: np.ndarray) -> bytes:
    return values.astype("<f4").tobytes()


def dequant_f32(block: bytes) -> np.ndarray:
    return np.frombuffer(block, dtype="<f4", count=1).astype(np.float64)


def build_f16(values: np.ndarray) -> bytes:
    return values.astype("<f2").tobytes()


def dequant_f16(block: bytes) -> np.ndarray:
    return np.frombuffer(block, dtype="<f2", count=1).astype(np.float64)


# --------------------------------------------------------------------------
# Synthetic GGUF fixture
# --------------------------------------------------------------------------


def make_tensor_payloads():
    """One payload per M0 dequantization type, sized to whole blocks."""
    tensors = []

    # Used in order; where the stream starts matters for reproducibility, so
    # q5_0 (the newest entry) draws after q4_0 and before q8_0.
    rng = np.random.default_rng(20261014)

    # f32: 8 elements.
    f32_vals = np.array([0.5, -1.25, 3.75, 0.0, -0.0625, 128.0, -256.5, 1.0], dtype=np.float32)
    tensors.append(("output.weight", [8], GGML_TYPE_F32, build_f32(f32_vals)))

    # f16: 16 elements.
    f16_vals = np.array([1.5, -2.25, 0.125, -0.5, 64.0, -0.0078125, 1024.0, 0.0,
                         7.0, -9.5, 0.25, -1.0, 2.0, -3.0, 0.0625, -128.0], dtype=np.float16)
    tensors.append(("token_embd.weight", [16], GGML_TYPE_F16, build_f16(f16_vals)))

    # q4_0: 32 elements = 1 block.
    q4_quants = np.array(
        [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
         7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8],
        dtype=np.int8,
    )
    q4_0_block = build_q4_0(q4_quants, 0.125)
    tensors.append(("blk.0.attn_q.weight", [32, 2], GGML_TYPE_Q4_0, q4_0_block * 2))

    # q5_0: 64 elements = 2 blocks.
    q5_quants = rng.integers(-16, 16, size=64, dtype=np.int8)
    q5_0_blocks = b"".join(build_q5_0(q5_quants[i * 32 : (i + 1) * 32], 0.09375) for i in range(2))
    tensors.append(("blk.0.attn_k.weight", [32, 2], GGML_TYPE_Q5_0, q5_0_blocks))

    # q8_0: 128 elements = 4 blocks.
    q8_quants = rng.integers(-128, 128, size=128, dtype=np.int8)
    q8_0_blocks = b"".join(build_q8_0(q8_quants[i * 32 : (i + 1) * 32], 0.03125) for i in range(4))
    tensors.append(("blk.0.ffn_gate.weight", [32, 4], GGML_TYPE_Q8_0, q8_0_blocks))

    # q4_K: 256 elements = 1 super-block.
    q4k_quants = rng.integers(0, 16, size=256, dtype=np.int8)
    q4k_scales = [1, 2, 3, 4, 5, 6, 7, 8]
    q4k_mins = [8, 7, 6, 5, 4, 3, 2, 1]
    q4k_block = build_q4_K(q4k_quants, q4k_scales, q4k_mins, 0.0625, 0.03125)
    tensors.append(("blk.0.ffn_down.weight", [256, 1], GGML_TYPE_Q4_K, q4k_block))

    # q6_K: 256 elements = 1 super-block.
    q6k_quants = rng.integers(-32, 32, size=256, dtype=np.int8)
    q6k_scales = list(range(-8, 8))
    q6k_block = build_q6_K(q6k_quants, q6k_scales, 0.0078125)
    tensors.append(("blk.0.ffn_up.weight", [256, 1], GGML_TYPE_Q6_K, q6k_block))

    return tensors


def build_synthetic_gguf():
    w = GgufWriter(alignment=64)

    # --- one key per metadata value type, so all 12 parse paths are exercised ---
    w.kv("test.u8", GGUF_TYPE_UINT8, GgufWriter._u8(200))
    w.kv("test.i8", GGUF_TYPE_INT8, GgufWriter._i8(-100))
    w.kv("test.u16", GGUF_TYPE_UINT16, GgufWriter._u16(60000))
    w.kv("test.i16", GGUF_TYPE_INT16, GgufWriter._i16(-30000))
    w.kv("test.u32", GGUF_TYPE_UINT32, GgufWriter._u32(4000000000))
    w.kv("test.i32", GGUF_TYPE_INT32, GgufWriter._i32(-2000000000))
    w.kv("test.f32", GGUF_TYPE_FLOAT32, GgufWriter._f32(3.5))
    w.kv("test.bool_true", GGUF_TYPE_BOOL, GgufWriter._u8(1))
    w.kv("test.bool_false", GGUF_TYPE_BOOL, GgufWriter._u8(0))
    w.kv("test.string", GGUF_TYPE_STRING, GgufWriter._string("hello, 世界"))
    w.kv("test.u64", GGUF_TYPE_UINT64, GgufWriter._u64(0xFFFFFFFFFFFF))
    w.kv("test.i64", GGUF_TYPE_INT64, GgufWriter._i64(-1234567890123))
    w.kv("test.f64", GGUF_TYPE_FLOAT64, GgufWriter._f64(-2.718281828459045))

    # --- arrays: strings, ints, nested, and one long enough to be summarized ---
    w.kv_array("test.array.strings", GGUF_TYPE_STRING,
               [GgufWriter._string(s) for s in ["alpha", "beta", "gamma"]])
    w.kv_array("test.array.i32", GGUF_TYPE_INT32,
               [GgufWriter._i32(v) for v in [10, 20, 30, 40]])
    w.kv_array("test.array.f32", GGUF_TYPE_FLOAT32,
               [GgufWriter._f32(v) for v in [0.5, 1.5, 2.5]])
    w.kv_array("test.array.empty", GGUF_TYPE_UINT8, [])

    # Nested: an array whose elements are themselves arrays.
    inner_a = GgufWriter._u32(GGUF_TYPE_INT32) + GgufWriter._u64(2) + GgufWriter._i32(1) + GgufWriter._i32(2)
    inner_b = GgufWriter._u32(GGUF_TYPE_INT32) + GgufWriter._u64(3) + GgufWriter._i32(3) + GgufWriter._i32(4) + GgufWriter._i32(5)
    w.kv_array_raw("test.array.nested", GGUF_TYPE_ARRAY, 2, inner_a + inner_b)

    # A 2048-entry string array: past the summarize threshold, so the parser
    # must keep only a bounded prefix instead of materializing all of it.
    long_tokens = [f"tok{i}" for i in range(2048)]
    w.kv_array("test.array.long", GGUF_TYPE_STRING,
               [GgufWriter._string(s) for s in long_tokens])

    # --- realistic model keys, so the CLI has something meaningful to print ---
    w.kv("general.architecture", GGUF_TYPE_STRING, GgufWriter._string("llama"))
    w.kv("general.alignment", GGUF_TYPE_UINT32, GgufWriter._u32(64))
    w.kv("llama.block_count", GGUF_TYPE_UINT32, GgufWriter._u32(12))
    w.kv("llama.attention.head_count", GGUF_TYPE_UINT32, GgufWriter._u32(9))
    w.kv("llama.attention.head_count_kv", GGUF_TYPE_UINT32, GgufWriter._u32(3))
    w.kv("llama.embedding_length", GGUF_TYPE_UINT32, GgufWriter._u32(576))
    w.kv("llama.context_length", GGUF_TYPE_UINT32, GgufWriter._u32(2048))

    for name, dims, type_id, payload in make_tensor_payloads():
        w.tensor(name, dims, type_id, payload)

    buf, facts = w.build()
    facts["long_array_count"] = 2048
    facts["long_array_prefix"] = [f"tok{i}" for i in range(8)]
    facts["tensor_nbytes"] = [len(t["payload"]) for t in w.tensors]
    return buf, facts


def gguf_header(tensor_count: int, kv_count: int, *, magic: int = GGUF_MAGIC, version: int = 3) -> bytes:
    """Frame a GGUF header. Used for the hand-aimed malformed fixtures."""
    return (
        GgufWriter._u32(magic)
        + GgufWriter._u32(version)
        + GgufWriter._u64(tensor_count)
        + GgufWriter._u64(kv_count)
    )


def kv_entry(key: str, type_id: int, payload: bytes, *, key_len: int | None = None) -> bytes:
    """One metadata entry, optionally lying about the key length."""
    raw = key.encode("utf-8")
    declared = len(raw) if key_len is None else key_len
    return struct.pack("<Q", declared) + raw + struct.pack("<I", type_id) + payload


def tensor_entry(
    name: str, dims: list[int], type_id: int, offset: int, *, n_dims: int | None = None
) -> bytes:
    """One tensor-directory entry, optionally lying about the rank."""
    raw = name.encode("utf-8")
    out = struct.pack("<Q", len(raw)) + raw
    out += struct.pack("<I", len(dims) if n_dims is None else n_dims)
    for d in dims:
        out += struct.pack("<Q", d)
    out += struct.pack("<I", type_id)
    out += struct.pack("<Q", offset)
    return out


def build_malformed_ggufs() -> dict[str, bytes]:
    """Purpose-built malformed containers, one per failure mode under test.

    Framing is done here rather than by hand in MoonBit: the u64 header counts
    are easy to miscount by hand, and a mis-framed buffer tests the wrong error.
    """
    out: dict[str, bytes] = {}

    out["bad_magic"] = b"MOON" + GgufWriter._u32(3) + GgufWriter._u64(0) + GgufWriter._u64(0)

    out["header_truncated"] = b"GGUF"

    out["version_1"] = gguf_header(0, 0, version=1)

    out["version_255"] = gguf_header(0, 0, version=255)

    out["empty_container"] = gguf_header(0, 0)

    # kv_count claims 2^40 entries in a 24-byte file.
    out["kv_count_huge"] = gguf_header(0, 1 << 40)

    # tensor_count claims 2^40 entries in a 24-byte file.
    out["tensor_count_huge"] = gguf_header(1 << 40, 0)

    # One entry whose value_type is 99, outside the enum.
    out["unknown_value_type"] = gguf_header(0, 1) + kv_entry("k", 99, b"")

    # A key length pointing past the end of the buffer.
    out["key_too_long"] = gguf_header(0, 1) + kv_entry("k", GGUF_TYPE_UINT8, b"", key_len=255)

    # An array whose declared element type (13) is not a value type.
    out["illegal_array_elem_type"] = (
        gguf_header(0, 1)
        + kv_entry(
            "k",
            GGUF_TYPE_ARRAY,
            GgufWriter._u32(13) + GgufWriter._u64(0),
        )
    )

    # An array claiming far more elements than the file can hold.
    out["array_count_overflow"] = (
        gguf_header(0, 1)
        + kv_entry(
            "k",
            GGUF_TYPE_ARRAY,
            GgufWriter._u32(GGUF_TYPE_INT32) + GgufWriter._u64(1 << 40),
        )
    )

    # general.alignment = u32 3, not a power of two.
    out["alignment_not_pow2"] = (
        gguf_header(0, 1) + kv_entry("general.alignment", GGUF_TYPE_UINT32, GgufWriter._u32(3))
    )

    # general.alignment present as f32 instead of u32.
    out["alignment_wrong_type"] = (
        gguf_header(0, 1)
        + kv_entry("general.alignment", GGUF_TYPE_FLOAT32, GgufWriter._f32(1.0))
    )

    # A tensor declaring five dimensions, past GGML_MAX_DIMS.
    out["tensor_five_dims"] = (
        gguf_header(1, 0)
        + tensor_entry("t", [1, 1, 1, 1, 1], GGML_TYPE_F32, 0, n_dims=5)
    )

    # A tensor entry cut off before its trailing offset field. The body is long
    # enough to satisfy the directory's minimum-size estimate, so the failure is
    # reported as Truncated at the offset field rather than as an implausible
    # count.
    out["tensor_dir_truncated"] = (
        gguf_header(1, 0)
        + struct.pack("<Q", 1)
        + b"t"
        + struct.pack("<I", 1)
        + struct.pack("<Q", 8)
        + struct.pack("<I", GGML_TYPE_F32)
    )

    # An f32 tensor of 8 elements (32 bytes) whose declared payload is larger
    # than the data section that follows it. The payload is short enough that
    # the data section itself is in range, so the failure is attributed to the
    # tensor rather than to the section start.
    out["tensor_out_of_range"] = (
        gguf_header(1, 0)
        + tensor_entry("t", [8], GGML_TYPE_F32, 0)
        + bytes(30)
    )

    # A tensor whose offset alone runs past a populated data section.
    out["tensor_offset_past_data"] = (
        gguf_header(1, 0)
        + tensor_entry("t", [8], GGML_TYPE_F32, 1 << 20)
        + bytes(32)
    )

    return out


def build_deep_nested_gguf() -> bytes:
    """A file whose metadata nests arrays 9 deep, past the parser's limit."""
    w = GgufWriter(alignment=32)
    body = GgufWriter._u32(GGUF_TYPE_INT32) + GgufWriter._u64(1) + GgufWriter._i32(0)
    for _ in range(9):
        body = GgufWriter._u32(GGUF_TYPE_ARRAY) + GgufWriter._u64(1) + body
    w.kv_array_raw("test.deep", GGUF_TYPE_ARRAY, 1, body)
    buf, _ = w.build()
    return buf


# --------------------------------------------------------------------------
# .mbt emission
# --------------------------------------------------------------------------


def emit_gguf_fixture(buf: bytes, facts: dict, deep_buf: bytes, malformed: dict[str, bytes]) -> None:
    parts = [
        "// GENERATED by tools/gen_fixtures.py -- do not edit by hand.",
        "//",
        "// Data-only fixture consumed by src/parse_test.mbt. Kept as compiled-in",
        "// bytes so `moon test` needs no filesystem on either target.",
        "",
        "///|",
        "/// A synthetic GGUF v3 file: all 12 metadata value types, nested arrays, a",
        "/// 2048-entry string array (past the summarize threshold), 64-byte",
        "/// alignment, and one tensor of each M0 dequantization type.",
        f"let synthetic_gguf : Bytes = {bytes_literal(buf)}",
        "",
        "///|",
        f"let synthetic_gguf_len : Int64 = {len(buf)}L",
        "",
        "///|",
        f"let synthetic_version : UInt = {facts['version']}U",
        "",
        "///|",
        f"let synthetic_alignment : UInt = {facts['alignment']}U",
        "",
        "///|",
        f"let synthetic_data_start : Int64 = {facts['data_start']}L",
        "",
        "///|",
        f"let synthetic_kv_count : Int = {facts['kv_count']}",
        "",
        "///|",
        "let synthetic_tensor_names : Array[String] = ["
        + ", ".join(f'"{n}"' for n in facts["tensor_names"])
        + "]",
        "",
        "///|",
        "let synthetic_tensor_type_ids : Array[Int] = ["
        + ", ".join(str(t) for t in facts["tensor_type_ids"])
        + "]",
        "",
        "///|",
        "let synthetic_tensor_nbytes : Array[Int64] = ["
        + ", ".join(f"{n}L" for n in facts["tensor_nbytes"])
        + "]",
        "",
        "///|",
        "let synthetic_tensor_offsets : Array[Int64] = ["
        + ", ".join(f"{o}L" for o in facts["tensor_offsets"])
        + "]",
        "",
        "///|",
        "let synthetic_tensor_dims : Array[Array[Int64]] = ["
        + ", ".join("[" + ", ".join(f"{d}L" for d in dims) + "]" for dims in facts["tensor_dims"])
        + "]",
        "",
        "///|",
        f"let long_array_count : Int = {facts['long_array_count']}",
        "",
        "///|",
        "let long_array_prefix : Array[String] = ["
        + ", ".join(f'"{s}"' for s in facts["long_array_prefix"])
        + "]",
        "",
        "///|",
        "/// A GGUF whose metadata nests arrays nine deep; parsing must refuse it",
        "/// rather than recursing without bound.",
        f"let deep_nested_gguf : Bytes = {bytes_literal(deep_buf)}",
        "",
    ]

    parts += [
        "///|",
        "/// Malformed containers, one per failure mode. Built here with explicit",
        "/// little-endian framing so each buffer exercises exactly one parser guard",
        "/// rather than accidentally failing earlier on a mis-sized count field.",
        "",
    ]
    for name, data in malformed.items():
        parts += [
            "///|",
            f"let malformed_{name} : Bytes = {bytes_literal(data)}",
            "",
        ]
    GGUF_OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {GGUF_OUT.relative_to(REPO_ROOT)} ({GGUF_OUT.stat().st_size} bytes)")


def emit_tensor_golden() -> None:
    """Emit block bytes plus exact expected values for all six M0 types."""
    rng = np.random.default_rng(777)
    sections: list[str] = []
    header = [
        "// GENERATED by tools/gen_fixtures.py -- do not edit by hand.",
        "//",
        "// Golden dequantization vectors consumed by src/tensor/golden_test.mbt.",
        "// Each `*_blocks` literal holds the raw ggml block bytes; the matching",
        "// `*_expected_bits` array holds the reference output as exact IEEE-754 f64",
        "// bit patterns computed by numpy from llama.cpp's dequantize_row_q* code.",
        "",
    ]

    def section(suffix: str, name: str, docs: list[str], blocks: bytes, expected: np.ndarray):
        out = ["///|", *[f"/// {d}" for d in docs], f"let {name}_blocks : Bytes = {bytes_literal(blocks)}", ""]
        out += ["///|", f"let {name}_expected_bits : Array[UInt64] = {f64_bits(expected)}", ""]
        out += ["///|", f"let {name}_elems : Int64 = {len(expected)}L", ""]
        sections.append("\n".join(out))

    # f32: 8 elements, no blocks involved.
    f32_vals = np.array([0.5, -1.25, 3.75, 0.0, -0.0625, 128.0, -256.5, 1.0], dtype=np.float32)
    f32_blocks = build_f32(f32_vals)
    f32_expected = np.concatenate([dequant_f32(f32_blocks[i * 4 : i * 4 + 4]) for i in range(8)])
    section("f32", "golden_f32", ["f32: 8 elements stored as raw IEEE 754 little-endian floats."],
            f32_blocks, f32_expected)

    # f16: 16 elements, one per block.
    f16_vals = np.array([1.5, -2.25, 0.125, -0.5, 64.0, -0.0078125, 1024.0, 0.0,
                         7.0, -9.5, 0.25, -1.0, 2.0, -3.0, 0.0625, -128.0], dtype=np.float16)
    f16_blocks = build_f16(f16_vals)
    f16_expected = np.concatenate([dequant_f16(f16_blocks[i * 2 : i * 2 + 2]) for i in range(16)])
    section("f16", "golden_f16",
            ["f16: 16 elements, one half-precision block each. Values were chosen to include",
             "zero, negatives and exact powers of two so the bit-level decode is unambiguous."],
            f16_blocks, f16_expected)

    # q4_0: 2 blocks.
    q4a = rng.integers(-8, 8, size=32, dtype=np.int8)
    q4b = rng.integers(-8, 8, size=32, dtype=np.int8)
    q4_0_blocks = build_q4_0(q4a, 0.125) + build_q4_0(q4b, 0.25)
    q4_0_expected = np.concatenate([dequant_q4_0(q4_0_blocks[:18]), dequant_q4_0(q4_0_blocks[18:])])
    section("q4_0", "golden_q4_0",
            ["q4_0: 2 blocks of 32 elements. Nibbles are centred on 8 and scaled by an f16 d."],
            q4_0_blocks, q4_0_expected)

    # q8_0: 4 blocks.
    q8_quants = rng.integers(-128, 128, size=128, dtype=np.int8)
    q8_0_blocks = b"".join(build_q8_0(q8_quants[i * 32 : (i + 1) * 32], 0.03125) for i in range(4))
    q8_0_expected = np.concatenate([dequant_q8_0(q8_0_blocks[i * 34 : (i + 1) * 34]) for i in range(4)])
    section("q8_0", "golden_q8_0",
            ["q8_0: 4 blocks of 32 int8 values, each scaled by an f16 d."],
            q8_0_blocks, q8_0_expected)

    # q4_K: 2 super-blocks, varied scales and mins so the 6-bit packing matters.
    q4k_blocks = b""
    q4k_expected = []
    for i in range(2):
        quants = rng.integers(0, 16, size=256, dtype=np.int8)
        scales = [int(v) for v in rng.integers(1, 64, size=8)]
        mins = [int(v) for v in rng.integers(1, 64, size=8)]
        d = float(np.float16(0.0625 * (i + 1)))
        dmin = float(np.float16(0.03125 * (i + 1)))
        blk = build_q4_K(quants, scales, mins, d, dmin)
        q4k_blocks += blk
        q4k_expected.append(dequant_q4_K(blk))
    section("q4_K", "golden_q4_K",
            ["q4_K: 2 super-blocks of 256 elements. Scales and mins use values above 15 in",
             "both the low and high nibble ranges, so the get_scale_min_k4 bit packing is",
             "exercised rather than trivially absent."],
            q4k_blocks, np.concatenate(q4k_expected))

    # q6_K: 2 super-blocks, negative and positive scales.
    q6k_blocks = b""
    q6k_expected = []
    for i in range(2):
        quants = rng.integers(-32, 32, size=256, dtype=np.int8)
        scales = [int(v) for v in rng.integers(-128, 128, size=16)]
        d = float(np.float16(0.0078125 * (i + 1)))
        blk = build_q6_K(quants, scales, d)
        q6k_blocks += blk
        q6k_expected.append(dequant_q6_K(blk))
    section("q6_K", "golden_q6_K",
            ["q6_K: 2 super-blocks of 256 elements. Scales span the full signed byte range,",
             "and the 2-bit high planes are all in use."],
            q6k_blocks, np.concatenate(q6k_expected))

    # q5_0: 2 blocks. Draws from the rng after every pre-existing section so
    # the committed bytes for those types are unchanged by this addition.
    q5a = rng.integers(-16, 16, size=32, dtype=np.int8)
    q5b = rng.integers(-16, 16, size=32, dtype=np.int8)
    q5_0_blocks = build_q5_0(q5a, 0.125) + build_q5_0(q5b, 0.0625)
    q5_0_expected = np.concatenate([dequant_q5_0(q5_0_blocks[:22]), dequant_q5_0(q5_0_blocks[22:])])
    section("q5_0", "golden_q5_0",
            ["q5_0: 2 blocks of 32 elements. Quants span -16..15, so the fifth-bit word has",
             "both zero and one bits set in every byte position."],
            q5_0_blocks, q5_0_expected)

    TENSOR_OUT.write_text("\n".join(header) + "\n".join(sections), encoding="utf-8")
    print(f"wrote {TENSOR_OUT.relative_to(REPO_ROOT)} ({TENSOR_OUT.stat().st_size} bytes)")


def main() -> None:
    buf, facts = build_synthetic_gguf()
    deep_buf = build_deep_nested_gguf()
    malformed = build_malformed_ggufs()
    emit_gguf_fixture(buf, facts, deep_buf, malformed)
    emit_tensor_golden()

    # Round-trip every builder through its reference dequantizer and compare
    # against the values the builder was handed. Without this, a builder and a
    # reference could be wrong in the same way and still agree in the .mbt file.
    selfcheck()

    print("fixtures written")


def selfcheck() -> None:
    """Verify each build_* / dequant_* pair against independently known values."""
    exact = 0.0  # tolerances are for f16 scale rounding only
    tol = 1e-6

    # f32 / f16: the payload is the value.
    vals32 = np.array([0.5, -1.25, 3.75, 0.0], dtype=np.float32)
    got32 = np.concatenate([dequant_f32(build_f32(vals32)[i * 4 : i * 4 + 4]) for i in range(4)])
    assert np.allclose(got32, vals32.astype(np.float64)), got32

    vals16 = np.array([1.5, -2.25, 0.125, -0.5], dtype=np.float16)
    got16 = np.concatenate([dequant_f16(build_f16(vals16)[i * 2 : i * 2 + 2]) for i in range(4)])
    assert np.allclose(got16, vals16.astype(np.float64)), got16

    # q4_0: quants in -8..7, value = d * q.
    q4q = np.array([-8, -1, 0, 1, 7, -4, 3, -7] * 4, dtype=np.int8)
    d0 = 0.5
    got = dequant_q4_0(build_q4_0(q4q, d0))
    assert np.allclose(got, q4q.astype(np.float64) * float(np.float16(d0)), atol=tol), (got, q4q)

    # q8_0: value = d * q.
    q8q = np.arange(-16, 16, dtype=np.int8)
    d8 = 0.25
    got = dequant_q8_0(build_q8_0(q8q, d8))
    assert np.allclose(got, q8q.astype(np.float64) * float(np.float16(d8)), atol=tol), got

    # q5_0: quants in -16..15, value = d * q, fifth bit in qh.
    q5q = np.array([-16, -1, 0, 1, 15, -8, 7, -15,
                    16 - 1, -3, 2, -5, 9, -11, 4, 13,
                    0, -2, 3, -4, 5, -6, 8, -9,
                    10, -12, 11, -13, 14, -15, 1, -1], dtype=np.int16)
    d5 = 0.5
    got = dequant_q5_0(build_q5_0(q5q, d5))
    assert np.allclose(got, q5q.astype(np.float64) * float(np.float16(d5)), atol=tol), (got, q5q)

    # q4_K: value = d * sc * q - dmin * m for every sub-block.
    quants = np.arange(16, dtype=np.int8).repeat(16)  # 256 entries, per-sub-block constant
    scales = [1, 2, 3, 4, 5, 6, 7, 8]
    mins = [8, 7, 6, 5, 4, 3, 2, 1]
    dk, dmink = 0.5, 0.25
    got = dequant_q4_K(build_q4_K(quants, scales, mins, dk, dmink))
    fd, fdm = float(np.float16(dk)), float(np.float16(dmink))
    for sub in range(8):
        for l in range(32):
            idx = sub * 32 + l
            want = fd * scales[sub] * float(quants[idx]) - fdm * mins[sub]
            assert abs(got[idx] - want) < tol, (idx, got[idx], want)

    # q6_K: value = d * sc * (q) with q in -32..31.
    q6q = np.array([-32, -1, 0, 1, 31, -17, 5, -9] * 32, dtype=np.int8)
    sc6 = list(range(-8, 8))
    d6 = 0.125
    got = dequant_q6_K(build_q6_K(q6q, sc6, d6))
    fd6 = float(np.float16(d6))
    for n in range(256):
        want = fd6 * sc6[n // 16] * float(q6q[n])
        assert abs(got[n] - want) < tol, (n, got[n], want)

    print("self-check passed for f32/f16/q4_0/q5_0/q8_0/q4_K/q6_K")


if __name__ == "__main__":
    main()
