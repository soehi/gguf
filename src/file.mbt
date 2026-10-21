// GGUF container assembly: header, metadata, tensor directory and data section.
//
// Layout, all little-endian:
//
//   magic            u32  "GGUF" (0x46554747 read as LE)
//   version          u32  2 or 3 (1 is refused)
//   tensor_count     u64
//   metadata_kv_count u64
//   kv[]             key(string) + value_type(u32) + value
//   tensor_info[]    name(string) + n_dims(u32) + dims(u64 * n_dims) + ggml_type(u32) + offset(u64)
//   padding          to `general.alignment` (default 32)
//   data             tensor payloads, each at data_start + offset
//
// Parsing is a single pass with no allocation proportional to total tensor
// bytes: the caller's `Bytes` buffer is retained and tensor payloads are
// reached through `data_start` plus each entry's offset.

///|
/// Maximum tensor rank GGUF can carry (`GGML_MAX_DIMS`).
pub let gguf_max_dims : Int = 4

///|
/// Alignment used when `general.alignment` is absent (`GGUF_DEFAULT_ALIGNMENT`).
pub let gguf_default_alignment : UInt = 32U

///|
/// The magic bytes `GGUF`, read as a little-endian u32.
pub let gguf_magic : UInt = 0x46554747U

///|
/// Highest GGUF version this parser accepts.
pub let gguf_version_max : UInt = 3U

///|
/// One entry of the tensor directory.
///
/// `n_elems` is the product of `dims`. `nbytes` is the on-disk payload size, or
/// `None` for type ids whose block size this build does not know (removed types
/// and ids past the enum) — such a tensor is still listed so a directory dump
/// can show it, but it cannot be range-checked or dequantized.
pub(all) struct TensorInfo {
  name : String
  dims : Array[UInt64]
  type_id : UInt
  offset : UInt64
  n_elems : Int64
  nbytes : Int64?
} derive(Eq, Debug)

///|
/// Number of dimensions, for symmetry with the on-disk `n_dims`.
pub fn TensorInfo::n_dims(self : TensorInfo) -> Int {
  self.dims.length()
}

///|
/// The ggml type of this tensor, resolved through the tensor package's table.
pub fn TensorInfo::ggml_type(self : TensorInfo) -> @tensor.GgmlType {
  @tensor.GgmlType::from_id(self.type_id.reinterpret_as_int())
}

///|
/// Whether this build can dequantize the tensor's type.
pub fn TensorInfo::is_dequantizable(self : TensorInfo) -> Bool {
  self.ggml_type().is_dequantizable()
}

///|
/// A parsed GGUF container. Holds a reference to the caller's buffer so tensor
/// payloads can be sliced on demand rather than copied at parse time.
pub(all) struct GgufFile {
  version : UInt
  alignment : UInt
  tensors : Array[TensorInfo]
  metadata : Array[(String, MetaValue)]
  data_start : Int64
  buf : Bytes
}

///|
pub fn GgufFile::tensor_count(self : GgufFile) -> Int {
  self.tensors.length()
}

///|
pub fn GgufFile::metadata_count(self : GgufFile) -> Int {
  self.metadata.length()
}

///|
/// Total buffer length, including the header, metadata and data sections.
pub fn GgufFile::len(self : GgufFile) -> Int64 {
  self.buf.length().to_int64()
}

///|
/// Bytes available from `data_start` to the end of the buffer.
///
/// This is the upper bound a tensor's `offset + nbytes` may reach; it can
/// exceed the sum of all tensor sizes when the file has trailing padding or
/// embedded alignment gaps.
pub fn GgufFile::data_len(self : GgufFile) -> Int64 {
  self.len() - self.data_start
}

///|
/// Look up a metadata value by key. Returns the first match, since GGUF keys
/// are unique in practice and a duplicate would be corruption.
pub fn GgufFile::get(self : GgufFile, key : String) -> MetaValue? {
  for entry in self.metadata {
    if entry.0 == key {
      return Some(entry.1)
    }
  }
  None
}

///|
/// Whether a key is present.
pub fn GgufFile::has_key(self : GgufFile, key : String) -> Bool {
  self.get(key) is Some(_)
}

///|
/// Look up a string-valued metadata entry.
pub fn GgufFile::get_string(self : GgufFile, key : String) -> String? {
  match self.get(key) {
    Some(v) => v.as_string()
    None => None
  }
}

///|
/// Look up an integer-valued metadata entry, widening any integer width.
pub fn GgufFile::get_int(self : GgufFile, key : String) -> Int64? {
  match self.get(key) {
    Some(v) => v.as_int()
    None => None
  }
}

///|
/// Look up a float-valued metadata entry.
pub fn GgufFile::get_double(self : GgufFile, key : String) -> Double? {
  match self.get(key) {
    Some(v) => v.as_double()
    None => None
  }
}

///|
/// Find a tensor directory entry by name.
pub fn GgufFile::tensor(self : GgufFile, name : String) -> TensorInfo? {
  for t in self.tensors {
    if t.name == name {
      return Some(t)
    }
  }
  None
}

///|
/// Find a tensor directory entry by name, returning its index.
pub fn GgufFile::index_of_tensor(self : GgufFile, name : String) -> Int? {
  for i, t in self.tensors {
    if t.name == name {
      return Some(i)
    }
  }
  None
}

///|
/// The whole buffer this file was parsed from.
///
/// Exposed so a caller can hold one shared reference and address tensors by
/// offset, instead of copying each tensor's payload out and holding a second
/// copy of the weights.
pub fn GgufFile::buffer(self : GgufFile) -> Bytes {
  self.buf
}

///|
/// Where a tensor's payload lives: an offset into the file buffer and its
/// length in bytes.
pub(all) struct TensorSpan {
  /// Byte offset of the payload within the file buffer.
  start : Int
  /// Payload length in bytes.
  byte_len : Int
} derive(Eq, Debug)

///|
/// Locate tensor `index`'s payload without copying it.
///
/// This is what a caller that keeps the file buffer alive should use; the
/// returned offsets address the same `Bytes` that `buffer()` exposes.
pub fn GgufFile::tensor_span(
  self : GgufFile,
  index : Int,
) -> Result[TensorSpan, GgufError] {
  let t = self.tensors[index]
  match t.nbytes {
    None =>
      Err(
        TensorOutOfRange(
          name=t.name,
          offset=0L,
          nbytes=-1L,
          data_len=self.data_len(),
        ),
      )
    Some(nbytes) => {
      let start = self.data_start + t.offset.reinterpret_as_int64()
      Ok({ start: start.to_int(), byte_len: nbytes.to_int() })
    }
  }
}

///|
/// The raw payload bytes of tensor `index`, spanning `nbytes` from the data
/// section.
///
/// Returns an owned copy. Callers that keep the file buffer alive should prefer
/// `tensor_span`, which avoids duplicating the payload.
pub fn GgufFile::tensor_bytes(
  self : GgufFile,
  index : Int,
) -> Result[Bytes, GgufError] {
  let t = self.tensors[index]
  match t.nbytes {
    None => Err(TensorOutOfRange(name=t.name, offset=0L, nbytes=-1L, data_len=self.data_len()))
    Some(nbytes) => {
      let start = self.data_start + t.offset.reinterpret_as_int64()
      let view = self.buf.view(start=start.to_int(), end=(start + nbytes).to_int())
      Ok(view.to_owned())
    }
  }
}

///|
/// Dequantize tensor `index` into a freshly allocated `Array[Float]`.
///
/// The buffer is sliced first so the tensor package sees a region it can
/// bounds-check on its own, and so a payload that sits past the buffer is
/// reported as a GGUF-level error rather than a generic buffer overrun.
pub fn GgufFile::dequant_tensor(
  self : GgufFile,
  index : Int,
) -> Result[Array[Float], GgufError] {
  let t = self.tensors[index]
  if !t.is_dequantizable() {
    return Err(
      TensorNotSupported(
        name=t.name,
        type_id=t.type_id,
        type_name=t.ggml_type().name,
      ),
    )
  }
  match self.tensor_bytes(index) {
    Err(e) => Err(e)
    Ok(payload) =>
      match @tensor.dequant(payload, 0L, t.ggml_type(), t.n_elems) {
        Ok(v) => Ok(v)
        Err(e) => Err(DequantFailed(name=t.name, reason=e.message()))
      }
  }
}

///|
/// Product of the tensor dimensions as `Int64`, or `None` if the product does
/// not fit.
fn product_dims(dims : Array[UInt64]) -> Int64? {
  let mut acc = 1L
  for d in dims {
    if d > 0x7FFFFFFFUL {
      return None
    }
    let n = d.reinterpret_as_int64()
    if n != 0L && acc > 0x7FFFFFFFFFFFFFFFL / n {
      return None
    }
    acc = acc * n
  }
  Some(acc)
}

///|
/// Read `count` metadata key-value pairs.
///
/// `materialize~` forwards to array-valued entries: when true, arrays beyond
/// the usual summarize threshold are decoded in full. `parse` keeps the default
/// (summarize) so a plain `inspect` stays cheap; `parse_materializing_metadata`
/// opts in for callers that consume whole arrays, like the tokenizer.
fn read_metadata(
  c : ByteCursor,
  count : UInt64,
  materialize~ : Bool = false,
) -> Result[Array[(String, MetaValue)], GgufError] {
  let out : Array[(String, MetaValue)] = []
  // Each pair needs at least a 8-byte key length, a 4-byte type and 1 value byte.
  let min_bytes = count * 13UL
  if min_bytes > c.remaining().reinterpret_as_uint64() {
    return Err(CountOverflow(offset=c.pos, count=count, elem_size=13L))
  }
  let n = count.to_int()
  for i = 0; i < n; i = i + 1 {
    let key = match c.read_string() {
      Ok(pair) => pair.0
      Err(e) => return Err(e)
    }
    let type_id = match c.read_u32() {
      Ok(t) => t
      Err(e) => return Err(e)
    }
    let value = match read_meta_value(c, type_id, materialize~) {
      Ok(v) => v
      Err(e) => return Err(e)
    }
    out.push((key, value))
  }
  Ok(out)
}

///|
/// Read `count` tensor directory entries.
fn read_tensor_infos(
  c : ByteCursor,
  count : UInt64,
) -> Result[Array[TensorInfo], GgufError] {
  let out : Array[TensorInfo] = []
  // Smallest possible entry: 8-byte name length, 4-byte n_dims (0),
  // 4-byte type, 8-byte offset.
  let min_bytes = count * 24UL
  if min_bytes > c.remaining().reinterpret_as_uint64() {
    return Err(TensorCountImplausible(count=count, remaining=c.remaining()))
  }
  let n = count.to_int()
  for i = 0; i < n; i = i + 1 {
    let info_start = c.pos
    let name = match c.read_string() {
      Ok(pair) => pair.0
      Err(e) => return Err(e)
    }
    let n_dims = match c.read_u32() {
      Ok(d) => d
      Err(e) => return Err(e)
    }
    if n_dims > gguf_max_dims.reinterpret_as_uint() {
      return Err(TooManyDims(offset=info_start, n_dims=n_dims, max_dims=gguf_max_dims))
    }
    let dims : Array[UInt64] = []
    for i = 0; i < n_dims.reinterpret_as_int(); i = i + 1 {
      match c.read_u64() {
        Ok(d) => dims.push(d)
        Err(e) => return Err(e)
      }
    }
    let type_id = match c.read_u32() {
      Ok(t) => t
      Err(e) => return Err(e)
    }
    let offset = match c.read_u64() {
      Ok(o) => o
      Err(e) => return Err(e)
    }
    let n_elems = match product_dims(dims) {
      Some(v) => v
      None => return Err(CountTooLarge(offset=info_start, count=offset))
    }
    // A tensor whose type this build cannot size stays in the directory with
    // `nbytes = None`; it is still valid input, just not dequantizable.
    let ty = @tensor.GgmlType::from_id(type_id.reinterpret_as_int())
    let nbytes = if ty.is_known() {
      match ty.nbytes(n_elems) {
        Ok(v) => Some(v)
        Err(_) => None
      }
    } else {
      None
    }
    out.push({ name, dims, type_id, offset, n_elems, nbytes })
  }
  Ok(out)
}

///|
/// Round `pos` up to the next multiple of `alignment`.
fn align_up(pos : Int64, alignment : UInt) -> Int64 {
  let a = alignment.reinterpret_as_int().to_int64()
  if a <= 1L {
    pos
  } else {
    (pos + a - 1L) / a * a
  }
}

///|
/// Parse a GGUF container from an in-memory buffer.
///
/// Accepts GGUF v2 and v3; v1 and anything newer are refused. Every failure
/// mode in the format — bad magic, truncation, unknown metadata or tensor type
/// ids, counts that overflow, tensor payloads outside the buffer — produces a
/// `GgufError` carrying the file offset, and no input can cause a panic or an
/// out-of-bounds read.
///
/// Array-valued metadata beyond the summarize threshold is folded into a
/// bounded summary; use `parse_materializing_metadata` to keep the full arrays.
pub fn parse(buf : Bytes) -> Result[GgufFile, GgufError] {
  parse_with(buf, materialize=false)
}

///|
/// Like `parse`, but materializes every array-valued metadata entry in full.
///
/// The tokenizer needs the complete `tokenizer.ggml.tokens` and
/// `tokenizer.ggml.merges` arrays (about 100k strings for SmolLM2), which the
/// default summary would truncate. Everything else — validation, tensor
/// directory, data section — behaves identically.
pub fn parse_materializing_metadata(
  buf : Bytes,
) -> Result[GgufFile, GgufError] {
  parse_with(buf, materialize=true)
}

///|
fn parse_with(buf : Bytes, materialize~ : Bool) -> Result[GgufFile, GgufError] {
  let c = ByteCursor::new(buf)
  let magic = match c.read_u32() {
    Ok(m) => m
    Err(_) => return Err(Truncated(offset=0L, needed=4L, remaining=c.len()))
  }
  if magic != gguf_magic {
    return Err(BadMagic(found=magic))
  }
  let version = match c.read_u32() {
    Ok(v) => v
    Err(e) => return Err(e)
  }
  if version == 1U {
    return Err(UnsupportedVersion(version=version, supported_max=gguf_version_max))
  }
  if version < 2U || version > gguf_version_max {
    return Err(UnsupportedVersion(version=version, supported_max=gguf_version_max))
  }
  let tensor_count = match c.read_u64() {
    Ok(n) => n
    Err(e) => return Err(e)
  }
  let kv_count = match c.read_u64() {
    Ok(n) => n
    Err(e) => return Err(e)
  }
  let metadata = match read_metadata(c, kv_count, materialize~) {
    Ok(m) => m
    Err(e) => return Err(e)
  }
  // `general.alignment` must be read before the data offset can be computed,
  // and it lives in the metadata that was just parsed.
  let alignment = match find_alignment(metadata) {
    Ok(a) => a
    Err(e) => return Err(e)
  }
  let tensors = match read_tensor_infos(c, tensor_count) {
    Ok(t) => t
    Err(e) => return Err(e)
  }
  // llama.cpp only aligns the data offset when there is at least one tensor:
  // a container with no tensors ends where its metadata ends, so padding it to
  // the alignment boundary would point the data section past the file.
  let data_start = if tensors.length() == 0 {
    c.pos
  } else {
    align_up(c.pos, alignment)
  }
  if data_start > c.len() {
    return Err(DataSectionOutOfRange(data_start=data_start, buffer_len=c.len()))
  }
  let file = { version, alignment, tensors, metadata, data_start, buf }
  // Validate every tensor payload against the data section up front, so the
  // CLI and the engine can index tensors without re-checking.
  for t in file.tensors {
    match t.nbytes {
      None => ()
      Some(nbytes) => {
        let end = t.offset.reinterpret_as_int64() + nbytes
        if end > file.data_len() {
          return Err(
            TensorOutOfRange(
              name=t.name,
              offset=t.offset.reinterpret_as_int64(),
              nbytes=nbytes,
              data_len=file.data_len(),
            ),
          )
        }
      }
    }
  }
  Ok(file)
}

///|
/// Resolve `general.alignment`: default 32, must be a positive power of two.
fn find_alignment(metadata : Array[(String, MetaValue)]) -> Result[UInt, GgufError] {
  for entry in metadata {
    if entry.0 == "general.alignment" {
      match entry.1 {
        U32(v) => {
          let a = v.reinterpret_as_int()
          if a <= 0 || (a & (a - 1)) != 0 {
            return Err(BadAlignment(value=v))
          }
          return Ok(v)
        }
        other => return Err(AlignmentWrongType(type_id=other.type_id()))
      }
    }
  }
  Ok(gguf_default_alignment)
}
