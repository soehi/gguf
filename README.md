# soehi/gguf

Read-only GGUF container parsing and ggml block dequantization, in pure
MoonBit. No FFI, no third-party dependencies; builds and tests on `native`
and `wasm-gc`.

This is the container + numeric layer of the MoonLM project, split out so it
can be reused on its own: anything that needs to open a GGUF and get floats
out of it can depend on this module without also taking the inference engine,
the tokenizer, or the agent runtime.

## Install

```bash
moon add soehi/gguf@0.1.1
```

## Usage

The module does no file I/O: you hand it a `Bytes` and it hands back metadata
and numbers. That is what lets the same code run in a browser.

```moonbit
fn inspect(buf : Bytes) -> Unit {
  // parse with full metadata when you need the vocabulary; plain `parse`
  // summarizes large arrays (a 100k-entry token list) instead of materializing.
  let file = match @gguf.parse(buf) {
    Ok(f) => f
    Err(e) => {
      println("parse failed: \{e.message()}")
      return
    }
  }
  // Metadata is typed, by key: get_string / get_int / get_double / get.
  let arch = file.get_string("general.architecture")   // String?
  println(arch.unwrap_or("unknown"))

  // Tensor lookup is an Option, not a Result.
  let idx = match file.index_of_tensor("blk.0.attn_q.weight") {
    Some(i) => i
    None => {
      println("no such tensor")
      return
    }
  }
  let info = file.tensors[idx]
  println("dims: \{info.n_dims()}, dequantizable: \{info.is_dequantizable()}")

  // Raw payload (owned copy), or decoded floats. Both are Result; an
  // unsupported storage type is an Err that names the type.
  let raw = match file.tensor_bytes(idx) {
    Ok(b) => b
    Err(e) => {
      println("cannot read payload: \{e.message()}")
      return
    }
  }
  println("payload bytes: \{raw.length()}")
  let vals = match file.dequant_tensor(idx) {
    Ok(v) => v
    Err(e) => {
      println("cannot dequantize: \{e.message()}")
      return
    }
  }
  println("decoded floats: \{vals.length()}")
}
```

Two details worth knowing up front:

- `index_of_tensor` returns `Option[Int]`, while `tensor_bytes`,
  `tensor_span` and `dequant_tensor` return `Result[_, GgufError]`. The error
  carries the file offset where the problem was found.
- `tensor_span` gives offsets into the buffer you already hold, so a caller
  that keeps `buf` alive can avoid the copy `tensor_bytes` makes.
- `dequant_tensor` returns `Err` (not a trap) for a storage type this build
  does not decode, naming the type; `is_dequantizable()` on `TensorInfo` asks
  the same question ahead of time.

## Packages

| Import | What it holds |
|---|---|
| `soehi/gguf` | The GGUF container: header, metadata key/values, tensor directory. |
| `soehi/gguf/tensor` | The ggml type table (all 43 ids) and block dequantization kernels. |

## Scope

**Read-only parsing.** Given a `Bytes` holding a GGUF file, you get the
version, the alignment, every metadata key/value, and the tensor directory
(name, dims, type, offset, element count, byte size). No file I/O is
performed here — the buffer comes from the caller, which is what lets this
package work unchanged on a browser, where there is no filesystem.

**Dequantization to f32.** Seven storage types are decoded (five block-quantized
plus f32/f16): f32, f16, q4_0, q5_0, q8_0, q4_K, q6_K. Every kernel is a
line-by-line transcription of the
matching `dequantize_row_*` in llama.cpp's `ggml/src/ggml-quants.c`, with
byte offsets taken from the `block_*` structs in `ggml/src/ggml-common.h`.
The expected values in the golden tests were produced by an independent numpy
reference and asserted to round-trip, so agreement is evidence rather than
tautology.

**Bounded failure.** A malformed container returns a named error with an
offset, never a trap. A short or misaligned payload is rejected rather than
truncated. A type this build cannot decode is reported by name, with its
block size still computed correctly so a directory listing stays usable.

## Non-goals

These are deliberately absent, and adding any of them would be a feature
request rather than a bug fix:

- **Writing GGUF.** This module does not produce files. Serializing a
  container needs a policy for alignment and metadata ordering that the
  reader cannot infer.
- **Quantization.** Only dequantization. Producing q4_0/q5_0 blocks means
  choosing a quantization objective, which is a different problem with a
  different set of tradeoffs.
- **Inference.** No matvec, no attention, no sampling. The consumer supplies
  the arithmetic; this module stops at "give me the numbers for this block".
- **Tokenizer, model config, or engine glue.** Those live in the MoonLM
  module and are what depend on this one, not the other way round.

## Verification

- Golden vectors per block type, at 1e-6 relative tolerance, on both targets.
- Cross-checked against llama.cpp's own Python `gguf` reader on real models
  (header, all metadata, the complete tensor directory, and decoded values).
- Malformed-input coverage: truncation at many offsets, byte-level mutation,
  and one hand-aimed case per parser guard.

See `docs/gguf-notes.md` for the byte layouts, the type table, and the
implementation notes.

## Regenerating the test fixtures

The golden vectors and the synthetic GGUF container are generated, not
hand-written. Their generator lives here:

```bash
pip install numpy
python tools/gen_fixtures.py
```

It writes `src/gen_fixture_test.mbt` and `src/tensor/gen_golden_test.mbt`.
CI re-runs it and fails if the checked-in files differ, so editing the
generator without regenerating is caught rather than silently accumulating.

## Maintenance

Maintained by **soehi** (the MoonLM author).

- **Semver.** Breaking changes bump the major version. Within a major
  version, new dequantizable types and new metadata helpers are additive and
  land in a minor; bug fixes and error-message improvements land in a patch.
- **Issues.** Please file them in the
  [gguf](https://github.com/soehi/gguf) repository.
- **License.** Apache-2.0, same as MoonLM.
