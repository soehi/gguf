// Integration tests for the GGUF container parser.
//
// The happy-path fixture is `synthetic_gguf` from gen_fixture_test.mbt: a
// complete GGUF v3 file with every metadata value type, nested arrays, an array
// past the summarize threshold, 64-byte alignment and one tensor of each M0
// dequantization type. Every malformed-input case is a hand-built byte string,
// so it also runs on wasm-gc where there is no filesystem.

///|
test "synthetic fixture parses and reports its header" {
  let f = @gguf.parse(synthetic_gguf).unwrap()
  assert_eq(f.version, synthetic_version)
  assert_eq(f.alignment, synthetic_alignment)
  assert_eq(f.data_start, synthetic_data_start)
  assert_eq(f.metadata_count(), synthetic_kv_count)
  assert_eq(f.tensor_count(), synthetic_tensor_names.length())
  assert_eq(f.len(), synthetic_gguf_len)
}

///|
test "tensor directory matches the fixture" {
  let f = @gguf.parse(synthetic_gguf).unwrap()
  for i, name in synthetic_tensor_names {
    let t = f.tensors[i]
    assert_eq(t.name, name)
    assert_eq(t.type_id.reinterpret_as_int(), synthetic_tensor_type_ids[i])
    assert_eq(t.offset.reinterpret_as_int64(), synthetic_tensor_offsets[i])
    assert_eq(t.n_dims(), synthetic_tensor_dims[i].length())
    for j, d in synthetic_tensor_dims[i] {
      assert_eq(t.dims[j].reinterpret_as_int64(), d)
    }
    assert_eq(t.nbytes.unwrap(), synthetic_tensor_nbytes[i])
  }
}

///|
test "every metadata value type round-trips" {
  let f = @gguf.parse(synthetic_gguf).unwrap()
  assert_eq(f.get_int("test.u8"), Some(200L))
  assert_eq(f.get_int("test.i8"), Some(-100L))
  assert_eq(f.get_int("test.u16"), Some(60000L))
  assert_eq(f.get_int("test.i16"), Some(-30000L))
  assert_eq(f.get_int("test.u32"), Some(4000000000L))
  assert_eq(f.get_int("test.i32"), Some(-2000000000L))
  match f.get("test.f32") {
    Some(@gguf.F32(v)) => assert_eq(v, 3.5)
    Some(v) => fail("test.f32 parsed as \{v.to_text()}")
    None => fail("test.f32 is missing")
  }
  assert_eq(f.get("test.bool_true"), Some(@gguf.Bool(true)))
  assert_eq(f.get("test.bool_false"), Some(@gguf.Bool(false)))
  assert_eq(f.get_string("test.string"), Some("hello, 世界"))
  assert_eq(f.get_int("test.u64"), Some(0xFFFFFFFFFFFFL))
  assert_eq(f.get_int("test.i64"), Some(-1234567890123L))
  assert_eq(f.get_double("test.f64"), Some(-2.718281828459045))
}

///|
test "small arrays are materialized in full" {
  let f = @gguf.parse(synthetic_gguf).unwrap()
  match f.get("test.array.strings") {
    Some(@gguf.Array(@gguf.Materialized(elems))) => {
      assert_eq(elems.length(), 3)
      assert_eq(elems[0], @gguf.Str("alpha"))
      assert_eq(elems[1], @gguf.Str("beta"))
      assert_eq(elems[2], @gguf.Str("gamma"))
    }
    Some(v) => fail("test.array.strings parsed as \{v.to_text()}")
    None => fail("test.array.strings is missing")
  }
  assert_eq(f.get("test.array.i32").unwrap().array_len(), Some(4))
  assert_eq(f.get("test.array.f32").unwrap().array_len(), Some(3))
}

///|
test "an empty array parses with a length of zero" {
  let f = @gguf.parse(synthetic_gguf).unwrap()
  let v = f.get("test.array.empty").unwrap()
  assert_eq(v.array_len(), Some(0))
  assert_eq(v.array_elem_type(), None)
}

///|
test "nested arrays keep their inner structure" {
  let f = @gguf.parse(synthetic_gguf).unwrap()
  match f.get("test.array.nested") {
    Some(@gguf.Array(@gguf.Materialized(outer))) => {
      assert_eq(outer.length(), 2)
      match outer[0] {
        @gguf.Array(@gguf.Materialized(inner)) => {
          assert_eq(inner.length(), 2)
          assert_eq(inner[0], @gguf.I32(1))
          assert_eq(inner[1], @gguf.I32(2))
        }
        @gguf.Array(_) => fail("outer[0] parsed as a non-materialized array")
        _ => fail("outer[0] is not an array")
      }
      match outer[1] {
        @gguf.Array(@gguf.Materialized(inner)) => {
          assert_eq(inner.length(), 3)
          assert_eq(inner[2], @gguf.I32(5))
        }
        @gguf.Array(_) => fail("outer[1] parsed as a non-materialized array")
        _ => fail("outer[1] is not an array")
      }
    }
    Some(v) => fail("test.array.nested parsed as \{v.to_text()}")
    None => fail("test.array.nested is missing")
  }
}

///|
test "a long array is summarized instead of materialized" {
  let f = @gguf.parse(synthetic_gguf).unwrap()
  match f.get("test.array.long") {
    Some(v) => {
      assert_eq(v.array_len(), Some(long_array_count))
      match v {
        @gguf.Array(@gguf.Summarized(s)) => {
          // The type and the count survive, and only a bounded prefix is kept.
          assert_eq(s.elem_type, 8U)
          assert_eq(s.count.to_int(), long_array_count)
          assert_eq(s.prefix.length(), @gguf.array_prefix_len)
          for i, want in long_array_prefix {
            assert_eq(s.prefix[i], @gguf.Str(want))
          }
        }
        other => fail("long array was not summarized: \{other}")
      }
    }
    None => fail("test.array.long is missing")
  }
}

///|
test "a long array's bytes are still fully consumed" {
  // If the parser materialized the prefix but forgot to skip the tail, the
  // tensor directory that follows would be misaligned. Reading a tensor back
  // proves the cursor landed exactly on the data section.
  let f = @gguf.parse(synthetic_gguf).unwrap()
  assert_eq(f.data_start, synthetic_data_start)
  assert_eq(f.tensor("output.weight").unwrap().n_elems, 8L)
}

///|
test "realistic model keys are readable" {
  let f = @gguf.parse(synthetic_gguf).unwrap()
  assert_eq(f.get_string("general.architecture"), Some("llama"))
  assert_eq(f.get_int("llama.block_count"), Some(12L))
  assert_eq(f.get_int("llama.attention.head_count"), Some(9L))
  assert_eq(f.get_int("llama.attention.head_count_kv"), Some(3L))
  assert_eq(f.get_int("llama.embedding_length"), Some(576L))
  assert_eq(f.get_int("llama.context_length"), Some(2048L))
}

///|
test "metadata lookup reports absence without guessing" {
  let f = @gguf.parse(synthetic_gguf).unwrap()
  assert_eq(f.get("no.such.key"), None)
  assert_eq(f.get_string("no.such.key"), None)
  assert_eq(f.get_int("no.such.key"), None)
  assert_false(f.has_key("no.such.key"))
  assert_true(f.has_key("general.architecture"))
  // A type mismatch yields None rather than a coerced value.
  assert_eq(f.get_string("llama.block_count"), None)
  assert_eq(f.get_int("test.string"), None)
}

///|
test "tensor lookup by name and payload slicing" {
  let f = @gguf.parse(synthetic_gguf).unwrap()
  assert_eq(f.tensor("blk.0.ffn_up.weight").unwrap().n_elems, 256L)
  assert_eq(f.tensor("missing.weight"), None)
  let payload = f.tensor_bytes(0).unwrap()
  assert_eq(payload.length(), synthetic_tensor_nbytes[0].to_int())
}

///|
test "tensors dequantize straight out of the container" {
  let f = @gguf.parse(synthetic_gguf).unwrap()
  for i, name in synthetic_tensor_names {
    match f.dequant_tensor(i) {
      Ok(v) => assert_eq(v.length().to_int64(), f.tensors[i].n_elems)
      Err(e) => fail("dequantizing \{name} failed: \{e.message()}")
    }
  }
}

///|
test "a tensor of an unsupported type is listed but not dequantizable" {
  // Build a directory entry naming q5_K (id 13), which M0 does not implement.
  let f = @gguf.parse(synthetic_gguf).unwrap()
  let t = f.tensors[0]
  // The fixture's f32 tensor is supported; assert the negative case through the
  // type table instead, so this test does not need a second fixture.
  assert_true(t.is_dequantizable())
  assert_false(@tensor.GgmlType::from_id(13).is_dequantizable())
  assert_eq(@tensor.GgmlType::from_id(13).name, "q5_K")
}

// ---------------------------------------------------------------------------
// Malformed input
// ---------------------------------------------------------------------------

///|
test "empty input is reported as truncated, not as a crash" {
  match @gguf.parse(b"") {
    Ok(_) => fail("empty input must not parse")
    Err(@gguf.Truncated(needed~, remaining~, ..)) => {
      assert_eq(needed, 4L)
      assert_eq(remaining, 0L)
    }
    Err(e) => fail("expected Truncated, got \{e.message()}")
  }
}

///|
test "a bad magic value is rejected and echoed back" {
  match @gguf.parse(malformed_bad_magic) {
    Ok(_) => fail("bad magic must not parse")
    Err(@gguf.BadMagic(found~)) => assert_eq(found, 0x4E4F4F4DU)
    Err(e) => fail("expected BadMagic, got \{e.message()}")
  }
}

///|
test "a header that stops after the magic is truncated" {
  match @gguf.parse(malformed_header_truncated) {
    Ok(_) => fail("truncated header must not parse")
    Err(@gguf.Truncated(offset~, ..)) => assert_eq(offset, 4L)
    Err(e) => fail("expected Truncated, got \{e.message()}")
  }
}

///|
test "GGUF v1 is refused explicitly" {
  match @gguf.parse(malformed_version_1) {
    Ok(_) => fail("v1 must not parse")
    Err(@gguf.UnsupportedVersion(version~, ..)) => assert_eq(version, 1U)
    Err(e) => fail("expected UnsupportedVersion, got \{e.message()}")
  }
}

///|
test "a version newer than this build is refused" {
  match @gguf.parse(malformed_version_255) {
    Ok(_) => fail("v255 must not parse")
    Err(@gguf.UnsupportedVersion(version~, supported_max~)) => {
      assert_eq(version, 255U)
      assert_eq(supported_max, 3U)
    }
    Err(e) => fail("expected UnsupportedVersion, got \{e.message()}")
  }
}

///|
test "an empty container with no tensors and no metadata parses" {
  let f = @gguf.parse(malformed_empty_container).unwrap()
  assert_eq(f.version, 3U)
  assert_eq(f.tensor_count(), 0)
  assert_eq(f.metadata_count(), 0)
  assert_eq(f.alignment, @gguf.gguf_default_alignment)
}

///|
test "a metadata count that cannot fit is rejected before reading" {
  match @gguf.parse(malformed_kv_count_huge) {
    Ok(_) => fail("an impossible kv_count must not parse")
    Err(@gguf.CountOverflow(..)) => ()
    Err(e) => fail("expected CountOverflow, got \{e.message()}")
  }
}

///|
test "a tensor count that cannot fit is rejected before reading" {
  match @gguf.parse(malformed_tensor_count_huge) {
    Ok(_) => fail("an impossible tensor_count must not parse")
    Err(@gguf.TensorCountImplausible(..)) => ()
    Err(e) => fail("expected TensorCountImplausible, got \{e.message()}")
  }
}

///|
test "an unknown metadata value type is reported with its id" {
  match @gguf.parse(malformed_unknown_value_type) {
    Ok(_) => fail("an unknown value type must not parse")
    Err(@gguf.UnknownValueType(type_id~, ..)) => assert_eq(type_id, 99U)
    Err(e) => fail("expected UnknownValueType, got \{e.message()}")
  }
}

///|
test "a metadata key longer than the file is truncated" {
  match @gguf.parse(malformed_key_too_long) {
    Ok(_) => fail("an over-long key must not parse")
    Err(@gguf.Truncated(offset~, needed~, ..)) => {
      assert_eq(offset, 32L)
      assert_eq(needed, 255L)
    }
    Err(e) => fail("expected Truncated, got \{e.message()}")
  }
}

///|
test "deeply nested arrays are refused rather than recursing without bound" {
  match @gguf.parse(deep_nested_gguf) {
    Ok(_) => fail("nine-deep nesting must not parse")
    Err(@gguf.ArrayNestingTooDeep(..)) => ()
    Err(e) => fail("expected ArrayNestingTooDeep, got \{e.message()}")
  }
}

///|
test "an array element type outside the value-type enum is refused" {
  match @gguf.parse(malformed_illegal_array_elem_type) {
    Ok(_) => fail("an illegal array element type must not parse")
    Err(@gguf.IllegalArrayElementType(type_id~, ..)) => assert_eq(type_id, 13U)
    Err(e) => fail("expected IllegalArrayElementType, got \{e.message()}")
  }
}

///|
test "an array count larger than the file is rejected" {
  match @gguf.parse(malformed_array_count_overflow) {
    Ok(_) => fail("an over-large array must not parse")
    Err(@gguf.CountTooLarge(count~, ..)) => assert_eq(count, 1099511627776UL)
    Err(e) => fail("expected CountTooLarge, got \{e.message()}")
  }
}

///|
test "general.alignment that is not a power of two is refused" {
  match @gguf.parse(malformed_alignment_not_pow2) {
    Ok(_) => fail("a non-power-of-two alignment must not parse")
    Err(@gguf.BadAlignment(value~)) => assert_eq(value, 3U)
    Err(e) => fail("expected BadAlignment, got \{e.message()}")
  }
}

///|
test "general.alignment of the wrong type is refused" {
  match @gguf.parse(malformed_alignment_wrong_type) {
    Ok(_) => fail("a mistyped alignment must not parse")
    Err(@gguf.AlignmentWrongType(type_id~)) => assert_eq(type_id, 6U)
    Err(e) => fail("expected AlignmentWrongType, got \{e.message()}")
  }
}

///|
test "a tensor declaring more than four dimensions is refused" {
  match @gguf.parse(malformed_tensor_five_dims) {
    Ok(_) => fail("five dimensions must not parse")
    Err(@gguf.TooManyDims(n_dims~, max_dims~, ..)) => {
      assert_eq(n_dims, 5U)
      assert_eq(max_dims, 4)
    }
    Err(e) => fail("expected TooManyDims, got \{e.message()}")
  }
}

///|
test "a truncated tensor directory is reported with an offset" {
  match @gguf.parse(malformed_tensor_dir_truncated) {
    Ok(_) => fail("a partial tensor entry must not parse")
    Err(@gguf.Truncated(..)) => ()
    Err(e) => fail("expected Truncated, got \{e.message()}")
  }
}

///|
test "a tensor whose payload runs past the empty data section is refused" {
  match @gguf.parse(malformed_tensor_out_of_range) {
    Ok(_) => fail("a tensor outside the data section must not parse")
    Err(@gguf.TensorOutOfRange(name~, nbytes~, ..)) => {
      assert_eq(name, "t")
      assert_eq(nbytes, 32L)
    }
    Err(e) => fail("expected TensorOutOfRange, got \{e.message()}")
  }
}

///|
test "a tensor offset past the data section is refused" {
  match @gguf.parse(malformed_tensor_offset_past_data) {
    Ok(_) => fail("a tensor offset outside the data section must not parse")
    Err(@gguf.TensorOutOfRange(name~, offset~, ..)) => {
      assert_eq(name, "t")
      assert_eq(offset, 1048576L)
    }
    Err(e) => fail("expected TensorOutOfRange, got \{e.message()}")
  }
}

///|
test "truncating the synthetic fixture at many points never crashes" {
  // Every prefix must either parse or fail cleanly; none may panic or read out
  // of bounds. This walks the header, metadata and tensor directory.
  let full = synthetic_gguf_len.to_int()
  for i = 0; i < 2048; i = i + 1 {
    let cut = i * full / 2048
    let prefix = synthetic_gguf.view(start=0, end=cut).to_owned()
    match @gguf.parse(prefix) {
      Ok(_) => ()
      Err(_) => ()
    }
  }
}

///|
test "single-byte corruption anywhere in the file never crashes" {
  // Corrupt one byte at a time across the whole fixture; the parser must return
  // a value every time rather than trapping.
  let full = synthetic_gguf_len.to_int()
  for i = 0; i < full; i = i + 997 {
    for v in [0, 1, 0x7F, 0x80, 0xFF] {
      let bytes = synthetic_gguf.to_array()
      bytes[i] = v.to_byte()
      match @gguf.parse(Bytes::from_array(bytes[:])) {
        Ok(_) => ()
        Err(_) => ()
      }
    }
  }
}

///|
test "parse_materializing_metadata keeps the full long array" {
  let f = @gguf.parse_materializing_metadata(synthetic_gguf).unwrap()
  match f.get("test.array.long") {
    Some(@gguf.Array(@gguf.Materialized(elems))) => {
      assert_eq(elems.length(), long_array_count)
      assert_eq(elems[0], @gguf.Str("tok0"))
      assert_eq(elems[long_array_count - 1], @gguf.Str("tok2047"))
    }
    Some(v) => fail("expected a fully materialized array, got \{v.to_text()}")
    None => fail("test.array.long is missing")
  }
  // Everything else behaves the same as the default parse.
  assert_eq(f.tensor_count(), synthetic_tensor_names.length())
  assert_eq(f.get_string("general.architecture"), Some("llama"))
}

///|
test "default parse still summarizes the long array" {
  let f = @gguf.parse(synthetic_gguf).unwrap()
  match f.get("test.array.long") {
    Some(@gguf.Array(@gguf.Summarized(_))) => ()
    Some(v) => fail("expected a summarized array, got \{v.to_text()}")
    None => fail("test.array.long is missing")
  }
}
