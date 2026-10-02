#pragma once

// Plans the images (model/GgufImageLayout.hpp) of a Qwen3.8 (qwen35)
// or Qwen3.6 MoE (qwen35moe) target read straight from a llama.cpp GGUF, from
// its metadata alone: section offsets, the header and descriptor bytes, and
// the source rows each tensor section is written from
// (model/GgufPreparation.hpp). A 3-D expert tensor is one quantized tensor of
// experts * N rows.

#include <cstdint>
#include <string>
#include <vector>

#include "model/GgufFile.hpp"
#include "model/QwenHybridLayout.hpp"

namespace splash::model::gguf {

// The general.architecture of a GGUF of a target with this FFN.
[[nodiscard]] constexpr const char *architecture(QwenFfnKind ffn) noexcept {
  return ffn == QwenFfnKind::SparseMoe ? "qwen35moe" : "qwen35";
}

// The general.architecture of a GGUF of a target of these dimensions:
// qwen4exp for Qwen3.8-Flash-Next, else by its FFN.
[[nodiscard]] constexpr const char *architecture(const QwenTargetDimensions &target) noexcept {
  return target.qwen4() ? "qwen4exp" : architecture(target.ffnKind);
}

// The order of a tensor's rows in the image. Rows below `from` keep their
// order; from there on, blocks of headRows rows are value heads, which
// llama.cpp stores tiled (value head of its key head * keyHeads + key head)
// and splash groups by key head (key head * valueHeadsPerKey + value head).
struct RowOrder {
  uint64_t from = UINT64_MAX; // UINT64_MAX: rows as stored
  uint32_t headRows = 0;
  uint32_t keyHeads = 0;
  uint32_t valueHeadsPerKey = 0;
};

// Rows [0, rows) of one source tensor in image order.
struct TensorRows {
  std::string name;
  uint32_t type = 0;   // ggml type
  uint64_t offset = 0; // in the file's tensor data
  uint64_t rows = 0;
  uint64_t rowBytes = 0;
  RowOrder order{};
  uint32_t file = 0;   // the file of a split GGUF it is in
};

// Header and descriptor bytes.
struct Fill {
  uint64_t offset = 0;
  std::vector<uint8_t> bytes;
};
// How a copy writes each value: as stored, narrowed from F32 to the bf16
// value it equals exactly (rows the kernels read as bf16), or widened from
// BF16 to the F32 value it equals (rows the kernels read as F32).
// DequantizeQ80: Q8_0 rows as the F32 values they equal (d * q, exact in
// F32), for small tensors a float kernel reads, as the MTP head's injection
// weights.
enum class Conversion : uint8_t { None, NarrowToBfloat16, WidenToFloat32, DequantizeQ80 };
// Rows written back to back, each value converted as `conversion` says.
struct Copy {
  uint64_t destination = 0;
  TensorRows source;
  Conversion conversion = Conversion::None;
};
// Quantized rows repacked into the planes of their format; the rows of the
// sources in order, then zero rows up to `rows`.
struct Repack {
  uint32_t format = 0; // GGUF_FMT_*
  uint64_t rows = 0;
  uint64_t columns = 0;
  uint64_t plane0 = 0, plane1 = 0, meta = 0; // image offsets; plane1 when the format has one
  std::vector<TensorRows> sources;
};
struct Image {
  std::string name; // layer-N.bin, head.bin, embedding.bin
  std::string magic; // kGgufImageMagic
  uint32_t layer = 0;
  uint32_t type = 0;
  uint64_t bytes = 0;
  std::vector<Fill> fills;
  std::vector<Copy> copies;
  std::vector<Repack> repacks;
};

// The target geometry a GGUF's metadata declares, its architecture included,
// with the rotary embedding and norms the kernels compute: the RoPE base and
// rotated dimensions, the RMS epsilon and no RoPE scaling. planImages checks
// the file's first, and model-check the installer's copy before any weight
// download. Throws GgufError naming every mismatch.
void requireMetadata(const GgufMetadata &metadata, const QwenTargetDimensions &geometry);

// The layers' images, then the head's and the embedding's, and for qwen4exp
// with a PLE layer the n-gram table's (ple.bin). Checks the metadata
// (requireMetadata, and a qwen4exp file's arrays) and each tensor's shape;
// throws GgufError naming every missing tensor and every tensor of a type
// this build cannot load.
[[nodiscard]] std::vector<Image> planImages(const GgufFile &file, const QwenTargetDimensions &geometry);

// A qwen4exp MTP head's image (mtp.bin): its one full-attention block
// (blk.<layers>, the block's tensors as a target layer's), then the
// nextn tensors: eh_proj, enorm, hnorm and the head's mix. The geometry is
// the target's; the MTP file has no PLE.
[[nodiscard]] Image planMtpImage(const GgufFile &file, const QwenTargetDimensions &geometry);

// The hash of a qwen4exp target's PLE n-gram embedding (llama.cpp
// qwen4exp.cpp llm_graph_input_qwen4exp_ple): per n-gram position the
// multiplier, per head its table rows' offset and count, and the table's rows.
struct PleHash {
  std::vector<uint64_t> multipliers;
  std::vector<uint32_t> headOffsets;
  std::vector<uint32_t> headVocabularies;
  uint64_t tableRows = 0;
  uint32_t eosToken = 0;
};
[[nodiscard]] PleHash readPleHash(const GgufFile &file, const QwenTargetDimensions &geometry);

} // namespace splash::model::gguf
