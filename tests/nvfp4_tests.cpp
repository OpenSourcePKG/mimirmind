// Unit tests for the NVFP4 runtime integration:
//   - runtime::nvfp4::resolveModelFormat (filesystem probe)
//   - runtime::nvfp4::loadNvfp4Model     (device upload + assembly)
//
// The loader is exercised against a hand-built miniature ModelOpt checkpoint
// written to a temp dir, uploaded through a malloc/memcpy FakeUploader — no
// GPU, no 23 GB model. Needs a real filesystem + mmap, so this is built on
// the box/Docker toolchain (like the other core_common-linked suites).

#include "TestFramework.hpp"

#include "compute/ComputeBuffer.hpp"
#include "runtime/nvfp4/ModelFormatResolver.hpp"
#include "runtime/nvfp4/NvFp4Model.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace nv  = mimirmind::runtime::nvfp4;
namespace cfg = mimirmind::core::config;
namespace fs  = std::filesystem;

namespace {

// --- FakeUploader: malloc as "device", memcpy as "upload" ----------------
void freeDeleter(void* p, std::size_t, void*) noexcept { std::free(p); }

struct FakeUploader final : nv::DeviceUploader {
    std::size_t allocations{0};
    std::size_t bytesUploaded{0};
    mimirmind::compute::ComputeBuffer allocate(std::size_t bytes) override {
        ++allocations;
        void* p = std::malloc(bytes ? bytes : 1);
        return mimirmind::compute::ComputeBuffer(p, bytes, &freeDeleter, nullptr);
    }
    void uploadHostBytes(void* dst, const void* src, std::size_t n) override {
        std::memcpy(dst, src, n);
        bytesUploaded += n;
    }
};

// --- tiny safetensors builder --------------------------------------------
struct Builder {
    std::string               header = "{";
    std::vector<std::uint8_t> data;
    bool                      first = true;

    void add(const std::string& name, const std::string& dtype,
             const std::string& shape, const std::vector<std::uint8_t>& bytes) {
        const std::size_t begin = data.size();
        data.insert(data.end(), bytes.begin(), bytes.end());
        const std::size_t end = data.size();
        if (!first) header += ",";
        first = false;
        header += "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":" + shape
                + ",\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
    }
    std::vector<std::uint8_t> finish() {
        header += "}";
        std::vector<std::uint8_t> buf(sizeof(std::uint64_t));
        const std::uint64_t n = header.size();
        std::memcpy(buf.data(), &n, sizeof(n));
        buf.insert(buf.end(), header.begin(), header.end());
        buf.insert(buf.end(), data.begin(), data.end());
        return buf;
    }
};

void writeFile(const fs::path& p, const std::string& text) {
    std::ofstream(p, std::ios::binary).write(text.data(),
                                             static_cast<std::streamsize>(text.size()));
}
void writeFile(const fs::path& p, const std::vector<std::uint8_t>& bytes) {
    std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()),
                                             static_cast<std::streamsize>(bytes.size()));
}

// Build a miniature mixed checkpoint (one NVFP4 module `w`, one FP8 module
// `attn.q`, one bf16 tensor) under a fresh temp directory. Returns the dir.
fs::path buildMiniCheckpoint() {
    const fs::path dir = fs::temp_directory_path()
                       / ("mm_nvfp4_test_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);

    Builder b;
    // NVFP4 module "w": in=16, out=2 -> weight U8 [2,8], scale F8 [2,1], global F32 scalar
    b.add("w.weight",         "U8",      "[2,8]", std::vector<std::uint8_t>(16, 0xAB));
    b.add("w.weight_scale",   "F8_E4M3", "[2,1]", std::vector<std::uint8_t>(2, 0x3C));
    b.add("w.weight_scale_2", "F32",     "[]",    std::vector<std::uint8_t>(4, 0x01));
    // FP8 module "attn.q": F8 [3,16], F32 weight + input scales
    b.add("attn.q.weight",       "F8_E4M3", "[3,16]", std::vector<std::uint8_t>(48, 0x7F));
    b.add("attn.q.weight_scale", "F32",     "[]",     std::vector<std::uint8_t>(4, 0x02));
    b.add("attn.q.input_scale",  "F32",     "[]",     std::vector<std::uint8_t>(4, 0x03));
    // unquantised bf16 tensor
    b.add("embed.weight",        "BF16",    "[2,4]",  std::vector<std::uint8_t>(16, 0x40));

    writeFile(dir / "model.safetensors", b.finish());
    writeFile(dir / "hf_quant_config.json",
              R"({"quantization":{"quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":"FP8",)"
              R"("quantized_layers":{"w":{"quant_algo":"W4A16_NVFP4","group_size":16},)"
              R"("attn.q":{"quant_algo":"FP8"}}}})");
    return dir;
}

} // namespace

// =======================================================================
// resolveModelFormat
// =======================================================================

TEST(resolve_explicit_wins) {
    // An explicit format is never probed.
    EXPECT_EQ(nv::resolveModelFormat(cfg::ModelFormat::Gguf, "/whatever/dir"),
              cfg::ModelFormat::Gguf);
    EXPECT_EQ(nv::resolveModelFormat(cfg::ModelFormat::Nvfp4, "/x.gguf"),
              cfg::ModelFormat::Nvfp4);
}

TEST(resolve_auto_by_extension) {
    EXPECT_EQ(nv::resolveModelFormat(cfg::ModelFormat::Auto, "/models/foo.gguf"),
              cfg::ModelFormat::Gguf);
    EXPECT_EQ(nv::resolveModelFormat(cfg::ModelFormat::Auto, "/models/foo.safetensors"),
              cfg::ModelFormat::Nvfp4);
}

TEST(resolve_auto_by_directory) {
    const fs::path dir = fs::temp_directory_path()
                       / ("mm_nvfp4_resolve_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    // empty dir -> Gguf
    EXPECT_EQ(nv::resolveModelFormat(cfg::ModelFormat::Auto, dir.string()),
              cfg::ModelFormat::Gguf);
    // dir with a safetensors index -> Nvfp4
    writeFile(dir / "model.safetensors.index.json", std::string("{}"));
    EXPECT_EQ(nv::resolveModelFormat(cfg::ModelFormat::Auto, dir.string()),
              cfg::ModelFormat::Nvfp4);
    fs::remove_all(dir);
}

// =======================================================================
// loadNvfp4Model
// =======================================================================

TEST(load_mini_checkpoint) {
    const fs::path dir = buildMiniCheckpoint();
    FakeUploader up;
    const nv::NvFp4Model model = nv::loadNvfp4Model(dir.string(), up);

    // 7 tensors uploaded, quant config is mixed.
    EXPECT_EQ(model.tensorCount(), std::size_t{7});
    EXPECT_TRUE(model.quantConfig().isMixed());
    EXPECT_EQ(up.allocations, std::size_t{7});
    // total device bytes = 16+2+4 + 48+4+4 + 16 = 94
    EXPECT_EQ(model.deviceBytes(), std::uint64_t{94});
    EXPECT_EQ(up.bytesUploaded,    std::size_t{94});

    // Two assembled quantised weights: NVFP4 `w`, FP8 `attn.q`.
    EXPECT_EQ(model.quantizedWeightCount(), std::size_t{2});

    // --- NVFP4 weight `w` -------------------------------------------------
    const auto* w = model.weight("w");
    EXPECT_TRUE(w != nullptr);
    EXPECT_EQ(w->layout.scheme,      mimirmind::core::modelopt::ModelOptQuantScheme::NVFP4_E2M1_BLK16);
    EXPECT_EQ(w->layout.outFeatures, std::uint64_t{2});
    EXPECT_EQ(w->layout.inFeatures,  std::uint64_t{16}); // 8 packed * 2
    EXPECT_EQ(w->layout.groupSize,   static_cast<std::uint16_t>(16));
    EXPECT_TRUE(w->packedWeight.devPtr != nullptr);
    EXPECT_EQ(w->packedWeight.nbytes, std::size_t{16});
    EXPECT_EQ(w->blockScale.nbytes,   std::size_t{2});
    EXPECT_EQ(w->globalScale.nbytes,  std::size_t{4});
    EXPECT_EQ(w->weightScale.devPtr,  static_cast<void*>(nullptr)); // FP8-only, unset
    // The assembled sub-tensor pointer must equal the same tensor via find().
    EXPECT_EQ(w->packedWeight.devPtr, model.find("w.weight")->devPtr);

    // --- FP8 weight `attn.q` ---------------------------------------------
    const auto* q = model.weight("attn.q");
    EXPECT_TRUE(q != nullptr);
    EXPECT_EQ(q->layout.scheme,      mimirmind::core::modelopt::ModelOptQuantScheme::FP8_E4M3);
    EXPECT_EQ(q->layout.outFeatures, std::uint64_t{3});
    EXPECT_EQ(q->layout.inFeatures,  std::uint64_t{16});
    EXPECT_TRUE(q->weightScale.devPtr != nullptr);
    EXPECT_TRUE(q->inputScale.devPtr  != nullptr);
    EXPECT_EQ(q->blockScale.devPtr,   static_cast<void*>(nullptr)); // NVFP4-only, unset

    // --- unquantised tensor is present but has no assembled weight -------
    EXPECT_TRUE(model.find("embed.weight") != nullptr);
    EXPECT_TRUE(model.weight("embed") == nullptr);

    // Uploaded bytes really are the source bytes (0xAB weight payload).
    EXPECT_EQ(static_cast<const std::uint8_t*>(w->packedWeight.devPtr)[0],
              std::uint8_t{0xAB});

    fs::remove_all(dir);
}

int main() {
    return mm::test::run();
}