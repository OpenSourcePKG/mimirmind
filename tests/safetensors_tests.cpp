// Pure-CPU unit tests for the safetensors container layer:
//   - core::safetensors dtype parsing
//   - parseSafetensorsHeader against hand-crafted file buffers
//
// The header parser is deliberately mmap-free so it builds and runs on any
// toolchain. The thin SafetensorsReader mmap wrapper is not exercised here
// (it needs a real file + modern libstdc++); its logic is trivial slicing
// over what this parser produces.

#include "TestFramework.hpp"

#include "core/safetensors/SafetensorsDtype.hpp"
#include "core/safetensors/SafetensorsHeader.hpp"
#include "core/safetensors/SafetensorsIndex.hpp"
#include "core/safetensors/SafetensorsModel.hpp"
#include "core/safetensors/SafetensorsReader.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace st = mimirmind::core::safetensors;

namespace {

// Build a safetensors file buffer: 8-byte LE header length, JSON header,
// then the raw data region.
std::vector<std::uint8_t> buildFile(const std::string&                header,
                                    const std::vector<std::uint8_t>&  data) {
    std::vector<std::uint8_t> buf(sizeof(std::uint64_t));
    const std::uint64_t n = header.size();
    std::memcpy(buf.data(), &n, sizeof(n));
    buf.insert(buf.end(), header.begin(), header.end());
    buf.insert(buf.end(), data.begin(), data.end());
    return buf;
}

// True iff calling `fn` throws std::runtime_error (what the parser uses).
bool threw(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

st::ParsedSafetensorsHeader parse(const std::vector<std::uint8_t>& buf) {
    return st::parseSafetensorsHeader(std::span<const std::uint8_t>(buf.data(), buf.size()));
}

} // namespace

// =======================================================================
// dtype parsing
// =======================================================================

TEST(dtype_fromString_and_width) {
    EXPECT_EQ(st::dtypeFromString("F32"),      st::SafetensorsDtype::F32);
    EXPECT_EQ(st::dtypeFromString("BF16"),     st::SafetensorsDtype::BF16);
    EXPECT_EQ(st::dtypeFromString("F8_E4M3"),  st::SafetensorsDtype::F8_E4M3);
    EXPECT_EQ(st::dtypeFromString("U8"),       st::SafetensorsDtype::U8);
    EXPECT_EQ(st::dtypeFromString("BOOL"),     st::SafetensorsDtype::Bool);
    EXPECT_EQ(st::dtypeFromString("nonsense"), st::SafetensorsDtype::Unknown);
    EXPECT_EQ(st::dtypeFromString("f32"),      st::SafetensorsDtype::Unknown); // case-sensitive

    EXPECT_EQ(st::dtypeWidth(st::SafetensorsDtype::F32),     std::size_t{4});
    EXPECT_EQ(st::dtypeWidth(st::SafetensorsDtype::BF16),    std::size_t{2});
    EXPECT_EQ(st::dtypeWidth(st::SafetensorsDtype::F8_E4M3), std::size_t{1});
    EXPECT_EQ(st::dtypeWidth(st::SafetensorsDtype::U8),      std::size_t{1});
    EXPECT_EQ(st::dtypeWidth(st::SafetensorsDtype::I64),     std::size_t{8});
    EXPECT_EQ(st::dtypeWidth(st::SafetensorsDtype::Unknown), std::size_t{0});

    EXPECT_EQ(st::dtypeName(st::SafetensorsDtype::F8_E4M3), std::string_view{"F8_E4M3"});
    EXPECT_EQ(st::dtypeName(st::SafetensorsDtype::U8),      std::string_view{"U8"});
}

// =======================================================================
// Valid header — mirrors the ModelOpt NVFP4 sidecar shape (packed weight +
// FP8 block scale + F32 scalar global scale).
// =======================================================================

TEST(parse_valid_nvfp4_like) {
    // gate_proj-like: weight U8 [4,2] (8 B), block scale F8_E4M3 [4,1] (4 B),
    // global scale F32 scalar (4 B). 16 data bytes total.
    const std::string header =
        R"({)"
        R"("w.weight":{"dtype":"U8","shape":[4,2],"data_offsets":[0,8]},)"
        R"("w.weight_scale":{"dtype":"F8_E4M3","shape":[4,1],"data_offsets":[8,12]},)"
        R"("w.weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[12,16]},)"
        R"("__metadata__":{"format":"pt","producer":"modelopt"})"
        R"(})";
    std::vector<std::uint8_t> data(16);
    for (std::uint8_t i = 0; i < 8; ++i) data[i] = static_cast<std::uint8_t>(0xA0 + i);

    const auto buf = buildFile(header, data);
    const auto p   = parse(buf);

    EXPECT_EQ(p.tensors.size(), std::size_t{3}); // __metadata__ excluded
    EXPECT_EQ(p.dataOffset,     std::size_t{8} + header.size());

    EXPECT_EQ(p.metadata.size(), std::size_t{2});
    EXPECT_EQ(p.metadata.at("format"),   std::string{"pt"});
    EXPECT_EQ(p.metadata.at("producer"), std::string{"modelopt"});

    // find via the parsed vector (ascending name order)
    const st::SafetensorsTensor* w = nullptr;
    const st::SafetensorsTensor* bs = nullptr;
    const st::SafetensorsTensor* gs = nullptr;
    for (const auto& t : p.tensors) {
        if (t.name == "w.weight")         w = &t;
        if (t.name == "w.weight_scale")   bs = &t;
        if (t.name == "w.weight_scale_2") gs = &t;
    }
    EXPECT_TRUE(w != nullptr);
    EXPECT_TRUE(bs != nullptr);
    EXPECT_TRUE(gs != nullptr);

    EXPECT_EQ(w->dtype,     st::SafetensorsDtype::U8);
    EXPECT_TRUE((w->shape == std::vector<std::uint64_t>{4, 2}));
    EXPECT_EQ(w->nelements, std::uint64_t{8});
    EXPECT_EQ(w->nbytes,    std::size_t{8});
    EXPECT_EQ(w->dataBegin, p.dataOffset + 0);
    EXPECT_EQ(w->dataEnd,   p.dataOffset + 8);

    EXPECT_EQ(bs->dtype,  st::SafetensorsDtype::F8_E4M3);
    EXPECT_EQ(bs->nbytes, std::size_t{4});

    // scalar global scale: empty shape, one element, 4 bytes.
    EXPECT_EQ(gs->dtype,     st::SafetensorsDtype::F32);
    EXPECT_TRUE(gs->shape.empty());
    EXPECT_EQ(gs->nelements, std::uint64_t{1});
    EXPECT_EQ(gs->nbytes,    std::size_t{4});

    // Byte slice at [dataBegin,dataEnd) is the weight payload we wrote.
    EXPECT_EQ(buf[w->dataBegin],     std::uint8_t{0xA0});
    EXPECT_EQ(buf[w->dataEnd - 1],   std::uint8_t{0xA7});
}

// =======================================================================
// Malformation — every branch must throw, not silently mis-parse.
// =======================================================================

TEST(parse_rejects_truncated_prefix) {
    std::vector<std::uint8_t> buf{0, 1, 2}; // < 8 bytes
    EXPECT_TRUE(threw([&] { parse(buf); }));
}

TEST(parse_rejects_header_len_past_eof) {
    // Declare a 4 KiB header but supply almost nothing.
    std::vector<std::uint8_t> buf(sizeof(std::uint64_t) + 4);
    const std::uint64_t n = 4096;
    std::memcpy(buf.data(), &n, sizeof(n));
    EXPECT_TRUE(threw([&] { parse(buf); }));
}

TEST(parse_rejects_bad_json) {
    const auto buf = buildFile("{ this is not json ", {});
    EXPECT_TRUE(threw([&] { parse(buf); }));
}

TEST(parse_rejects_unknown_dtype) {
    const std::string header =
        R"({"t":{"dtype":"F4","shape":[2],"data_offsets":[0,2]}})";
    const auto buf = buildFile(header, {1, 2});
    EXPECT_TRUE(threw([&] { parse(buf); }));
}

TEST(parse_rejects_offsets_past_region) {
    const std::string header =
        R"({"t":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";
    const auto buf = buildFile(header, {1, 2}); // only 2 data bytes, end=4 > 2
    EXPECT_TRUE(threw([&] { parse(buf); }));
}

TEST(parse_rejects_byte_length_mismatch) {
    // shape [2,4] U8 = 8 elements = 8 bytes, but data_offsets span only 4.
    const std::string header =
        R"({"t":{"dtype":"U8","shape":[2,4],"data_offsets":[0,4]}})";
    const auto buf = buildFile(header, {1, 2, 3, 4});
    EXPECT_TRUE(threw([&] { parse(buf); }));
}

TEST(parse_rejects_begin_gt_end) {
    const std::string header =
        R"({"t":{"dtype":"U8","shape":[2],"data_offsets":[4,2]}})";
    const auto buf = buildFile(header, {1, 2, 3, 4});
    EXPECT_TRUE(threw([&] { parse(buf); }));
}

TEST(parse_rejects_negative_shape_dim) {
    const std::string header =
        R"({"t":{"dtype":"U8","shape":[-1],"data_offsets":[0,0]}})";
    const auto buf = buildFile(header, {});
    EXPECT_TRUE(threw([&] { parse(buf); }));
}

// =======================================================================
// index.json (multi-shard weight map) parsing
// =======================================================================

TEST(parse_index_valid) {
    const std::string idx =
        R"({)"
        R"("metadata":{"total_size":23407580856},)"
        R"("weight_map":{)"
        R"("model.embed_tokens.weight":"model-00001-of-00003.safetensors",)"
        R"("lm_head.weight":"model-00003-of-00003.safetensors",)"
        R"("model.layers.0.mlp.gate.weight":"model-00001-of-00003.safetensors",)"
        R"("model.layers.39.self_attn.q_proj.weight":"model-00002-of-00003.safetensors")"
        R"(}})";
    const auto p = st::parseSafetensorsIndex(idx);

    EXPECT_EQ(p.weightMap.size(), std::size_t{4});
    EXPECT_EQ(p.totalSize,        std::uint64_t{23407580856});
    EXPECT_EQ(p.weightMap.at("lm_head.weight"),
              std::string{"model-00003-of-00003.safetensors"});

    // Three distinct shards, ascending order, de-duplicated.
    const auto shards = p.shardFiles();
    EXPECT_EQ(shards.size(), std::size_t{3});
    EXPECT_TRUE((shards == std::vector<std::string>{
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors"}));
}

TEST(parse_index_no_metadata_ok) {
    const std::string idx =
        R"({"weight_map":{"t":"a.safetensors"}})";
    const auto p = st::parseSafetensorsIndex(idx);
    EXPECT_EQ(p.weightMap.size(), std::size_t{1});
    EXPECT_EQ(p.totalSize,        std::uint64_t{0}); // absent -> 0
}

TEST(parse_index_rejects_malformed) {
    EXPECT_TRUE(threw([] { (void)st::parseSafetensorsIndex("not json"); }));
    // missing weight_map
    EXPECT_TRUE(threw([] { (void)st::parseSafetensorsIndex(R"({"metadata":{}})"); }));
    // empty weight_map
    EXPECT_TRUE(threw([] { (void)st::parseSafetensorsIndex(R"({"weight_map":{}})"); }));
    // non-string shard value
    EXPECT_TRUE(threw([] {
        (void)st::parseSafetensorsIndex(R"({"weight_map":{"t":123}})");
    }));
    // total_size present but not an unsigned int
    EXPECT_TRUE(threw([] {
        (void)st::parseSafetensorsIndex(
            R"({"metadata":{"total_size":-5},"weight_map":{"t":"a"}})");
    }));
}

// =======================================================================
// SafetensorsReader::openBytes — in-memory shard source (M-Munin shm attach)
// =======================================================================

TEST(reader_openBytes_parsesAndViewsData) {
    // Two U8 tensors, 4 bytes each, packed back to back.
    const std::string header =
        R"({"a.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},)"
        R"("b.weight":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}})";
    std::vector<std::uint8_t> data(8);
    for (std::uint8_t i = 0; i < 8; ++i) data[i] = static_cast<std::uint8_t>(0x10 + i);
    const auto buf = buildFile(header, data);

    st::SafetensorsReader reader;
    reader.openBytes(std::span<const std::uint8_t>(buf.data(), buf.size()),
                     "shard-0.safetensors");

    EXPECT_TRUE(reader.isOpen());
    EXPECT_EQ(reader.path(), std::string_view{"shard-0.safetensors"});
    EXPECT_EQ(reader.tensorCount(), std::size_t{2});

    const auto* a = reader.find("a.weight");
    EXPECT_TRUE(a != nullptr);
    if (a != nullptr) {
        const auto ab = reader.tensorBytes(*a);
        EXPECT_EQ(ab.size(), std::size_t{4});
        EXPECT_EQ(ab[0], std::uint8_t{0x10});
        EXPECT_EQ(ab[3], std::uint8_t{0x13});
    }
    const auto* b = reader.find("b.weight");
    EXPECT_TRUE(b != nullptr);
    if (b != nullptr) {
        const auto bb = reader.tensorBytes(*b);
        EXPECT_EQ(bb.size(), std::size_t{4});
        EXPECT_EQ(bb[0], std::uint8_t{0x14});
        EXPECT_EQ(bb[3], std::uint8_t{0x17});
    }

    // close() clears the external view.
    reader.close();
    EXPECT_TRUE(!reader.isOpen());
}

TEST(reader_openBytes_rejectsMalformed) {
    // 8-byte header length pointing past the buffer -> parser throws.
    std::vector<std::uint8_t> bad(sizeof(std::uint64_t), 0);
    const std::uint64_t huge = 1u << 20;
    std::memcpy(bad.data(), &huge, sizeof(huge));
    st::SafetensorsReader reader;
    EXPECT_TRUE(threw([&] {
        reader.openBytes(std::span<const std::uint8_t>(bad.data(), bad.size()), "bad");
    }));
    EXPECT_TRUE(!reader.isOpen());
}

// =======================================================================
// SafetensorsModel::openFromShards — multi-shard from in-memory images
// =======================================================================

TEST(model_openFromShards_unifiesNamespaceAndData) {
    // Shard 0: x,y (4 B each). Shard 1: z (4 B).
    const std::string h0 =
        R"({"x.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},)"
        R"("y.weight":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}})";
    std::vector<std::uint8_t> d0(8);
    for (std::uint8_t i = 0; i < 8; ++i) d0[i] = static_cast<std::uint8_t>(0x10 + i);
    const auto buf0 = buildFile(h0, d0);

    const std::string h1 =
        R"({"z.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";
    std::vector<std::uint8_t> d1(4);
    for (std::uint8_t i = 0; i < 4; ++i) d1[i] = static_cast<std::uint8_t>(0xC0 + i);
    const auto buf1 = buildFile(h1, d1);

    const st::SafetensorsModel::ShardImage shards[2] = {
        {"model-00001.safetensors", std::span<const std::uint8_t>(buf0.data(), buf0.size())},
        {"model-00002.safetensors", std::span<const std::uint8_t>(buf1.data(), buf1.size())},
    };

    st::SafetensorsModel model;
    model.openFromShards(std::span<const st::SafetensorsModel::ShardImage>(shards, 2),
                         /*declaredTotalSize=*/12);

    EXPECT_TRUE(model.isOpen());
    EXPECT_EQ(model.shardCount(),  std::size_t{2});
    EXPECT_EQ(model.tensorCount(), std::size_t{3});
    EXPECT_EQ(model.declaredTotalSize(), std::uint64_t{12});

    EXPECT_TRUE(model.find("x.weight") != nullptr);
    EXPECT_TRUE(model.find("z.weight") != nullptr);   // resolves across shards
    EXPECT_TRUE(model.find("nope")     == nullptr);

    const auto zb = model.tensorBytes("z.weight");
    EXPECT_EQ(zb.size(), std::size_t{4});
    EXPECT_EQ(zb[0], std::uint8_t{0xC0});
    EXPECT_EQ(zb[3], std::uint8_t{0xC3});

    model.close();
    EXPECT_TRUE(!model.isOpen());
}

TEST(model_openFromShards_rejectsDuplicateTensorAndEmpty) {
    // Same tensor name in two shards -> reject.
    const std::string h =
        R"({"dup.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";
    std::vector<std::uint8_t> d(4, 0x11);
    const auto buf = buildFile(h, d);
    const st::SafetensorsModel::ShardImage dupShards[2] = {
        {"a.safetensors", std::span<const std::uint8_t>(buf.data(), buf.size())},
        {"b.safetensors", std::span<const std::uint8_t>(buf.data(), buf.size())},
    };
    st::SafetensorsModel m1;
    EXPECT_TRUE(threw([&] {
        m1.openFromShards(
            std::span<const st::SafetensorsModel::ShardImage>(dupShards, 2));
    }));

    // Empty shard list -> reject.
    st::SafetensorsModel m2;
    EXPECT_TRUE(threw([&] {
        m2.openFromShards(std::span<const st::SafetensorsModel::ShardImage>());
    }));
}

int main() {
    return mm::test::run();
}