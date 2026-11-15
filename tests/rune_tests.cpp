// Pure-CPU unit tests for the Rune format and the host-side applier
// (roadmap 7.1 — Futhark core). No GPU, no model file: target tensors are
// host buffers resolved through a lambda.

#include "TestFramework.hpp"

#include "core/gguf/GgufReader.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "core/rune/Futhark.hpp"
#include "core/rune/RuneApplier.hpp"
#include "core/rune/RuneFile.hpp"
#include "core/rune/RuneFormat.hpp"
#include "core/safetensors/CheckpointFingerprint.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace rn = mimirmind::core::rune;

namespace {

constexpr const char* kBase = "base-fp-1";

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

/// A host "model": named tensors with their own byte storage.
struct FakeTensor {
    rn::RuneElemType           type{rn::RuneElemType::F32};
    std::vector<std::uint64_t> dims;
    std::vector<std::byte>     bytes;
    bool                       derived{false};
};

struct FakeModel {
    std::map<std::string, FakeTensor> tensors;

    void addF32(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                const std::vector<float>& vals) {
        FakeTensor t;
        t.type = rn::RuneElemType::F32;
        t.dims = {cols, rows};
        t.bytes.resize(vals.size() * 4);
        std::memcpy(t.bytes.data(), vals.data(), t.bytes.size());
        tensors[name] = std::move(t);
    }

    void addBf16(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                 const std::vector<float>& vals) {
        FakeTensor t;
        t.type = rn::RuneElemType::BF16;
        t.dims = {cols, rows};
        t.bytes.resize(vals.size() * 2);
        for (std::size_t i = 0; i < vals.size(); ++i) {
            const std::uint16_t b = rn::floatToBf16Bits(vals[i]);
            std::memcpy(t.bytes.data() + i * 2, &b, 2);
        }
        tensors[name] = std::move(t);
    }

    std::string hash(const std::string& name) const {
        const auto& b = tensors.at(name).bytes;
        return rn::tensorContentHash(b.data(), b.size());
    }

    std::vector<float> f32(const std::string& name) const {
        const auto& b = tensors.at(name).bytes;
        std::vector<float> v(b.size() / 4);
        std::memcpy(v.data(), b.data(), b.size());
        return v;
    }

    float bf16At(const std::string& name, std::size_t i) const {
        std::uint16_t b = 0;
        std::memcpy(&b, tensors.at(name).bytes.data() + i * 2, 2);
        return rn::bf16BitsToFloat(b);
    }

    rn::RuneTensorResolver resolver() {
        return [this](std::string_view name) -> std::optional<rn::RuneTensorView> {
            const auto it = tensors.find(std::string{name});
            if (it == tensors.end()) {
                return std::nullopt;
            }
            rn::RuneTensorView v;
            v.name             = it->first;
            v.type             = it->second.type;
            v.dims             = it->second.dims;
            v.bytes            = std::span<std::byte>(it->second.bytes);
            v.hasDerivedCopies = it->second.derived;
            return v;
        };
    }
};

rn::RuneOp writeSlice(const std::string& target, const std::string& hash,
                      const std::string& payload, std::uint64_t b, std::uint64_t e) {
    rn::RuneOp op;
    op.kind       = rn::RuneOpKind::WriteSlice;
    op.target     = target;
    op.tensorHash = hash;
    op.payload    = payload;
    op.rowBegin   = b;
    op.rowEnd     = e;
    return op;
}

rn::RuneOp scaleOp(const std::string& target, const std::string& hash,
                   const std::string& payload, rn::RuneScaleAxis axis) {
    rn::RuneOp op;
    op.kind       = rn::RuneOpKind::Scale;
    op.target     = target;
    op.tensorHash = hash;
    op.payload    = payload;
    op.axis       = axis;
    return op;
}

rn::RuneFile makeRune(std::vector<rn::RuneOp> ops,
                      std::map<std::string, std::vector<float>> payloads,
                      const std::string& base = kBase) {
    rn::RuneManifest m;
    m.name            = "test-rune";
    m.baseFingerprint = base;
    m.provenanceJson  = R"({"by":"rune_tests"})";
    m.ops             = std::move(ops);
    return rn::RuneFile{std::move(m), std::move(payloads)};
}

std::vector<std::uint8_t> buildContainer(const std::string& metadataJson,
                                         const std::string& tensorsJson,
                                         const std::vector<std::uint8_t>& data) {
    nlohmann::json meta;
    meta[std::string{rn::kRuneMetadataKey}] = metadataJson;
    std::string header = "{" + tensorsJson + (tensorsJson.empty() ? "" : ",") +
                         "\"__metadata__\":" + meta.dump() + "}";
    std::vector<std::uint8_t> buf(8);
    const std::uint64_t n = header.size();
    std::memcpy(buf.data(), &n, 8);
    buf.insert(buf.end(), header.begin(), header.end());
    buf.insert(buf.end(), data.begin(), data.end());
    return buf;
}

} // namespace

// ======================================================================
// Format basics
// ======================================================================

TEST(opKind_names_roundTrip) {
    for (auto k : {rn::RuneOpKind::WriteSlice, rn::RuneOpKind::Scale, rn::RuneOpKind::Mask,
                   rn::RuneOpKind::LowRank, rn::RuneOpKind::Steer,
                   rn::RuneOpKind::WriteQuantBlocks}) {
        const auto back = rn::opKindFromName(rn::opKindName(k));
        EXPECT_TRUE(back.has_value());
        EXPECT_EQ(*back, k);
    }
    EXPECT_TRUE(!rn::opKindFromName("explode").has_value());
    EXPECT_TRUE(rn::opKindApplicable(rn::RuneOpKind::WriteSlice));
    EXPECT_TRUE(rn::opKindApplicable(rn::RuneOpKind::Scale));
    EXPECT_TRUE(!rn::opKindApplicable(rn::RuneOpKind::LowRank));
}

TEST(contentHash_formatAndDeterminism) {
    const std::uint8_t a[3] = {1, 2, 3};
    const std::uint8_t b[3] = {1, 2, 4};
    const std::string  ha   = rn::tensorContentHash(a, 3);
    EXPECT_EQ(ha.size(), std::size_t{24});
    EXPECT_EQ(ha.substr(0, 8), std::string{"fnv1a64:"});
    EXPECT_EQ(ha, rn::tensorContentHash(a, 3));
    EXPECT_TRUE(ha != rn::tensorContentHash(b, 3));
    // FNV-1a 64 of the empty input is the offset basis.
    EXPECT_EQ(rn::tensorContentHash(a, 0), std::string{"fnv1a64:cbf29ce484222325"});
}

TEST(halfConversion_knownValues) {
    EXPECT_EQ(rn::floatToHalfBits(1.0f), std::uint16_t{0x3c00});
    EXPECT_EQ(rn::floatToHalfBits(-2.0f), std::uint16_t{0xc000});
    EXPECT_EQ(rn::floatToHalfBits(65504.0f), std::uint16_t{0x7bff});
    EXPECT_EQ(rn::floatToHalfBits(65520.0f), std::uint16_t{0x7c00});   // rounds to inf
    EXPECT_EQ(rn::floatToHalfBits(65519.0f), std::uint16_t{0x7bff});
    EXPECT_EQ(rn::floatToHalfBits(std::ldexp(1.0f, -24)), std::uint16_t{0x0001});
    EXPECT_EQ(rn::floatToHalfBits(std::ldexp(1.0f, -14)), std::uint16_t{0x0400});
    EXPECT_EQ(rn::floatToHalfBits(0.0f), std::uint16_t{0x0000});
    // Tie at 1 + 2^-11 rounds to even (1.0).
    EXPECT_EQ(rn::floatToHalfBits(1.0f + std::ldexp(1.0f, -11)), std::uint16_t{0x3c00});
    EXPECT_EQ(rn::halfBitsToFloat(0x3c00), 1.0f);
    EXPECT_EQ(rn::halfBitsToFloat(0x0001), std::ldexp(1.0f, -24));
    EXPECT_TRUE(std::isinf(rn::halfBitsToFloat(0x7c00)));
    EXPECT_TRUE(std::isnan(rn::halfBitsToFloat(rn::floatToHalfBits(std::nanf("")))));
    for (std::uint32_t h = 0; h < 0x7c00; ++h) {   // every finite positive half round-trips
        const auto bits = static_cast<std::uint16_t>(h);
        EXPECT_EQ(rn::floatToHalfBits(rn::halfBitsToFloat(bits)), bits);
    }
}

TEST(bf16Conversion_knownValues) {
    EXPECT_EQ(rn::floatToBf16Bits(1.0f), std::uint16_t{0x3f80});
    EXPECT_EQ(rn::bf16BitsToFloat(0x3f80), 1.0f);
    // 1 + 2^-8 is a tie between 1.0 and 1+2^-7 -> rounds to even (1.0).
    EXPECT_EQ(rn::floatToBf16Bits(1.0f + std::ldexp(1.0f, -8)), std::uint16_t{0x3f80});
    // Slightly above the tie rounds up.
    EXPECT_EQ(rn::floatToBf16Bits(1.0f + std::ldexp(1.0f, -8) + std::ldexp(1.0f, -20)),
              std::uint16_t{0x3f81});
    EXPECT_TRUE(std::isnan(rn::bf16BitsToFloat(rn::floatToBf16Bits(std::nanf("")))));
}

// ======================================================================
// Container round trip + parse rejections
// ======================================================================

TEST(runeFile_serializeParse_roundTrip) {
    const auto rune = makeRune(
        {writeSlice("blk.0.attn_q.weight", "fnv1a64:0123456789abcdef", "p0", 1, 2),
         scaleOp("blk.1.ffn_up.weight", "fnv1a64:fedcba9876543210", "p1", rn::RuneScaleAxis::Cols)},
        {{"p0", {1.5f, -2.0f, 3.25f}}, {"p1", {0.5f, 2.0f}}});
    const std::vector<std::uint8_t> bytes = rune.serialize();
    EXPECT_EQ((bytes.size() - 8 - 0) % 4, std::size_t{0});

    const auto back = rn::RuneFile::parse(std::span<const std::uint8_t>(bytes));
    const auto& m   = back.manifest();
    EXPECT_EQ(m.name, std::string{"test-rune"});
    EXPECT_EQ(m.baseFingerprint, std::string{kBase});
    EXPECT_EQ(m.version, rn::kRuneFormatVersion);
    EXPECT_EQ(nlohmann::json::parse(m.provenanceJson)["by"].get<std::string>(),
              std::string{"rune_tests"});
    EXPECT_EQ(m.ops.size(), std::size_t{2});
    EXPECT_EQ(m.ops[0].kind, rn::RuneOpKind::WriteSlice);
    EXPECT_EQ(m.ops[0].rowBegin, std::uint64_t{1});
    EXPECT_EQ(m.ops[0].rowEnd, std::uint64_t{2});
    EXPECT_EQ(m.ops[1].kind, rn::RuneOpKind::Scale);
    EXPECT_EQ(m.ops[1].axis, rn::RuneScaleAxis::Cols);
    EXPECT_EQ(m.ops[1].tensorHash, std::string{"fnv1a64:fedcba9876543210"});
    const auto p0 = back.payload("p0");
    EXPECT_EQ(p0.size(), std::size_t{3});
    EXPECT_EQ(p0[2], 3.25f);
    // Re-serialising the parsed rune is byte-identical.
    EXPECT_TRUE(back.serialize() == bytes);
}

TEST(runeFile_rejectsMalformedManifests) {
    const std::string ok =
        R"({"format":"mimirmind-rune","version":1,"name":"x","base_fingerprint":"b",)"
        R"("ops":[{"op":"scale","target":"t","tensor_hash":"fnv1a64:0000000000000000",)"
        R"("payload":"p","axis":"rows"}]})";
    const std::string tensors = R"("p":{"dtype":"F32","shape":[1],"data_offsets":[0,4]})";
    std::vector<std::uint8_t> one(4);
    const float f = 1.0f;
    std::memcpy(one.data(), &f, 4);

    auto parseWith = [&](const std::string& meta, const std::string& t,
                         const std::vector<std::uint8_t>& d) {
        const auto buf = buildContainer(meta, t, d);
        return rn::RuneFile::parse(std::span<const std::uint8_t>(buf));
    };
    auto replaced = [&](const std::string& from, const std::string& to) {
        std::string s = ok;
        const auto  pos = s.find(from);
        s.replace(pos, from.size(), to);
        return s;
    };

    EXPECT_TRUE(!threw([&] { (void)parseWith(ok, tensors, one); }));
    EXPECT_TRUE(threw([&] { (void)parseWith("not json", tensors, one); }));
    EXPECT_TRUE(threw([&] { (void)parseWith(replaced("mimirmind-rune", "other"), tensors, one); }));
    EXPECT_TRUE(threw([&] { (void)parseWith(replaced("\"version\":1", "\"version\":99"), tensors, one); }));
    EXPECT_TRUE(threw([&] { (void)parseWith(replaced("\"base_fingerprint\":\"b\"", "\"base_fingerprint\":\"\""), tensors, one); }));
    EXPECT_TRUE(threw([&] { (void)parseWith(replaced("\"op\":\"scale\"", "\"op\":\"melt\""), tensors, one); }));
    EXPECT_TRUE(threw([&] { (void)parseWith(replaced("fnv1a64:0000000000000000", "md5:00"), tensors, one); }));
    EXPECT_TRUE(threw([&] { (void)parseWith(replaced("\"axis\":\"rows\"", "\"axis\":\"diag\""), tensors, one); }));
    EXPECT_TRUE(threw([&] { (void)parseWith(replaced("\"payload\":\"p\"", "\"payload\":\"missing\""), tensors, one); }));
    // Non-F32 payload.
    const std::string i32 = R"("p":{"dtype":"I32","shape":[1],"data_offsets":[0,4]})";
    EXPECT_TRUE(threw([&] { (void)parseWith(ok, i32, one); }));
    // No rune metadata at all (plain safetensors file).
    {
        const std::string h = "{" + tensors + "}";
        std::vector<std::uint8_t> buf(8);
        const std::uint64_t n = h.size();
        std::memcpy(buf.data(), &n, 8);
        buf.insert(buf.end(), h.begin(), h.end());
        buf.insert(buf.end(), one.begin(), one.end());
        EXPECT_TRUE(threw([&] { (void)rn::RuneFile::parse(std::span<const std::uint8_t>(buf)); }));
    }
    // write_slice with row_end <= row_begin.
    EXPECT_TRUE(threw([&] {
        (void)makeRune({writeSlice("t", "fnv1a64:0000000000000000", "p", 3, 3)}, {{"p", {1.0f}}});
    }));
}

// ======================================================================
// Applier
// ======================================================================

TEST(apply_writeSlice_f32_exact) {
    FakeModel model;
    model.addF32("w", 3, 2, {1, 2, 3, 4, 5, 6});
    const auto rune = makeRune({writeSlice("w", model.hash("w"), "p", 1, 2)},
                               {{"p", {7.0f, 8.0f, 9.0f}}});
    rn::RuneApplier applier;
    const auto r = applier.apply(rune, kBase, model.resolver());
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r->opsApplied, std::size_t{1});
    EXPECT_EQ(r->touched.size(), std::size_t{1});
    EXPECT_TRUE(model.f32("w") == (std::vector<float>{1, 2, 3, 7, 8, 9}));
    EXPECT_EQ(r->touched[0].hashAfter, model.hash("w"));
}

TEST(apply_identityScale_isBitIdentical) {
    // "Null rune" acceptance: an all-ones scale must not change a single byte,
    // for every supported encoding.
    FakeModel model;
    model.addF32("f", 4, 2, {0.1f, -3.5f, 1e-20f, 7.0f, 2.0f, -0.0f, 123.456f, 9.0f});
    model.addBf16("b", 4, 2, {0.1f, -3.5f, 1e-20f, 7.0f, 2.0f, -0.0f, 123.456f, 9.0f});
    const std::string hf = model.hash("f");
    const std::string hb = model.hash("b");
    const auto rune = makeRune(
        {scaleOp("f", hf, "ones2", rn::RuneScaleAxis::Rows),
         scaleOp("b", hb, "ones4", rn::RuneScaleAxis::Cols)},
        {{"ones2", {1.0f, 1.0f}}, {"ones4", {1.0f, 1.0f, 1.0f, 1.0f}}});
    rn::RuneApplier applier;
    EXPECT_TRUE(applier.apply(rune, kBase, model.resolver()).has_value());
    EXPECT_EQ(model.hash("f"), hf);
    EXPECT_EQ(model.hash("b"), hb);
}

TEST(apply_scaleRows_bf16_matchesHandComputed) {
    FakeModel model;
    model.addBf16("w", 2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    const auto rune = makeRune({scaleOp("w", model.hash("w"), "s", rn::RuneScaleAxis::Rows)},
                               {{"s", {0.5f, 3.0f}}});
    rn::RuneApplier applier;
    EXPECT_TRUE(applier.apply(rune, kBase, model.resolver()).has_value());
    EXPECT_EQ(model.bf16At("w", 0), 0.5f);
    EXPECT_EQ(model.bf16At("w", 1), 1.0f);
    EXPECT_EQ(model.bf16At("w", 2), 9.0f);
    EXPECT_EQ(model.bf16At("w", 3), 12.0f);
}

TEST(apply_rejectsWithoutTouchingAnything) {
    FakeModel model;
    model.addF32("w", 2, 2, {1, 2, 3, 4});
    const std::string h = model.hash("w");
    const auto good     = makeRune({scaleOp("w", h, "s", rn::RuneScaleAxis::Rows)},
                                   {{"s", {2.0f, 2.0f}}});
    rn::RuneApplier applier;

    // Wrong base model.
    EXPECT_TRUE(!applier.apply(good, "other-base", model.resolver()).has_value());
    // Missing target.
    EXPECT_TRUE(!applier.apply(makeRune({scaleOp("nope", h, "s", rn::RuneScaleAxis::Rows)},
                                        {{"s", {2.0f, 2.0f}}}),
                               kBase, model.resolver()).has_value());
    // Wrong payload length.
    EXPECT_TRUE(!applier.apply(makeRune({scaleOp("w", h, "s", rn::RuneScaleAxis::Rows)},
                                        {{"s", {2.0f, 2.0f, 2.0f}}}),
                               kBase, model.resolver()).has_value());
    // Out-of-range slice.
    EXPECT_TRUE(!applier.apply(makeRune({writeSlice("w", h, "p", 1, 3)},
                                        {{"p", {1, 1, 1, 1}}}),
                               kBase, model.resolver()).has_value());
    // Non-finite payload.
    EXPECT_TRUE(!applier.apply(makeRune({scaleOp("w", h, "s", rn::RuneScaleAxis::Rows)},
                                        {{"s", {2.0f, std::numeric_limits<float>::infinity()}}}),
                               kBase, model.resolver()).has_value());
    // Content hash mismatch.
    EXPECT_TRUE(!applier.apply(makeRune({scaleOp("w", "fnv1a64:0000000000000000", "s",
                                                 rn::RuneScaleAxis::Rows)},
                                        {{"s", {2.0f, 2.0f}}}),
                               kBase, model.resolver()).has_value());
    // Not yet applicable op kind.
    {
        rn::RuneOp lr;
        lr.kind       = rn::RuneOpKind::LowRank;
        lr.target     = "w";
        lr.tensorHash = h;
        EXPECT_TRUE(!applier.apply(makeRune({lr}, {}), kBase, model.resolver()).has_value());
    }
    // Derived copies -> refuse (patch would be a silent no-op).
    model.tensors["w"].derived = true;
    EXPECT_TRUE(!applier.apply(good, kBase, model.resolver()).has_value());
    model.tensors["w"].derived = false;
    // Quantised target -> refuse.
    model.tensors["w"].type = rn::RuneElemType::Unsupported;
    EXPECT_TRUE(!applier.apply(good, kBase, model.resolver()).has_value());
    model.tensors["w"].type = rn::RuneElemType::F32;

    EXPECT_EQ(model.hash("w"), h);
    EXPECT_EQ(applier.appliedCount(), std::size_t{0});
}

TEST(apply_midwayFailure_restoresEarlierOps) {
    FakeModel model;
    model.addF32("a", 2, 1, {1, 2});
    model.addF32("b", 2, 1, {3, 4});
    const std::string ha = model.hash("a");
    const std::string hb = model.hash("b");
    // op0 is valid; op1 carries a stale hash -> whole rune must roll back.
    const auto rune = makeRune(
        {scaleOp("a", ha, "s", rn::RuneScaleAxis::Cols),
         scaleOp("b", "fnv1a64:1111111111111111", "s", rn::RuneScaleAxis::Cols)},
        {{"s", {10.0f, 10.0f}}});
    rn::RuneApplier applier;
    const auto r = applier.apply(rune, kBase, model.resolver());
    EXPECT_TRUE(!r.has_value());
    EXPECT_EQ(model.hash("a"), ha);
    EXPECT_EQ(model.hash("b"), hb);
}

TEST(apply_twoOpsOnOneTensor_chainHashes) {
    FakeModel model;
    model.addF32("w", 2, 1, {1, 2});
    const std::string h0 = model.hash("w");
    // Precompute the hash after op0 to author op1 against it.
    FakeModel scratch = model;
    const float mid[2] = {2, 4};
    std::memcpy(scratch.tensors["w"].bytes.data(), mid, sizeof(mid));
    const std::string h1 = scratch.hash("w");

    const auto rune = makeRune(
        {scaleOp("w", h0, "two", rn::RuneScaleAxis::Cols),
         writeSlice("w", h1, "v", 0, 1)},
        {{"two", {2.0f, 2.0f}}, {"v", {5.0f, 6.0f}}});
    rn::RuneApplier applier;
    const auto r = applier.apply(rune, kBase, model.resolver());
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r->touched.size(), std::size_t{1});
    EXPECT_EQ(r->touched[0].hashBefore, h0);
    EXPECT_TRUE(model.f32("w") == (std::vector<float>{5, 6}));
}

TEST(apply_stackedRunes_thenRollbackRestoresOriginal) {
    FakeModel model;
    model.addF32("w", 2, 2, {1, 2, 3, 4});
    const std::string h0 = model.hash("w");
    rn::RuneApplier applier;

    const auto r1 = makeRune({scaleOp("w", h0, "s", rn::RuneScaleAxis::Rows)},
                             {{"s", {2.0f, 3.0f}}});
    EXPECT_TRUE(applier.apply(r1, kBase, model.resolver()).has_value());
    EXPECT_TRUE(model.f32("w") == (std::vector<float>{2, 4, 9, 12}));

    // Second rune authored against the post-r1 state; applying it out of
    // order (before r1) would fail its hash check.
    const std::string h1 = model.hash("w");
    const auto r2 = makeRune({writeSlice("w", h1, "p", 0, 1)}, {{"p", {0.0f, 0.0f}}});
    EXPECT_TRUE(applier.apply(r2, kBase, model.resolver()).has_value());
    EXPECT_TRUE(model.f32("w") == (std::vector<float>{0, 0, 9, 12}));
    EXPECT_EQ(applier.appliedCount(), std::size_t{2});

    EXPECT_EQ(applier.rollback(), std::size_t{2});
    EXPECT_EQ(model.hash("w"), h0);
    EXPECT_EQ(applier.appliedCount(), std::size_t{0});
}

TEST(apply_viaSerializedFile_endToEnd) {
    FakeModel model;
    model.addF32("blk.0.attn_output.weight", 3, 1, {1, 1, 1});
    const auto authored = makeRune(
        {writeSlice("blk.0.attn_output.weight", model.hash("blk.0.attn_output.weight"),
                    "row0", 0, 1)},
        {{"row0", {0.25f, -0.5f, 4.0f}}});
    const auto bytes = authored.serialize();
    const auto loaded = rn::RuneFile::parse(std::span<const std::uint8_t>(bytes));
    rn::RuneApplier applier;
    EXPECT_TRUE(applier.apply(loaded, kBase, model.resolver()).has_value());
    EXPECT_TRUE(model.f32("blk.0.attn_output.weight") == (std::vector<float>{0.25f, -0.5f, 4.0f}));
}

// ======================================================================
// Futhark — staged apply over a WeightsMap with injected device I/O
// ======================================================================

namespace {

namespace gg = mimirmind::core::gguf;

/// "Device" memory: separate host buffers the engine would only reach via
/// readback/upload. Counts transfers and can fail one upload on demand.
struct FakeDevice {
    std::map<std::string, std::vector<float>> mem;
    int         readbacks{0};
    int         uploads{0};
    const void* failUploadTo{nullptr};

    rn::FutharkDeviceIo io() {
        rn::FutharkDeviceIo d;
        d.readback = [this](void* dst, const void* src, std::size_t n) {
            ++readbacks;
            std::memcpy(dst, src, n);
        };
        d.upload = [this](void* dst, const void* src, std::size_t n) {
            if (dst == failUploadTo) {
                failUploadTo = nullptr;   // fail once, then let the restore through
                std::memcpy(dst, src, n / 2);   // simulate a torn copy
                throw std::runtime_error("injected upload failure");
            }
            ++uploads;
            std::memcpy(dst, src, n);
        };
        return d;
    }

    gg::GgufTensor f32Tensor(const std::string& name, std::uint64_t cols, std::uint64_t rows) {
        gg::GgufTensor t;
        t.name       = name;
        t.type       = gg::GgmlType::F32;
        t.dimensions = {cols, rows};
        t.nelements  = cols * rows;
        t.nbytes     = cols * rows * 4;
        t.usmPtr     = mem.at(name).data();
        return t;
    }

    std::string hash(const std::string& name) const {
        const auto& v = mem.at(name);
        return rn::tensorContentHash(v.data(), v.size() * 4);
    }
};

} // namespace

TEST(futhark_stackedRunes_uploadOncePerTensor) {
    FakeDevice dev;
    dev.mem["w"] = {1, 2, 3, 4};
    const auto weights = gg::WeightsMap::fromOwnedTensors({dev.f32Tensor("w", 2, 2)});
    const std::string h0 = dev.hash("w");
    const std::vector<float> mid = {2, 4, 6, 8};
    const std::string h1 = rn::tensorContentHash(mid.data(), mid.size() * 4);

    std::vector<rn::RuneFile> runes;
    runes.push_back(makeRune({scaleOp("w", h0, "s", rn::RuneScaleAxis::Cols)},
                             {{"s", {2.0f, 2.0f}}}));
    runes.push_back(makeRune({writeSlice("w", h1, "r", 1, 2)}, {{"r", {9.0f, 9.0f}}}));

    rn::Futhark f{dev.io()};
    const auto r = f.apply(runes, kBase, weights);
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r->reports.size(), std::size_t{2});
    EXPECT_EQ(r->tensorsUploaded, std::size_t{1});
    EXPECT_EQ(dev.readbacks, 1);
    EXPECT_EQ(dev.uploads, 1);
    EXPECT_TRUE(dev.mem["w"] == (std::vector<float>{2, 4, 9, 9}));
}

TEST(futhark_failingRune_neverTouchesDevice) {
    FakeDevice dev;
    dev.mem["a"] = {1, 2};
    dev.mem["b"] = {3, 4};
    const auto weights = gg::WeightsMap::fromOwnedTensors(
        {dev.f32Tensor("a", 2, 1), dev.f32Tensor("b", 2, 1)});
    std::vector<rn::RuneFile> runes;
    runes.push_back(makeRune({scaleOp("a", dev.hash("a"), "s", rn::RuneScaleAxis::Cols)},
                             {{"s", {5.0f, 5.0f}}}));
    runes.push_back(makeRune({scaleOp("b", "fnv1a64:2222222222222222", "s",
                                      rn::RuneScaleAxis::Cols)},
                             {{"s", {5.0f, 5.0f}}}));
    rn::Futhark f{dev.io()};
    const auto r = f.apply(runes, kBase, weights);
    EXPECT_TRUE(!r.has_value());
    EXPECT_EQ(dev.uploads, 0);
    EXPECT_TRUE(dev.mem["a"] == (std::vector<float>{1, 2}));
    EXPECT_TRUE(dev.mem["b"] == (std::vector<float>{3, 4}));
}

TEST(futhark_refusesDerivedSiblingAndQuantTargets) {
    FakeDevice dev;
    dev.mem["output.weight"]    = {1, 2};
    dev.mem["output.weight.nv"] = {0, 0};
    dev.mem["q"]                = {1, 2};
    auto q = dev.f32Tensor("q", 2, 1);
    q.type = gg::GgmlType::Q8_0;
    const auto weights = gg::WeightsMap::fromOwnedTensors(
        {dev.f32Tensor("output.weight", 2, 1), dev.f32Tensor("output.weight.nv", 2, 1), q});
    rn::Futhark f{dev.io()};

    std::vector<rn::RuneFile> r1;
    r1.push_back(makeRune({scaleOp("output.weight", dev.hash("output.weight"), "s",
                                   rn::RuneScaleAxis::Cols)},
                          {{"s", {2.0f, 2.0f}}}));
    const auto a = f.apply(r1, kBase, weights);
    EXPECT_TRUE(!a.has_value());
    EXPECT_TRUE(a.error().find("derived copies") != std::string::npos);

    std::vector<rn::RuneFile> r2;
    r2.push_back(makeRune({scaleOp("q", dev.hash("q"), "s", rn::RuneScaleAxis::Cols)},
                          {{"s", {2.0f, 2.0f}}}));
    const auto b = f.apply(r2, kBase, weights);
    EXPECT_TRUE(!b.has_value());
    EXPECT_TRUE(b.error().find("quantised") != std::string::npos);

    EXPECT_EQ(dev.readbacks, 0);   // refused targets are never even read back
    EXPECT_EQ(dev.uploads, 0);
}

TEST(futhark_uploadFailure_restoresDevice) {
    FakeDevice dev;
    dev.mem["a"] = {1, 2};
    dev.mem["b"] = {3, 4};
    const auto weights = gg::WeightsMap::fromOwnedTensors(
        {dev.f32Tensor("a", 2, 1), dev.f32Tensor("b", 2, 1)});
    std::vector<rn::RuneFile> runes;
    runes.push_back(makeRune({scaleOp("a", dev.hash("a"), "s", rn::RuneScaleAxis::Cols),
                              scaleOp("b", dev.hash("b"), "s", rn::RuneScaleAxis::Cols)},
                             {{"s", {7.0f, 7.0f}}}));
    dev.failUploadTo = dev.mem["b"].data();
    rn::Futhark f{dev.io()};
    const auto r = f.apply(runes, kBase, weights);
    EXPECT_TRUE(!r.has_value());
    EXPECT_TRUE(r.error().find("restored to the base model") != std::string::npos);
    EXPECT_TRUE(dev.mem["a"] == (std::vector<float>{1, 2}));
    EXPECT_TRUE(dev.mem["b"] == (std::vector<float>{3, 4}));
}

TEST(futhark_wrongBaseFingerprint_refused) {
    FakeDevice dev;
    dev.mem["w"] = {1, 2};
    const auto weights = gg::WeightsMap::fromOwnedTensors({dev.f32Tensor("w", 2, 1)});
    std::vector<rn::RuneFile> runes;
    runes.push_back(makeRune({scaleOp("w", dev.hash("w"), "s", rn::RuneScaleAxis::Cols)},
                             {{"s", {2.0f, 2.0f}}}));
    rn::Futhark f{dev.io()};
    EXPECT_TRUE(!f.apply(runes, "some-other-model", weights).has_value());
    EXPECT_EQ(dev.uploads, 0);
}

// ======================================================================
// Checkpoint fingerprint (shared with Munin)
// ======================================================================

TEST(checkpointFingerprint_matchesShardSetFormula) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
                         ("rune_tests_ckpt_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    {
        std::ofstream(dir / "model.safetensors", std::ios::binary) << std::string(100, 'x');
    }
    const std::vector<mimirmind::core::safetensors::ShardSize> one = {{"model.safetensors", 100}};
    // Pinned against an independent standard FNV-1a 64 of
    // "model.safetensors:100;total:0" — external authoring tools rely on it.
    EXPECT_EQ(mimirmind::core::safetensors::shardSetFingerprint(one, 0),
              std::string{"nvfp4:07eb5b978ad457b3"});
    EXPECT_EQ(mimirmind::core::safetensors::checkpointFingerprint(dir.string()),
              mimirmind::core::safetensors::shardSetFingerprint(one, 0));

    {
        std::ofstream(dir / "a.safetensors", std::ios::binary) << std::string(10, 'a');
        std::ofstream(dir / "b.safetensors", std::ios::binary) << std::string(20, 'b');
        std::ofstream(dir / "model.safetensors.index.json")
            << R"({"metadata":{"total_size":30},"weight_map":{"x":"a.safetensors","y":"b.safetensors"}})";
    }
    const std::vector<mimirmind::core::safetensors::ShardSize> two = {{"a.safetensors", 10},
                                                                     {"b.safetensors", 20}};
    const std::string fp = mimirmind::core::safetensors::checkpointFingerprint(dir.string());
    EXPECT_EQ(fp, mimirmind::core::safetensors::shardSetFingerprint(two, 30));
    EXPECT_EQ(fp.substr(0, 6), std::string{"nvfp4:"});
    fs::remove_all(dir);
}

int main() {
    return mm::test::run();
}
