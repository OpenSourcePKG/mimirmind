// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/RuneStaging.hpp"

#include "core/gguf/GgufTypes.hpp"
#include "core/rune/RuneFile.hpp"
#include "core/rune/RuneFormat.hpp"
#include "core/safetensors/CheckpointFingerprint.hpp"
#include "hugin/ForwardSupport.hpp"
#include "hugin/ForwardTools.hpp"
#include "hugin/ModelView.hpp"
#include "hugin/SurgeryTools.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>

namespace mimirmind::hugin {

namespace {

using nlohmann::json;
using namespace forward_support;
namespace fs = std::filesystem;
namespace rn = core::rune;
namespace gg = core::gguf;

std::string nowIso() {
    const auto        t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm           tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

/// Rune names become file names: a conservative charset, no traversal.
std::string checkedName(const json& a) {
    const std::string n = a.at("name").get<std::string>();
    if (n.empty() || n.size() > 96 || n.front() == '.') {
        throw std::runtime_error("rune name must be 1-96 chars and not start with '.'");
    }
    for (const char c : n) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
        if (!ok) {
            throw std::runtime_error("rune name may only contain [A-Za-z0-9._-]");
        }
    }
    return n;
}

std::vector<std::uint8_t> readFile(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read '" + p.string() + "'");
    }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void writeFile(const fs::path& p, const void* data, std::size_t n) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    if (!out) {
        throw std::runtime_error("cannot write '" + p.string() + "'");
    }
}

std::string fileHash(const fs::path& p) {
    const auto b = readFile(p);
    return rn::tensorContentHash(b.data(), b.size());
}

/// Budget argument: default, clamped to the hard ceiling.
double budgetArg(const json& a, const char* key, double dflt, double ceiling) {
    const double v = a.value(key, dflt);
    if (!(v > 0.0) || v > ceiling) {
        throw std::runtime_error(std::string{"'"} + key + "' must be in (0, " +
                                 std::to_string(ceiling) + "]");
    }
    return v;
}

/// 'override_reason' (empty when absent).
std::string overrideReason(const json& a) {
    return a.contains("override_reason") ? a["override_reason"].get<std::string>() : std::string{};
}

/// 7.12 evidence gate for a steer edit: held-out separation of the direction
/// (or condition) that decides where the edit acts. Throws unless the
/// evidence passes or an 'override_reason' is given; returns the provenance
/// record either way. The shuffled-label null is a diagnostic only: with
/// d_model >> prompts the in-sample / val metric separates random labels as
/// well (null p95 = 1), so only held-out prompts can show a real effect.
json evidenceGate(const json& a, const std::string& what,
                  const std::optional<SeparationReport>& holdout, const NullStats& null,
                  double fitted, const char* fittedName, bool checkNegativeFire) {
    const double minAuc  = a.value("min_holdout_auc", 0.8);
    const double maxFire = a.value("max_holdout_negative_fire", 0.05);
    if (!(minAuc >= 0.5 && minAuc <= 1.0)) {
        throw std::runtime_error("'min_holdout_auc' must be in [0.5, 1]");
    }
    if (!(maxFire >= 0.0 && maxFire <= 1.0)) {
        throw std::runtime_error("'max_holdout_negative_fire' must be in [0, 1]");
    }
    constexpr std::size_t kMinHoldoutPerClass = 4;
    std::vector<std::string> fails;
    if (!holdout) {
        fails.push_back(what + " has no held-out evidence (fit it with holdout_positive / "
                        "holdout_negative)");
    } else {
        // Count prompts, not rows: one prompt with many generated rows is one sample.
        const bool        grouped = holdout->positivePrompts + holdout->negativePrompts > 0;
        const std::size_t hp = grouped ? holdout->positivePrompts : holdout->positives;
        const std::size_t hn = grouped ? holdout->negativePrompts : holdout->negatives;
        if (hp < kMinHoldoutPerClass || hn < kMinHoldoutPerClass) {
            fails.push_back(what + " held-out set too small (" + std::to_string(hp) + " / " +
                            std::to_string(hn) + " prompts, need >= " +
                            std::to_string(kMinHoldoutPerClass) + " per class)");
        }
        if (holdout->auc < minAuc) {
            fails.push_back(what + " held-out AUC " + std::to_string(holdout->auc) +
                            " < min_holdout_auc " + std::to_string(minAuc));
        }
        if (checkNegativeFire && holdout->negativeFireRate > maxFire) {
            fails.push_back(what + " fires on " + std::to_string(holdout->negativeFireRate) +
                            " of the held-out negative rows > max_holdout_negative_fire " +
                            std::to_string(maxFire));
        }
    }
    const std::string reason = overrideReason(a);
    json ev{{"what", what}, {"holdout", holdout ? SurgeryTools::toJson(*holdout) : json(nullptr)},
            {"null", SurgeryTools::toJson(null)}, {fittedName, fitted},
            {"fit_metric_informative", null.permutations > 0 && fitted > null.p95},
            {"min_holdout_auc", minAuc}, {"failed", fails}};
    if (checkNegativeFire) {
        ev["max_holdout_negative_fire"] = maxFire;
    }
    if (!fails.empty()) {
        if (reason.empty()) {
            std::string msg = "evidence gate (7.12) failed:";
            for (const auto& f : fails) msg += "\n  - " + f;
            msg += "\npass 'override_reason' to stage anyway (recorded in the provenance)";
            throw std::runtime_error(msg);
        }
        ev["override_reason"] = reason;
    }
    return ev;
}

/// 7.10: provenance of an edit_search front point; the staged edit must be
/// exactly the trial's edit (same headsKey / same vector and alpha).
json searchProvenance(const json& a, const SurgeryTools& surgery, const std::string& stagedKey) {
    if (!a.contains("search") && !a.contains("trial")) {
        return nullptr;
    }
    if (!a.contains("search") || !a.contains("trial")) {
        throw std::runtime_error("pass 'search' and 'trial' together");
    }
    json pr = surgery.searchTrial(a["search"].get<std::string>(), a["trial"].get<std::size_t>());
    const json& edit = pr["result"]["edit"];
    std::string trialKey;
    if (edit.is_array()) {
        std::vector<SurgeryTools::HeadEditKey> keys;
        for (const auto& he : edit) {
            keys.emplace_back(he["block"].get<std::uint32_t>(), he["heads"].get<std::vector<std::size_t>>(),
                              he["scale"].get<double>());
        }
        trialKey = SurgeryTools::headsKey(keys);
    } else {
        trialKey = SurgeryTools::vectorKey(edit["vector"].get<std::string>(), edit["alpha"].get<double>());
    }
    if (trialKey != stagedKey) {
        throw std::runtime_error("the staged edit differs from edit_search trial " +
                                 std::to_string(a["trial"].get<std::size_t>()) + " (stage its 'edit' as returned)");
    }
    return pr;
}

/// 7.12 causality gate: the same edit must have a causality_check verdict of
/// at least "weak"; with `checkDose` its measured dose must stay <= the floor.
json causalityGate(const json& a, const std::string& what, const SurgeryTools::Causality* c,
                   bool checkDose) {
    std::vector<std::string> fails;
    if (c == nullptr) {
        fails.push_back(what + " has no causality_check verdict for exactly this edit");
    } else {
        if (c->verdict != "causal" && c->verdict != "weak") {
            fails.push_back(what + " causality verdict is '" + c->verdict + "' (ratio " +
                            std::to_string(c->ratio) + " vs random edits)");
        }
        if (checkDose && c->dose > RuneStaging::kSteerDoseReasonFloor) {
            fails.push_back(what + " dose " + std::to_string(c->dose) + " of the residual norm > " +
                            std::to_string(RuneStaging::kSteerDoseReasonFloor));
        }
    }
    const std::string reason = overrideReason(a);
    json rec{{"what", what}, {"causality", c ? SurgeryTools::toJson(*c) : json(nullptr)},
             {"failed", fails}};
    if (!fails.empty()) {
        if (reason.empty()) {
            std::string msg = "causality gate (7.12) failed:";
            for (const auto& f : fails) msg += "\n  - " + f;
            msg += "\nrun causality_check on this edit, or pass 'override_reason' (recorded in "
                   "the provenance)";
            throw std::runtime_error(msg);
        }
        rec["override_reason"] = reason;
    }
    return rec;
}

/// 7.12 dose / placement limits shared by every steer kind.
void steerLimits(const json& a, std::uint32_t block, double rel, double maxRel) {
    if (block == 0) {
        throw std::runtime_error("steering after block 0 is refused (7.12): it silences or "
                                 "derails the model — pick a later block");
    }
    if (rel > RuneStaging::kSteerDoseReasonFloor && overrideReason(a).empty()) {
        throw std::runtime_error("steer dose " + std::to_string(rel) + " of the residual norm > " +
                                 std::to_string(RuneStaging::kSteerDoseReasonFloor) +
                                 " needs an 'override_reason' (7.12)");
    }
    if (rel > maxRel) {
        throw std::runtime_error("edit budget exceeded: relative norm " + std::to_string(rel) +
                                 " > max_relative_norm " + std::to_string(maxRel));
    }
}

const char* stateName(int s) {
    static const char* kNames[] = {"staged", "committed", "rejected"};
    return kNames[s];
}

} // namespace

RuneStaging::RuneStaging(ForwardTools& forward, SurgeryTools& surgery, fs::path dir,
                         std::string huginVersion)
    : _forward(forward), _surgery(surgery), _dir(std::move(dir)), _version(std::move(huginVersion)) {
    // 7.10: edit_search evaluates head trials as real runes (engine load path,
    // KV-cached generation) instead of uncached probe forwards.
    _surgery.setHeadRuneWriter([this](const std::vector<SurgeryTools::HeadEditKey>& edits, const std::string& tag) {
        std::vector<HeadScale> specs;
        for (const auto& [b, h, sc] : edits) specs.push_back(HeadScale{b, h, sc});
        return writeSearchHeadRune(std::move(specs), tag);
    });
}

fs::path RuneStaging::dirOf(State s) const {
    return _dir / stateName(static_cast<int>(s));
}

fs::path RuneStaging::runePath(State s, const std::string& name) const {
    return dirOf(s) / (name + ".rune");
}

fs::path RuneStaging::sidecarPath(State s, const std::string& name) const {
    return dirOf(s) / (name + ".json");
}

bool RuneStaging::exists(const std::string& name) const {
    for (const State s : {State::Staged, State::Committed, State::Rejected}) {
        if (fs::exists(runePath(s, name))) {
            return true;
        }
    }
    return false;
}

RuneStaging::State RuneStaging::locate(const std::string& name) const {
    for (const State s : {State::Staged, State::Committed, State::Rejected}) {
        if (fs::exists(runePath(s, name))) {
            return s;
        }
    }
    throw std::runtime_error("no rune named '" + name + "' under '" + _dir.string() + "'");
}

json RuneStaging::readSidecar(State s, const std::string& name) const {
    const auto b = readFile(sidecarPath(s, name));
    return json::parse(b.begin(), b.end());
}

void RuneStaging::writeSidecar(State s, const std::string& name, const json& j) const {
    const std::string text = j.dump(2);
    writeFile(sidecarPath(s, name), text.data(), text.size());
}

void RuneStaging::move(State from, State to, const std::string& name) const {
    fs::create_directories(dirOf(to));
    fs::rename(runePath(from, name), runePath(to, name));
    fs::rename(sidecarPath(from, name), sidecarPath(to, name));
}

void RuneStaging::audit(const std::string& event, const std::string& name, json details) const {
    fs::create_directories(_dir);
    std::ofstream out(_dir / "audit.jsonl", std::ios::app);
    out << json{{"at", nowIso()}, {"event", event}, {"name", name},
                {"details", std::move(details)}}.dump()
        << "\n";
}

std::string RuneStaging::baseFingerprint() const {
    const auto t = _forward.target();
    if (t.path.empty()) {
        throw std::runtime_error("no model loaded — call open_model first");
    }
    if (fs::is_directory(t.path)) {
        return core::safetensors::checkpointFingerprint(t.path);
    }
    return ModelView{t.path}.fingerprint();
}

json RuneStaging::stage(const std::string& name, const std::string& kind,
                        const std::vector<std::uint8_t>& bytes, json params) const {
    if (exists(name)) {
        throw std::runtime_error("a rune named '" + name + "' already exists (" +
                                 stateName(static_cast<int>(locate(name))) +
                                 ") — pick a new name or rune_rollback it");
    }
    fs::create_directories(dirOf(State::Staged));
    const fs::path p = runePath(State::Staged, name);
    writeFile(p, bytes.data(), bytes.size());
    const std::string hash = rn::tensorContentHash(bytes.data(), bytes.size());
    json side{{"name", name}, {"kind", kind}, {"status", "staged"}, {"rune_hash", hash},
              {"model", _forward.target().path}, {"params", params},
              {"history", json::array({json{{"at", nowIso()}, {"event", "staged"}}})}};
    writeSidecar(State::Staged, name, side);
    audit("staged", name, {{"kind", kind}, {"rune_hash", hash}, {"params", std::move(params)}});
    return json{{"name", name}, {"status", "staged"}, {"path", fs::absolute(p).string()},
                {"rune_hash", hash}, {"next", "rune_gate"}};
}

json RuneStaging::stageLowRank(const std::string& name, const std::vector<LowRankSpec>& specs,
                               json params) const {
    (void)checkedName(json{{"name", name}});
    if (specs.empty()) {
        throw std::runtime_error("no adapters to stage");
    }
    const std::string base = baseFingerprint();
    rn::RuneManifest man;
    man.name            = name;
    man.baseFingerprint = base;
    man.provenanceJson  = json{{"tool", "hugin"}, {"hugin_version", _version}, {"kind", "lowrank"},
                               {"created_at", nowIso()}, {"params", params}}.dump();
    std::map<std::string, std::vector<float>> payloads;
    for (const auto& s : specs) {
        rn::RuneOp op;
        op.kind     = rn::RuneOpKind::LowRank;
        op.target   = s.target;
        op.payload  = "lora." + s.target + ".a";
        op.payloadB = "lora." + s.target + ".b";
        op.rank     = s.rank;
        op.alpha    = s.alpha;
        payloads[op.payload]  = s.a;
        payloads[op.payloadB] = s.b;
        man.ops.push_back(std::move(op));
    }
    const rn::RuneFile rune{std::move(man), std::move(payloads)};
    json out = stage(name, "lowrank", rune.serialize(), std::move(params));
    out["base_fingerprint"] = base;
    out["adapters"]         = specs.size();
    return out;
}

RuneStaging::HeadRune RuneStaging::buildHeadRune(const json& a, std::vector<HeadScale>& specs) const {
    const auto t = _forward.target();
    if (t.path.empty()) {
        throw std::runtime_error("no model loaded — call open_model first");
    }
    if (specs.empty()) {
        throw std::runtime_error("no head edits given");
    }
    // Per-target cache of what a head rune binds to (fingerprint, head count,
    // attn_output type / columns / content hash per block): edit_search builds
    // one rune per trial, and re-reading the weights would reload the base
    // engine under --single-engine every time.
    const std::string key = t.path + "|" + t.tokenizerGguf;
    if (_headCache.key != key) {
        _headCache = HeadCache{};
        _headCache.key = key;
    }
    const bool checkpoint = fs::is_directory(t.path);
    // GGUF: the file bytes. Checkpoint: the bytes the engine holds after load
    // (e.g. FP8 attention requantised to 32-blocks) — what Futhark hashes.
    std::optional<ModelView> mv;
    const auto view = [&]() -> ModelView& {
        if (!mv) mv.emplace(t.path);
        return *mv;
    };
    if (_headCache.fingerprint.empty()) {
        _headCache.fingerprint = checkpoint ? baseFingerprint() : view().fingerprint();
        _headCache.heads       = _forward.engine().config().headCount;
        _headCache.blocks      = _forward.engine().config().blockCount;
    }
    const std::size_t H  = _headCache.heads;
    const double maxFrac  = budgetArg(a, "max_head_fraction", 0.25, kMaxHeadFraction);
    const double maxScale = budgetArg(a, "max_scale", 8.0, kMaxHeadScale);

    std::sort(specs.begin(), specs.end(),
              [](const HeadScale& x, const HeadScale& y) { return x.block < y.block; });
    HeadRune out;
    out.manifest.baseFingerprint = _headCache.fingerprint;
    out.heads    = H;
    out.maxFrac  = maxFrac;
    out.maxScale = maxScale;
    out.edits    = json::array();
    for (std::size_t i = 0; i < specs.size(); ++i) {
        HeadScale& sp = specs[i];
        if (i > 0 && specs[i - 1].block == sp.block) {
            throw std::runtime_error("block " + std::to_string(sp.block) +
                                     " appears twice — one head_edits entry per block");
        }
        if (sp.block >= _headCache.blocks) {
            throw std::runtime_error("block " + std::to_string(sp.block) + " out of range");
        }
        const auto blk = sp.block;
        std::sort(sp.heads.begin(), sp.heads.end());
        sp.heads.erase(std::unique(sp.heads.begin(), sp.heads.end()), sp.heads.end());
        if (sp.heads.empty()) {
            throw std::runtime_error("'heads' must not be empty (block " + std::to_string(blk) + ")");
        }
        if (sp.heads.back() >= H) {
            throw std::runtime_error("head " + std::to_string(sp.heads.back()) + " out of range");
        }
        if (!std::isfinite(sp.scale) || sp.scale < 0.0 || sp.scale > maxScale) {
            throw std::runtime_error("edit budget exceeded: scale " + std::to_string(sp.scale) +
                                     " must be in [0, max_scale " + std::to_string(maxScale) + "]");
        }
        const double frac = static_cast<double>(sp.heads.size()) / static_cast<double>(H);
        if (frac > maxFrac) {
            throw std::runtime_error("edit budget exceeded: " + std::to_string(sp.heads.size()) + "/" +
                                     std::to_string(H) + " heads in block " + std::to_string(blk) +
                                     " > max_head_fraction " + std::to_string(maxFrac));
        }
        const std::string target = "blk." + std::to_string(blk) + ".attn_output.weight";
        auto it = _headCache.tensors.find(blk);
        if (it == _headCache.tensors.end()) {
            HeadCache::Tensor ct;
            std::vector<std::uint8_t> bytes;
            if (!checkpoint) {
                const auto& tensor = view().require(target);
                ct.type = tensor.type;
                ct.cols = ModelView::colsOf(tensor);
                const auto b = view().bytes(tensor);
                bytes.assign(b.begin(), b.end());
            } else {
                Engine& e = _forward.engine();
                const auto* tensor = e.weights().find(target);
                if (tensor == nullptr || tensor->usmPtr == nullptr || tensor->dimensions.empty()) {
                    throw std::runtime_error("'" + target + "' not found in the loaded model");
                }
                ct.type = tensor->type;
                ct.cols = tensor->dimensions[0];
                bytes.resize(tensor->nbytes);
                e.ops()->readbackToHost(bytes.data(), tensor->usmPtr, bytes.size());
            }
            ct.hash = rn::tensorContentHash(bytes.data(), bytes.size());
            it = _headCache.tensors.emplace(blk, std::move(ct)).first;
        }
        const gg::GgmlType  type = it->second.type;
        const std::uint64_t cols = it->second.cols;
        const bool blockScaled = type == gg::GgmlType::Q8_0 || type == gg::GgmlType::FP8_E4M3;
        const bool okType = blockScaled || type == gg::GgmlType::F32 || type == gg::GgmlType::F16 ||
                            type == gg::GgmlType::BF16;
        if (!okType) {
            throw std::runtime_error("'" + target + "' is " + std::string{gg::typeInfo(type).name} +
                                     " — head runes support F32/F16/BF16/Q8_0/FP8_E4M3 only");
        }
        if (H == 0 || cols % H != 0) {
            throw std::runtime_error("'" + target + "' has " + std::to_string(cols) +
                                     " input columns, not a multiple of the head count " +
                                     std::to_string(H));
        }
        const std::size_t hd = static_cast<std::size_t>(cols / H);
        if (blockScaled) {
            if (hd % 32 != 0) {
                throw std::runtime_error("head_dim is not a multiple of the 32-element block");
            }
            if (sp.scale != 0.0 && sp.scale != 1.0 &&
                std::exp2(std::round(std::log2(sp.scale))) != sp.scale) {
                throw std::runtime_error("'" + target + "' is block-scaled (" +
                                         std::string{gg::typeInfo(type).name} +
                                         "): the scale must be 0 or a power of two to stay exact");
            }
        }
        std::vector<float> factors(static_cast<std::size_t>(cols), 1.0F);
        for (const auto h : sp.heads) {
            std::fill_n(factors.begin() + static_cast<std::ptrdiff_t>(h * hd), hd,
                        static_cast<float>(sp.scale));
        }
        rn::RuneOp op;
        op.kind       = rn::RuneOpKind::Scale;
        op.target     = target;
        op.tensorHash = it->second.hash;
        op.payload    = "heads." + target;
        op.axis       = rn::RuneScaleAxis::Cols;
        out.manifest.ops.push_back(op);
        out.payloads.emplace(op.payload, std::move(factors));
        out.edits.push_back({{"block", blk}, {"heads", sp.heads}, {"scale", sp.scale},
                             {"head_fraction", frac}, {"target", target},
                             {"target_type", std::string{gg::typeInfo(type).name}}});
    }
    return out;
}

std::string RuneStaging::writeSearchHeadRune(std::vector<HeadScale> specs, const std::string& tag) const {
    HeadRune hr = buildHeadRune(json::object(), specs);
    hr.manifest.name           = tag;
    hr.manifest.provenanceJson = json{{"tool", "hugin"}, {"hugin_version", _version},
                                      {"kind", "edit_search_trial"}, {"created_at", nowIso()},
                                      {"params", {{"head_edits", hr.edits}}}}.dump();
    const rn::RuneFile rune{std::move(hr.manifest), std::move(hr.payloads)};
    const auto bytes = rune.serialize();
    const fs::path dir = _dir / "search";
    fs::create_directories(dir);
    const fs::path p = dir / (tag + ".rune");
    writeFile(p, bytes.data(), bytes.size());
    return fs::absolute(p).string();
}

json RuneStaging::stageHeadScales(const json& a, const std::string& name, const std::string& kind,
                                  std::vector<HeadScale> specs) const {
    HeadRune hr = buildHeadRune(a, specs);
    std::vector<SurgeryTools::HeadEditKey> keys;
    for (const auto& sp : specs) keys.emplace_back(sp.block, sp.heads, sp.scale);
    const json causal = causalityGate(a, "head edits", _surgery.causality(SurgeryTools::headsKey(keys)), true);
    rn::RuneManifest man = std::move(hr.manifest);
    man.name = name;
    const json& edits = hr.edits;
    json params{{"head_edits", edits}, {"block_heads", hr.heads}};
    if (edits.size() == 1) {   // single-block summary (rune_stage_ablate compatibility)
        params["block"]         = edits[0]["block"];
        params["heads"]         = edits[0]["heads"];
        params["head_fraction"] = edits[0]["head_fraction"];
        params["target"]        = edits[0]["target"];
        params["target_type"]   = edits[0]["target_type"];
    }
    params["causality"] = causal;
    if (const json sp = searchProvenance(a, _surgery, SurgeryTools::headsKey(keys)); !sp.is_null()) {
        params["edit_search"] = sp;
    }
    man.provenanceJson = json{{"tool", "hugin"}, {"hugin_version", _version},
                              {"kind", kind}, {"created_at", nowIso()},
                              {"note", a.value("note", std::string{})},
                              {"params", params},
                              {"budget", {{"max_head_fraction", hr.maxFrac}, {"max_scale", hr.maxScale}}}}.dump();
    const rn::RuneFile rune{std::move(man), std::move(hr.payloads)};
    json out = stage(name, kind, rune.serialize(), params);
    out["base_fingerprint"] = rune.manifest().baseFingerprint;
    return out;
}

void RuneStaging::registerOn(McpServer& server) {
    server.addTool(
        "rune_stage_steer",
        "Write a session steering vector (steer_vector) as a staged Rune: a Steer op that adds "
        "alpha x vector to the residual stream after its block on every forward path. Guardrail: "
        "|alpha|·||v|| / mean residual norm must stay within 'max_relative_norm' (default 0.5, "
        "ceiling 2.0; above 0.5 only with 'override_reason'); block 0 is refused; the vector "
        "needs held-out evidence from steer_vector (>= 4 held-out prompts per class, held-out "
        "AUC >= 'min_holdout_auc', default 0.8) and a causality_check verdict >= weak for this "
        "vector at this alpha, or an 'override_reason' "
        "(roadmap 7.12, recorded in the provenance). Nothing is activated — run rune_gate next.",
        schema({{"name", {{"type", "string"}}},
                {"vector", {{"type", "string"}}},
                {"alpha", {{"type", "number"}}},
                {"max_relative_norm", {{"type", "number"}}},
                {"min_holdout_auc", {{"type", "number"}}},
                {"max_holdout_negative_fire", {{"type", "number"}}},
                {"override_reason", {{"type", "string"}}},
                {"search", {{"type", "string"}}},
                {"trial", {{"type", "integer"}}},
                {"note", {{"type", "string"}, {"description", "why this edit (kept in provenance)"}}}},
               {"name", "vector", "alpha"}),
        [this](const json& a) {
            const std::string name = checkedName(a);
            const std::string base = baseFingerprint();   // also: model loaded?
            const auto& v     = _surgery.steerVector(a.at("vector").get<std::string>());
            const double alpha = a.at("alpha").get<double>();
            const double maxRel = budgetArg(a, "max_relative_norm", 0.5, kMaxSteerRelativeNorm);
            if (!std::isfinite(alpha) || alpha == 0.0) {
                throw std::runtime_error("'alpha' must be finite and non-zero");
            }
            const double rel = v.residualNorm > 0 ? std::fabs(alpha) * v.norm / v.residualNorm : 1e300;
            steerLimits(a, v.block, rel, maxRel);
            const json evidence = evidenceGate(a, "steering vector '" + a["vector"].get<std::string>() + "'",
                                               v.holdout, v.null, v.inSampleAuc, "in_sample_auc", false);
            const json causal = causalityGate(
                a, "steer '" + a["vector"].get<std::string>() + "' at alpha " + std::to_string(alpha),
                _surgery.causality(SurgeryTools::vectorKey(a["vector"].get<std::string>(), alpha)), false);
            const std::string block   = "blk." + std::to_string(v.block);
            const std::string payload = "steer." + block;
            json params{{"vector", a["vector"]}, {"block", v.block}, {"alpha", alpha},
                        {"vector_norm", v.norm}, {"mean_residual_norm", v.residualNorm},
                        {"relative_norm", rel}, {"positives", v.positives},
                        {"negatives", v.negatives}, {"evidence", evidence}, {"causality", causal}};
            if (const json sp = searchProvenance(a, _surgery, SurgeryTools::vectorKey(a["vector"].get<std::string>(), alpha));
                !sp.is_null()) {
                params["edit_search"] = sp;
            }
            rn::RuneManifest man;
            man.name            = name;
            man.baseFingerprint = base;
            man.provenanceJson  = json{{"tool", "hugin"}, {"hugin_version", _version},
                                       {"kind", "steer"}, {"created_at", nowIso()},
                                       {"note", a.value("note", std::string{})},
                                       {"params", params},
                                       {"budget", {{"max_relative_norm", maxRel}}}}.dump();
            rn::RuneOp op;
            op.kind    = rn::RuneOpKind::Steer;
            op.target  = block;
            op.payload = payload;
            op.alpha   = alpha;
            man.ops.push_back(op);
            const rn::RuneFile rune{std::move(man), {{payload, v.values}}};
            json out = stage(name, "steer", rune.serialize(), params);
            out["base_fingerprint"] = rune.manifest().baseFingerprint;
            return out;
        });

    server.addTool(
        "rune_stage_steer_if",
        "Write a projection or conditional steer (roadmap 7.11) as a staged Rune — exactly the "
        "spec steer_if_preview shows: 'vector' (steer_vector), 'mode' add | project, 'alpha', "
        "optional 'condition' (condition_vector) + 'threshold'. With a condition: a steer_if op "
        "(only rows with cos(h_condition_block, c) > threshold are changed, per token); "
        "without: a steer_project op (mode project only). Guardrails: projection alpha in "
        "(0, 1]; add mode |alpha|·||v|| / mean residual norm <= 'max_relative_norm' (default "
        "0.5, ceiling 2.0; above 0.5 only with 'override_reason'); block 0 refused. Evidence "
        "(roadmap 7.12): a steer_if needs its condition's held-out evidence from "
        "condition_vector — held-out AUC >= 'min_holdout_auc' (default 0.8), fire rate over every "
        "gated row of the held-out negatives <= 'max_holdout_negative_fire' (default 0.05), >= 4 "
        "held-out rows per class; an add-mode steer also needs a causality_check verdict >= weak "
        "for the vector at this alpha; a steer_project needs the vector's held-out "
        "evidence. Otherwise only with 'override_reason' (recorded in the provenance). Nothing is "
        "activated — run rune_gate next.",
        schema({{"name", {{"type", "string"}}},
                {"vector", {{"type", "string"}}},
                {"alpha", {{"type", "number"}}},
                {"mode", {{"type", "string"}, {"enum", {"add", "project"}}}},
                {"condition", {{"type", "string"}}},
                {"threshold", {{"type", "number"}}},
                {"max_relative_norm", {{"type", "number"}}},
                {"min_holdout_auc", {{"type", "number"}}},
                {"max_holdout_negative_fire", {{"type", "number"}}},
                {"override_reason", {{"type", "string"}}},
                {"note", {{"type", "string"}, {"description", "why this edit (kept in provenance)"}}}},
               {"name", "vector", "alpha"}),
        [this](const json& a) {
            const std::string name = checkedName(a);
            const std::string base = baseFingerprint();   // also: model loaded?
            auto spec = _surgery.condSteerSpec(a);
            const double maxRel = budgetArg(a, "max_relative_norm", 0.5, kMaxSteerRelativeNorm);
            const auto steerBlock = spec.params["block"].get<std::uint32_t>();
            if (spec.op.steerMode == rn::RuneSteerMode::Add &&
                spec.op.kind == rn::RuneOpKind::SteerIf) {
                const double rel = spec.params["relative_norm"].get<double>();
                if (!(rel > 0)) {
                    throw std::runtime_error("relative norm must be > 0");
                }
                steerLimits(a, steerBlock, rel, maxRel);
            } else {
                // Projection: dose is bounded by alpha <= 1 (checked in the spec).
                steerLimits(a, steerBlock, 0.0, maxRel);
            }
            if (spec.op.steerMode == rn::RuneSteerMode::Add) {
                const std::string vn    = a.at("vector").get<std::string>();
                const double      alpha = a.at("alpha").get<double>();
                spec.params["causality"] = causalityGate(
                    a, "steer '" + vn + "' at alpha " + std::to_string(alpha),
                    _surgery.causality(SurgeryTools::vectorKey(vn, alpha)), false);
            }
            if (spec.op.kind == rn::RuneOpKind::SteerIf) {
                const std::string cn = a.at("condition").get<std::string>();
                const auto& c = _surgery.conditionVector(cn);
                if (a.contains("threshold") && c.holdout) {
                    throw std::runtime_error("'threshold' would bypass the held-out evidence of "
                                             "condition '" + cn + "' — refit the condition instead");
                }
                spec.params["evidence"] = evidenceGate(a, "condition '" + cn + "'", c.holdout, c.null,
                                                       c.valAccuracy, "val_accuracy", true);
            } else {
                const auto& v = _surgery.steerVector(a.at("vector").get<std::string>());
                spec.params["evidence"] = evidenceGate(a, "steering vector '" + a["vector"].get<std::string>() + "'",
                                                       v.holdout, v.null, v.inSampleAuc, "in_sample_auc", false);
            }
            const std::string kind{rn::opKindName(spec.op.kind)};
            rn::RuneManifest man;
            man.name            = name;
            man.baseFingerprint = base;
            man.provenanceJson  = json{{"tool", "hugin"}, {"hugin_version", _version},
                                       {"kind", kind}, {"created_at", nowIso()},
                                       {"note", a.value("note", std::string{})},
                                       {"params", spec.params},
                                       {"budget", {{"max_relative_norm", maxRel}}}}.dump();
            man.ops.push_back(spec.op);
            const rn::RuneFile rune{std::move(man), std::move(spec.payloads)};
            json out = stage(name, kind, rune.serialize(), spec.params);
            out["base_fingerprint"] = rune.manifest().baseFingerprint;
            return out;
        });

    server.addTool(
        "rune_stage_ablate",
        "Write a head ablation as a staged Rune: a 0/1 column Scale on blk.<block>.attn_output."
        "weight that zeroes the given heads' input columns of the O-projection (= the "
        "ablate_preview intervention baked into the weight). Same targets as rune_stage_heads "
        "(F32/F16/BF16, exact Q8_0 / FP8_E4M3 block masking; GGUF and NVFP4 checkpoints). "
        "Guardrails: heads / block heads <= 'max_head_fraction' (default 0.25, ceiling 0.5); "
        "causality_check on exactly these heads (scale 0) must say causal or weak, with a dose "
        "<= 0.5 of the residual norm (roadmap 7.12), else only with 'override_reason'.",
        schema({{"name", {{"type", "string"}}},
                {"block", {{"type", "integer"}}},
                {"heads", {{"type", "array"}, {"items", {{"type", "integer"}}}}},
                {"max_head_fraction", {{"type", "number"}}},
                {"override_reason", {{"type", "string"}}},
                {"search", {{"type", "string"}}},
                {"trial", {{"type", "integer"}}},
                {"note", {{"type", "string"}}}},
               {"name", "block", "heads"}),
        [this](const json& a) {
            const std::string name = checkedName(a);
            HeadScale spec;
            spec.block = static_cast<std::uint32_t>(a.at("block").get<std::int64_t>());
            spec.heads = a.at("heads").get<std::vector<std::size_t>>();
            spec.scale = 0.0;
            return stageHeadScales(a, name, "ablate_heads", {spec});
        });

    server.addTool(
        "rune_stage_heads",
        "Write a head scaling as a staged Rune: for each {block, heads, scale} a column Scale on "
        "blk.<block>.attn_output.weight that multiplies those heads' input columns of the "
        "O-projection by 'scale' (0 = ablate, > 1 = amplify) — the ablate_preview / eval_preview "
        "head_edits intervention baked into the weight. Targets: F32/F16/BF16 attn_output "
        "(rounded to the encoding), Q8_0 / FP8_E4M3 block scales (exact: 0 or power-of-two "
        "factors), GGUF and NVFP4 checkpoints. Guardrails: heads per block / block heads <= "
        "'max_head_fraction' (default 0.25, ceiling 0.5) and scale <= 'max_scale' (default 8, "
        "ceiling 16); causality_check on exactly these head_edits must say causal or weak, with "
        "a dose <= 0.5 of the residual norm (roadmap 7.12), else only with 'override_reason'.",
        schema({{"name", {{"type", "string"}}},
                {"head_edits", {{"type", "array"}, {"items", {{"type", "object"}}}}},
                {"max_head_fraction", {{"type", "number"}}},
                {"max_scale", {{"type", "number"}}},
                {"override_reason", {{"type", "string"}}},
                {"search", {{"type", "string"}}},
                {"trial", {{"type", "integer"}}},
                {"note", {{"type", "string"}}}},
               {"name", "head_edits"}),
        [this](const json& a) {
            const std::string name = checkedName(a);
            std::vector<HeadScale> specs;
            for (const auto& he : a.at("head_edits")) {
                HeadScale spec;
                spec.block = static_cast<std::uint32_t>(he.at("block").get<std::int64_t>());
                spec.heads = he.at("heads").get<std::vector<std::size_t>>();
                spec.scale = he.value("scale", 0.0);
                specs.push_back(std::move(spec));
            }
            return stageHeadScales(a, name, "scale_heads", std::move(specs));
        });

    server.addTool(
        "rune_stage_experts",
        "Write an expert disable as a staged Rune: a Mask op on blk.<block>.ffn_gate_inp.weight "
        "(the MoE router). The engine pushes the disabled experts' router logits to -inf before "
        "top-K on every path, so they are never routed to; no weight changes, works for any "
        "weight format (GGUF and NVFP4). Use expert_usage first to see which experts fire, "
        "rune_compare to preview. Guardrails: disabled / block experts <= 'max_expert_fraction' "
        "(default 0.125, ceiling 0.25), and at least top-K experts stay enabled.",
        schema({{"name", {{"type", "string"}}},
                {"block", {{"type", "integer"}}},
                {"experts", {{"type", "array"}, {"items", {{"type", "integer"}}}}},
                {"max_expert_fraction", {{"type", "number"}}},
                {"note", {{"type", "string"}}}},
               {"name", "block", "experts"}),
        [this](const json& a) {
            const std::string name = checkedName(a);
            const std::string base = baseFingerprint();   // also: model loaded?
            Engine& e = _forward.engine();
            const std::size_t nExp = e.config().expertCount;
            const std::size_t topK = e.config().expertUsedCount;
            if (nExp == 0) {
                throw std::runtime_error("the loaded model has no MoE experts");
            }
            const auto blk = blockList(e, json{{"blocks", json::array({a.at("block")})}}).front();
            std::vector<std::size_t> experts;
            for (const auto& x : a.at("experts")) {
                const auto v = x.get<std::int64_t>();
                if (v < 0 || v >= static_cast<std::int64_t>(nExp)) {
                    throw std::runtime_error("expert " + std::to_string(v) + " out of range (0.." +
                                             std::to_string(nExp - 1) + ")");
                }
                experts.push_back(static_cast<std::size_t>(v));
            }
            std::sort(experts.begin(), experts.end());
            experts.erase(std::unique(experts.begin(), experts.end()), experts.end());
            if (experts.empty()) {
                throw std::runtime_error("'experts' must not be empty");
            }
            const double maxFrac = budgetArg(a, "max_expert_fraction", 0.125, kMaxExpertFraction);
            const double frac    = static_cast<double>(experts.size()) / static_cast<double>(nExp);
            if (frac > maxFrac) {
                throw std::runtime_error("edit budget exceeded: " + std::to_string(experts.size()) +
                                         "/" + std::to_string(nExp) +
                                         " experts > max_expert_fraction " + std::to_string(maxFrac));
            }
            if (nExp - experts.size() < topK) {
                throw std::runtime_error("disabling " + std::to_string(experts.size()) +
                                         " experts leaves fewer than top-K " + std::to_string(topK));
            }
            const std::string target = "blk." + std::to_string(blk) + ".ffn_gate_inp.weight";
            std::vector<float> keep(nExp, 1.0F);
            for (const auto x : experts) {
                keep[x] = 0.0F;
            }
            json params{{"block", blk}, {"experts", experts}, {"block_experts", nExp},
                        {"top_k", topK}, {"expert_fraction", frac}, {"target", target}};
            rn::RuneManifest man;
            man.name            = name;
            man.baseFingerprint = base;
            man.provenanceJson  = json{{"tool", "hugin"}, {"hugin_version", _version},
                                       {"kind", "disable_experts"}, {"created_at", nowIso()},
                                       {"note", a.value("note", std::string{})},
                                       {"params", params},
                                       {"budget", {{"max_expert_fraction", maxFrac}}}}.dump();
            rn::RuneOp op;
            op.kind    = rn::RuneOpKind::Mask;
            op.target  = target;
            op.payload = "experts." + target;
            man.ops.push_back(op);
            const rn::RuneFile rune{std::move(man), {{op.payload, std::move(keep)}}};
            json out = stage(name, "disable_experts", rune.serialize(), params);
            out["base_fingerprint"] = base;
            return out;
        });

    server.addTool(
        "rune_gate",
        "Gate a staged Rune: forward-only goldset eval on the base model and on the model with "
        "the Rune applied by Futhark (the engine's own load path), on a 'target' set (what the "
        "edit should improve) and an optional 'control' set (what must not regress). Green iff "
        "target accuracy does not drop and (accuracy or mean logprob) improves by more than "
        "'min_target_logprob_gain', control accuracy drops by at most 'max_control_accuracy_drop' "
        "(default 0) and control mean logprob by at most 'max_control_logprob_drop' (default 0.05), "
        "and the rune adds no non-finite items. Free-generation items ({prompt, generate:{...}}, "
        "see eval) are checked per check: a check (answer language, no code-mixing, a fact "
        "regex) that passed on the base and fails with the rune is a regression; more than "
        "'max_check_regressions' (default 0) in a set is red, in the target set as well. "
        "Red moves the rune to rejected/ unless "
        "'auto_reject' is false. Sets: {items:[...]} or {path:\".jsonl\"} (see eval).",
        schema({{"name", {{"type", "string"}}},
                {"target", {{"type", "object"}}},
                {"control", {{"type", "object"}}},
                {"min_target_logprob_gain", {{"type", "number"}}},
                {"max_control_accuracy_drop", {{"type", "number"}}},
                {"max_control_logprob_drop", {{"type", "number"}}},
                {"max_check_regressions", {{"type", "integer"}}},
                {"auto_reject", {{"type", "boolean"}}},
                {"add_bos", {{"type", "boolean"}}}},
               {"name", "target"}),
        [this](const json& a) {
            const std::string name = checkedName(a);
            if (locate(name) != State::Staged) {
                throw std::runtime_error("rune '" + name + "' is not staged");
            }
            const fs::path    path   = fs::absolute(runePath(State::Staged, name));
            const std::string hash   = fileHash(path);
            const bool        addBos = a.value("add_bos", true);
            const double minGain   = a.value("min_target_logprob_gain", 0.0);
            const double maxAccDrop = a.value("max_control_accuracy_drop", 0.0);
            const double maxLpDrop  = a.value("max_control_logprob_drop", 0.05);
            const auto   maxCheckReg = a.value("max_check_regressions", std::int64_t{0});
            if (maxCheckReg < 0) {
                throw std::runtime_error("'max_check_regressions' must be >= 0");
            }

            json reasons = json::array();
            json result{{"rune_hash", hash}, {"at", nowIso()}};
            // Base runs first for every set, then the rune runs: with a single
            // resident engine (--single-engine, models too large for base + variant)
            // that is two loads instead of four.
            struct SetEval {
                json items;
                json rowsB = json::array();
                json b;
            };
            const auto evalBase = [&](const json& set, const char* label) {
                SetEval s;
                s.items = loadGoldset(set);
                if (!s.items.is_array() || s.items.empty()) {
                    throw std::runtime_error(std::string{label} + " set is empty");
                }
                s.b = evalRun(_forward.engine(), s.items, addBos, s.rowsB);
                return s;
            };
            const auto evalRune = [&](const SetEval& s) -> json {
                json rowsR = json::array();
                const json r = evalRun(_forward.engineWithRunes({path.string()}), s.items, addBos, rowsR);
                json flips = json::array();
                for (std::size_t i = 0; i < s.items.size(); ++i) {
                    const bool cb = s.rowsB[i]["base_or_run"]["correct"].get<bool>();
                    const bool cr = rowsR[i]["base_or_run"]["correct"].get<bool>();
                    if (cb != cr) {
                        flips.push_back({{"index", i}, {"id", s.rowsB[i]["id"]}, {"now_correct", cr}});
                    }
                }
                return json{{"base", s.b}, {"rune", r},
                            {"delta_accuracy", r["accuracy"].get<double>() - s.b["accuracy"].get<double>()},
                            {"delta_mean_logprob",
                             r["mean_logprob"].get<double>() - s.b["mean_logprob"].get<double>()},
                            {"new_non_finite", r["non_finite_items"].get<std::size_t>() >
                                                   s.b["non_finite_items"].get<std::size_t>()},
                            {"flips", flips},
                            {"check_regressions", checkRegressions(s.rowsB, rowsR)}};
            };
            const auto checkBudget = [&](const json& set, const char* label) {
                const auto n = set["check_regressions"].size();
                if (n > static_cast<std::size_t>(maxCheckReg)) {
                    reasons.push_back(std::string{label} + ": " + std::to_string(n) +
                                      " free-generation check regression(s)");
                }
            };
            _forward.dropVariant();   // never reuse an engine built from older bytes
            try {
                const SetEval tgtBase = evalBase(a.at("target"), "target");
                std::optional<SetEval> ctlBase;
                if (a.contains("control")) {
                    ctlBase = evalBase(a["control"], "control");
                }
                const json tgt = evalRune(tgtBase);
                result["target"] = tgt;
                const double dAcc = tgt["delta_accuracy"].get<double>();
                const double dLp  = tgt["delta_mean_logprob"].get<double>();
                if (dAcc < 0) reasons.push_back("target accuracy dropped");
                if (!(dAcc > 0 || dLp > minGain)) reasons.push_back("target did not improve");
                if (tgt["new_non_finite"].get<bool>()) reasons.push_back("target: new non-finite items");
                checkBudget(tgt, "target");
                if (ctlBase) {
                    const json ctl = evalRune(*ctlBase);
                    result["control"] = ctl;
                    if (-ctl["delta_accuracy"].get<double>() > maxAccDrop)
                        reasons.push_back("control accuracy dropped beyond budget");
                    if (-ctl["delta_mean_logprob"].get<double>() > maxLpDrop)
                        reasons.push_back("control mean logprob dropped beyond budget");
                    if (ctl["new_non_finite"].get<bool>()) reasons.push_back("control: new non-finite items");
                    checkBudget(ctl, "control");
                } else {
                    result["warning"] = "no control set — regressions outside the target are unchecked";
                }
            } catch (const std::exception& ex) {
                // A rune the engine refuses to load (Futhark) is a red gate, not a tool error.
                reasons.push_back(std::string{"evaluation failed: "} + ex.what());
            }
            _forward.dropVariant();
            const bool green = reasons.empty();
            result["green"]   = green;
            result["reasons"] = reasons;
            result["thresholds"] = {{"min_target_logprob_gain", minGain},
                                    {"max_control_accuracy_drop", maxAccDrop},
                                    {"max_control_logprob_drop", maxLpDrop},
                                    {"max_check_regressions", maxCheckReg}};
            json side = readSidecar(State::Staged, name);
            side["gate"]   = result;
            const bool autoReject = !green && a.value("auto_reject", true);
            side["status"] = green ? "staged (gate green)"
                                   : (autoReject ? "rejected (gate red)" : "staged (gate red)");
            side["history"].push_back({{"at", nowIso()}, {"event", green ? "gate_green" : "gate_red"}});
            writeSidecar(State::Staged, name, side);
            audit(green ? "gate_green" : "gate_red", name, {{"rune_hash", hash}, {"reasons", reasons}});
            if (autoReject) {
                move(State::Staged, State::Rejected, name);
                audit("auto_rollback", name, {{"to", "rejected"}});
                result["moved_to"] = "rejected";
            }
            return json{{"name", name}, {"gate", result}};
        });

    server.addTool(
        "rune_commit",
        "Commit a staged Rune whose gate is green and whose file is byte-identical to what the "
        "gate evaluated. Moves it to committed/ and returns the `runes` config line. Hugin never "
        "edits the serving config: activation happens by adding the path to the model's `runes` "
        "and restarting in a maintenance window.",
        schema({{"name", {{"type", "string"}}}}, {"name"}),
        [this](const json& a) {
            const std::string name = checkedName(a);
            if (locate(name) != State::Staged) {
                throw std::runtime_error("rune '" + name + "' is not staged");
            }
            json side = readSidecar(State::Staged, name);
            if (!side.contains("gate") || !side["gate"].value("green", false)) {
                throw std::runtime_error("rune '" + name + "' has no green gate — run rune_gate");
            }
            const std::string hash = fileHash(runePath(State::Staged, name));
            if (hash != side["gate"]["rune_hash"].get<std::string>()) {
                throw std::runtime_error("rune '" + name + "' changed since its gate (" + hash +
                                         ") — re-gate it");
            }
            side["status"] = "committed";
            side["history"].push_back({{"at", nowIso()}, {"event", "committed"}});
            writeSidecar(State::Staged, name, side);
            move(State::Staged, State::Committed, name);
            audit("committed", name, {{"rune_hash", hash}});
            const std::string p = fs::absolute(runePath(State::Committed, name)).string();
            return json{{"name", name}, {"status", "committed"}, {"path", p}, {"rune_hash", hash},
                        {"activate", {{"config", {{"runes", json::array({p})}}},
                                      {"how", "append to models[].runes for this model, restart in a "
                                              "maintenance window, check the Futhark log line"}}}};
        });

    server.addTool(
        "rune_rollback",
        "Undo a lifecycle step: committed -> staged (gate cleared; also remove it from any "
        "models[].runes config and restart to deactivate), staged -> rejected (discard). "
        "Rejected runes stay on disk for the audit trail.",
        schema({{"name", {{"type", "string"}}}, {"reason", {{"type", "string"}}}}, {"name"}),
        [this](const json& a) {
            const std::string name = checkedName(a);
            const State s = locate(name);
            if (s == State::Rejected) {
                throw std::runtime_error("rune '" + name + "' is already rejected");
            }
            const State to = s == State::Committed ? State::Staged : State::Rejected;
            json side = readSidecar(s, name);
            side.erase("gate");
            side["status"] = to == State::Staged ? "staged (rolled back from committed)" : "rejected (discarded)";
            side["history"].push_back({{"at", nowIso()}, {"event", "rollback"},
                                       {"reason", a.value("reason", std::string{})}});
            writeSidecar(s, name, side);
            move(s, to, name);
            audit("rollback", name, {{"from", stateName(static_cast<int>(s))},
                                     {"to", stateName(static_cast<int>(to))},
                                     {"reason", a.value("reason", std::string{})}});
            json out{{"name", name}, {"status", stateName(static_cast<int>(to))}};
            if (s == State::Committed) {
                out["deactivate"] = "remove '" + fs::absolute(runePath(State::Committed, name)).string() +
                                    "' from models[].runes and restart";
            }
            return out;
        });

    server.addTool(
        "rune_list", "Every Rune under the rune directory with status, kind and gate summary.",
        schema(json::object()),
        [this](const json&) {
            json out = json::array();
            for (const State s : {State::Staged, State::Committed, State::Rejected}) {
                if (!fs::is_directory(dirOf(s))) {
                    continue;
                }
                for (const auto& entry : fs::directory_iterator(dirOf(s))) {
                    if (entry.path().extension() != ".rune") {
                        continue;
                    }
                    const std::string name = entry.path().stem().string();
                    json row{{"name", name}, {"state", stateName(static_cast<int>(s))},
                             {"path", fs::absolute(entry.path()).string()}};
                    try {
                        const json side = readSidecar(s, name);
                        row["kind"]   = side.value("kind", std::string{});
                        row["status"] = side.value("status", std::string{});
                        if (side.contains("gate")) {
                            row["gate_green"] = side["gate"].value("green", false);
                            row["gate_reasons"] = side["gate"].value("reasons", json::array());
                        }
                    } catch (const std::exception& ex) {
                        row["sidecar_error"] = ex.what();
                    }
                    out.push_back(std::move(row));
                }
            }
            return json{{"dir", fs::absolute(_dir).string()}, {"runes", out}};
        });
}

} // namespace mimirmind::hugin
