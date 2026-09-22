// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/DecisionHead.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::encoder {

namespace {

std::string readTextFile(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        throw std::runtime_error("DecisionHead: cannot read " + p.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Read a little-endian float32 array of exactly `expected` elements. The target
// platforms (x86-64, Grace ARM, RDNA3) are all little-endian, matching the
// trainer's `.astype('<f4').tofile()` output byte-for-byte.
std::vector<float> readF32Exact(const std::filesystem::path& p, std::size_t expected) {
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("DecisionHead: cannot read " + p.string());
    }
    const std::streamoff bytes = in.tellg();
    if (bytes < 0 || static_cast<std::size_t>(bytes) != expected * sizeof(float)) {
        throw std::runtime_error(
            "DecisionHead: '" + p.string() + "' has " + std::to_string(bytes) +
            " bytes, expected " + std::to_string(expected * sizeof(float)) +
            " (" + std::to_string(expected) + " float32)");
    }
    std::vector<float> out(expected);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()),
            static_cast<std::streamsize>(expected * sizeof(float)));
    if (!in) {
        throw std::runtime_error("DecisionHead: short read on " + p.string());
    }
    return out;
}

// Write a float32 array little-endian (matches readF32Exact + the trainer's
// `.astype('<f4').tofile()`).
void writeF32(const std::filesystem::path& p, const std::vector<float>& v) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("DecisionHead: cannot write " + p.string());
    }
    out.write(reinterpret_cast<const char*>(v.data()),
              static_cast<std::streamsize>(v.size() * sizeof(float)));
    if (!out) {
        throw std::runtime_error("DecisionHead: short write on " + p.string());
    }
}

} // namespace

DecisionHead DecisionHead::loadFromDir(const std::filesystem::path& dir) {
    const nlohmann::json j = nlohmann::json::parse(
        readTextFile(dir / "head.json"), nullptr,
        /*allow_exceptions=*/true, /*ignore_comments=*/true);

    DecisionHead h{};
    h._name = j.at("name").get<std::string>();
    if (h._name.empty()) {
        throw std::runtime_error("DecisionHead: 'name' must be non-empty in " +
                                 (dir / "head.json").string());
    }
    h._labels = j.at("labels").get<std::vector<std::string>>();
    if (h._labels.size() < 2) {
        throw std::runtime_error(
            "DecisionHead '" + h._name + "': needs >= 2 labels, got " +
            std::to_string(h._labels.size()));
    }
    h._hidden      = j.at("hidden").get<std::size_t>();
    h._temperature = j.value("temperature", 1.0F);
    h._threshold   = j.value("threshold", 0.0F);
    if (h._hidden == 0) {
        throw std::runtime_error("DecisionHead '" + h._name + "': hidden must be > 0");
    }
    if (!(h._temperature > 0.0F)) {
        throw std::runtime_error("DecisionHead '" + h._name +
                                 "': temperature must be > 0");
    }

    const std::size_t k = h._labels.size();
    h._weight = readF32Exact(dir / "weight.f32", k * h._hidden);
    h._bias   = readF32Exact(dir / "bias.f32", k);
    return h;
}

void DecisionHead::writeToDir(const std::filesystem::path& dir, const Spec& spec) {
    if (spec.name.empty()) {
        throw std::runtime_error("DecisionHead::writeToDir: 'name' must be non-empty");
    }
    if (spec.labels.size() < 2) {
        throw std::runtime_error("DecisionHead::writeToDir '" + spec.name +
                                 "': needs >= 2 labels");
    }
    if (spec.hidden == 0) {
        throw std::runtime_error("DecisionHead::writeToDir '" + spec.name +
                                 "': hidden must be > 0");
    }
    if (!(spec.temperature > 0.0F)) {
        throw std::runtime_error("DecisionHead::writeToDir '" + spec.name +
                                 "': temperature must be > 0");
    }
    const std::size_t k = spec.labels.size();
    if (spec.weight.size() != k * spec.hidden) {
        throw std::runtime_error(
            "DecisionHead::writeToDir '" + spec.name + "': weight has " +
            std::to_string(spec.weight.size()) + " floats, expected " +
            std::to_string(k * spec.hidden) + " (labels*hidden)");
    }
    if (spec.bias.size() != k) {
        throw std::runtime_error(
            "DecisionHead::writeToDir '" + spec.name + "': bias has " +
            std::to_string(spec.bias.size()) + " floats, expected " +
            std::to_string(k));
    }

    std::filesystem::create_directories(dir);

    nlohmann::json j;
    j["name"]        = spec.name;
    j["labels"]      = spec.labels;
    j["hidden"]      = spec.hidden;
    j["temperature"] = spec.temperature;
    j["threshold"]   = spec.threshold;
    if (!spec.encoder.empty()) {
        j["encoder"] = spec.encoder;
    }
    {
        std::ofstream out(dir / "head.json", std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("DecisionHead: cannot write " +
                                     (dir / "head.json").string());
        }
        out << j.dump(2);
    }
    writeF32(dir / "weight.f32", spec.weight);
    writeF32(dir / "bias.f32", spec.bias);
}

DecisionHead::Result DecisionHead::decide(std::span<const float> emb) const {
    if (emb.size() != _hidden) {
        throw std::invalid_argument(
            "DecisionHead '" + _name + "': embedding dim " +
            std::to_string(emb.size()) + " != head hidden " +
            std::to_string(_hidden));
    }
    const std::size_t k = _labels.size();

    // logits = (W·e + b) / temperature, computed in F64 for a stable,
    // Python-parity-clean dot product over 1024 terms.
    std::vector<float> logits(k);
    for (std::size_t c = 0; c < k; ++c) {
        const float* row = _weight.data() + c * _hidden;
        double acc = static_cast<double>(_bias[c]);
        for (std::size_t d = 0; d < _hidden; ++d) {
            acc += static_cast<double>(row[d]) * static_cast<double>(emb[d]);
        }
        logits[c] = static_cast<float>(acc / static_cast<double>(_temperature));
    }

    // Numerically stable softmax (subtract max).
    float maxLogit = logits[0];
    std::size_t argmax = 0;
    for (std::size_t c = 1; c < k; ++c) {
        if (logits[c] > maxLogit) {
            maxLogit = logits[c];
            argmax   = c;
        }
    }
    double sum = 0.0;
    std::vector<double> exps(k);
    for (std::size_t c = 0; c < k; ++c) {
        exps[c] = std::exp(static_cast<double>(logits[c] - maxLogit));
        sum += exps[c];
    }

    Result r{};
    r.dist.reserve(k);
    for (std::size_t c = 0; c < k; ++c) {
        r.dist.push_back({_labels[c], static_cast<float>(exps[c] / sum)});
    }
    r.choice    = _labels[argmax];
    r.p         = r.dist[argmax].p;
    r.confident = r.p >= _threshold;
    return r;
}

} // namespace mimirmind::runtime::encoder
