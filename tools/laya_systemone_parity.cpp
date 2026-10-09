// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// laya_systemone_parity — end-to-end Laya typed decision (tokenizer +
// build_sequence + backbone + head + decode) via LayaSystemOne, for parity
// against the python Agent.predict (roadmap 8.27.4). Box-only (CUDA).
//
// Usage: laya_systemone_parity <model_dir> [state_file]
// Hardcodes the 4-way Pegenaut mailbox-routing question (support/vertrieb/
// buchhaltung/spam) to match the python reference.

#include "compute/cuda/GpuMatmul.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "runtime/encoder/laya/LayaSystemOne.hpp"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    using mimirmind::runtime::encoder::laya::LayaSystemOne;
    if (argc < 2) {
        std::fprintf(stderr, "usage: laya_systemone_parity <model_dir> [state_file]\n");
        return 2;
    }
    std::string state =
        "Ich habe eine Frage zur Rechnung 2026-0931, bitte um eine korrigierte Rechnung.";
    if (argc >= 3) {
        std::ifstream in(argv[2]);
        if (in) {
            std::ostringstream ss;
            ss << in.rdbuf();
            state = ss.str();
            while (!state.empty() && (state.back() == '\n' || state.back() == '\r')) state.pop_back();
        }
    }

    try {
        mimirmind::core::cuda::CudaComputeContext ctx{};
        mimirmind::compute::cuda::GpuOps          ops{ctx};
        mimirmind::compute::cuda::GpuMatmul       mm{ctx, ops};

        LayaSystemOne so{ops, mm};
        so.load(argv[1], mimirmind::core::gguf::GgmlType::F32, /*nHeadLayers=*/2);

        LayaSystemOne::Question q;
        q.type = LayaSystemOne::QType::Choice;
        q.instructions = "Ordne diese E-Mail genau einem Postfach zu.";
        q.choice = {
            {"support", "Technische Hilfe, Stoerung, Fehler, Reklamation an einem gekauften Produkt."},
            {"vertrieb", "Kaufinteresse, Angebot, Preis, Produktberatung vor dem Kauf."},
            {"buchhaltung", "Rechnungen, Mahnungen, Zahlungen, Gutschriften, Steuer, Bankdaten."},
            {"spam", "Werbung, Phishing, Betrug, Massenmail."},
        };

        const std::string arg2 = (argc >= 3) ? std::string(argv[2]) : std::string();
        const bool goldset = arg2.size() > 6 && arg2.substr(arg2.size() - 6) == ".jsonl";
        if (goldset) {
            // Goldset mode: each line {"text","gold"}; report accuracy + confusion.
            std::ifstream in(arg2);
            std::string line;
            int n = 0, correct = 0;
            std::map<std::pair<std::string, std::string>, int> conf;
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                const auto j = nlohmann::json::parse(line);
                const std::string text = j.at("text").get<std::string>();
                const std::string gold = j.value("gold", std::string{});
                const auto a = so.decide(text, q);
                ++n;
                if (a.choice == gold) ++correct;
                conf[{gold, a.choice}]++;
            }
            std::printf("accuracy %.3f (%d/%d)\n",
                        n ? static_cast<double>(correct) / n : 0.0, correct, n);
            std::printf("confusion (gold -> pred):\n");
            for (const auto& [kv, c] : conf) {
                std::printf("  %-12s -> %-12s : %d\n", kv.first.c_str(), kv.second.c_str(), c);
            }
        } else {
            const auto a = so.decide(state, q);
            std::printf("choice: %s\n", a.choice.c_str());
            std::printf("probabilities:");
            for (const auto& [key, p] : a.probabilities) std::printf(" %s=%.4f", key.c_str(), p);
            std::printf("\nconfidence: %.4f\nanswer_confidence: %.4f\nact_probability: %.4f\n",
                        a.confidence, a.answerConfidence, a.actProbability);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "laya_systemone_parity: %s\n", e.what());
        return 1;
    }
    return 0;
}
