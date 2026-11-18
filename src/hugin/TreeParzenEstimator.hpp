// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mimirmind::hugin {

/**
 * Multi-objective Tree-structured Parzen Estimator (roadmap 7.10), own
 * implementation — no Optuna. Univariate TPE: after `startupTrials` random
 * trials, the observations are ranked (feasible before infeasible,
 * non-dominated sorting, then the sum of min-max-normalised objectives) and
 * split into the best `gamma` fraction ("good") and the rest ("bad"). Each
 * parameter draws candidates from the good Parzen density and keeps the one
 * with the highest l(x) / g(x). All objectives are minimised. Deterministic
 * for a given seed. Pure host code, no engine dependency.
 */
class TreeParzenEstimator {
public:
    struct Param {
        enum class Kind { Float, Int, Choice };
        std::string name;
        Kind        kind{Kind::Float};
        double      lo{0};          ///< Float / Int bounds (inclusive)
        double      hi{1};
        bool        log{false};     ///< sample in log space (lo > 0)
        std::size_t choices{0};     ///< Choice: values are indices 0..choices-1
    };

    struct Config {
        std::size_t   startupTrials{8};
        double        gamma{0.25};
        std::size_t   candidates{24};
        std::uint64_t seed{0x7100ULL};
    };

    struct Trial {
        std::vector<double> params;       ///< one value per Param (Int / Choice integral)
        std::vector<double> objectives;   ///< minimised
        bool                feasible{true};
    };

    TreeParzenEstimator(std::vector<Param> space, std::size_t objectives, Config cfg);

    /// Next parameter vector to evaluate.
    [[nodiscard]] std::vector<double> suggest();

    /// Record an evaluated trial (objectives.size() must equal the count).
    void tell(std::vector<double> params, std::vector<double> objectives, bool feasible);

    [[nodiscard]] const std::vector<Trial>& trials() const noexcept { return _trials; }
    [[nodiscard]] const std::vector<Param>& space() const noexcept { return _space; }

    /// Indices of the non-dominated feasible trials (minimisation).
    [[nodiscard]] static std::vector<std::size_t> paretoFront(const std::vector<Trial>& trials);

    /// Trial indices best-first: feasible first, by non-domination rank,
    /// then by the sum of min-max normalised objectives.
    [[nodiscard]] static std::vector<std::size_t> rank(const std::vector<Trial>& trials);

private:
    [[nodiscard]] double uniform01();
    [[nodiscard]] double normal();
    [[nodiscard]] double randomValue(const Param& p);
    [[nodiscard]] double sampleParam(std::size_t pi, const std::vector<std::size_t>& good,
                                     const std::vector<std::size_t>& bad);

    std::vector<Param> _space;
    std::size_t        _objectives;
    Config             _cfg;
    std::uint64_t      _state;
    std::vector<Trial> _trials;
};

} // namespace mimirmind::hugin
