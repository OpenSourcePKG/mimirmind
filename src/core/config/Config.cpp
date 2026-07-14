#include "core/config/Config.hpp"

#include <algorithm>
#include <fstream>
#include <initializer_list>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace mimirmind::core::config {

using ::mimirmind::runtime::ThermalProfile;

namespace {

[[noreturn]] void fail(std::string_view where, std::string_view msg) {
    std::ostringstream os;
    os << "Config '" << where << "': " << msg;
    throw std::runtime_error(os.str());
}

// Enforce a whitelist of keys at the current JSON object. Any key present
// in `j` but not in `allowed` triggers a fail() — typo protection.
void checkKnownKeys(std::string_view                          path,
                    std::string_view                          section,
                    const nlohmann::json&                     j,
                    std::initializer_list<const char*>        allowed) {
    if (!j.is_object()) {
        return; // callers already validated shape
    }
    const std::unordered_set<std::string> allow(allowed.begin(), allowed.end());
    for (const auto& item : j.items()) {
        if (allow.find(item.key()) == allow.end()) {
            std::ostringstream os;
            os << "unknown field '" << section << "." << item.key() << "' — "
               << "allowed: [";
            bool first = true;
            for (const char* k : allowed) {
                if (!first) os << ", ";
                os << k;
                first = false;
            }
            os << "]";
            fail(path, os.str());
        }
    }
}

// ---- Small typed readers with clear error messages --------------------------

template <typename T>
std::optional<T> readOpt(std::string_view                             path,
                         std::string_view                             section,
                         const nlohmann::json&                        j,
                         const char*                                  key);

template <>
std::optional<bool> readOpt<bool>(std::string_view      path,
                                  std::string_view      section,
                                  const nlohmann::json& j,
                                  const char*           key) {
    if (!j.contains(key) || j[key].is_null()) return std::nullopt;
    if (!j[key].is_boolean()) {
        std::ostringstream os;
        os << section << "." << key << " must be boolean";
        fail(path, os.str());
    }
    return j[key].get<bool>();
}

template <>
std::optional<int> readOpt<int>(std::string_view      path,
                                std::string_view      section,
                                const nlohmann::json& j,
                                const char*           key) {
    if (!j.contains(key) || j[key].is_null()) return std::nullopt;
    if (!j[key].is_number_integer()) {
        std::ostringstream os;
        os << section << "." << key << " must be integer";
        fail(path, os.str());
    }
    return j[key].get<int>();
}

template <>
std::optional<std::size_t> readOpt<std::size_t>(std::string_view      path,
                                                std::string_view      section,
                                                const nlohmann::json& j,
                                                const char*           key) {
    if (!j.contains(key) || j[key].is_null()) return std::nullopt;
    if (!j[key].is_number_unsigned() && !(j[key].is_number_integer() && j[key].get<int>() >= 0)) {
        std::ostringstream os;
        os << section << "." << key << " must be a non-negative integer";
        fail(path, os.str());
    }
    return static_cast<std::size_t>(j[key].get<long long>());
}

template <>
std::optional<float> readOpt<float>(std::string_view      path,
                                    std::string_view      section,
                                    const nlohmann::json& j,
                                    const char*           key) {
    if (!j.contains(key) || j[key].is_null()) return std::nullopt;
    if (!j[key].is_number()) {
        std::ostringstream os;
        os << section << "." << key << " must be a number";
        fail(path, os.str());
    }
    return j[key].get<float>();
}

template <>
std::optional<std::string> readOpt<std::string>(std::string_view      path,
                                                std::string_view      section,
                                                const nlohmann::json& j,
                                                const char*           key) {
    if (!j.contains(key) || j[key].is_null()) return std::nullopt;
    if (!j[key].is_string()) {
        std::ostringstream os;
        os << section << "." << key << " must be string";
        fail(path, os.str());
    }
    return j[key].get<std::string>();
}

TriState parseTriState(std::string_view      path,
                       std::string_view      section,
                       const nlohmann::json& j,
                       const char*           key,
                       TriState              defaultValue) {
    if (!j.contains(key) || j[key].is_null()) return defaultValue;
    if (!j[key].is_string()) {
        std::ostringstream os;
        os << section << "." << key << " must be one of \"auto\" | \"force\" | \"disable\"";
        fail(path, os.str());
    }
    const auto s = j[key].get<std::string>();
    if (s == "auto")    return TriState::Auto;
    if (s == "force")   return TriState::Force;
    if (s == "disable") return TriState::Disable;
    std::ostringstream os;
    os << section << "." << key << "='" << s << "' — expected \"auto\" | \"force\" | \"disable\"";
    fail(path, os.str());
}

// ---- Section parsers --------------------------------------------------------

RuntimeSettings parseRuntime(std::string_view      path,
                             std::string_view      section,
                             const nlohmann::json& j) {
    checkKnownKeys(path, section, j,
                   {"kvDtype", "maxContextTokens", "usmProbeTotalGib",
                    "preserveThinking", "spvDir"});
    RuntimeSettings r{};
    r.kvDtype           = readOpt<std::string>(path, section, j, "kvDtype");
    r.maxContextTokens  = readOpt<std::size_t>(path, section, j, "maxContextTokens");
    r.usmProbeTotalGib  = readOpt<int>(path, section, j, "usmProbeTotalGib");
    r.preserveThinking  = readOpt<bool>(path, section, j, "preserveThinking");
    r.spvDir            = readOpt<std::string>(path, section, j, "spvDir");
    if (r.kvDtype.has_value()) {
        const auto& v = *r.kvDtype;
        if (!v.empty() && v != "f32" && v != "q8_0") {
            std::ostringstream os;
            os << section << ".kvDtype='" << v
               << "' — expected \"\", \"f32\", or \"q8_0\"";
            fail(path, os.str());
        }
    }
    return r;
}

ModelEntry parseModel(std::string_view      path,
                      const nlohmann::json& j,
                      std::size_t           index) {
    std::ostringstream sec;
    sec << "models[" << index << "]";
    const std::string section = sec.str();
    if (!j.is_object()) {
        fail(path, section + " must be an object");
    }
    checkKnownKeys(path, section, j, {"id", "title", "path", "loadOnStart", "runtime"});

    ModelEntry m{};
    if (!j.contains("id") || !j["id"].is_string() || j["id"].get<std::string>().empty()) {
        fail(path, section + ".id is required (non-empty string)");
    }
    m.id = j["id"].get<std::string>();
    if (const auto v = readOpt<std::string>(path, section, j, "title"); v.has_value()) {
        m.title = *v;
    }
    if (!j.contains("path") || !j["path"].is_string() || j["path"].get<std::string>().empty()) {
        fail(path, section + ".path is required (non-empty string)");
    }
    m.path = j["path"].get<std::string>();
    if (const auto v = readOpt<bool>(path, section, j, "loadOnStart"); v.has_value()) {
        m.loadOnStart = *v;
    }
    if (j.contains("runtime")) {
        m.runtime = parseRuntime(path, section + ".runtime", j["runtime"]);
    }
    return m;
}

ServerSettings parseServer(std::string_view      path,
                           const nlohmann::json& j) {
    checkKnownKeys(path, "server", j, {"port", "log"});
    ServerSettings s{};
    if (const auto v = readOpt<int>(path, "server", j, "port"); v.has_value()) {
        if (*v <= 0 || *v > 65535) {
            fail(path, "server.port must be in 1..65535");
        }
        s.port = *v;
    }
    if (j.contains("log")) {
        const auto& lj = j["log"];
        if (!lj.is_object()) fail(path, "server.log must be an object");
        checkKnownKeys(path, "server.log", lj, {"file", "level"});
        if (const auto v = readOpt<std::string>(path, "server.log", lj, "file"); v.has_value()) {
            s.log.file = *v;
        }
        if (const auto v = readOpt<std::string>(path, "server.log", lj, "level"); v.has_value()) {
            s.log.level = *v;
        }
    }
    return s;
}

FeatureSettings parseFeatures(std::string_view      path,
                              const nlohmann::json& j) {
    checkKnownKeys(path, "features", j,
                   {"clr", "flashPrefill", "flashPrefillGqaQ8",
                    "flashPrefillKTileQ8",
                    "fusedQkv", "moeGroup",
                    "gemm", "gemmV2", "gemmMinM", "dp4a",
                    "q8_0Reorder", "moeFusedDown"});
    FeatureSettings f{};
    if (const auto v = readOpt<bool>(path, "features", j, "clr");               v) f.clr               = *v;
    if (const auto v = readOpt<bool>(path, "features", j, "flashPrefill");      v) f.flashPrefill      = *v;
    if (const auto v = readOpt<bool>(path, "features", j, "flashPrefillGqaQ8"); v) f.flashPrefillGqaQ8 = *v;
    if (const auto v = readOpt<std::size_t>(path, "features", j, "flashPrefillKTileQ8"); v) {
        if (*v != 0 && *v != 64 && *v != 128) {
            fail(path,
                 "features.flashPrefillKTileQ8 must be 0 (autotune), 64, "
                 "or 128 (got " + std::to_string(*v) + ")");
        }
        f.flashPrefillKTileQ8 = *v;
    }
    if (const auto v = readOpt<bool>(path, "features", j, "fusedQkv");          v) f.fusedQkv          = *v;
    if (const auto v = readOpt<bool>(path, "features", j, "moeGroup");     v) f.moeGroup     = *v;
    if (const auto v = readOpt<bool>(path, "features", j, "gemmV2");       v) f.gemmV2       = *v;
    f.gemm     = parseTriState(path, "features", j, "gemm",  TriState::Auto);
    f.dp4a     = parseTriState(path, "features", j, "dp4a",  TriState::Disable);
    f.q8_0Reorder = parseTriState(path, "features", j, "q8_0Reorder",
                                  TriState::Disable);
    f.moeFusedDown = parseTriState(path, "features", j, "moeFusedDown",
                                   TriState::Disable);
    f.gemmMinM = readOpt<std::size_t>(path, "features", j, "gemmMinM");
    return f;
}

SpeculativeSettings parseSpeculative(std::string_view      path,
                                     const nlohmann::json& j) {
    checkKnownKeys(path, "speculative", j, {"enabled", "target", "draft", "n"});
    SpeculativeSettings s{};
    if (const auto v = readOpt<bool>(path, "speculative", j, "enabled"); v) s.enabled = *v;
    if (const auto v = readOpt<std::string>(path, "speculative", j, "target"); v) s.target = *v;
    if (const auto v = readOpt<std::string>(path, "speculative", j, "draft");  v) s.draft  = *v;
    if (const auto v = readOpt<int>(path, "speculative", j, "n"); v) {
        if (*v < 1 || *v > 32) fail(path, "speculative.n must be in 1..32");
        s.n = *v;
    }
    return s;
}

ThermalProfile parseThermalInline(std::string_view      path,
                                  const nlohmann::json& j) {
    // Same schema as the standalone thermal-profile files, inlined.
    checkKnownKeys(path, "governor.thermal", j,
                   {"name", "description", "package_temp_soft_c",
                    "package_temp_hard_c", "package_throttle_max_ms",
                    "gpu_target_temp_c"});
    ThermalProfile t{};
    if (const auto v = readOpt<std::string>(path, "governor.thermal", j, "name"); v)
        t.name = *v;
    if (const auto v = readOpt<std::string>(path, "governor.thermal", j, "description"); v)
        t.description = *v;
    t.package_temp_soft_c = readOpt<float>(path, "governor.thermal", j, "package_temp_soft_c");
    t.package_temp_hard_c = readOpt<float>(path, "governor.thermal", j, "package_temp_hard_c");
    if (t.package_temp_soft_c.has_value() != t.package_temp_hard_c.has_value()) {
        fail(path,
             "governor.thermal: package_temp_soft_c and package_temp_hard_c must be "
             "set together (both or neither)");
    }
    if (t.hasPackageLimits() && !(*t.package_temp_soft_c < *t.package_temp_hard_c)) {
        std::ostringstream os;
        os << "governor.thermal: package_temp_soft_c (" << *t.package_temp_soft_c
           << ") must be strictly less than package_temp_hard_c ("
           << *t.package_temp_hard_c << ")";
        fail(path, os.str());
    }
    if (const auto v = readOpt<int>(path, "governor.thermal", j, "package_throttle_max_ms"); v) {
        if (*v < 0) fail(path, "governor.thermal.package_throttle_max_ms must be >= 0");
        t.package_throttle_max_ms = *v;
    }
    t.gpu_target_temp_c = readOpt<float>(path, "governor.thermal", j, "gpu_target_temp_c");
    if (t.gpu_target_temp_c.has_value() && t.hasPackageLimits() &&
        *t.gpu_target_temp_c >= *t.package_temp_soft_c) {
        std::ostringstream os;
        os << "governor.thermal.gpu_target_temp_c (" << *t.gpu_target_temp_c
           << ") must be strictly less than package_temp_soft_c ("
           << *t.package_temp_soft_c << ")";
        fail(path, os.str());
    }
    return t;
}

GovernorSettings parseGovernor(std::string_view      path,
                               const nlohmann::json& j) {
    checkKnownKeys(path, "governor", j,
                   {"gpuClockPin", "tickLog", "tickLogFile", "fan", "thermal"});
    GovernorSettings g{};
    g.gpuClockPin = readOpt<std::string>(path, "governor", j, "gpuClockPin");
    if (const auto v = readOpt<bool>(path, "governor", j, "tickLog"); v) g.tickLog = *v;
    if (const auto v = readOpt<std::string>(path, "governor", j, "tickLogFile"); v)
        g.tickLogFile = *v;

    if (j.contains("fan")) {
        const auto& fj = j["fan"];
        if (!fj.is_object()) fail(path, "governor.fan must be an object");
        checkKnownKeys(path, "governor.fan", fj, {"boost", "pwmBoost", "pwmMin"});
        if (const auto v = readOpt<bool>(path, "governor.fan", fj, "boost"); v)
            g.fan.boost = *v;
        g.fan.pwmBoost = readOpt<int>(path, "governor.fan", fj, "pwmBoost");
        g.fan.pwmMin   = readOpt<int>(path, "governor.fan", fj, "pwmMin");
        auto validatePwm = [&](const char* which, const std::optional<int>& v) {
            if (!v.has_value()) return;
            if (*v < 0 || *v > 255) {
                std::ostringstream os;
                os << "governor.fan." << which << " must be in 0..255";
                fail(path, os.str());
            }
        };
        validatePwm("pwmBoost", g.fan.pwmBoost);
        validatePwm("pwmMin",   g.fan.pwmMin);
    }
    if (j.contains("thermal") && !j["thermal"].is_null()) {
        if (!j["thermal"].is_object()) fail(path, "governor.thermal must be an object");
        g.thermal = parseThermalInline(path, j["thermal"]);
    }
    return g;
}

DiagnosticsSettings parseDiagnostics(std::string_view      path,
                                     const nlohmann::json& j) {
    checkKnownKeys(path, "diagnostics", j,
                   {"parityDump", "traceBlock0", "traceDecodeFile",
                    "traceOpTimes", "gpuBench", "regressionAlert"});
    DiagnosticsSettings d{};
    if (const auto v = readOpt<std::string>(path, "diagnostics", j, "parityDump"); v)
        d.parityDump = *v;
    if (const auto v = readOpt<bool>(path, "diagnostics", j, "traceBlock0"); v)
        d.traceBlock0 = *v;
    if (const auto v = readOpt<std::string>(path, "diagnostics", j, "traceDecodeFile"); v)
        d.traceDecodeFile = *v;
    if (const auto v = readOpt<bool>(path, "diagnostics", j, "traceOpTimes"); v)
        d.traceOpTimes = *v;
    if (const auto v = readOpt<bool>(path, "diagnostics", j, "gpuBench"); v)
        d.gpuBench = *v;
    if (const auto v = readOpt<bool>(path, "diagnostics", j, "regressionAlert"); v)
        d.regressionAlert = *v;
    return d;
}

} // namespace

// ---- merge / accessors ------------------------------------------------------

RuntimeSettings mergeRuntime(const RuntimeSettings& base,
                             const RuntimeSettings& override_) {
    RuntimeSettings r = base;
    if (override_.kvDtype.has_value())          r.kvDtype          = override_.kvDtype;
    if (override_.maxContextTokens.has_value()) r.maxContextTokens = override_.maxContextTokens;
    if (override_.usmProbeTotalGib.has_value()) r.usmProbeTotalGib = override_.usmProbeTotalGib;
    if (override_.preserveThinking.has_value()) r.preserveThinking = override_.preserveThinking;
    if (override_.spvDir.has_value())           r.spvDir           = override_.spvDir;
    return r;
}

RuntimeSettings Config::effectiveRuntime(std::string_view modelId) const {
    for (const auto& m : models) {
        if (m.id == modelId) {
            return mergeRuntime(runtime, m.runtime);
        }
    }
    std::ostringstream os;
    os << "effectiveRuntime: no model with id='" << modelId << "'";
    throw std::runtime_error(os.str());
}

const ModelEntry& Config::model(std::string_view id) const {
    for (const auto& m : models) {
        if (m.id == id) return m;
    }
    std::ostringstream os;
    os << "Config::model: no model with id='" << id << "'";
    throw std::runtime_error(os.str());
}

const ModelEntry& Config::defaultModelEntry() const {
    if (models.empty()) {
        throw std::runtime_error("Config::defaultModelEntry: `models` is empty");
    }
    if (!defaultModel.empty()) {
        return model(defaultModel);
    }
    return models.front();
}

// ---- loader -----------------------------------------------------------------

Config loadConfig(std::string_view path) {
    std::ifstream in{std::string{path}};
    if (!in.is_open()) {
        std::ostringstream os;
        os << "cannot open file '" << path << "' — copy config.example.json to "
              "config.json and edit for your host";
        throw std::runtime_error(os.str());
    }

    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        std::ostringstream os;
        os << "Config '" << path << "': invalid JSON: " << e.what();
        throw std::runtime_error(os.str());
    }
    if (!j.is_object()) {
        fail(path, "top-level JSON must be an object");
    }

    checkKnownKeys(path, "<root>", j,
                   {"defaultModel", "models", "server", "runtime", "features",
                    "speculative", "governor", "diagnostics"});

    Config cfg{};

    if (const auto v = readOpt<std::string>(path, "<root>", j, "defaultModel"); v) {
        cfg.defaultModel = *v;
    }

    if (!j.contains("models") || !j["models"].is_array()) {
        fail(path, "'models' is required and must be an array");
    }
    const auto& mj = j["models"];
    if (mj.empty()) {
        fail(path, "'models' must contain at least one entry");
    }
    cfg.models.reserve(mj.size());
    for (std::size_t i = 0; i < mj.size(); ++i) {
        cfg.models.push_back(parseModel(path, mj[i], i));
    }
    // No duplicate ids.
    for (std::size_t i = 0; i < cfg.models.size(); ++i) {
        for (std::size_t k = i + 1; k < cfg.models.size(); ++k) {
            if (cfg.models[i].id == cfg.models[k].id) {
                std::ostringstream os;
                os << "duplicate model id '" << cfg.models[i].id
                   << "' at models[" << i << "] and models[" << k << "]";
                fail(path, os.str());
            }
        }
    }
    // Multi-load is now wired: each loadOnStart:true entry gets its own
    // InferenceEngine and chat/completions dispatches on request.model.
    // Duplicate ids already blocked above.
    const auto loadOnStartCount = std::count_if(
        cfg.models.begin(), cfg.models.end(),
        [](const ModelEntry& m) { return m.loadOnStart; });
    if (loadOnStartCount == 0) {
        fail(path,
             "no model has loadOnStart:true — set at least one entry to "
             "loadOnStart:true so the server has something to serve");
    }
    // Default model must resolve.
    if (!cfg.defaultModel.empty()) {
        const auto exists = std::any_of(
            cfg.models.begin(), cfg.models.end(),
            [&](const ModelEntry& m) { return m.id == cfg.defaultModel; });
        if (!exists) {
            std::ostringstream os;
            os << "defaultModel='" << cfg.defaultModel
               << "' does not match any models[].id";
            fail(path, os.str());
        }
    }

    if (j.contains("server"))       cfg.server       = parseServer(path, j["server"]);
    if (j.contains("runtime"))      cfg.runtime      = parseRuntime(path, "runtime", j["runtime"]);
    if (j.contains("features"))     cfg.features     = parseFeatures(path, j["features"]);
    if (j.contains("speculative"))  cfg.speculative  = parseSpeculative(path, j["speculative"]);
    if (j.contains("governor"))     cfg.governor     = parseGovernor(path, j["governor"]);
    if (j.contains("diagnostics"))  cfg.diagnostics  = parseDiagnostics(path, j["diagnostics"]);

    // Cross-section validation: speculative model ids must resolve.
    if (cfg.speculative.enabled) {
        auto exists = [&](const std::string& id) {
            return std::any_of(cfg.models.begin(), cfg.models.end(),
                               [&](const ModelEntry& m) { return m.id == id; });
        };
        if (cfg.speculative.target.empty()) {
            fail(path, "speculative.enabled=true requires speculative.target (model id)");
        }
        if (!exists(cfg.speculative.target)) {
            std::ostringstream os;
            os << "speculative.target='" << cfg.speculative.target
               << "' does not match any models[].id";
            fail(path, os.str());
        }
        if (cfg.speculative.draft.empty()) {
            fail(path, "speculative.enabled=true requires speculative.draft (model id)");
        }
        if (!exists(cfg.speculative.draft)) {
            std::ostringstream os;
            os << "speculative.draft='" << cfg.speculative.draft
               << "' does not match any models[].id";
            fail(path, os.str());
        }
        if (cfg.speculative.target == cfg.speculative.draft) {
            fail(path, "speculative.target and speculative.draft must differ");
        }
    }

    return cfg;
}

void applyCliOverrides(Config& cfg, const CliOverrides& cli) {
    if (cli.modelPath.has_value()) {
        if (cfg.models.empty()) {
            ModelEntry m{};
            m.id   = "primary";
            m.path = *cli.modelPath;
            cfg.models.push_back(std::move(m));
        } else {
            const std::string targetId =
                cfg.defaultModel.empty() ? cfg.models.front().id : cfg.defaultModel;
            bool applied = false;
            for (auto& m : cfg.models) {
                if (m.id == targetId) {
                    m.path  = *cli.modelPath;
                    applied = true;
                    break;
                }
            }
            if (!applied) {
                std::ostringstream os;
                os << "applyCliOverrides: default model id '" << targetId
                   << "' not found in models[]";
                throw std::runtime_error(os.str());
            }
        }
    }
    if (cli.port.has_value()) {
        if (*cli.port <= 0 || *cli.port > 65535) {
            throw std::runtime_error("--port must be in 1..65535");
        }
        cfg.server.port = *cli.port;
    }
    if (cli.logFile.has_value())  cfg.server.log.file  = *cli.logFile;
    if (cli.logLevel.has_value()) cfg.server.log.level = *cli.logLevel;
}

} // namespace mimirmind::core::config