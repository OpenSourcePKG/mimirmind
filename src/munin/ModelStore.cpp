#include "munin/ModelStore.hpp"

#include "core/gguf/TensorFingerprint.hpp"
#include "core/log/Log.hpp"

#include <cstdint>
#include <stdexcept>

namespace mimirmind::munin {

::mimirmind::core::ipc::TensorManifest LoadedModel::buildManifest() const {
    using ::mimirmind::core::ipc::ManifestEntry;
    using ::mimirmind::core::ipc::TensorManifest;

    TensorManifest m{};
    m.modelId          = id;
    m.modelFingerprint = fingerprint;

    const auto& ts = reader->tensors();
    m.tensors.reserve(ts.size());
    for (std::size_t i = 0; i < ts.size(); ++i) {
        const auto& t = ts[i];
        ManifestEntry e{};
        e.name        = t.name;
        e.type        = t.type;
        e.dims        = t.dimensions;
        e.bytes       = static_cast<std::uint64_t>(t.nbytes);
        e.handleIndex = static_cast<std::uint32_t>(i);
        m.tensors.push_back(std::move(e));
    }
    return m;
}

ModelStore::ModelStore(const ::mimirmind::core::config::Config& cfg,
                       ::mimirmind::core::l0::UsmAllocator&    allocator) {
    std::size_t loaded = 0;
    for (const auto& m : cfg.models) {
        if (!m.loadOnStart) {
            continue;
        }
        if (m.id.empty() || m.path.empty()) {
            throw std::runtime_error{
                "ModelStore: model entry with loadOnStart:true has empty id "
                "or path (id='" + m.id + "', path='" + m.path + "')"};
        }
        if (_byId.contains(m.id)) {
            throw std::runtime_error{
                "ModelStore: duplicate model id '" + m.id +
                "' — every loadable entry must have a unique id"};
        }
        MM_LOG_INFO("munin",
                    "ModelStore: loading model id='{}' path='{}'",
                    m.id, m.path);

        auto lm    = std::make_unique<LoadedModel>();
        lm->id     = m.id;
        lm->path   = m.path;
        lm->reader = std::make_unique<::mimirmind::core::gguf::GgufReader>();
        lm->reader->open(m.path);
        lm->reader->loadTensors(allocator);
        lm->weights = std::make_unique<::mimirmind::core::gguf::WeightsMap>(
            *lm->reader);
        lm->totalBytes  = lm->reader->totalTensorBytes();
        lm->fingerprint = ::mimirmind::core::gguf::tensorFingerprint(*lm->reader);

        MM_LOG_INFO("munin",
                    "ModelStore: loaded id='{}' tensors={} bytes={} "
                    "fingerprint='{}'",
                    lm->id, lm->reader->tensorCount(), lm->totalBytes,
                    lm->fingerprint);

        _byId.emplace(m.id, std::move(lm));
        ++loaded;
    }

    if (loaded == 0) {
        throw std::runtime_error{
            "ModelStore: no models with loadOnStart:true in config.json — "
            "Munin has nothing to hold. Set at least one model to "
            "loadOnStart:true or start standalone mimirmind instead."};
    }
    MM_LOG_INFO("munin", "ModelStore: {} model(s) resident in USM", loaded);
}

ModelStore::~ModelStore() {
    // ~LoadedModel → ~GgufReader → close() releases USM through the
    // allocator. Attached workers' IPC pointers become invalid at this
    // instant — that is exactly the shutdown semantics Munin promises.
}

const LoadedModel*
ModelStore::find(std::string_view modelId) const noexcept {
    // unordered_map::find over string_view needs C++20 transparent lookup;
    // to keep the interface `noexcept` regardless of compiler support we
    // materialise a std::string. Cost is a small alloc per healthz/attach,
    // negligible at Munin's request rate.
    const std::string key{modelId};
    const auto it = _byId.find(key);
    if (it == _byId.end()) {
        return nullptr;
    }
    return it->second.get();
}

std::vector<ModelStore::ModelSummary> ModelStore::summaries() const {
    std::vector<ModelSummary> out;
    out.reserve(_byId.size());
    for (const auto& [id, m] : _byId) {
        ModelSummary s{};
        s.id          = id;
        s.fingerprint = m->fingerprint;
        s.totalBytes  = m->totalBytes;
        s.tensorCount = static_cast<std::uint32_t>(m->reader->tensorCount());
        out.push_back(std::move(s));
    }
    return out;
}

} // namespace mimirmind::munin