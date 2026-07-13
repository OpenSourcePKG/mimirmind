#include "munin/ModelStore.hpp"

#include "core/log/Log.hpp"

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace mimirmind::munin {

namespace {

// FNV-1a 64-bit over a sequence of bytes. Deterministic, no allocation,
// good-enough distribution to distinguish two GGUFs that differ by any
// tensor's metadata. Not a cryptographic hash — the fingerprint is an
// identity check, not a security boundary.
constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t kFnvPrime  = 0x100000001b3ULL;

inline void fnvUpdate(std::uint64_t& h, const void* data, std::size_t n) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= static_cast<std::uint64_t>(p[i]);
        h *= kFnvPrime;
    }
}

std::string toHex64(std::uint64_t v) {
    std::ostringstream os;
    os << std::hex << std::setw(16) << std::setfill('0') << v;
    return os.str();
}

} // namespace

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

std::string ModelStore::computeFingerprint(
        const ::mimirmind::core::gguf::GgufReader& reader) {
    std::uint64_t h = kFnvOffset;

    // Version + alignment + tensor-data-offset — cheap identity anchor
    // that catches header rewrites.
    const std::uint32_t v = reader.version();
    fnvUpdate(h, &v, sizeof(v));
    const std::uint64_t a = reader.alignment();
    fnvUpdate(h, &a, sizeof(a));
    const std::uint64_t off = reader.tensorDataOffset();
    fnvUpdate(h, &off, sizeof(off));
    const std::uint64_t total = reader.totalTensorBytes();
    fnvUpdate(h, &total, sizeof(total));

    // Metadata count is a signal but not the values — full metadata hash
    // would be another walk over strings/arrays we don't need.
    const std::uint64_t mdCount = reader.metadataCount();
    fnvUpdate(h, &mdCount, sizeof(mdCount));

    // Every tensor's name + type + dims + bytes + file-offset. That is the
    // "shape" of the model as far as attach semantics care — two GGUFs
    // that differ in any tensor's dtype or dims produce different
    // fingerprints, and a worker configured for one refuses the other.
    const auto& ts = reader.tensors();
    const std::uint64_t nT = ts.size();
    fnvUpdate(h, &nT, sizeof(nT));
    for (const auto& t : ts) {
        fnvUpdate(h, t.name.data(), t.name.size());
        const std::uint32_t ty = static_cast<std::uint32_t>(t.type);
        fnvUpdate(h, &ty, sizeof(ty));
        const std::uint64_t nd = t.dimensions.size();
        fnvUpdate(h, &nd, sizeof(nd));
        if (!t.dimensions.empty()) {
            fnvUpdate(h, t.dimensions.data(),
                      t.dimensions.size() * sizeof(std::uint64_t));
        }
        const std::uint64_t nb = t.nbytes;
        fnvUpdate(h, &nb, sizeof(nb));
        fnvUpdate(h, &t.fileOffset, sizeof(t.fileOffset));
    }

    std::ostringstream os;
    os << ts.size() << "." << reader.totalTensorBytes() << "." << toHex64(h);
    return os.str();
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
        lm->fingerprint = computeFingerprint(*lm->reader);

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