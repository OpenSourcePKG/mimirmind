// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gguf/GgufReader.hpp"

#include "core/log/Log.hpp"
#include "core/l0/UsmAllocator.hpp"
#include "munin/ChunkAllocator.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace mimirmind::core::gguf {

namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747u;   // 'G','G','U','F' LE
constexpr std::uint32_t kSupportedVersion = 3;

double bytesToMiB(std::size_t b) noexcept {
    return static_cast<double>(b) / (1024.0 * 1024.0);
}

class Cursor {
public:
    explicit Cursor(std::span<const std::uint8_t> bytes) noexcept
        : _bytes{bytes}, _pos{0} {}

    template <typename T>
    T read() {
        static_assert(std::is_trivially_copyable_v<T>);
        require(sizeof(T));
        T v;
        std::memcpy(&v, _bytes.data() + _pos, sizeof(T));
        _pos += sizeof(T);
        return v;
    }

    std::string readString() {
        const auto n = read<std::uint64_t>();
        if (n > _bytes.size() || _pos + n > _bytes.size()) {
            throw std::runtime_error(
                "gguf: truncated string (n=" + std::to_string(n) +
                ", pos=" + std::to_string(_pos) +
                ", file_size=" + std::to_string(_bytes.size()) + ")");
        }
        std::string s{reinterpret_cast<const char*>(_bytes.data() + _pos),
                      static_cast<std::size_t>(n)};
        _pos += static_cast<std::size_t>(n);
        return s;
    }

    void readBytes(void* dst, std::size_t n) {
        require(n);
        std::memcpy(dst, _bytes.data() + _pos, n);
        _pos += n;
    }

    [[nodiscard]] std::size_t pos() const noexcept { return _pos; }

private:
    void require(std::size_t n) const {
        if (_pos + n > _bytes.size()) {
            throw std::runtime_error(
                "gguf: truncated read (need " + std::to_string(n) +
                " bytes at pos " + std::to_string(_pos) +
                ", file_size=" + std::to_string(_bytes.size()) + ")");
        }
    }

    std::span<const std::uint8_t> _bytes;
    std::size_t                   _pos;
};

MetadataValue readValue(Cursor& c, GgufValueType type);

GgufArray readArray(Cursor& c) {
    const auto elemType = static_cast<GgufValueType>(c.read<std::uint32_t>());
    const auto count    = c.read<std::uint64_t>();

    GgufArray arr;
    arr.elementType = elemType;
    arr.count       = count;

    if (elemType == GgufValueType::String) {
        arr.strings.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            arr.strings.push_back(c.readString());
        }
        return arr;
    }
    if (elemType == GgufValueType::Array) {
        throw std::runtime_error("gguf: nested arrays not supported by v3");
    }

    const std::size_t width = valueElementWidth(elemType);
    if (width == 0) {
        throw std::runtime_error(
            "gguf: unknown array element type tag=" +
            std::to_string(static_cast<std::uint32_t>(elemType)));
    }
    const std::size_t bytes = static_cast<std::size_t>(count) * width;
    arr.raw.resize(bytes);
    if (bytes > 0) {
        c.readBytes(arr.raw.data(), bytes);
    }
    return arr;
}

MetadataValue readValue(Cursor& c, GgufValueType type) {
    using VT = GgufValueType;
    switch (type) {
        case VT::UInt8:   return c.read<std::uint8_t>();
        case VT::Int8:    return c.read<std::int8_t>();
        case VT::UInt16:  return c.read<std::uint16_t>();
        case VT::Int16:   return c.read<std::int16_t>();
        case VT::UInt32:  return c.read<std::uint32_t>();
        case VT::Int32:   return c.read<std::int32_t>();
        case VT::UInt64:  return c.read<std::uint64_t>();
        case VT::Int64:   return c.read<std::int64_t>();
        case VT::Float32: return c.read<float>();
        case VT::Float64: return c.read<double>();
        case VT::Bool:    return c.read<std::uint8_t>() != 0;
        case VT::String:  return c.readString();
        case VT::Array:   return readArray(c);
    }
    throw std::runtime_error("gguf: unknown value type tag=" +
                             std::to_string(static_cast<std::uint32_t>(type)));
}

std::string formatDims(const std::vector<std::uint64_t>& dims) {
    std::string s = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            s += ",";
        }
        s += std::to_string(dims[i]);
    }
    s += "]";
    return s;
}

} // namespace

GgufReader::~GgufReader() {
    close();
}

void GgufReader::open(std::string_view path) {
    close();

    _file = core::l0::MappedFile{path};

    Cursor c{_file.bytes()};

    const auto magic = c.read<std::uint32_t>();
    if (magic != kGgufMagic) {
        MM_LOG_ERROR("gguf",
                     "bad magic: got 0x{:08x}, expected 0x{:08x} ('GGUF')",
                     magic, kGgufMagic);
        throw std::runtime_error("GgufReader: bad magic for " + std::string{path});
    }

    _version = c.read<std::uint32_t>();
    if (_version != kSupportedVersion) {
        MM_LOG_WARN("gguf",
                    "GGUF version {} — reader is validated for v{} only; "
                    "proceeding but expect possible parse errors",
                    _version, kSupportedVersion);
    }

    const auto tensorCount   = c.read<std::uint64_t>();
    const auto metadataCount = c.read<std::uint64_t>();

    MM_LOG_INFO("gguf",
                "header: version={} tensors={} metadata={} file_size={} bytes ({:.2f} MiB)",
                _version, tensorCount, metadataCount,
                _file.size(), bytesToMiB(_file.size()));

    // --- Metadata --------------------------------------------------------

    for (std::uint64_t i = 0; i < metadataCount; ++i) {
        std::string key  = c.readString();
        const auto type  = static_cast<GgufValueType>(c.read<std::uint32_t>());
        MetadataValue v  = readValue(c, type);

        MM_LOG_TRACE("gguf", "metadata[{}] '{}' : {}",
                     i, key, valueTypeName(type));

        if (key == "general.alignment") {
            if (std::holds_alternative<std::uint32_t>(v)) {
                _alignment = std::get<std::uint32_t>(v);
            } else if (std::holds_alternative<std::uint64_t>(v)) {
                _alignment = static_cast<std::size_t>(std::get<std::uint64_t>(v));
            }
        }

        _metadata.emplace(std::move(key), std::move(v));
    }
    MM_LOG_DEBUG("gguf", "metadata done at pos={} bytes", c.pos());

    // --- Tensor index ----------------------------------------------------

    _tensors.reserve(static_cast<std::size_t>(tensorCount));
    for (std::uint64_t i = 0; i < tensorCount; ++i) {
        GgufTensor t;
        t.name = c.readString();
        const auto ndim = c.read<std::uint32_t>();
        if (ndim == 0 || ndim > 8) {
            MM_LOG_ERROR("gguf",
                         "tensor '{}' has implausible ndim={}",
                         t.name, ndim);
            throw std::runtime_error("GgufReader: implausible tensor ndim");
        }
        t.dimensions.reserve(ndim);
        std::uint64_t nelem = 1;
        for (std::uint32_t d = 0; d < ndim; ++d) {
            const auto dim = c.read<std::uint64_t>();
            t.dimensions.push_back(dim);
            nelem *= dim;
        }
        t.nelements  = nelem;
        t.type       = static_cast<GgmlType>(c.read<std::uint32_t>());
        t.fileOffset = c.read<std::uint64_t>();
        t.nbytes     = bytesForTensor(t.type, nelem);

        if (t.nbytes == 0) {
            MM_LOG_ERROR("gguf",
                         "tensor '{}' has unknown/incompatible type tag={} "
                         "nelements={} — cannot compute byte size",
                         t.name, static_cast<std::uint32_t>(t.type), nelem);
            throw std::runtime_error("GgufReader: unsupported tensor type for " + t.name);
        }

        MM_LOG_TRACE("gguf",
                     "tensor[{}] '{}' type={} dims={} nelem={} bytes={} offset={}",
                     i, t.name, typeInfo(t.type).name, formatDims(t.dimensions),
                     t.nelements, t.nbytes, t.fileOffset);

        _totalTensorBytes += t.nbytes;
        _tensors.push_back(std::move(t));
    }

    // --- Align to data section ------------------------------------------

    const auto posAfterIndex = c.pos();
    const auto pad = (_alignment - (posAfterIndex % _alignment)) % _alignment;
    _tensorDataOffset = posAfterIndex + pad;

    if (_tensorDataOffset > _file.size()) {
        MM_LOG_ERROR("gguf",
                     "data offset {} exceeds file size {} — file truncated?",
                     _tensorDataOffset, _file.size());
        throw std::runtime_error("GgufReader: data offset past EOF");
    }

    MM_LOG_INFO("gguf",
                "parsed — {} metadata, {} tensors, alignment={}, "
                "data at offset {} ({:.2f} MiB tensor payload)",
                _metadata.size(), _tensors.size(), _alignment,
                _tensorDataOffset, bytesToMiB(_totalTensorBytes));
}

void GgufReader::loadTensors(core::l0::UsmAllocator& allocator) {
    if (!_file.isOpen()) {
        throw std::runtime_error("GgufReader::loadTensors: not open");
    }
    if (_chunkLoaded) {
        throw std::runtime_error(
            "GgufReader::loadTensors: reader already loaded via "
            "loadTensorsIntoChunks — mixing ownership regimes is not "
            "supported");
    }
    if (_allocator != nullptr && _allocator != &allocator) {
        throw std::runtime_error(
            "GgufReader::loadTensors: already loaded with a different allocator");
    }
    _allocator = &allocator;

    MM_LOG_INFO("gguf",
                "loading {} tensor(s) into USM ({:.2f} MiB total)",
                _tensors.size(), bytesToMiB(_totalTensorBytes));

    std::size_t loadedBytes = 0;
    std::size_t loadedCount = 0;

    for (auto& t : _tensors) {
        if (t.usmPtr != nullptr) {
            continue;
        }

        const std::size_t absOffset = _tensorDataOffset +
                                      static_cast<std::size_t>(t.fileOffset);
        if (absOffset + t.nbytes > _file.size()) {
            MM_LOG_ERROR("gguf",
                         "tensor '{}' walks off EOF: abs_offset={} bytes={} "
                         "file_size={}",
                         t.name, absOffset, t.nbytes, _file.size());
            throw std::runtime_error(
                "GgufReader::loadTensors: tensor data out of bounds for " + t.name);
        }

        t.usmPtr = allocator.allocate(t.nbytes);
        std::memcpy(t.usmPtr, _file.data() + absOffset, t.nbytes);

        loadedBytes += t.nbytes;
        ++loadedCount;

        MM_LOG_TRACE("gguf",
                     "loaded tensor[{}] '{}' {} bytes -> ptr={}",
                     loadedCount, t.name, t.nbytes, t.usmPtr);

        if (loadedCount % 50 == 0) {
            MM_LOG_DEBUG("gguf",
                         "progress: {}/{} tensors, {:.2f} MiB loaded",
                         loadedCount, _tensors.size(), bytesToMiB(loadedBytes));
        }
    }

    MM_LOG_INFO("gguf",
                "load complete: {} tensor(s), {:.2f} MiB now in USM",
                loadedCount, bytesToMiB(loadedBytes));

    // The mmap'd file is no longer needed — every byte we care about is
    // either in the metadata map (parsed) or in USM (copied). Dropping it
    // here releases ~file_size of Linux page cache that would otherwise
    // double-bill alongside the USM copy. Subsequent reads of t.fileOffset
    // would be invalid; nothing in the engine does that.
    if (_file.isOpen()) {
        const std::size_t freed = _file.size();
        _file.close();
        MM_LOG_INFO("gguf",
                    "dropped mmap of source GGUF, freed ~{:.2f} MiB of page cache",
                    static_cast<double>(freed) / (1024.0 * 1024.0));
    }
}

void GgufReader::loadTensorsIntoChunks(::mimirmind::munin::ChunkAllocator& chunks) {
    if (!_file.isOpen()) {
        throw std::runtime_error(
            "GgufReader::loadTensorsIntoChunks: not open");
    }
    if (_allocator != nullptr) {
        throw std::runtime_error(
            "GgufReader::loadTensorsIntoChunks: reader already loaded via "
            "loadTensors — mixing ownership regimes is not supported");
    }
    if (_chunkLoaded) {
        return; // idempotent
    }

    MM_LOG_INFO("gguf",
                "loading {} tensor(s) into chunk-USM ({:.2f} MiB total)",
                _tensors.size(), bytesToMiB(_totalTensorBytes));

    std::size_t loadedBytes = 0;
    std::size_t loadedCount = 0;

    for (auto& t : _tensors) {
        if (t.usmPtr != nullptr) {
            continue;
        }

        const std::size_t absOffset = _tensorDataOffset +
                                      static_cast<std::size_t>(t.fileOffset);
        if (absOffset + t.nbytes > _file.size()) {
            MM_LOG_ERROR("gguf",
                         "tensor '{}' walks off EOF: abs_offset={} bytes={} "
                         "file_size={}",
                         t.name, absOffset, t.nbytes, _file.size());
            throw std::runtime_error(
                "GgufReader::loadTensorsIntoChunks: tensor data out of bounds "
                "for " + t.name);
        }

        const auto a  = chunks.allocate(t.nbytes);
        t.usmPtr      = a.ptr;
        t.chunkIndex  = a.chunkIndex;
        t.chunkOffset = a.chunkOffset;
        std::memcpy(t.usmPtr, _file.data() + absOffset, t.nbytes);

        loadedBytes += t.nbytes;
        ++loadedCount;

        MM_LOG_TRACE("gguf",
                     "chunk-loaded tensor[{}] '{}' {} bytes -> "
                     "chunk={} offset={} ptr={}",
                     loadedCount, t.name, t.nbytes,
                     t.chunkIndex, t.chunkOffset, t.usmPtr);

        if (loadedCount % 50 == 0) {
            MM_LOG_DEBUG("gguf",
                         "chunk-progress: {}/{} tensors, {:.2f} MiB loaded, "
                         "chunks={}",
                         loadedCount, _tensors.size(), bytesToMiB(loadedBytes),
                         chunks.chunkCount());
        }
    }

    _chunkLoaded = true;

    MM_LOG_INFO("gguf",
                "chunk-load complete: {} tensor(s), {:.2f} MiB packed into "
                "{} chunk(s) ({:.2f} MiB / chunk max)",
                loadedCount, bytesToMiB(loadedBytes),
                chunks.chunkCount(), bytesToMiB(chunks.chunkBytes()));

    if (_file.isOpen()) {
        const std::size_t freed = _file.size();
        _file.close();
        MM_LOG_INFO("gguf",
                    "dropped mmap of source GGUF, freed ~{:.2f} MiB of page cache",
                    static_cast<double>(freed) / (1024.0 * 1024.0));
    }
}

void GgufReader::close() noexcept {
    if (_allocator != nullptr) {
        for (auto& t : _tensors) {
            if (t.usmPtr != nullptr) {
                _allocator->deallocate(t.usmPtr, t.nbytes);
                t.usmPtr = nullptr;
            }
        }
        _allocator = nullptr;
    }
    // Chunk-loaded tensors: the ChunkAllocator owns the memory. Just
    // null the raw pointers so the reader forgets its view — the actual
    // release happens when the outside owner (LoadedModel) drops the
    // ChunkAllocator.
    if (_chunkLoaded) {
        for (auto& t : _tensors) {
            t.usmPtr      = nullptr;
            t.chunkIndex  = 0;
            t.chunkOffset = 0;
        }
        _chunkLoaded = false;
    }
    _file.close();
    _tensors.clear();
    _metadata.clear();
    _version          = 0;
    _alignment        = 32;
    _tensorDataOffset = 0;
    _totalTensorBytes = 0;
}

const GgufTensor* GgufReader::findTensor(std::string_view name) const noexcept {
    for (const auto& t : _tensors) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

const MetadataValue* GgufReader::findMetadata(std::string_view key) const noexcept {
    const auto it = _metadata.find(std::string{key});
    return it == _metadata.end() ? nullptr : &it->second;
}

} // namespace mimirmind::core::gguf