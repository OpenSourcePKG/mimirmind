#include "runtime/MappedFile.hpp"

#include "runtime/Log.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace mimirmind::runtime {

MappedFile::MappedFile(std::string_view path)
    : _path{path}
{
    MM_LOG_INFO("mmap", "opening {}", _path);

    _fd = ::open(_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (_fd < 0) {
        const int err = errno;
        MM_LOG_ERROR("mmap", "open({}) failed: {} ({})",
                     _path, std::strerror(err), err);
        throw std::runtime_error("MappedFile: open(" + _path + ") failed: " +
                                 std::strerror(err));
    }

    struct stat st{};
    if (::fstat(_fd, &st) != 0) {
        const int err = errno;
        ::close(_fd);
        _fd = -1;
        MM_LOG_ERROR("mmap", "fstat({}) failed: {} ({})",
                     _path, std::strerror(err), err);
        throw std::runtime_error("MappedFile: fstat failed: " +
                                 std::string{std::strerror(err)});
    }

    _size = static_cast<std::size_t>(st.st_size);
    MM_LOG_DEBUG("mmap", "{} stat: size={} bytes", _path, _size);

    if (_size == 0) {
        ::close(_fd);
        _fd = -1;
        MM_LOG_ERROR("mmap", "{} is empty", _path);
        throw std::runtime_error("MappedFile: file is empty: " + _path);
    }

    void* p = ::mmap(nullptr, _size, PROT_READ, MAP_PRIVATE, _fd, 0);
    if (p == MAP_FAILED) {
        const int err = errno;
        ::close(_fd);
        _fd = -1;
        MM_LOG_ERROR("mmap", "mmap({}) failed: {} ({})",
                     _path, std::strerror(err), err);
        throw std::runtime_error("MappedFile: mmap failed: " +
                                 std::string{std::strerror(err)});
    }
    _data = static_cast<std::uint8_t*>(p);

    MM_LOG_INFO("mmap",
                "mapped {} bytes ({:.2f} MiB) ptr={}",
                _size,
                static_cast<double>(_size) / (1024.0 * 1024.0),
                static_cast<const void*>(_data));

    if (::madvise(_data, _size, MADV_RANDOM) != 0) {
        MM_LOG_DEBUG("mmap", "madvise(MADV_RANDOM) failed: {} (non-fatal)",
                     std::strerror(errno));
    }
}

MappedFile::~MappedFile() {
    close();
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : _data{std::exchange(other._data, nullptr)},
      _size{std::exchange(other._size, std::size_t{0})},
      _fd{std::exchange(other._fd, -1)},
      _path{std::move(other._path)}
{}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        close();
        _data = std::exchange(other._data, nullptr);
        _size = std::exchange(other._size, std::size_t{0});
        _fd   = std::exchange(other._fd, -1);
        _path = std::move(other._path);
    }
    return *this;
}

void MappedFile::close() noexcept {
    if (_data != nullptr) {
        MM_LOG_DEBUG("mmap", "munmap({} bytes) for {}", _size, _path);
        // Hint the kernel that we don't need these pages cached anymore.
        // Without this, even after munmap the file pages linger in the
        // page cache (counted as buff/cache by free/top) until memory
        // pressure forces eviction. On a 6.5 GiB-USM-resident process
        // this shows up as ~4 GiB phantom "used" memory.
        ::madvise(_data, _size, MADV_DONTNEED);
        ::munmap(_data, _size);
        _data = nullptr;
        _size = 0;
    }
    if (_fd >= 0) {
        // Same hint at the fd level — POSIX_FADV_DONTNEED on (0, 0)
        // means the whole file. Best-effort; some filesystems ignore it
        // (tmpfs in particular) but ext4 / xfs respect it.
        ::posix_fadvise(_fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(_fd);
        _fd = -1;
    }
}

} // namespace mimirmind::runtime