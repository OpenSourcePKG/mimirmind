#include "model/WeightsMap.hpp"

#include "runtime/Log.hpp"

#include <stdexcept>
#include <string>

namespace mimirmind::model {

WeightsMap::WeightsMap(const GgufReader& reader) {
    const auto& tensors = reader.tensors();
    _byName.reserve(tensors.size() * 2);
    for (const auto& t : tensors) {
        _byName.emplace(t.name, &t);
    }
    MM_LOG_INFO("weights", "indexed {} tensors for O(1) lookup", _byName.size());
}

const GgufTensor* WeightsMap::find(std::string_view name) const noexcept {
    const auto it = _byName.find(std::string{name});
    return it == _byName.end() ? nullptr : it->second;
}

const GgufTensor& WeightsMap::require(std::string_view name) const {
    if (const auto* t = find(name)) {
        return *t;
    }
    MM_LOG_ERROR("weights", "required tensor '{}' missing", name);
    throw std::runtime_error("WeightsMap::require: tensor '" +
                             std::string{name} + "' not in model");
}

const GgufTensor* WeightsMap::findBlock(std::size_t blockIdx,
                                        std::string_view suffix) const {
    std::string key = "blk.";
    key += std::to_string(blockIdx);
    key += '.';
    key.append(suffix);
    return find(key);
}

} // namespace mimirmind::model