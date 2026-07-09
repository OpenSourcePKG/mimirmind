#include "server/ApiHelpers.hpp"

#include <chrono>
#include <random>

namespace mimirmind::server {

using nlohmann::json;

std::string makeRequestId() {
    static std::mt19937_64 rng{std::random_device{}()};
    static constexpr char kAlpha[] =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string out = "chatcmpl-";
    out.reserve(out.size() + 24);
    std::uniform_int_distribution<std::size_t> d{0, sizeof(kAlpha) - 2};
    for (int i = 0; i < 24; ++i) {
        out.push_back(kAlpha[d(rng)]);
    }
    return out;
}

std::int64_t unixNow() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void sendJson(httplib::Response& res, int status, const json& body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

void sendError(httplib::Response& res, int status,
               std::string_view type, std::string_view message) {
    json body = {
        {"error", {
            {"message", std::string{message}},
            {"type",    std::string{type}},
            {"code",    nullptr},
        }},
    };
    sendJson(res, status, body);
}

} // namespace mimirmind::server