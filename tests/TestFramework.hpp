#pragma once

// Tiny header-only test framework shared by the pure-CPU test binaries.
//
// Usage:
//   #include "TestFramework.hpp"
//   TEST(my_test_name) { EXPECT_EQ(2 + 2, 4); }
//   int main() { return mm::test::run(); }
//
// Failure throws std::runtime_error with file:line + expr context; the
// runner catches and continues to the next test. No external deps.

#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mm::test {

struct TestCase {
    const char*           name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

// Print enums via their underlying integer; everything else streams as-is.
template <typename T>
auto streamable(const T& v) {
    if constexpr (std::is_enum_v<T>) {
        return static_cast<std::underlying_type_t<T>>(v);
    } else {
        return v;
    }
}

template <typename A, typename B>
void expectEqImpl(const char* file, int line,
                  const char* exprA, const char* exprB,
                  const A& a, const B& b) {
    if (!(a == b)) {
        std::ostringstream os;
        os << file << ":" << line << " EXPECT_EQ(" << exprA << ", " << exprB
           << ") failed: lhs=" << streamable(a) << " rhs=" << streamable(b);
        throw std::runtime_error(os.str());
    }
}

inline void expectNearImpl(const char* file, int line,
                           const char* exprA, const char* exprB,
                           float a, float b, float tol) {
    const float diff = std::fabs(a - b);
    if (!(diff <= tol)) {
        std::ostringstream os;
        os << file << ":" << line << " EXPECT_NEAR(" << exprA << ", " << exprB
           << ", tol=" << tol << ") failed: lhs=" << a << " rhs=" << b
           << " diff=" << diff;
        throw std::runtime_error(os.str());
    }
}

inline void expectTrueImpl(const char* file, int line,
                           const char* expr, bool cond) {
    if (!cond) {
        std::ostringstream os;
        os << file << ":" << line << " EXPECT_TRUE(" << expr << ") failed";
        throw std::runtime_error(os.str());
    }
}

inline int run() {
    int passed = 0;
    int failed = 0;
    for (auto& t : registry()) {
        std::printf("[RUN ] %s\n", t.name);
        try {
            t.fn();
            std::printf("[ OK ] %s\n", t.name);
            ++passed;
        } catch (const std::exception& e) {
            std::printf("[FAIL] %s\n        %s\n", t.name, e.what());
            ++failed;
        }
    }
    std::printf("\n=== %d passed, %d failed (of %zu) ===\n",
                passed, failed, registry().size());
    return failed == 0 ? 0 : 1;
}

} // namespace mm::test

#define TEST(name)                                                       \
    static void name();                                                  \
    static const ::mm::test::Registrar reg_##name{#name, name};          \
    static void name()

#define EXPECT_EQ(a, b)      ::mm::test::expectEqImpl(  __FILE__, __LINE__, #a, #b, (a), (b))
#define EXPECT_NEAR(a, b, t) ::mm::test::expectNearImpl(__FILE__, __LINE__, #a, #b, (a), (b), (t))
#define EXPECT_TRUE(c)       ::mm::test::expectTrueImpl(__FILE__, __LINE__, #c, (c))