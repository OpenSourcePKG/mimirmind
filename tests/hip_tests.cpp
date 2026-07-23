// HIP-backed test suite. Runs each `hip_*_probe` binary as a subprocess
// and surfaces pass/fail as a `TestFramework` TEST() case. Aggregating
// this way keeps the kernel-side code (the actual parity math) in the
// per-probe files where it already lives — with the shared TEST()
// runner it now shows up as ONE ctest-visible target (`hip_tests`) with
// N sub-cases instead of N independent binaries the CI has to launch
// separately.
//
// This is the "un-gate for CI-visibility" step of the HIP-to-main
// track. A future revision will migrate the kernel bodies into the
// TEST() functions directly and drop the subprocess layer — that's a
// bigger refactor (~2-3 weeks per the plan) and is intentionally out
// of scope here. See Synaipse
// `todos/hip-tests-inline-migration.md` for the follow-up.

#include "TestFramework.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>

namespace {

std::filesystem::path binDir() {
    // Resolve THIS binary's directory once so probes are looked up
    // relative to the same build tree — CI runs from arbitrary CWDs.
    char self[4096]{};
    const auto n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0) {
        return std::filesystem::current_path();
    }
    return std::filesystem::path{std::string{self, static_cast<std::size_t>(n)}}
        .parent_path();
}

// Run `probeName` (relative to hip_tests' dir), block, return exit code.
// stdout is captured into `outBuf` and appended to the failure message
// on error so operators see the kernel's own diagnostic. Throws on
// spawn / wait failures.
int runProbe(std::string_view probeName, std::string& outBuf) {
    const auto probePath = binDir() / std::string{probeName};
    if (!std::filesystem::exists(probePath)) {
        std::ostringstream os;
        os << "probe binary missing at " << probePath.string()
           << " — did the build produce " << probeName << "?";
        throw std::runtime_error(os.str());
    }

    int pipefd[2]{};
    if (::pipe(pipefd) != 0) {
        throw std::runtime_error(std::string{"pipe() failed: "} + std::strerror(errno));
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        throw std::runtime_error(std::string{"fork() failed: "} + std::strerror(errno));
    }

    if (pid == 0) {
        // Child — redirect stdout to the pipe, exec the probe.
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDOUT_FILENO);
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);
        const auto pathStr = probePath.string();
        char* argv[] = { const_cast<char*>(pathStr.c_str()), nullptr };
        ::execv(pathStr.c_str(), argv);
        // exec returned — failure. Write to the pipe so parent sees it.
        std::string err = "execv failed: ";
        err += std::strerror(errno);
        (void)!::write(STDERR_FILENO, err.data(), err.size());
        _exit(127);
    }

    // Parent — drain the pipe, wait.
    ::close(pipefd[1]);
    char buf[4096]{};
    while (true) {
        const auto n = ::read(pipefd[0], buf, sizeof(buf));
        if (n <= 0) break;
        outBuf.append(buf, static_cast<std::size_t>(n));
    }
    ::close(pipefd[0]);

    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        throw std::runtime_error(std::string{"waitpid() failed: "}
                                 + std::strerror(errno));
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return -WTERMSIG(status);
    return 255;
}

// TEST-body helper: runs the probe, throws with the captured output on
// non-zero exit so the TestFramework catches it and prints the kernel's
// own diagnostic to stderr.
void expectProbeOk(std::string_view probeName) {
    std::string out;
    const int rc = runProbe(probeName, out);
    if (rc != 0) {
        std::ostringstream os;
        os << probeName << " exited with code " << rc
           << "\n---- probe stdout / stderr ----\n" << out;
        throw std::runtime_error(os.str());
    }
}

} // namespace

// -----------------------------------------------------------------------
// Core kernel families — one TEST per family so a broken kernel port
// shows up in exactly one place. The coverage matrix is intentionally
// narrower than gpu_tests (which has ~110 cases against the L0 kernels);
// each probe here is already a Q8_0/Q6_K/etc parity check with
// combined-tolerance abs+rel gates, so a green ctest run says "kernels
// still numerically match compute:: reference bodies on THIS device."
// -----------------------------------------------------------------------

// Element-wise + reduction primitives
TEST(hip_rmsnorm)                    { expectProbeOk("hip_rmsnorm_probe"); }
TEST(hip_rmsnorm_gemma)              { expectProbeOk("hip_rmsnorm_gemma_probe"); }
TEST(hip_add_residual)               { expectProbeOk("hip_add_residual_probe"); }
TEST(hip_silu_mul)                   { expectProbeOk("hip_silu_mul_probe"); }
TEST(hip_gelu_mul)                   { expectProbeOk("hip_gelu_mul_probe"); }

// Positional encoding + shape ops
TEST(hip_rope)                       { expectProbeOk("hip_rope_probe"); }
TEST(hip_qkv_split)                  { expectProbeOk("hip_qkv_split_probe"); }

// Matmul families — one representative per quant type
TEST(hip_matmul_q8_0_vec)            { expectProbeOk("hip_matmul_q8_probe"); }
TEST(hip_matmul_q8_0_gemm)           { expectProbeOk("hip_matmul_q8_gemm_probe"); }
TEST(hip_matmul_q3k_vec)             { expectProbeOk("hip_matmul_q3k_probe"); }

// MoE (from Blocker 2)
TEST(hip_moe_down_fused_k_q8_0)      { expectProbeOk("hip_moe_down_fused_k_q8_0_probe"); }
TEST(hip_moe_down_fused_k_q6k)       { expectProbeOk("hip_moe_down_fused_k_q6k_probe"); }
TEST(hip_moe_topk)                   { expectProbeOk("hip_moe_topk_probe"); }
TEST(hip_ffn_gate_up_q8)             { expectProbeOk("hip_ffn_gate_up_q8_probe"); }

// Qwen3-Next GatedDeltaNet linear-attention (gate + AR recurrence, wave32)
TEST(hip_deltanet)                   { expectProbeOk("hip_deltanet_probe"); }

// Attention families — flash prefill covers the largest compute share
TEST(hip_attention)                  { expectProbeOk("hip_attention_probe"); }
TEST(hip_attention_prefill_flash)    { expectProbeOk("hip_attention_prefill_flash_probe"); }

int main() {
    return mm::test::run();
}
