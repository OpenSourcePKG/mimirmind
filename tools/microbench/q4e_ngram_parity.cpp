// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Qwen4-Exp PLE n-gram hashing parity harness (5.27 I-4b). Pure host int64
// arithmetic — the production PLE host-gather hashing lifted verbatim. Recomputes
// layer_multipliers (splitmix64), per-head prime vocab sizes/offsets, the
// EOS-aware shift, the XOR mix and the per-head (mod prime + offset) exactly as
// transformers Qwen4ExpTextNGramEmbedding, and checks the resulting ngram_ids
// bit-for-bit against the numpy reference dump (q4e_ngram_ref.py).
//
// Build + run (host, no GPU):
//   g++ -O2 -std=c++20 tools/microbench/q4e_ngram_parity.cpp -o /tmp/q4e_ngram_parity
//   python3 tools/microbench/q4e_ngram_ref.py /tmp/q4e_ngram_dump
//   /tmp/q4e_ngram_parity /tmp/q4e_ngram_dump

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t GAMMA = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t M1    = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t M2    = 0x94D049BB133111EBULL;
constexpr std::int64_t  PRIME_1 = 10007;

std::uint64_t splitmix64(std::uint64_t v) {
    v = v + GAMMA;
    v = (v ^ (v >> 30)) * M1;
    v = (v ^ (v >> 27)) * M2;
    return v ^ (v >> 31);
}

std::vector<std::int64_t> buildLayerMultipliers(std::int64_t unigramVocab,
                                                int ngramSize, int pleLayerIndex,
                                                std::int64_t seed) {
    const std::uint64_t maxLong = (1ULL << 63) - 1;
    const std::uint64_t multiplierMax = maxLong / static_cast<std::uint64_t>(std::max<std::int64_t>(unigramVocab, 1));
    const std::uint64_t halfBound = std::max<std::uint64_t>(1, multiplierMax / 2);
    const std::uint64_t baseSeed = static_cast<std::uint64_t>(seed)
                                 + static_cast<std::uint64_t>(PRIME_1) * static_cast<std::uint64_t>(pleLayerIndex);
    std::vector<std::int64_t> out;
    for (int index = 0; index < ngramSize; ++index) {
        const std::uint64_t value = baseSeed + GAMMA * static_cast<std::uint64_t>(index + 1);
        out.push_back(static_cast<std::int64_t>(2 * (splitmix64(value) % halfBound) + 1));
    }
    return out;
}

bool isPrime(std::int64_t v) {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    for (std::int64_t d = 3; d * d <= v; d += 2)
        if (v % d == 0) return false;
    return true;
}

std::int64_t findNthPrimeAfter(std::int64_t start, std::int64_t count) {
    std::int64_t prime = start;
    for (std::int64_t i = 0; i < count; ++i) {
        ++prime;
        while (!isPrime(prime)) ++prime;
    }
    return prime;
}

std::vector<std::int64_t> shiftRightIgnoreEos(const std::vector<std::int64_t>& tok,
                                              int shift, std::int64_t eos) {
    const int L = static_cast<int>(tok.size());
    if (shift == 0) return tok;
    std::vector<std::int64_t> out(L);
    std::int64_t run = -1;  // cummax of eos_positions
    std::int64_t prevMaxPrev = -1;  // prevMax[i-1]; for i=0 previous_eos=-1
    for (int i = 0; i < L; ++i) {
        const std::int64_t eospos = (tok[i] == eos) ? i : -1;
        const std::int64_t previousEos = (i == 0) ? -1 : prevMaxPrev;
        // advance cummax AFTER reading prevMax[i-1]
        if (eospos > run) run = eospos;
        const std::int64_t segmentStart = previousEos + 1;
        const std::int64_t positionInSegment = i - segmentStart;
        const std::int64_t sourcePositions = static_cast<std::int64_t>(i) - shift;
        const std::int64_t gpos = sourcePositions < 0 ? 0 : sourcePositions;
        const std::int64_t shifted = tok[gpos];
        const bool valid = (positionInSegment >= shift) && (sourcePositions >= 0);
        out[i] = valid ? shifted : eos;
        prevMaxPrev = run;  // becomes prevMax[i] for the next iteration
    }
    return out;
}

std::vector<std::int64_t> readI64(const std::string& p, std::size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(2); }
    std::vector<std::int64_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 8));
    if (!f) { std::fprintf(stderr, "short read %s\n", p.c_str()); std::exit(2); }
    return v;
}

} // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp/q4e_ngram_dump";
    std::map<std::string, std::int64_t> meta;
    {
        std::ifstream f(dir + "/meta.txt");
        if (!f) { std::fprintf(stderr, "no meta.txt\n"); return 2; }
        std::string k; std::int64_t v;
        while (f >> k >> v) meta[k] = v;
    }
    const int ngramSize      = static_cast<int>(meta["ngram_size"]);
    const int headsPerNgram  = static_cast<int>(meta["heads_per_ngram"]);
    const int ngramHeads     = (ngramSize - 1) * headsPerNgram;
    const std::int64_t vocab = meta["vocab_size"];
    const std::int64_t eos   = meta["eos"];
    const std::int64_t base  = meta["ngram_vocab_base"];
    const std::int64_t seed  = meta["seed"];
    const int pleIdx         = static_cast<int>(meta["ple_layer_index"]);
    const int T              = static_cast<int>(meta["T"]);
    const int ctx            = static_cast<int>(meta["context_len"]);

    const auto mult = buildLayerMultipliers(vocab, ngramSize, pleIdx, seed);
    std::vector<std::int64_t> headVocab(ngramHeads), headOff(ngramHeads);
    std::int64_t total = 0;
    for (int h = 0; h < ngramHeads; ++h) {
        const std::int64_t globalH = static_cast<std::int64_t>(pleIdx) * ngramHeads + h;
        headVocab[h] = findNthPrimeAfter(base - 1, globalH + 1);
        headOff[h] = total;
        total += headVocab[h];
    }

    const auto inputIds = readI64(dir + "/input_ids.bin", T);
    const auto refIds   = readI64(dir + "/ngram_ids.bin", static_cast<std::size_t>(T) * ngramHeads);
    // Cross-check the recomputed tables against the reference dump too.
    const auto refMult  = readI64(dir + "/mult.bin", ngramSize);
    const auto refHV    = readI64(dir + "/headvocab.bin", ngramHeads);
    const auto refHO    = readI64(dir + "/headoff.bin", ngramHeads);

    // history = [ctx * eos | input_ids]
    std::vector<std::int64_t> history(ctx, eos);
    history.insert(history.end(), inputIds.begin(), inputIds.end());
    const int Lh = static_cast<int>(history.size());

    std::vector<std::vector<std::int64_t>> shifted(ngramSize);
    for (int s = 0; s < ngramSize; ++s) shifted[s] = shiftRightIgnoreEos(history, s, eos);

    // ngram_ids_full [Lh, ngramHeads]; then take last T rows.
    std::vector<std::int64_t> full(static_cast<std::size_t>(Lh) * ngramHeads);
    for (int ngram = 2; ngram <= ngramSize; ++ngram) {
        const int start = (ngram - 2) * headsPerNgram;
        for (int i = 0; i < Lh; ++i) {
            std::uint64_t mixed = static_cast<std::uint64_t>(shifted[0][i])
                                * static_cast<std::uint64_t>(mult[0]);
            for (int pos = 1; pos < ngram; ++pos) {
                const std::uint64_t term = static_cast<std::uint64_t>(shifted[pos][i])
                                         * static_cast<std::uint64_t>(mult[pos]);
                mixed ^= term;
            }
            const std::int64_t m = static_cast<std::int64_t>(mixed);
            for (int j = 0; j < headsPerNgram; ++j) {
                const std::int64_t hv = headVocab[start + j];
                const std::int64_t r = ((m % hv) + hv) % hv;
                full[static_cast<std::size_t>(i) * ngramHeads + start + j] = r + headOff[start + j];
            }
        }
    }

    // Compare last-T rows vs refIds; tables vs ref.
    std::size_t mism = 0, first = 0; bool haveFirst = false;
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < ngramHeads; ++j) {
            const std::int64_t got = full[static_cast<std::size_t>(Lh - T + t) * ngramHeads + j];
            const std::int64_t exp = refIds[static_cast<std::size_t>(t) * ngramHeads + j];
            if (got != exp) { if (!haveFirst) { first = static_cast<std::size_t>(t) * ngramHeads + j; haveFirst = true; } ++mism; }
        }
    }
    std::size_t tblMism = 0;
    for (int i = 0; i < ngramSize; ++i) tblMism += (mult[i] != refMult[i]);
    for (int h = 0; h < ngramHeads; ++h) tblMism += (headVocab[h] != refHV[h]) + (headOff[h] != refHO[h]);

    std::printf("q4e_ngram_parity: T=%d ngramHeads=%d\n", T, ngramHeads);
    std::printf("  mult[0]=%lld headVocab[0]=%lld headOff[15]=%lld\n",
                (long long)mult[0], (long long)headVocab[0], (long long)headOff[ngramHeads - 1]);
    std::printf("  table mismatches=%zu\n", tblMism);
    std::printf("  ngram_ids mismatches=%zu / %d\n", mism, T * ngramHeads);
    if (mism && haveFirst)
        std::printf("  first mismatch at flat idx %zu\n", first);
    const bool pass = (mism == 0 && tblMism == 0);
    std::printf("%s\n", pass ? "NGRAM PARITY PASS" : "NGRAM PARITY FAIL");
    return pass ? 0 : 1;
}
