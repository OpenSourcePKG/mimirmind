#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Stefan Werfling
#
# Qwen4-Exp PLE n-gram hashing reference (5.27 I-4b). Reimplements the EXACT
# integer hashing from transformers models/qwen4_exp Qwen4ExpTextNGramEmbedding
# (_splitmix64 / _build_layer_multipliers / _find_nth_prime_after /
# _shift_right_ignore_eos / forward), in pure Python int + numpy int64, so the
# C++ host gather (q4e_ngram_parity.cpp) can be bit-checked against it. Hashing
# is host int64 arithmetic — no GPU/torch needed.
#
# Dumps into an output dir:
#   meta.txt        ngram_size heads_per_ngram vocab_size eos ngram_vocab_base seed ple_layer_index T context_len
#   mult.bin        [ngram_size]  int64  layer_multipliers
#   headvocab.bin   [ngram_heads] int64  per-head prime vocab sizes
#   headoff.bin     [ngram_heads] int64  per-head offsets
#   input_ids.bin   [T]           int64
#   ngram_ids.bin   [T, ngram_heads] int64  expected gathered ids

import os
import sys

import numpy as np

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB
PRIME_1 = 10007


def splitmix64(v: int) -> int:
    v = (v + GAMMA) & MASK64
    v = ((v ^ (v >> 30)) * M1) & MASK64
    v = ((v ^ (v >> 27)) * M2) & MASK64
    return (v ^ (v >> 31)) & MASK64


def build_layer_multipliers(unigram_vocab, ngram_size, ple_layer_index, seed):
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = (seed + PRIME_1 * ple_layer_index) & MASK64
    out = []
    for index in range(ngram_size):
        value = (base_seed + GAMMA * (index + 1)) & MASK64
        out.append(2 * (splitmix64(value) % half_bound) + 1)
    return out


def is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    d = 3
    while d * d <= v:
        if v % d == 0:
            return False
        d += 2
    return True


def find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not is_prime(prime):
            prime += 1
    return prime


# int64 two's-complement wrap for signed multiply (matches torch/numpy int64).
def to_i64(x: int) -> int:
    x &= MASK64
    return x - (1 << 64) if x >= (1 << 63) else x


def shift_right_ignore_eos(tokens: np.ndarray, shift: int, eos: int) -> np.ndarray:
    # tokens: [L] int64 (B=1). Mirrors the transformers per-row logic.
    if shift == 0:
        return tokens.copy()
    L = tokens.shape[0]
    positions = np.arange(L, dtype=np.int64)
    eos_positions = np.where(tokens == eos, positions, np.int64(-1))
    prev_eos_incl = np.maximum.accumulate(eos_positions)          # cummax
    previous_eos = np.concatenate([[np.int64(-1)], prev_eos_incl[:-1]])
    segment_start = previous_eos + 1
    position_in_segment = positions - segment_start
    source_positions = positions - shift
    gather_positions = np.clip(source_positions, 0, None)
    shifted = tokens[gather_positions]
    valid = (position_in_segment >= shift) & (source_positions >= 0)
    return np.where(valid, shifted, np.int64(eos))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/q4e_ngram_dump"
    os.makedirs(out_dir, exist_ok=True)

    ngram_size = 3
    heads_per_ngram = 8
    ngram_heads = (ngram_size - 1) * heads_per_ngram   # 16
    vocab_size = 248320
    eos = 248044
    ngram_vocab_base = 20_000_000
    seed = 1234
    ple_layer_index = 0
    context_len = ngram_size - 1                        # 2

    mult = build_layer_multipliers(vocab_size, ngram_size, ple_layer_index, seed)

    head_vocab, head_off = [], []
    total = 0
    for h in range(ngram_heads):
        global_h = ple_layer_index * ngram_heads + h
        size = find_nth_prime_after(ngram_vocab_base - 1, global_h + 1)
        head_vocab.append(size)
        head_off.append(total)
        total += size

    # Deterministic input with a couple of EOS to exercise segment resets.
    rng = np.random.default_rng(1234)
    T = 12
    input_ids = rng.integers(0, vocab_size, size=T, dtype=np.int64)
    input_ids[4] = eos
    input_ids[9] = eos

    previous_context = np.full(context_len, eos, dtype=np.int64)
    history = np.concatenate([previous_context, input_ids])        # [context_len + T]
    shifted = [shift_right_ignore_eos(history, s, eos) for s in range(ngram_size)]

    Lh = history.shape[0]
    blocks = []  # each [Lh, heads_per_ngram]
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        # Accumulate the mix in UNSIGNED 64-bit space (mult wraps mod 2^64 like
        # torch/numpy int64; XOR is bitwise on the 64-bit rep), then reinterpret
        # as signed for the positive-divisor remainder.
        mixed_u = [(int(shifted[0][i]) * mult[0]) & MASK64 for i in range(Lh)]
        for pos in range(1, ngram):
            for i in range(Lh):
                mixed_u[i] ^= (int(shifted[pos][i]) * mult[pos]) & MASK64
        blk = np.zeros((Lh, heads_per_ngram), dtype=np.int64)
        for j in range(heads_per_ngram):
            hv = head_vocab[start + j]
            ho = head_off[start + j]
            for i in range(Lh):
                m = to_i64(mixed_u[i])
                r = ((m % hv) + hv) % hv    # torch.remainder positive-divisor semantics
                blk[i, j] = r + ho
        blocks.append(blk)

    ngram_ids_full = np.concatenate(blocks, axis=1)                # [Lh, 16]
    ngram_ids = ngram_ids_full[-T:]                                # [T, 16]

    def dump(name, arr):
        arr.astype("<i8").tofile(os.path.join(out_dir, name))

    dump("mult.bin", np.array(mult, dtype=np.int64))
    dump("headvocab.bin", np.array(head_vocab, dtype=np.int64))
    dump("headoff.bin", np.array(head_off, dtype=np.int64))
    dump("input_ids.bin", input_ids)
    dump("ngram_ids.bin", ngram_ids)
    with open(os.path.join(out_dir, "meta.txt"), "w") as f:
        f.write(f"ngram_size {ngram_size}\nheads_per_ngram {heads_per_ngram}\n"
                f"vocab_size {vocab_size}\neos {eos}\nngram_vocab_base {ngram_vocab_base}\n"
                f"seed {seed}\nple_layer_index {ple_layer_index}\nT {T}\n"
                f"context_len {context_len}\n")

    print(f"wrote {out_dir}: T={T} ngram_heads={ngram_heads}")
    print(f"mult={mult}")
    print(f"head_vocab[0..3]={head_vocab[:4]} head_off[0..3]={head_off[:4]}")
    print(f"ngram_ids[0]={ngram_ids[0].tolist()}")


if __name__ == "__main__":
    main()
