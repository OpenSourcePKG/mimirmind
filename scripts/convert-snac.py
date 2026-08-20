#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Stefan Werfling
#
# Convert the reference SNAC 24 kHz codec (hubertsiuzdak/snac_24khz) into a
# flat safetensors file the in-house C++ SnacDecoder (src/runtime/audio/
# SnacDecoder.cpp, roadmap 8.13.2) reads directly. The reference ships weights
# under torch weight-norm *parametrizations*; accessing `module.weight` returns
# the already-resolved effective weight, so the export materialises plain
# `.weight`/`.bias` tensors and the C++ never reconstructs the norm.
#
# Only the DECODE path is exported (quantizer.from_codes + Decoder) — the
# encoder is unused for TTS.
#
# Tensor naming (must match SnacDecoder.cpp exactly):
#   q.{i}.codebook            [4096, 8]        embedding weight
#   q.{i}.out_proj.weight     [768, 8, 1]      WNConv1d(8->768, k1)
#   q.{i}.out_proj.bias       [768]
#   dec.in.dw.weight          [768, 1, 7]      depthwise WNConv1d(768,768,k7,g768)
#   dec.in.dw.bias            [768]
#   dec.in.pw.weight          [1024, 768, 1]   WNConv1d(768->1024, k1)
#   dec.in.pw.bias            [1024]
#   dec.block.{b}.snake.alpha [inD]            (inD = 1024 >> b)
#   dec.block.{b}.up.weight   [inD, outD, 2s]  WNConvTranspose1d (outD = 1024 >> (b+1))
#   dec.block.{b}.up.bias     [outD]
#   dec.block.{b}.noise.weight[outD, outD, 1]  NoiseBlock linear (bias=False)
#   dec.block.{b}.res.{r}.snake1.alpha [outD]
#   dec.block.{b}.res.{r}.dw.weight    [outD, 1, 7]  depthwise dilated conv
#   dec.block.{b}.res.{r}.dw.bias      [outD]
#   dec.block.{b}.res.{r}.snake2.alpha [outD]
#   dec.block.{b}.res.{r}.pw.weight    [outD, outD, 1]
#   dec.block.{b}.res.{r}.pw.bias      [outD]
#   dec.out.snake.alpha       [64]
#   dec.out.conv.weight       [1, 64, 7]
#   dec.out.conv.bias         [1]
#
# Usage:
#   pip install snac safetensors torch numpy
#   python scripts/convert-snac.py <snac_24khz_dir_or_hf_repo> <out.safetensors>
#     e.g. python scripts/convert-snac.py hubertsiuzdak/snac_24khz \
#          /opt/mimirmind/models/snac-24khz/model.safetensors
#
# Also emits, next to the output, a `snac_fixture_codes.json` +
# `snac_fixture_pcm.f32` when a --fixture WAV is given: the reference decode of
# that clip's codes, so the C++ decoder can be byte-parity-checked off-box.

import sys
import json

import torch
from snac import SNAC
from safetensors.torch import save_file


def alpha(m):
    # Snake1d.alpha is [1, C, 1] -> [C].
    return m.alpha.detach().reshape(-1).contiguous()


def conv_w(m):
    return m.weight.detach().contiguous()   # effective (weight-norm resolved)


def conv_b(m):
    return None if m.bias is None else m.bias.detach().contiguous()


def add(out, name, tensor):
    if tensor is not None:
        out[name] = tensor.to(torch.float32)


def export_residual_unit(out, prefix, ru):
    # ResidualUnit.block = [Snake1d, WNConv1d(k7,dil,gdim), Snake1d, WNConv1d(k1)]
    blk = ru.block
    add(out, prefix + "snake1.alpha", alpha(blk[0]))
    add(out, prefix + "dw.weight", conv_w(blk[1]))
    add(out, prefix + "dw.bias",   conv_b(blk[1]))
    add(out, prefix + "snake2.alpha", alpha(blk[2]))
    add(out, prefix + "pw.weight", conv_w(blk[3]))
    add(out, prefix + "pw.bias",   conv_b(blk[3]))


def export_decoder_block(out, prefix, db):
    # DecoderBlock.block = [Snake1d, WNConvTranspose1d, NoiseBlock,
    #                       ResidualUnit x3]
    blk = db.block
    add(out, prefix + "snake.alpha", alpha(blk[0]))
    add(out, prefix + "up.weight", conv_w(blk[1]))
    add(out, prefix + "up.bias",   conv_b(blk[1]))
    add(out, prefix + "noise.weight", conv_w(blk[2].linear))
    export_residual_unit(out, prefix + "res.0.", blk[3])
    export_residual_unit(out, prefix + "res.1.", blk[4])
    export_residual_unit(out, prefix + "res.2.", blk[5])


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]

    model = SNAC.from_pretrained(src).eval() if not src.endswith(".json") \
        else SNAC.from_config(src)
    q = model.quantizer
    dm = model.decoder.model

    assert model.vq_strides == [4, 2, 1], f"unexpected vq_strides {model.vq_strides}"
    assert model.decoder_rates == [8, 8, 4, 2], \
        f"unexpected decoder_rates {model.decoder_rates}"

    out = {}
    # Quantizer decode path.
    for i, vq in enumerate(q.quantizers):
        add(out, f"q.{i}.codebook", vq.codebook.weight.detach().contiguous())
        add(out, f"q.{i}.out_proj.weight", conv_w(vq.out_proj))
        add(out, f"q.{i}.out_proj.bias",   conv_b(vq.out_proj))

    # Decoder head (depthwise): model[0] dw conv, model[1] pw conv.
    add(out, "dec.in.dw.weight", conv_w(dm[0])); add(out, "dec.in.dw.bias", conv_b(dm[0]))
    add(out, "dec.in.pw.weight", conv_w(dm[1])); add(out, "dec.in.pw.bias", conv_b(dm[1]))

    # 4 DecoderBlocks at model[2..5] (attn_window_size is null -> no LocalMHA).
    for b in range(4):
        export_decoder_block(out, f"dec.block.{b}.", dm[2 + b])

    # Tail: Snake1d model[6], WNConv1d model[7], Tanh model[8].
    add(out, "dec.out.snake.alpha", alpha(dm[6]))
    add(out, "dec.out.conv.weight", conv_w(dm[7]))
    add(out, "dec.out.conv.bias",   conv_b(dm[7]))

    save_file(out, dst)
    print(f"wrote {len(out)} tensors -> {dst}")

    # Optional parity fixture: decode a clip's codes with the reference and dump
    # both the codes and the reference PCM for the C++ byte-parity check.
    if "--fixture" in sys.argv:
        import numpy as np
        from snac.snac import SNAC as _S  # noqa
        wav_path = sys.argv[sys.argv.index("--fixture") + 1]
        import torchaudio
        wav, sr = torchaudio.load(wav_path)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        wav = wav.mean(0, keepdim=True).unsqueeze(0)   # [1,1,T]
        # Disable the stochastic NoiseBlock so the reference decode is
        # DETERMINISTIC and byte-comparable to the C++ zero-noise decode.
        import types
        for m in model.decoder.modules():
            if type(m).__name__ == "NoiseBlock":
                m.forward = types.MethodType(lambda self, x: x, m)
        with torch.inference_mode():
            codes = model.encode(wav)
            pcm = model.decode(codes).squeeze().cpu().numpy().astype(np.float32)
        base = dst.rsplit("/", 1)[0] if "/" in dst else "."
        json.dump([c.squeeze(0).cpu().tolist() for c in codes],
                  open(base + "/snac_fixture_codes.json", "w"))
        pcm.tofile(base + "/snac_fixture_pcm.f32")
        print(f"fixture: {[c.numel() for c in codes]} codes, {pcm.size} pcm samples "
              f"(noise disabled in ref -> byte-parity vs C++ zero-noise decode)")


if __name__ == "__main__":
    main()
