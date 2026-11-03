# deepseek-v41-upstream

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-%E2%89%A5%20de5122d8-orange)](https://github.com/ggml-org/llama.cpp)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-GGUF%20weights-yellow)](https://huggingface.co/HumanjavaEnterprises/DeepSeek-V4.1-Flash-1017-GSQ-RCO-upstream-GGUF)

**Run ISTA-DASLab's DeepSeek-V4.1-Flash GSQ-RCO quant (748B) on stock upstream llama.cpp.** You don't need
Strata, a fork, or any third-party engine.

ISTA-DASLab's [DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
is a very good 2-bit quant, but it ships in Strata's layout and upstream llama.cpp refuses it:

```
error loading model: array key not found: deepseek41.attention.kv_source_layers
```

`tools/repack-dsv41-ista.py` rewrites it into the layout upstream's `deepseek41` architecture expects. It does
**not requantize**: every quantized weight is copied byte for byte.

**Ready to run:** the repacked GGUF is on Hugging Face (`hf download … --local-dir ds41/`):
[HumanjavaEnterprises/DeepSeek-V4.1-Flash-1017-GSQ-RCO-upstream-GGUF](https://huggingface.co/HumanjavaEnterprises/DeepSeek-V4.1-Flash-1017-GSQ-RCO-upstream-GGUF)

**Build your own from BF16:** [build/](build/README.md) takes any DeepSeek-V4.1-Flash checkpoint (a fine-tune, an
abliterated model, a merge) to a one-card GGUF in the same size class, with our own imatrix and ISTA's type map.
The first build made this way is
[HumanjavaEnterprises/DeepSeek-V4.1-Flash-1017-obliterated-upstream-GGUF](https://huggingface.co/HumanjavaEnterprises/DeepSeek-V4.1-Flash-1017-obliterated-upstream-GGUF)
(abliterated: refusals reduced, not removed; capability matches the base quant. What it is and isn't, including what we haven't verified: [EVALS.md](EVALS.md#what-this-build-is-and-isnt)).

## Why this one

| | quantization | on disk | in RAM + VRAM | stock llama.cpp |
|---|---|---|---|---|
| ISTA GSQ-RCO IQ2_XXS (original) | GSQ-RCO | 129 GiB + 103 GB Engram file | ~138 GB | ❌ needs Strata |
| unsloth UD-IQ1_M | Unsloth Dynamic | 250 GB | ~140 GB | ✅ |
| **this repack** | **GSQ-RCO (ISTA's, untouched)** | 248 GB | **~138 GB** | ✅ |

Same size class as unsloth's smallest upstream quant. The difference is the quantization: this is **ISTA-DASLab's
GSQ-RCO**, now runnable on stock llama.cpp. Our head-to-head is below and in [EVALS.md](EVALS.md).

In both upstream-format files the 103 GB Engram table is read **lazily from disk** (`per_layer_token_embd`), so
it stays on your SSD. Only ~138 GB has to fit in RAM + VRAM.

## Run it

You need upstream llama.cpp **`de5122d8` (2026-10-27) or newer** (the first with `deepseek41`); **2026-11-03 or newer** for the fast command below.

**One 96 GB card + ~130 GB of free RAM** (what we run, 2026-11-03: **66-72 tok/s decode**, ~590 tok/s prefill warm).
Every routed expert lives in RAM and `--moe-cache-mib` keeps the most-used ones in VRAM. Needs llama.cpp from
2026-11-03 or newer (we use `ca8daad1`, which has the [#30944](https://github.com/ggml-org/llama.cpp/issues/30944) prompt-cache fix):
```bash
llama-server -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-upstream-IQ2_XXS-00001-of-00004.gguf \
  -c 32768 -np 1 -ngl 99 -t 16 -ub 2048 -fa on \
  -ot '_exps=CPU' --moe-cache-mib 80000 -lm none \
  --chat-template-kwargs '{"enable_thinking":false}'
```
Decode climbs over the first few hundred tokens while the cache fills. On a smaller card, shrink `--moe-cache-mib`.

**Juice it: + DSpark speculative decoding (code ~80-100 tok/s).** Add unsloth's DSpark drafter (MIT, 8 GB) on the
same card and trim the cache to make room:
```bash
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf --local-dir dspark/
# add to the command above (and use --moe-cache-mib 72000 instead of 80000):
  --spec-type draft-dspark -md dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf -ngld 99 --spec-draft-n-max 2
```
Measured 2026-11-03 on one 96 GB card: code **80-102 tok/s** (+30-45%, ~78-84% of drafts accepted), prose about the
same (68-74). Drafting 3 tokens was slower than 2. ngram-mod gave nothing with the cache on. Measured on the
abliterated build; same architecture and drafter here. Speculative output at temperature 0 isn't bit-identical to
plain decoding ([#31472](https://github.com/ggml-org/llama.cpp/issues/31472)), so run evals with it off.

**No cache / older llama.cpp:** keep all routed experts in RAM (~125 GB of RAM, ~18 tok/s):
```bash
llama-server -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-upstream-IQ2_XXS-00001-of-00004.gguf \
  -c 16384 -ngl 99 -t <physical cores> -ub 2048 -ot 'exps=CPU'
```

See [RECIPE.md](RECIPE.md) for the full tuning sweep and what each flag is worth.

## Verified

The published files were loaded on stock llama.cpp `de5122d8` with one 96 GB GPU. They
answered 4/4 sanity prompts correctly at temperature 0: a fact, arithmetic, code and science. Every quantized tensor
is byte-identical to ISTA's. All 17 derived metadata keys match unsloth's independent upstream conversion of the
same model.

Check your download:
```bash
cd ds41/ && sha256sum -c SHA256SUMS
```

## Quality vs unsloth and full precision

Head-to-head against unsloth's UD-IQ1_M (same size class), identical setup. Details and raw data in
[EVALS.md](EVALS.md).

| | ISTA GSQ-RCO (this) | unsloth UD-IQ1_M | DeepSeek API (full precision) |
|---|---|---|---|
| wikitext-2 PPL | 3.507 ± 0.040 | **3.333 ± 0.037** | n/a |
| GSM8K (250) | 97.2% | 96.4% | 96.8% (all three tie) |
| MMLU (500) | 84.4% | 84.6% | **87.0%** (beats both quants, p < 0.05) |
| decode tok/s | **31.9** | 24.3 | n/a |

Same task accuracy, ~31% faster decode. unsloth's quant has lower perplexity. Against the full-precision model
(DeepSeek's API), math is intact and both 2-bit quants give up ~2.5 points of MMLU knowledge recall.

## Benchmarks

Hardware: 1× RTX PRO 6000 Blackwell Max-Q (96 GB), Xeon w5-3525 (16 cores), 256 GB DDR5, NVMe. Upstream llama.cpp
`de5122d8`, CUDA 12.9, sm_120.

| config | prefill tok/s | decode tok/s |
|---|---|---|
| all experts in RAM, 16 threads | 70 | 18.5 |
| experts of layers 0-21 on card | 123 | 31.9 |
| experts of layers 0-27 on card, `-ub 2048` (short ctx only; 96.1 GB peak at 64k) | **448** | 39-40 |
| layers 0-24 on card + `ngram-mod`, 64k ctx (85.9 GB peak, `de5122d8`) | ~250 | 37.8 code edits / 32 prose |
| **all experts in RAM + `--moe-cache-mib 80000 -lm none` (`ca8daad1`, 2026-11-03)** | **~590** | **66-72** |

Sanity check: 4/4 correct on a small fact, arithmetic, code and science set. This is a smoke test, not a quality
benchmark. For quality, see [EVALS.md](EVALS.md).

## Repack it yourself

```bash
# 1. ISTA's files (pin a revision; we used 653a987)
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF --revision 653a9874aa38c356233c5f3cc8c96b19a43b57eb --local-dir ista/
# 2. repack: metadata first (--dry-run writes nothing), then for real (~250 GB of free disk)
GGUF_PY=/path/to/llama.cpp/gguf-py python3 tools/repack-dsv41-ista.py --src ista/ --out out/DeepSeek-V4.1-Flash-1017-GSQ-RCO-upstream-IQ2_XXS.gguf --dry-run
GGUF_PY=/path/to/llama.cpp/gguf-py python3 tools/repack-dsv41-ista.py --src ista/ --out out/DeepSeek-V4.1-Flash-1017-GSQ-RCO-upstream-IQ2_XXS.gguf
```
Optional `--check-against <unsloth upstream-format shard 1>` asserts every derived key and type equals unsloth's
own upstream conversion of the same model. All 17 matched for us.

Unit tests cover the metadata math (no model files needed):
`GGUF_PY=/path/to/llama.cpp/gguf-py python3 -m unittest discover tests`

### What the repack changes
- **Key renames** to upstream names: `kv_source_layer_ids` → `attention.kv_source_layers`, `engram_*` →
  `ple.*` / `engram.*`, and so on (see `RENAME` in the script).
- **Value types**: four keys need u32/i32 to match upstream's loader (`UPSTREAM_TYPES`).
- **`engram_gate = engram_q × engram_k`** (BF16 → F32), the same rule as upstream's `conversion/deepseek41.py`.
- **`engram_wkv` → `engram_kv`**; `attn_output_a` 3-D → 2-D (same bytes).
- **Engram metadata tensors → KV arrays**: multipliers, primes, token map, and offsets made absolute over the joined
  table.
- **Engram file → one lazily-read `per_layer_token_embd`**, streamed in chunks so it is never held in memory.

## Security
See [SECURITY.md](SECURITY.md): how to report issues, what the tool does and doesn't touch, and how to
verify the weights' provenance.

## Credits & license
- Model: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT).
- Quantization: [ISTA-DASLab](https://huggingface.co/ISTA-DASLab) GSQ-RCO (MIT). All the quality is theirs; this
  repo only changes the container.
- Runtime: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
- Obliterated build: weights by [OBLITERATUS](https://huggingface.co/OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated)
  (refusal-direction ablation after [Arditi et al. 2024](https://arxiv.org/abs/2406.11717)). The bootstrap imatrix is
  [unsloth's](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (MIT). The calibration data's sources and
  licenses are in [build/calibration/SOURCES.md](build/calibration/SOURCES.md).
- This repo: MIT, by [@vveerrgg](https://github.com/vveerrgg) / Humanjava Enterprises.
