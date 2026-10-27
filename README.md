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

## Why this one

| | size | runs on upstream llama.cpp |
|---|---|---|
| ISTA GSQ-RCO IQ2_XXS (original) | 129 GiB + 103 GB Engram file | ❌ needs Strata |
| unsloth UD-IQ1_M (upstream format) | ~250 GB, Engram embedded | ✅, but needs ~250 GB of RAM + VRAM |
| **this repack** | 4 shards, 248 GB on disk, **~130 GB resident** | ✅ |

The Engram table (103 GB) becomes one `per_layer_token_embd` tensor, which upstream reads **lazily from disk**. It
stays on the SSD, so the part that has to fit in RAM + VRAM is about 130 GB.

## Run it

You need upstream llama.cpp **`de5122d8` (2026-10-27) or newer**, the first version with `deepseek41`.

**One 96 GB card + RAM** (what we run: 64k context, experts of layers 0-24 on the card, 25-39 in RAM):
```bash
llama-server -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-upstream-IQ2_XXS-00001-of-00004.gguf \
  -c 65536 -np 1 -ngl 99 -t 16 -ub 2048 --spec-type ngram-mod \
  -ot 'blk\.(2[5-9]|3[0-9])\.ffn_(up|down|gate)_exps=CPU' \
  --chat-template-kwargs '{"enable_thinking":false}'
```

**Smaller card:** keep all routed experts in RAM (~125 GB of RAM):
```bash
llama-server -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-upstream-IQ2_XXS-00001-of-00004.gguf \
  -c 16384 -ngl 99 -t <physical cores> -ot 'exps=CPU'
```

See [RECIPE.md](RECIPE.md) for the full tuning sweep and what each flag is worth.

## Verified

The published files were loaded with exactly the command above (stock llama.cpp `de5122d8`, one 96 GB GPU). They
answered 4/4 sanity prompts correctly at temperature 0: a fact, arithmetic, code and science. Every quantized tensor
is byte-identical to ISTA's. All 17 derived metadata keys match unsloth's independent upstream conversion of the
same model.

Check your download:
```bash
cd ds41/ && sha256sum -c SHA256SUMS
```

## Benchmarks

Hardware: 1× RTX PRO 6000 Blackwell Max-Q (96 GB), Xeon w5-3525 (16 cores), 256 GB DDR5, NVMe. Upstream llama.cpp
`de5122d8`, CUDA 12.9, sm_120.

| config | prefill tok/s | decode tok/s |
|---|---|---|
| all experts in RAM, 16 threads | 70 | 18.5 |
| experts of layers 0-21 on card | 123 | 31.9 |
| experts of layers 0-27 on card, `-ub 2048` | **448** | **39-40** |
| layers 0-24 on card + `ngram-mod`, 64k ctx (fits a 96 GB card with headroom) | ~250 | 37.8 code edits / 32 prose |

Sanity check: 4/4 correct on a small fact, arithmetic, code and science set. This is a smoke test, not a quality
benchmark. For quality, see ISTA's model card.

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
- This repo: MIT, by [@vveerrgg](https://github.com/vveerrgg) / Humanjava Enterprises.
