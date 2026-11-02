# Build your own: any DeepSeek-V4.1-Flash checkpoint → a one-card GGUF

Commands below run from the repo root.

The [repack](../README.md#repack-it-yourself) only re-containers ISTA's finished quant. This path starts from **full
BF16 weights** (a fine-tune, an abliterated model, a merge) and ends with a quant in ISTA's GSQ-RCO size class that
runs on one 96 GB card + RAM under stock llama.cpp. It is how we built
[DeepSeek-V4.1-Flash-1017-obliterated-upstream-GGUF](https://huggingface.co/HumanjavaEnterprises/DeepSeek-V4.1-Flash-1017-obliterated-upstream-GGUF)
from [OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated](https://huggingface.co/OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated).

Everything below ran on 1× RTX PRO 6000 (96 GB), a 16-core Xeon w5-3525, 256 GB DDR5 and NVMe/RAID storage, with
upstream llama.cpp `de5122d8`.

| step | tool | runs on | time (ours) | disk |
|---|---|---|---|---|
| 1. download BF16 safetensors | `hf` | network | — | 1.5 TB |
| 2. convert to BF16 GGUF | `build/tools/convert-ds41-hf.py` | CPU | 2 h 35 min | 1.4 TB |
| 3. bootstrap quant (no imatrix) | `llama-quantize` | CPU | ~2.5 h | 232 GB |
| 4. calibration text | `build/calibration/` | CPU | 30 s (+6 GB download) | 3 MB |
| 5. imatrix | `llama-imatrix` on the bootstrap quant | 1 GPU + RAM | 2 h 9 min | 1.6 GB |
| 6. fold the imatrix | `build/tools/fold-imatrix-groups.py` | CPU | 1 min | 1.6 GB |
| 7. final quant | `llama-quantize --imatrix` | CPU | ~2.5 h | 232 GB |
| 8. check types | `build/tools/tensor-types.py diff` | CPU | 1 min | — |

Peak disk is about 3.2 TB (download + BF16 GGUF + one quant). Delete the safetensors after step 2 to stay under 2 TB.

## 1. Download
```bash
hf download OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated --revision 29a9f998fe73e25c4f22734f5f312ba22d19e238 --local-dir src/
```

## 2. Convert (HF-transformers layout → BF16 GGUF)
Upstream's converter reads DeepSeek's native checkpoint layout (`layers.N.attn.wq_a.weight`, per-expert `w1/w2/w3`).
Re-exported checkpoints use the transformers layout (`model.layers.N.self_attn.q_a_proj.weight`, stacked
`mlp.experts.gate_up_proj`). `build/tools/convert-ds41-hf.py` runs upstream's converter unchanged and only:
- renames every tensor to its native name. An unknown name stops the run with the full list, never a silent skip;
- streams the stacked experts one at a time and splits `gate_up_proj` into gate (first half) and up (second half);
- rebuilds the Engram table as `weight × weight_scale_inv` (BF16 values × one BF16 scale per 32);
- loads the tokenizer straight from `tokenizer.json` (transformers has no config class for `deepseek_v41_text`).

```bash
python3 -m venv venv && venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu \
  && venv/bin/pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt "transformers>=5"
LLAMA_CPP=llama.cpp venv/bin/python build/tools/convert-ds41-hf.py src/ --outtype bf16 --split-max-size 50G \
  --outfile bf16/model-BF16.gguf            # add --dry-run first: plans all shards, writes nothing
```

## 3. Bootstrap quant
The imatrix has to be measured by running the model, and 1.4 TB of BF16 doesn't run on this box. So we quantize once
with a borrowed imatrix, measure on that build, then quantize again with our own. The bootstrap only supplies
activations, so it doesn't have to be good. We used unsloth's published imatrix for the base model
([unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF),
`imatrix_unsloth.gguf_file` at rev `8a6ec9c`, MIT), folded as in step 6:
```bash
llama-quantize --imatrix unsloth.folded.gguf --tensor-type-file build/recipes/ds41-gsq-rco-typemap.txt --keep-split \
  bf16/model-BF16-00001-of-00026.gguf boot/model-boot.gguf IQ2_XXS 16
```
Without any imatrix, llama-quantize refuses IQ1/IQ2 types. We haven't tried that route. A bootstrap at a type that
needs no imatrix (Q2_K, ~250 GB) should work as well.

**The type map** (`build/recipes/ds41-gsq-rco-typemap.txt`) gives every tensor ISTA-DASLab's GSQ-RCO type. That means
attention at Q8_0/Q5_K, routed experts at a per-layer IQ1_M / IQ2_XXS / IQ2_XS mix, Engram at IQ4_NL, and the
indexer, compressor and hyper-connection tensors kept at BF16/F32. We only changed the router, which stays F32.
Regenerate it from any reference quant:
```bash
GGUF_PY=llama.cpp/gguf-py python3 build/tools/tensor-types.py map <reference-00001-of-N.gguf> --keep 'ffn_gate_inp=f32' > typemap.txt
```
Two traps we hit:
- **Anchor every pattern** (`^name$`). `llama-quantize` matches by regex search, so `token_embd\.weight` also
  matched `per_layer_token_embd.weight` and pushed the 105 GB Engram table to Q5_K (+23 GB).
- **List the high-precision tensors too.** A map built only from the reference's *quantized* tensors leaves the
  BF16/F32 ones to the base type: 107 small tensors silently became IQ2_XXS. That build still answered sanity
  prompts correctly, so only step 8 catches this.

## 4. Calibration text
710k tokens of permissively licensed text: 25% code in 10 languages, 20% coding-agent and chat transcripts, 20% prose,
15% math, 11% multilingual, 9% JSON/YAML/markdown. Only train splits are used, so wikitext, GSM8K and MATH test sets
stay clean for evals. Sources, revisions and licenses are listed in [calibration/SOURCES.md](calibration/SOURCES.md).
```bash
cd build/calibration && python3 fetch.py raw && python3 -I build_calibration.py raw ../../src/tokenizer.json calibration.txt
md5sum calibration.txt && cd ../..   # e89e9a932b225f4f38cb57c9ef1fcef0
```
Swap in your own data the same way, for example your team's code. More variety covers more of the 384 experts.

## 5. imatrix
Use the same placement as serving (experts of layers 25-39 in RAM):
```bash
llama-imatrix -m boot/model-boot-00001-of-00026.gguf -f build/calibration/calibration.txt -o imatrix.gguf \
  -c 2048 -b 2048 -ub 2048 -ngl 999 -fa on -t 16 \
  -ot 'blk\.(2[5-9]|3[0-9])\.ffn_(up|down|gate)_exps=CPU'
```
351 chunks at ~20 s each. Check expert coverage afterwards. Ours reached **all 15,360 experts** (384 × 40 layers),
each with at least 11 tokens.

## 6. Fold the imatrix
`llama-imatrix` records the grouped output projection `attn_output_a` as 8 groups × 4096 = 32768 values, but
`llama-quantize` wants one per column (4096) and aborts with
`imatrix size 32768 is different from tensor size 4096`. Every V4.1 imatrix has this, including unsloth's.
Average the groups:
```bash
GGUF_PY=llama.cpp/gguf-py python3 build/tools/fold-imatrix-groups.py imatrix.gguf bf16/model-BF16-00001-of-00026.gguf imatrix.folded.gguf
```

## 7. Final quant
```bash
llama-quantize --imatrix imatrix.folded.gguf --tensor-type-file build/recipes/ds41-gsq-rco-typemap.txt --keep-split \
  bf16/model-BF16-00001-of-00026.gguf out/model-IQ2_XXS.gguf IQ2_XXS 16
```
Expect `quant size = 237451 MiB (2.66 BPW)`. **Judge the run by its exit code and that line, not by the shard
count.** `--keep-split` creates all 26 shard files up front, so a run that died at tensor 128 looked complete.

## 8. Check types against the reference
```bash
GGUF_PY=llama.cpp/gguf-py python3 build/tools/tensor-types.py diff <reference-00001-of-N.gguf> out/model-IQ2_XXS-00001-of-00026.gguf
```
The only expected difference is the router (`ffn_gate_inp`: reference BF16, ours F32). Then serve it with the
[README command](../README.md#run-it) and run [the evals](../EVALS.md).

Tests for the converter's name mapping and the imatrix fold: `GGUF_PY=llama.cpp/gguf-py python3 -m unittest discover build/tests`.
