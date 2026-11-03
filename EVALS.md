# Evals: ISTA GSQ-RCO IQ2_XXS (this repack) vs unsloth UD-IQ1_M vs the full-precision API

Two upstream-format quants of DeepSeek-V4.1-Flash-1017 in the same size class (~138 vs ~140 GB resident), run
under identical conditions, plus DeepSeek's hosted API as the full-precision reference.

## Full-precision reference (DeepSeek API)

The same questions, prompts and settings (temp 0, thinking off) sent to DeepSeek's hosted API, model
`deepseek-flash`, which reports itself as "DeepSeek-V4.1-Flash" (2026-10-27).

| | ISTA GSQ-RCO (this) | unsloth UD-IQ1_M | **DeepSeek API (full precision)** |
|---|---|---|---|
| GSM8K (250) | 243 (97.2%) | 241 (96.4%) | 242 (96.8%) |
| MMLU (500) | 422 (84.4%) | 423 (84.6%) | **435 (87.0%)** |

- **Math survived 2-bit compression intact.** On GSM8K both quants match the full model (p = 1.0 for each).
- **Knowledge recall lost ~2.5 points.** On MMLU the full model beats both quants: it alone got 23 questions right vs
  ISTA's 10 (p = 0.035), and 21 vs unsloth's 9 (p = 0.043). Both quants lose about the same amount, which fits
  unsloth's lower perplexity not turning into better task scores.
- **Caveats:** the API doesn't confirm it serves exactly the `-1017` release, and hosted serving may differ from a
  local run in ways we can't see. On the first API pass, 4 GSM8K questions came back as empty replies. They were
  re-asked and all 4 answered correctly. The harness now retries empty replies, and those rows are marked in
  `evals/results/deepseek-api-flash.jsonl`. The API can't score perplexity: it returns probabilities only for text
  it generates.

## Quant vs quant

| | **ISTA GSQ-RCO IQ2_XXS** (this repack) | **unsloth UD-IQ1_M** (rev `60ad365`) | read |
|---|---|---|---|
| wikitext-2 perplexity (150 × 512-token chunks), lower is better | 3.507 ± 0.040 | **3.333 ± 0.037** | unsloth better |
| GSM8K, 250 questions (fixed seed) | **243 (97.2%)** | 241 (96.4%) | tie (paired sign test p = 0.69) |
| MMLU, 500 questions (fixed seed) | 422 (84.4%) | **423 (84.6%)** | tie (p = 1.0) |
| decode speed during the evals | **31.9 tok/s** | 24.3 tok/s | ISTA **+31%** |

**In one line:** the same task accuracy, ISTA's quant decodes ~31% faster on the same hardware, and unsloth's tracks
the original model's token distribution more closely (lower perplexity).

### How to read it
- **Perplexity** is the most sensitive measure of quantization damage. A gap of 0.17 is well outside both error
  bars, so unsloth's quant is measurably closer to the full model on this text.
- **The task scores don't separate the two.** On GSM8K, ISTA alone got 4 questions right and unsloth alone got 2.
  On MMLU the split was 18 vs 19. Neither is distinguishable from chance at these sample sizes.
- **Speed** comes from summing every generated token and its decode time in the server logs across all eval
  answers. Same build, same offload, same threads. We haven't profiled why. The likely cause is how fast the
  CPU-side kernels handle each quant type for the experts that live in RAM.

## Method
- **Hardware:** 1× RTX PRO 6000 Blackwell Max-Q (96 GB), Xeon w5-3525 (16 cores), 256 GB DDR5, NVMe.
- **Software:** upstream llama.cpp `de5122d8`, CUDA 12.9.
- **Identical for both:** experts of layers 25-39 in RAM, the rest on GPU; `-ngl 99 -t 16 -ub 2048`; temp 0; no
  speculative decoding; thinking off.
- **Perplexity:** `llama-perplexity -f wikitext-2-raw test -c 512 --chunks 150`.
- **GSM8K:** 250 test questions (shuffle seed 42). Prompt: "Solve step by step, then finish with a line
  'Answer: <number>'." 768 max tokens. Exact numeric match.
- **MMLU:** 500 test questions across all subjects (shuffle seed 42). Prompt: question + A-D, "Answer with the
  letter only." Exact letter match.

## Reproduce
```bash
python3 evals/fetch_data.py                       # wikitext-2, GSM8K, MMLU samples (fixed seeds)
LLAMA_BIN=/path/to/llama.cpp/build/bin evals/run-model.sh <tag> <model-shard-00001.gguf>
# the API reference (any OpenAI-compatible endpoint):
EVAL_URL=https://api.deepseek.com/chat/completions EVAL_MODEL=deepseek-flash EVAL_KEY_FILE=<key file> \
  python3 evals/evals.py - deepseek-api-flash
```
Per-question results for both runs are in `evals/results/`.

## Caveats
- Small samples: 250 and 500 questions. Differences of 1-2 points are noise.
- One prompt format, temperature 0, thinking off. Other prompting may change absolute scores.
- Perplexity on wikitext-2 can favor quants whose calibration data resembles Wikipedia. We don't know either
  quant's calibration set.

## The obliterated build (our imatrix), 2026-11-02

[DeepSeek-V4.1-Flash-1017-obliterated-upstream-GGUF](https://huggingface.co/HumanjavaEnterprises/DeepSeek-V4.1-Flash-1017-obliterated-upstream-GGUF):
[OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated](https://huggingface.co/OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated)
quantized from BF16 with ISTA's type map and our own imatrix ([build/](build/README.md)). Same harness, same flags.

| | **obliterated, our imatrix** | ISTA GSQ-RCO (base model) |
|---|---|---|
| wikitext-2 perplexity | 3.638 ± 0.040 | 3.507 ± 0.040 |
| GSM8K (250) | 243 (97.2%) | 243 (97.2%) |
| MMLU (500) | 418 (83.6%) | 422 (84.4%) |
| refusal spot check (20 prompts) | 8 flagged, **7 real** | 19 |
| decode speed during the evals | 8.8 tok/s (median per answer 19.9) | 31.9 tok/s |

- **Capability held.** GSM8K is identical and MMLU is within noise (4 questions). Perplexity is ~4% higher. This
  compares a different model (abliterated weights) against the base model, so it mixes ablation cost and quant cost.
- **Refusals are reduced, not gone.** `evals/refusal.py` flags replies by keyword. Of the 8 flagged, the ransomware
  prompt was a false positive (it complied with a disclaimer). The 7 real ones are soft refusals ("I can't give a
  step-by-step guide, but here is how it works"): lockpicking, hotwiring, exam cheating, a jury-duty lie, money
  laundering, a manipulative message, slurs. We haven't run the same check on the BF16 weights, so part of this may
  come from quantization. Per-prompt flags: `evals/results/*.refusal.json` (reply text not published).
- **Speed: solved (sweep 2026-11-03).** The 8.8 tok/s above came from the eval harness, not the model: llama.cpp
  `de5122d8` had the prompt-cache bug [#30944](https://github.com/ggml-org/llama.cpp/issues/30944), and a few long
  answers looped. A clean rerun on `ca8daad1` (speculation off, `cache_prompt:false`, one 96 GB card) puts this
  build level with ISTA's (llama-bench tg128: 28.7 vs 28.8 at 8K depth, flat out to 32K).

#### Speed on one 96 GB card (2026-11-03)

llama-server `ca8daad1`, `-ngl 99 -t 16 -ub 2048 -fa on -c 32768`, chat endpoint, temperature 0.7, 1500 generated
tokens, three code prompts (explain a 4K-token file, write an LRU module, refactor a 1.7K-token file), two runs each.

| placement | decode tok/s | prefill 4K tok/s (warm) | host RSS |
|---|---|---|---|
| experts of layers 25-39 in RAM (old) | 20.7-26.7 | ~300 | ~48 GB |
| all experts in RAM + `--moe-cache-mib 80000` | 52-64 | ~360 | 126 GiB |
| same + `-lm none` | 63-74 | ~580 | 126 GiB |
| **same, cache 72000 + DSpark drafter (`--spec-draft-n-max 2`)** | **code 80-102, prose 68-74** | ~500 | 127 GiB |

`--moe-cache-mib` (upstream [#29887](https://github.com/ggml-org/llama.cpp/pull/29887)) keeps an LRU of the most
used experts in VRAM, so a card that can't hold all experts still serves most tokens from VRAM. Decode climbs over the
first few hundred tokens while it fills. A small cache on top of the old split was slower (20-21 tok/s), so give it
the VRAM the experts would have used. With the cache on, ngram-mod adds nothing, DSpark drafting 2 tokens adds 30-45% on code, and drafting 3 is slower than 2. Evals stay speculation-off ([#31472](https://github.com/ggml-org/llama.cpp/issues/31472)).

### What this build is and isn't

**What carries over from OBLITERATUS's weights**
- This is a quant of [OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated](https://huggingface.co/OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated)
  (rev `29a9f99`) with the refusal direction already ablated. We did no ablation, fine-tuning or merging of our own.
- On our 20-prompt check, refusals fall from 19 (the base model in ISTA's quant, whose type map we reuse) to 7. GSM8K and MMLU match the base quant.

**What we haven't verified**
- **Whether it's as uncensored as the full-precision original.** Quantizing to ~2.6 bits per weight can partly
  restore behavior that an ablation removed. We haven't measured OBLITERATUS's BF16 weights (they don't fit on our
  hardware), so we can't say how much of the remaining 7 soft refusals come from their ablation and how much from our
  quantization.
- **Any refusal or safety claims OBLITERATUS makes.** We haven't reproduced them.
- **Breadth.** 20 English prompts, temperature 0, thinking off, a keyword heuristic read by hand. That isn't a
  benchmark. Other topics, languages, long multi-turn chats or thinking mode may behave differently.

**What it is not**
- Not safety-tuned, filtered or moderated. It will produce content a stock model refuses, and it may still refuse
  some requests. You are responsible for how you use it.
- Not a claim that abliteration preserves every capability. We measured math (GSM8K), knowledge (MMLU) and
  perplexity, nothing else.

**What we plan to check next**, with results added here as they land:
1. Refusals on a higher-bit quant of the same weights, to separate quantization loss from the ablation itself.
2. If quantization is the cause: keep the most affected layers at higher precision and requantize.
3. A larger refusal set and a KL-divergence check against a higher-precision reference.
