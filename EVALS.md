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
