# Evals: ISTA GSQ-RCO IQ2_XXS (this repack) vs unsloth UD-IQ1_M

Two upstream-format quants of DeepSeek-V4.1-Flash-1017 in the same size class (~138 vs ~140 GB resident), run
under identical conditions.

## Results

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
```
Per-question results for both runs are in `evals/results/`.

## Caveats
- Small samples: 250 and 500 questions. Differences of 1-2 points are noise.
- One prompt format, temperature 0, thinking off. Other prompting may change absolute scores.
- Perplexity on wikitext-2 can favor quants whose calibration data resembles Wikipedia. We don't know either
  quant's calibration set.
