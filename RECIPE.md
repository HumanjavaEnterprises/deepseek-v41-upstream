# Tuning recipe: DeepSeek-V4.1-Flash on one GPU + RAM

How we got from ~10 to ~40 decode tok/s on one 96 GB card. All numbers come from upstream llama.cpp `de5122d8` with
CUDA, on 1× RTX PRO 6000 Blackwell Max-Q, a Xeon w5-3525 (16 cores / 32 threads) and 256 GB DDR5. Throughput was
measured with `llama-bench` (pp4096 / tg128) on an idle CPU unless noted.

## The model in one paragraph
V4.1-Flash is a 748B mixture-of-experts. Each token reads only a few experts, so decode is **bound by memory
bandwidth**: every routed-expert byte that lives in system RAM costs RAM bandwidth on every token. The game is to
put as many expert layers on the GPU as fit, and to make the CPU side as efficient as possible.

## The sweep

| config | prefill | decode |
|---|---|---|
| all experts on CPU, 8 threads | 70 (pp512) | 11.8 |
| all experts on CPU, **16 threads** | 70 (pp512) | **18.5** (24 or 32 threads: 15.3) |
| experts of layers 0-9 on GPU | 91 (pp512) | 21.8 |
| 0-15 on GPU | 105 (pp512) | 26.2 |
| 0-21 on GPU | 123 (pp512) | 31.9 |
| 0-24 on GPU | 145 / 363 (ub 512 / **2048**) | 33.5 |
| 0-26 on GPU | 168 / 422 | 36.3 |
| 0-27 on GPU | 182 / **448** | **39-40** |

## Rules that fell out
1. **Threads: bench physical cores vs nearly all hyperthreads.** For this IQ2_XXS quant, 16 beat 24 and 32: the
   extra threads fight over the same memory bandwidth.
   It depends on the quant, though: on the same box, DeepSeek-V4-Flash at unsloth UD-Q4_K_XL (all experts in RAM)
   ran **30 threads at 23.4 tok/s vs 19.6 at 16**. Heavier-to-decode formats can use the extra threads.
2. **Every expert layer you move onto the GPU helps**, roughly linearly. Fill the card, but leave room for KV cache
   and compute buffers.
3. **`-ub 2048` gives about 2.5x prefill** when experts are split across CPU and GPU (bigger batches per expert
   fetch), and **3.2x** (51.6 -> 166 tok/s pp4096) with all experts in RAM on V4-Flash Q4_K_XL, for ~0.5 GB more
   VRAM. Decode is unchanged.
4. **Mind the real peak, not the file size.** Layers 0-27 at 64k context peaked at **96.1 GB** of VRAM, which is
   too tight on a 96 GB card. Layers 0-24 peak at **85.9 GB**, which is what we run.
5. **A cold cache lies.** Our first partial-offload test showed 1.8 tok/s: 30 threads with the CPU busy. Bench on an
   idle box, warm.

## Speculative decoding (bench: 16k ctx, temp 0)

| config | code edit | prose | drafts accepted |
|---|---|---|---|
| layers 0-25 on card, no drafting | 33.5 | 33.2 | — |
| + `--spec-type ngram-mod` | **40.2 (+20%)** | 34.2 | 100% code |
| + `ngram-simple` | 36.0 | 33.6 | 100% code |
| layers 0-22 on card + DSpark drafter (unsloth Q8_0, 8 GB), n_max 3 | 39.9 | **24.0 (-22%)** | 95% code / 41% prose |
| + DSpark, n_max 5 | 37.5 | 19.4 | 85% / 32% |

- **Use `ngram-mod`.** It needs no VRAM, never slows prose, and gives +20% on edits and refactors (code agents
  rewrite what they just read).
- **Skip the DSpark drafter when experts live in RAM.** Verifying a draft batch pulls more RAM-resident experts per
  step, and prose acceptance can't pay for it. It may win when the whole model is on GPUs.

## Expert-offload pattern
Experts of layers N-39 go to the CPU:
```
-ot 'blk\.(2[5-9]|3[0-9])\.ffn_(up|down|gate)_exps=CPU'     # layers 25-39 in RAM
```
If your launcher refuses regex brackets, the same 45 tensors as a comma list:
```
-ot 'blk.25.*_exps=CPU,blk.26.*_exps=CPU,blk.27.*_exps=CPU,blk.28.*_exps=CPU,blk.29.*_exps=CPU,blk.3..ffn.*_exps=CPU'
```

## Ideas we haven't tried yet
- `--load-mode none` vs mmap (llama.cpp warns that mmap with CPU overrides is slower).
- Splitting across two 96 GB cards so no experts are in RAM (estimated 70+ tok/s).
