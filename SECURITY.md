# Security Policy

This project publishes two things: a **GGUF repack tool** (`tools/repack-dsv41-ista.py`) and **model weights** on
Hugging Face. People run them on machines that hold API keys, SSH keys and private data, so we take integrity and
supply-chain reports seriously and handle them with priority.

> **Use the latest version of the tool, and verify every download** (`sha256sum -c SHA256SUMS`). Fixes ship on
> `main` and in the Hugging Face repo's history.

## Reporting a vulnerability

**Please report security issues privately. Do not open a public GitHub issue.**

- Email **security@humanjava.com** with details and reproduction steps.
- You can also use GitHub's **private vulnerability reporting** (Security tab → "Report a vulnerability").

Please include:
- A clear description and the impact (what an attacker could do).
- Steps to reproduce, or a proof of concept.
- The tool commit, the Hugging Face revision, and the llama.cpp commit you used.

### What to expect
- Acknowledgement within **3 business days**.
- An initial assessment and severity within **7 business days**.
- Coordinated disclosure: we'll agree a timeline with you before any public detail, and credit you (if you wish)
  once a fix ships.

## Supported versions

| Component | Supported |
|---|---|
| `tools/repack-dsv41-ista.py` on `main` | ✅ |
| Weights at the latest Hugging Face revision | ✅ |
| Older commits or revisions | ❌ (please update) |

## In scope
- **Weight integrity:** published files that differ from what the card and `SHA256SUMS` describe, or that were not
  produced by the documented repack of ISTA-DASLab's files.
- **Embedded content:** anything in the GGUF metadata beyond the model (unexpected strings, payloads), including
  the chat template.
- **The repack tool:** code execution, path traversal, or unbounded resource use when it parses a crafted GGUF;
  any network access or file writes outside `--out`.
- **Supply chain:** this repo's CI, dependencies, or release process.

## Out of scope
- Vulnerabilities in llama.cpp itself (report to [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/security)),
  in ISTA-DASLab's quantization, or in DeepSeek's model.
- Model *behavior*: jailbreaks, harmful or incorrect output, prompt injection. These are properties of the base model.
- Reports generated solely by automated scanners without a demonstrated impact.

## Good-faith safe harbor
We will not pursue or support legal action against researchers who act in good faith, avoid privacy violations
and service disruption, and give us reasonable time to remediate before public disclosure.

## What's in the box (verify it yourself)

**The tool is small and offline.** One file, about 200 lines. It imports only the Python standard library
(`argparse`, `glob`, `os`, `sys`), `numpy`, and llama.cpp's own `gguf-py`. It makes **no network calls** and runs
**no subprocesses**. It reads `--src` (and `--check-against`) and writes only the files under `--out`. With
`--dry-run` it writes nothing. You can read it in a few minutes.

**The weights are data, not code.** GGUF has no pickle and no executable sections. The tool copies every quantized
tensor **byte for byte** from ISTA-DASLab's files. It computes exactly one tensor per Engram layer
(`engram_gate = engram_q × engram_k`, upstream llama.cpp's own conversion rule), and it rewrites metadata as
documented in the README.

**Provenance chain:**
1. `deepseek-ai/DeepSeek-V4.1-Flash-1017` (MIT)
2. → ISTA-DASLab GSQ-RCO GGUF, revision `653a9874aa38c356233c5f3cc8c96b19a43b57eb`
3. → this tool
4. → the Hugging Face files, with `SHA256SUMS` published beside them.

You can rerun the repack yourself with the commands in the README and compare. The quantized tensors must match
exactly. Every derived key was also checked against unsloth's independent upstream conversion of the same model
(17/17 equal).

**No third-party inference engine.** The whole point of this project is to run on stock upstream llama.cpp, which
you build from source, rather than on a fork or a closed engine.

## If we have a security incident

This is a small open-source project, not a 24/7 security operation. Here is what we commit to, so you can hold us
to it.

**If we confirm a problem with published weights or the tool, we will:**
- Publish an advisory **at or before** the fix, not after.
- State plainly which Hugging Face revisions and tool commits are affected, with dates, so you can tell whether
  your download is.
- Replace or remove affected files and publish new `SHA256SUMS`.
- Publish a technical write-up within **14 days** of the fix, even if it is unflattering or incomplete (marked as
  such).

**What we cannot promise.** We do not run a bug bounty and cannot compensate for losses. We cannot reach people
who downloaded and never return. We announce on this repo, on the Hugging Face repo's discussions, and on our
social channels, and ask the community to amplify.

**What we ask of you.** Give us a reasonable window before public disclosure. **But if a flaw is being actively
exploited, publish immediately.** User safety beats our coordination.
