#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Humanjava Enterprises
"""Rebuild calibration.txt (llama-imatrix calibration text) from the raw sources
listed in SOURCES.md. Deterministic: fixed seed, pinned source revisions.

Usage:  python -I build_calibration.py RAW_DIR TOKENIZER_JSON OUT_TXT
Needs:  pyarrow, tokenizers.  Raw files are fetched by scripts/fetch.py + fetch2.py.

Each emitted chunk is 512..2048 tokens (DeepSeek-V4.1 tokenizer), chunks are
shuffled and separated by a blank line.  Chat-like data is rendered with the
model's own role tokens (<｜User｜>, <｜Assistant｜>, </think>, <｜end▁of▁sentence｜>).
"""
import difflib, gzip, json, os, random, re, sys
import pyarrow.parquet as pq
from tokenizers import Tokenizer

SEED = 20261101
RAW, TOK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
tok = Tokenizer.from_file(TOK)
U, A, E, T, S = "<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>", "</think>", "<｜System｜>"
PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0"}
MIN_T, MAX_T = 512, 2048
SEP = "\n\n"

def P(*a): return os.path.join(RAW, *a)
def ntok(s): return len(tok.encode(s, add_special_tokens=False).ids)

def window(text, rng, lo=600, hi=1900):
    """Cut one random token-aligned window of lo..hi tokens out of a long doc."""
    enc = tok.encode(text, add_special_tokens=False)
    n = len(enc.ids)
    L = rng.randint(lo, hi)
    if n <= L:
        return text
    st = rng.randint(0, n - L)
    a, b = enc.offsets[st][0], enc.offsets[st + L - 1][1]
    return text[a:b]

def take(docs, budget, rng, pack_sep="\n\n"):
    """docs: iterator of strings (already shuffled). Long docs -> one window;
    short docs are packed together until >= MIN_T. Returns list of chunks."""
    out, used, buf, bt = [], 0, [], 0
    for d in docs:
        d = d.strip()
        if not d or "\n\n\n\n\n" in d[:200]:
            continue
        n = ntok(d)
        if n > MAX_T:
            d = window(d, rng); n = ntok(d)
        if n >= MIN_T:
            ch, cn = d, n
        else:
            if bt + n > MAX_T:
                buf, bt = [], 0  # drop an under-filled pack rather than overflow
            buf.append(d); bt += n + 1
            if bt < MIN_T:
                continue
            ch = pack_sep.join(buf); cn = ntok(ch); buf, bt = [], 0
            if cn > MAX_T:
                continue
        if not (MIN_T <= cn <= MAX_T):
            continue
        out.append(ch); used += cn
        if used >= budget:
            break
    return out, used

def shuffled(seq, rng):
    seq = list(seq); rng.shuffle(seq); return seq

# ---------------- sources ----------------
CPF_CODE = ["python", "javascript", "typescript", "rust", "go", "c", "c++", "shell", "sql", "java"]
CPF_STRUCT = ["json", "yaml", "markdown", "toml", "dockerfile"]

def cpf_rows(lang, rng, k=4000):
    rows = []
    with open(P("bigcode__commitpackft", "data", lang, "data.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["license"] in PERMISSIVE:
                rows.append(d)
    return shuffled(rows, rng)[:k]

def cpf_file(d):
    return f"{d['new_file']}\n{d['new_contents']}"

def cpf_diff(d):
    diff = "".join(difflib.unified_diff(
        d["old_contents"].splitlines(True), d["new_contents"].splitlines(True),
        fromfile="a/" + d["old_file"], tofile="b/" + d["new_file"]))
    if not diff:
        return ""
    return (f"{U}Make this change to the repository: {d['subject']}\n"
            f"{A}{T}Here is the patch.\n\n```diff\n{diff}```\n\nCommit message:\n{d['message']}{E}")

def wikitext_articles():
    t = pq.read_table(P("Salesforce__wikitext", "wikitext-103-raw-v1", "train-00000-of-00002.parquet"))
    arts, cur = [], []
    for line in t.column("text").to_pylist():
        if re.match(r"^ = [^=].* = \n$", line) and cur:
            arts.append("".join(cur)); cur = []
        cur.append(line)
    arts.append("".join(cur))
    fix = lambda s: (s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
                      .replace(" , ", ", ").replace(" . ", ". ").replace(" 's", "'s"))
    return [fix(a) for a in arts if len(a) > 3000]

def swe_smith(rng):
    t = pq.read_table(P("SWE-bench__SWE-smith-trajectories", "data", "tool-00004-of-00008.parquet"),
                      columns=["messages"])
    out = []
    for m in t.column("messages").to_pylist():
        msgs = json.loads(m)
        parts = []
        for x in msgs:
            c = x.get("content")
            if isinstance(c, list):
                c = "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
            c = c or ""
            if x.get("tool_calls"):
                c += "\n" + json.dumps([tc.get("function", tc) for tc in x["tool_calls"]], ensure_ascii=False)
            r = x["role"]
            if r == "system": parts.append(S + c)
            elif r == "user": parts.append(U + c)
            elif r == "assistant": parts.append(A + T + c + E)
            else: parts.append(U + "[tool output]\n" + c)
        out.append("".join(parts))
    return shuffled(out, rng)

def glaive(rng):
    g = json.load(open(P("glaiveai__glaive-function-calling-v2", "glaive-function-calling-v2.json"), encoding="utf-8"))
    out = []
    for d in shuffled(g, rng)[:6000]:
        chat = d["chat"].replace("<|endoftext|>", "")
        segs = re.split(r"\n*(USER|ASSISTANT|FUNCTION RESPONSE): ", chat)
        s = S + d["system"].replace("SYSTEM: ", "", 1)
        for i in range(1, len(segs) - 1, 2):
            role, c = segs[i], segs[i + 1].strip()
            if role == "USER": s += U + c
            elif role == "ASSISTANT": s += A + T + c + E
            else: s += U + "[function response]\n" + c
        out.append(s)
    return out

def oasst_threads(rng):
    """Walk each tree along top-ranked replies -> one multi-turn thread per tree."""
    en, other = [], []
    with gzip.open(P("OpenAssistant__oasst2", "2023-11-05_oasst2_ready.trees.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            node = json.loads(line)["prompt"]
            lang, s = node["lang"], ""
            while node:
                s += (U if node["role"] == "prompter" else A + T) + node["text"] + (E if node["role"] == "assistant" else "")
                reps = [r for r in node.get("replies", []) if not r.get("deleted")]
                reps.sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 0))
                node = reps[0] if reps else None
            (en if lang == "en" else other).append(s)
    return shuffled(en, rng), shuffled(other, rng)

def gsm8k(rng):
    t = pq.read_table(P("openai__gsm8k", "main", "train-00000-of-00001.parquet")).to_pylist()
    return [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in shuffled(t, rng)]

MATH_SUBJ = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
             "number_theory", "prealgebra", "precalculus"]
def hendrycks(subj, rng):
    t = pq.read_table(P("EleutherAI__hendrycks_math", subj, "train-00000-of-00001.parquet")).to_pylist()
    return [f"Problem ({r['type']}, {r['level']}):\n{r['problem']}\n\nSolution:\n{r['solution']}" for r in shuffled(t, rng)]

WIKI = [("zh", "00006"), ("es", "00013"), ("fr", "00017"), ("de", "00020"), ("ja", "00015"),
        ("ru", "00021"), ("ko", "00003"), ("ar", "00007"), ("hi", "00002"), ("vi", "00004")]
def wiki(lang, n, rng):
    pf = pq.ParquetFile(P("wikimedia__wikipedia", f"20231101.{lang}", f"train-00000-of-{n}.parquet"))
    rg = sorted(rng.sample(range(pf.num_row_groups), min(3, pf.num_row_groups)))
    rows = pf.read_row_groups(rg, columns=["title", "text"]).to_pylist()
    return [f"{r['title']}\n\n{r['text']}" for r in shuffled(rows, rng) if len(r["text"]) > 1500]

# ---------------- plan (token budgets) ----------------
def main():
    rng = random.Random(SEED)
    plan = []  # (category, subsource, docs, budget)
    for lang in CPF_CODE:
        plan.append(("code", f"commitpackft/{lang} files", [cpf_file(d) for d in cpf_rows(lang, rng)], 17500))
    plan.append(("agent", "SWE-smith tool trajectories", swe_smith(rng), 62000))
    plan.append(("agent", "glaive function-calling v2", glaive(rng), 30000))
    diffs = [cpf_diff(d) for lang in CPF_CODE for d in cpf_rows(lang, rng, 600)]
    plan.append(("agent", "commitpackft commit->diff", shuffled(diffs, rng), 26000))
    en, other = oasst_threads(rng)
    plan.append(("agent", "oasst2 en threads", en, 22000))
    plan.append(("prose", "wikitext-103", shuffled(wikitext_articles(), rng), 140000))
    plan.append(("math", "gsm8k train", gsm8k(rng), 45000))
    for s in MATH_SUBJ:
        plan.append(("math", f"MATH/{s}", hendrycks(s, rng), 8600))
    for lang, n in WIKI:
        plan.append(("multilingual", f"wikipedia/{lang}", wiki(lang, n, rng), 5000))
    plan.append(("multilingual", "oasst2 non-en threads", other, 20000))
    for lang in CPF_STRUCT:
        plan.append(("structured", f"commitpackft/{lang} files", [cpf_file(d) for d in cpf_rows(lang, rng)], 14000))

    chunks, stats = [], {}
    for cat, sub, docs, budget in plan:
        ch, used = take(iter(docs), budget, rng)
        chunks += ch
        stats.setdefault(cat, {})[sub] = (len(ch), used)
    rng.shuffle(chunks)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(SEP.join(c.strip("\n") for c in chunks) + "\n")
    total = sum(u for c in stats.values() for _, u in c.values())
    print(f"chunks={len(chunks)} tokens={total} bytes={os.path.getsize(OUT)}")
    for cat, subs in stats.items():
        ct = sum(u for _, u in subs.values())
        print(f"{cat:13s} {ct:8d} {100*ct/total:5.1f}%")
        for sub, (n, u) in subs.items():
            print(f"    {sub:34s} chunks={n:4d} tokens={u}")

if __name__ == "__main__":
    main()
