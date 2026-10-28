#!/usr/bin/env python3
"""GSM8K + MMLU against a running llama-server (temp 0, thinking off). usage: evals.py PORT TAG"""
import json, re, sys, time, urllib.request, os
port, tag = sys.argv[1], sys.argv[2]
E = os.path.dirname(os.path.abspath(__file__))
def chat(msg, n):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps({
        "messages": [{"role": "user", "content": msg}], "max_tokens": n, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))["choices"][0]["message"]["content"]
def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None
res = {"tag": tag}; out = open(f"{E}/{tag}.jsonl", "w")
t0 = time.time(); ok = 0; G = json.load(open(f"{E}/gsm8k-250.json"))
for i, x in enumerate(G):
    a = chat(x["question"] + "\n\nSolve step by step, then finish with a line 'Answer: <number>'.", 768)
    m = re.search(r"Answer:\s*(.+)", a); got = num(m.group(1)) if m else num(a)
    want = num(x["answer"].split("####")[-1]); hit = got is not None and abs(got - want) < 1e-6
    ok += hit; out.write(json.dumps({"set": "gsm8k", "i": i, "hit": hit, "got": got, "want": want}) + "\n")
res["gsm8k"] = [ok, len(G)]; print(f"{tag} gsm8k {ok}/{len(G)} ({time.time()-t0:.0f}s)", flush=True)
t0 = time.time(); ok = 0; M = json.load(open(f"{E}/mmlu-500.json"))
for i, x in enumerate(M):
    q = x["question"] + "\n" + "\n".join(f"{c}. {t}" for c, t in zip("ABCD", x["choices"]))
    a = chat(q + "\n\nAnswer with the letter only.", 8)
    m = re.search(r"\b([ABCD])\b", a.strip().upper()); hit = bool(m) and "ABCD".index(m.group(1)) == x["answer"]
    ok += hit; out.write(json.dumps({"set": "mmlu", "i": i, "hit": hit, "got": a.strip()[:8], "want": "ABCD"[x["answer"]]}) + "\n")
res["mmlu"] = [ok, len(M)]; print(f"{tag} mmlu {ok}/{len(M)} ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(f"{E}/{tag}.result.json", "w"))
