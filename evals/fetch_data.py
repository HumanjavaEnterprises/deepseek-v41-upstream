#!/usr/bin/env python3
"""Fetch the eval data via the Hugging Face datasets-server (stdlib only) and draw the fixed-seed samples.
Writes wikitext-2-test.txt, gsm8k-250.json and mmlu-500.json next to this file."""
import json, os, random, time, urllib.request
OUT = os.path.dirname(os.path.abspath(__file__))
def rows(ds, cfg, split):
    out, off = [], 0
    while True:
        u = f"https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length=100"
        for _ in range(5):
            try:
                d = json.load(urllib.request.urlopen(u, timeout=60)); break
            except Exception:
                time.sleep(3)
        r = [x["row"] for x in d["rows"]]; out += r; off += 100
        if len(r) < 100:
            return out
w = rows("Salesforce/wikitext", "wikitext-2-raw-v1", "test")
open(os.path.join(OUT, "wikitext-2-test.txt"), "w").write("".join(x["text"] for x in w))
g = rows("openai/gsm8k", "main", "test"); random.Random(42).shuffle(g)
json.dump(g[:250], open(os.path.join(OUT, "gsm8k-250.json"), "w"))
m = rows("cais/mmlu", "all", "test"); random.Random(42).shuffle(m)
json.dump(m[:500], open(os.path.join(OUT, "mmlu-500.json"), "w"))
print("wrote wikitext-2-test.txt, gsm8k-250.json, mmlu-500.json")
