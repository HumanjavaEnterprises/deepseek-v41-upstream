#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Download the calibration sources listed in SOURCES.md at their pinned revisions into raw/.

  python3 fetch.py [RAW_DIR]      (~6 GB; needs huggingface_hub)
"""
import sys
from huggingface_hub import HfApi, hf_hub_download

RAW = sys.argv[1] if len(sys.argv) > 1 else "raw"
MATH = "algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus".split()
CODE = "python javascript typescript rust go c c++ shell sql java json yaml markdown toml dockerfile".split()
WIKI = [("zh", "00006"), ("es", "00013"), ("fr", "00017"), ("de", "00020"), ("ja", "00015"), ("ru", "00021"),
        ("ko", "00003"), ("ar", "00007"), ("hi", "00002"), ("vi", "00004")]
SOURCES = [  # (dataset, revision prefix, files) — train splits only, so eval sets stay clean
    ("openai/gsm8k", "740312add88f", ["main/train-00000-of-00001.parquet"]),
    ("EleutherAI/hendrycks_math", "21a5633873b6", [f"{s}/train-00000-of-00001.parquet" for s in MATH]),
    ("Salesforce/wikitext", "b08601e04326", ["wikitext-103-raw-v1/train-00000-of-00002.parquet"]),
    ("OpenAssistant/oasst2", "179dd21fc551", ["2023-11-05_oasst2_ready.trees.jsonl.gz"]),
    ("glaiveai/glaive-function-calling-v2", "e7f4b6456019", ["glaive-function-calling-v2.json"]),
    ("SWE-bench/SWE-smith-trajectories", "08e109b4a59e", ["data/tool-00004-of-00008.parquet"]),
    ("bigcode/commitpackft", "fc56fe33c030", [f"data/{l}/data.jsonl" for l in CODE]),
    ("wikimedia/wikipedia", "b04c8d1ceb2f", [f"20231101.{l}/train-00000-of-{n}.parquet" for l, n in WIKI]),
]

api = HfApi()
for repo, rev, files in SOURCES:
    full = api.dataset_info(repo, revision=rev).sha
    if not full.startswith(rev):
        raise SystemExit(f"{repo}: revision {rev} resolved to {full}")
    for f in files + ["README.md"]:
        hf_hub_download(repo, f, repo_type="dataset", revision=full, local_dir=f"{RAW}/{repo.replace('/', '__')}")
        print(repo, full[:12], f, flush=True)
