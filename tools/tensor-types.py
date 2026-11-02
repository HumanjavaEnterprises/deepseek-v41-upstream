#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""tensor-types.py — copy a reference quant's per-tensor types, and check a new quant against it.

  tensor-types.py map  REF-00001-of-N.gguf [--keep NAME=TYPE ...] > typemap.txt
      one anchored `^name$=type` line per tensor with 2+ dimensions, for
      `llama-quantize --tensor-type-file`. Anchors matter: an unanchored `token_embd\\.weight`
      also matches `per_layer_token_embd.weight`. Tensors the reference kept at BF16/F32 are
      listed too, otherwise llama-quantize gives them the low-bit default type.
  tensor-types.py diff REF-00001-of-N.gguf NEW-00001-of-N.gguf
      every tensor whose type differs (exit 1 if any). Run it before you ship a quant.

Needs gguf-py: GGUF_PY=/path/to/llama.cpp/gguf-py (or pip install gguf).
"""
import argparse, collections, glob, os, re, sys

if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
import gguf  # noqa: E402


def shards(first):
    pat = re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", first)
    return sorted(glob.glob(pat)) or [first]


def types(first):
    """{name: (type, n_dims > 1)} over all shards."""
    out = {}
    for f in shards(first):
        for t in gguf.GGUFReader(f).tensors:
            out[t.name] = (t.tensor_type.name.lower(), sum(1 for x in t.shape if x > 1))
    return out


def cmd_map(a):
    keep = dict(k.split("=", 1) for k in a.keep)
    for name, (ty, nd) in sorted(types(a.ref).items()):
        if nd < 2:
            continue                       # norms / biases stay as converted
        for pat, kty in keep.items():
            if re.search(pat, name):
                ty = kty
        print(f"^{re.escape(name)}$={ty}")


def cmd_diff(a):
    ref, new = types(a.ref), types(a.new)
    bad = collections.Counter()
    for name, (ty, _) in ref.items():
        got = new.get(name, ("missing",))[0]
        if got != ty:
            bad[(re.sub(r"blk\.\d+\.", "blk.N.", name), ty, got)] += 1
    for (name, ty, got), n in sorted(bad.items()):
        print(f"{n:4d} x {name}: reference {ty}, new {got}")
    extra = sorted(set(new) - set(ref))
    if extra:
        print(f"{len(extra)} tensors only in the new file, e.g. {extra[:3]}")
    print("types match" if not bad else f"{sum(bad.values())} tensors differ")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map")
    m.add_argument("ref")
    m.add_argument("--keep", nargs="*", default=[], help="REGEX=TYPE overrides, e.g. 'ffn_gate_inp=f32'")
    d = sub.add_parser("diff")
    d.add_argument("ref")
    d.add_argument("new")
    a = ap.parse_args()
    return cmd_map(a) if a.cmd == "map" else cmd_diff(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
