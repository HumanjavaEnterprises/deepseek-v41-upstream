#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""fold-imatrix-groups.py — make an imatrix usable by llama-quantize for DeepSeek-V4.1 grouped tensors.

llama-imatrix records the grouped output projection (`attn_output_a`, applied per head group) as
n_group x ne0 values (32768 = 8 x 4096), but llama-quantize expects one value per column of the
stored tensor (ne0 = 4096) and aborts:  "imatrix size 32768 is different from tensor size 4096".
This folds each such entry to ne0 by averaging the groups (each group sees the same tokens, so the
mean keeps the scale). Every other entry is copied unchanged.

  fold-imatrix-groups.py IN.imatrix.gguf MODEL-00001-of-N.gguf OUT.imatrix.gguf
Needs numpy and gguf-py: GGUF_PY=/path/to/llama.cpp/gguf-py (or pip install gguf).
"""
import glob, os, re, sys

if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
import numpy as np   # noqa: E402
import gguf          # noqa: E402

SUMS = (".in_sum2", ".in_sum")


def model_ne0(first_shard):
    """{tensor name: ne0} over every shard of a split GGUF."""
    pat = re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", str(first_shard))
    out = {}
    for f in sorted(glob.glob(pat)) or [first_shard]:
        for t in gguf.GGUFReader(f).tensors:
            out[t.name] = int(t.shape[0])
    return out


def main(src, model, dst):
    ne0 = model_ne0(model)
    r = gguf.GGUFReader(src)
    w = gguf.GGUFWriter(dst, arch=None)
    for f in r.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture":
            continue
        val = f.contents()
        if f.types[0] == gguf.GGUFValueType.ARRAY:
            w.add_key_value(f.name, val, f.types[0], sub_type=f.types[-1])
        else:
            w.add_key_value(f.name, val, f.types[0])
    folded = 0
    for t in r.tensors:
        data = np.array(t.data).reshape(-1)
        shape = [int(x) for x in reversed(t.shape)]          # numpy order of the stored ggml shape
        base = next((t.name[:-len(s)] for s in SUMS if t.name.endswith(s)), None)
        n = ne0.get(base) if base else None
        if n and len(shape) == 1 and data.size != n:
            if data.size % n:
                raise SystemExit(f"{base}: {data.size} values do not fold onto ne0={n}")
            data, shape = data.reshape(-1, n).mean(axis=0).astype(data.dtype), [n]
            folded += 1
        w.add_tensor(t.name, data.reshape(shape))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"folded {folded} grouped sums -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
