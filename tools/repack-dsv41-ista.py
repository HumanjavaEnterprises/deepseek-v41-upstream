#!/usr/bin/env python3
"""repack-dsv41-ista.py — repack ISTA-DASLab's DeepSeek-V4.1-Flash-1017 GSQ-RCO GGUFs (Strata layout) into the
layout UPSTREAM llama.cpp (ggml-org, deepseek41 arch) loads. No third-party engine involved: pure gguf-py.

No requantization: every weight is copied bit-exact except
  - blk.N.engram_q + engram_k (BF16)   -> blk.N.engram_gate (F32) = q * k   (upstream conversion/deepseek41.py rule)
  - blk.N.engram_wkv                   -> blk.N.engram_kv                   (rename)
  - blk.N.attn_output_a [d, r, g]      -> [d, r*g]                          (same bytes, upstream's 2-D shape)
  - engram_{multipliers,offsets,primes,token_map} tensors -> deepseek41.ple.* / engram.* KV arrays
    (offsets made absolute over the concatenated table, as upstream does)
  - the separate Engram file's two tables -> ONE per_layer_token_embd (rows concatenated in layer order),
    which upstream reads lazily (TENSOR_READ_LAZY) — it can stay on disk.
Key renames follow upstream's names. --check-against <unsloth shard 1> asserts every derived array equals the
values unsloth's upstream-format conversion of the same model holds (tokenizer map, primes, offsets, multipliers).

  repack-dsv41-ista.py --src DIR --out DIR/name.gguf [--check-against us1.gguf] [--dry-run]
--check-against is optional: the four key types upstream needs are built in (UPSTREAM_TYPES).
Needs upstream llama.cpp's gguf-py (de5122d8 or newer, the first with deepseek41):
  GGUF_PY=/path/to/llama.cpp/gguf-py python3 repack-dsv41-ista.py ...     (or `pip install` that gguf-py)
"""
import argparse, glob, os, sys
import numpy as np

if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.path.expanduser(os.environ["GGUF_PY"]))
import gguf                                    # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGUFValueType as VT, GGMLQuantizationType as QT  # noqa: E402

A = "deepseek41"
RENAME = {
    f"{A}.kv_source_layer_ids": f"{A}.attention.kv_source_layers",
    f"{A}.index_source_layer_ids": f"{A}.attention.indexer.source_layers",
    f"{A}.attention.candidate_source_layer_id": f"{A}.attention.indexer.candidate_source_layer",
    f"{A}.attention.candidate_topk_blocks": f"{A}.attention.indexer.candidate_top_k_blocks",
    f"{A}.attention.candidate_block_size": f"{A}.attention.indexer.candidate_block_size",
    f"{A}.engram.layer_ids": f"{A}.ple.layers",
    f"{A}.engram_max_ngram_size": f"{A}.ple.ngram_size",
    f"{A}.engram_n_heads": f"{A}.ple.heads_per_ngram",
    f"{A}.engram_head_dim": f"{A}.embedding_length_per_layer_input",
    f"{A}.engram_pad_token_id": f"{A}.engram.pad_token_id",
    f"{A}.engram_vocab_size": f"{A}.engram.vocab_size",
    f"{A}.engram_compressed_vocab_size": f"{A}.engram.compressed_vocab_size",
}
DROP = {f"{A}.engram.present", f"{A}.engram.num_embeddings", f"{A}.engram.file", f"{A}.engram.source",
        f"{A}.dspark_target_layer_ids", f"{A}.expert_scoring_func",       # no drafter shipped; scoring = gating_func 4
        "split.no", "split.count", "split.tensors.count", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
# value types upstream's deepseek41 loader requires where ISTA's header differs (found by --check-against unsloth's
# upstream-format conversion, rev 60ad365); applied always, so no reference file is needed
UPSTREAM_TYPES = {
    f"{A}.attention.indexer.candidate_source_layer": (VT.INT32, None),
    f"{A}.attention.kv_source_layers": (VT.ARRAY, VT.UINT32),
    f"{A}.attention.indexer.source_layers": (VT.ARRAY, VT.UINT32),
    f"{A}.ple.layers": (VT.ARRAY, VT.UINT32),
}
TABLES = ("engram_multipliers.weight", "engram_offsets.weight", "engram_primes.weight", "engram_token_map.weight")


class Streamed:
    """Raw bytes written chunk by chunk (the Engram table is ~100 GB): never held in memory."""
    def __init__(self, byte_shape, parts):
        self.shape, self.dtype, self.parts = tuple(byte_shape), np.dtype(np.uint8), parts
        self.nbytes = int(np.prod(byte_shape))

    def tofile(self, fout):
        n = 0
        for p in self.parts:
            for i in range(0, p.shape[0], 1 << 20):          # 1M rows per chunk
                c = np.ascontiguousarray(p[i:i + (1 << 20)]); c.tofile(fout); n += c.nbytes
        if n != self.nbytes:
            raise ValueError(f"streamed {n} bytes, expected {self.nbytes}")


def bf16(t):
    return (np.asarray(t.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def shards(src, pat):
    s = sorted(glob.glob(os.path.join(src, pat)))
    if not s:
        sys.exit(f"repack: no {pat} in {src}")
    return s


def derived(tables, kv):
    mult = np.asarray(tables["engram_multipliers.weight"].data).reshape(-1)
    primes = np.asarray(tables["engram_primes.weight"].data).reshape(-1)
    tmap = np.asarray(tables["engram_token_map.weight"].data).reshape(-1)
    offs = np.concatenate([[0], np.cumsum(primes)[:-1]])
    ista_offs = np.asarray(tables["engram_offsets.weight"].data).reshape(len(kv[f"{A}.engram.layer_ids"]), -1)
    # ISTA's offsets are per layer (each restarts at 0); upstream's are absolute over the concatenated table
    for li, row in enumerate(ista_offs):
        p = primes.reshape(ista_offs.shape)[li]
        if list(row) != [int(x) for x in np.concatenate([[0], np.cumsum(p)[:-1]])]:
            raise ValueError(f"engram layer {li}: ISTA offsets are not the per-layer prime prefix sums")
    sums = primes.reshape(ista_offs.shape).sum(axis=1)
    if [int(x) for x in sums] != list(kv[f"{A}.engram.num_embeddings"]):
        raise ValueError("engram prime sums != engram.num_embeddings")
    return {
        f"{A}.ple.layer_multipliers": ([int(x) for x in mult], VT.UINT64),
        f"{A}.ple.head_offsets": ([int(x) for x in offs], VT.UINT64),
        f"{A}.ple.head_vocab_sizes": ([int(x) for x in primes], VT.UINT64),
        f"{A}.engram.token_map": ([int(x) for x in tmap], VT.INT32),
        f"{A}.embedding_length_out": (kv[f"{A}.embedding_length"] * kv[f"{A}.hyper_connection.count"], VT.UINT32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True, help="output path; shards get -0000N-of-0000M")
    ap.add_argument("--check-against", help="unsloth upstream-format shard 1 (header only is fine)")
    ap.add_argument("--split-gib", type=int, default=46)
    ap.add_argument("--dry-run", action="store_true", help="derive + check metadata only, write nothing")
    a = ap.parse_args()

    readers = [GGUFReader(p) for p in shards(a.src, "*-IQ*-0000?-of-0000?.gguf")]
    r0 = readers[0]
    kv = {k: f.contents() for k, f in r0.fields.items()}
    tens = {t.name: t for r in readers for t in r.tensors}
    tables = {n: tens.pop(n) for n in TABLES}
    d = derived(tables, kv)

    if a.check_against:
        ref = {k: f.contents() for k, f in GGUFReader(a.check_against).fields.items()} \
            if os.path.getsize(a.check_against) < (1 << 30) else None
        bad = 0
        for k, (v, _) in d.items():
            rv = ref.get(k)
            if (list(rv) if isinstance(rv, list) else rv) != v:
                print(f"MISMATCH {k}: ours {str(v)[:80]} vs ref {str(rv)[:80]}"); bad += 1
        for old, new in RENAME.items():
            if kv.get(old) != ref.get(new):
                print(f"MISMATCH {old}->{new}: {kv.get(old)} vs {ref.get(new)}"); bad += 1
        if bad:
            sys.exit(f"repack: {bad} derived keys disagree with the reference — not writing")
        print(f"repack: all {len(d) + len(RENAME)} derived/renamed keys match the reference")

    if a.dry_run and not glob.glob(os.path.join(a.src, "*Engram*.gguf")):
        print("repack: dry-run — Engram file not on disk yet, metadata checked only"); return 0
    reftypes = dict(UPSTREAM_TYPES)
    if a.check_against:
        for k, f in GGUFReader(a.check_against).fields.items():
            reftypes[k] = (f.types[0], f.types[-1] if f.types[0] == VT.ARRAY else None)
        for k, f in r0.fields.items():
            n = RENAME.get(k, k)
            if k in DROP or k in d or n not in reftypes:
                continue
            ours = (f.types[0], f.types[-1] if f.types[0] == VT.ARRAY else None)
            if ours != reftypes[n]:
                print(f"repack: type fix {n}: {ours} -> {reftypes[n]}")
            if n in UPSTREAM_TYPES and reftypes[n] != UPSTREAM_TYPES[n]:
                sys.exit(f"repack: {n}: reference type {reftypes[n]} != built-in {UPSTREAM_TYPES[n]}")
    eng = GGUFReader(shards(a.src, "*Engram*.gguf")[0])
    etabs = sorted((t for t in eng.tensors if t.name.endswith("engram_embd.weight")),
                   key=lambda t: int(t.name.split(".")[1]))
    if [int(t.name.split(".")[1]) for t in etabs] != list(kv[f"{A}.engram.layer_ids"]):
        sys.exit("repack: Engram file layers != backbone engram.layer_ids")
    if kv.get(f"{A}.engram.source") != eng.fields[f"{A}.engram.source"].contents():
        sys.exit("repack: Engram file was made for a different backbone (engram.source differs)")
    rows = sum(int(t.shape[1]) for t in etabs)
    print(f"repack: per_layer_token_embd {etabs[0].tensor_type.name} [{int(etabs[0].shape[0])}, {rows}]")
    if a.dry_run:
        print("repack: dry-run — nothing written"); return 0

    w = GGUFWriter(a.out, A, split_max_size=a.split_gib << 30)
    for k, f in r0.fields.items():
        if k in DROP or k in d or k.startswith("GGUF."):
            continue
        name = RENAME.get(k, k)
        if name == "general.architecture":
            continue                                              # GGUFWriter writes it
        vt, sub = reftypes.get(name, (f.types[0], f.types[-1] if f.types[0] == VT.ARRAY else None))
        w.add_key_value(name, f.contents(), vt, sub_type=sub)
    w.add_key_value("general.repacked_by", "deepseek-v41-upstream repack-dsv41-ista.py", VT.STRING)
    for k, (v, t) in d.items():
        if k in reftypes:
            t = reftypes[k][1] if isinstance(v, list) else reftypes[k][0]
        if isinstance(v, list):
            w.add_key_value(k, v, VT.ARRAY, sub_type=t)
        else:
            w.add_key_value(k, v, t)

    for name, t in tens.items():
        raw = np.asarray(t.data)
        if name.endswith(".engram_k.weight"):
            continue
        if name.endswith(".engram_q.weight"):
            k = tens[name.replace("engram_q", "engram_k")]
            w.add_tensor(name.replace("engram_q", "engram_gate"), (bf16(t) * bf16(k)).reshape(raw.shape[:-1] + (-1,)))   # raw is BF16 bytes
            continue
        if name.endswith(".engram_wkv.weight"):
            name = name.replace("engram_wkv", "engram_kv")
        if name.endswith(".attn_output_a.weight") and raw.ndim == 3:
            raw = raw.reshape(raw.shape[0] * raw.shape[1], raw.shape[2])   # [g, r, bytes] -> [g*r, bytes]
        if t.tensor_type in (QT.F32, QT.F16, QT.I32, QT.I64):
            w.add_tensor(name, raw)
        else:
            w.add_tensor(name, raw, raw_dtype=t.tensor_type)
    bpr = np.asarray(etabs[0].data).shape[1]
    w.add_tensor("per_layer_token_embd.weight",
                 Streamed((rows, bpr), [np.asarray(t.data) for t in etabs]), raw_dtype=etabs[0].tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    print(f"repack: wrote {a.out} (+shards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
