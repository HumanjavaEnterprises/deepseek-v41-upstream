#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""convert-ds41-hf.py — convert a DeepSeek-V4.1-Flash checkpoint saved in the HF-transformers layout (e.g. a
re-exported / edited BF16 model: model.layers.N.mlp.experts.gate_up_proj stacked, engram_tables + weight_scale_inv)
with UPSTREAM llama.cpp's converter (conversion/deepseek41.py), which only reads DeepSeek's native layout.

Our own adapter; upstream code runs unchanged. It only:
  - renames HF tensor names to the native names upstream maps (layers.N.attn.wq_a.weight, ...)
  - writes routed experts from the stacked BF16 tensors (gate_up_proj split gate|up; down_proj) as BF16
    ffn_{gate,up,down}_exps, streamed one expert at a time (upstream writes them as MXFP4 from per-expert w1/w2/w3)
  - writes the Engram table from BF16 values x BF16 per-32 scale (upstream: e4m3 x e8m0) as one BF16 table

  convert-ds41-hf.py SRC_DIR --outfile OUT.gguf [--split-max-size 50G] [any convert_hf_to_gguf.py flag]
Needs torch, transformers >= 5 and llama.cpp's convert requirements. LLAMA_CPP=/path/to/llama.cpp (>= de5122d8,
the first tree with conversion/deepseek41.py).
"""
import json, os, re, sys
from pathlib import Path

LCPP = Path(os.path.expanduser(os.environ.get("LLAMA_CPP", "llama.cpp")))
ROOT = {"model.embed_tokens.weight": "embed.weight", "model.norm.weight": "norm.weight", "lm_head.weight": "head.weight"}
LAYER = [  # (HF suffix, native suffix)
    ("attn_hc.base", "hc_attn_base"), ("attn_hc.fn", "hc_attn_fn"), ("attn_hc.scale", "hc_attn_scale"),
    ("ffn_hc.base", "hc_ffn_base"), ("ffn_hc.fn", "hc_ffn_fn"), ("ffn_hc.scale", "hc_ffn_scale"),
    ("input_layernorm.weight", "attn_norm.weight"), ("post_attention_layernorm.weight", "ffn_norm.weight"),
    ("mlp.gate.weight", "ffn.gate.weight"), ("mlp.gate.e_score_correction_bias", "ffn.gate.bias"),
    ("mlp.gate.e_score_correction_bias_vl", "ffn.gate.bias_vl"),
    ("mlp.gate.e_score_correction.bias", "ffn.gate.bias"),           # names as upstream's indexer hands them over
    ("mlp.gate.e_score_correction.bias_vl", "ffn.gate.bias_vl"),
    ("mlp.shared_experts.gate_proj.weight", "ffn.shared_experts.w1.weight"),
    ("mlp.shared_experts.down_proj.weight", "ffn.shared_experts.w2.weight"),
    ("mlp.shared_experts.up_proj.weight", "ffn.shared_experts.w3.weight"),
    ("self_attn.sinks", "attn.attn_sink"),
    ("self_attn.q_a_proj.weight", "attn.wq_a.weight"), ("self_attn.q_a_norm.weight", "attn.q_norm.weight"),
    ("self_attn.q_b_proj.weight", "attn.wq_b.weight"),
    ("self_attn.kv_proj.weight", "attn.wkv.weight"), ("self_attn.kv_norm.weight", "attn.kv_norm.weight"),
    ("self_attn.o_a_proj.weight", "attn.wo_a.weight"), ("self_attn.o_b_proj.weight", "attn.wo_b.weight"),
    ("self_attn.compressor.kv_proj.weight", "attn.compressor.wkv.weight"),
    ("self_attn.compressor.gate_proj.weight", "attn.compressor.wgate.weight"),
    ("self_attn.compressor.kv_norm.weight", "attn.compressor.norm.weight"),
    ("self_attn.indexer.k_proj.weight", "attn.indexer.wk.weight"),
    ("self_attn.indexer.k_norm.weight", "attn.indexer.k_norm.weight"),
    ("self_attn.indexer.q_b_proj.weight", "attn.indexer.wq_b.weight"),
    ("self_attn.indexer.weights_proj.weight", "attn.indexer.weights_proj.weight"),
    ("engram.wkv.weight", "engram.wkv.weight"), ("engram.q_weight", "engram.q_weight"),
    ("engram.k_weight", "engram.k_weight"),
]
LAYER_MAP = dict(LAYER)
STACKED = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


def native_name(hf: str) -> str | None:
    """HF name -> native name; None for tensors written by the stacked/engram writers below."""
    if hf in ROOT:
        return ROOT[hf]
    if hf.startswith("model.engram_tables."):
        return None
    m = re.match(r"model\.layers\.(\d+)\.(.+)$", hf)
    if m is None:
        raise ValueError(f"unmapped tensor {hf!r}")
    if m.group(2) in STACKED:
        return None
    if m.group(2) not in LAYER_MAP:
        raise ValueError(f"unmapped layer tensor {hf!r}")
    return f"layers.{m.group(1)}.{LAYER_MAP[m.group(2)]}"


def patch():
    """Import upstream's converter and patch DeepseekV41Model to read the HF layout. Returns the converter module."""
    global np, torch, gguf, ds41, cast, _ORIG_INDEX
    sys.path[:0] = [str(LCPP), str(LCPP / "gguf-py")]
    import numpy as np
    import torch
    import gguf
    import convert_hf_to_gguf as conv
    from conversion import deepseek41 as ds41
    from typing import cast
    _ORIG_INDEX = ds41.DeepseekV41Model.index_tensors
    _patch_tokenizer()
    for m in ("index_tensors", "_hf_slice", "_write_mxfp4_experts", "_write_engram_tables"):
        setattr(ds41.DeepseekV41Model, m, getattr(HFLayout, m))
    return conv


class HFLayout:
    """Methods patched onto upstream's DeepseekV41Model."""
    def index_tensors(self, remote_hf_model_id=None):
        hf = _ORIG_INDEX(self, remote_hf_model_id=remote_hf_model_id)
        with open(self.dir_model / "model.safetensors.index.json", encoding="utf-8") as f:
            self._hf_files = json.load(f)["weight_map"]
        out, bad = {}, []
        for name, fn in hf.items():
            try:
                n = native_name(name)
            except ValueError:
                bad.append(name)
                continue
            if n is not None:
                out[n] = fn
        if bad:
            raise ValueError("unmapped tensors: " + ", ".join(sorted({re.sub(r"\.\d+\.", ".N.", b) for b in bad})))
        return out

    def _hf_slice(self, name: str):
        from safetensors import safe_open
        f = safe_open(self.dir_model / self._hf_files[name], framework="pt")
        return f, f.get_slice(name)

    def _write_mxfp4_experts(self, bid, proj, key, n_expert=None):
        n_expert = self.hparams["n_routed_experts"] if n_expert is None else n_expert
        hf = f"model.layers.{bid}.mlp.experts." + ("down_proj" if proj == "w2" else "gate_up_proj")
        f, sl = self._hf_slice(hf)
        e, rows, cols = sl.get_shape()
        if e != n_expert:
            raise ValueError(f"{hf}: {e} experts, config says {n_expert}")
        if proj != "w2":
            if rows % 2:
                raise ValueError(f"{hf}: odd gate_up rows {rows}")
            rows //= 2
        r0 = rows if proj == "w3" else 0                      # gate_up = [gate; up] on the row axis

        def chunks():
            for i in range(e):
                t = sl[i:i + 1, r0:r0 + rows, :] if proj != "w2" else sl[i:i + 1]
                yield t.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().reshape(rows, cols * 2)
        data = cast(np.ndarray, ds41._StreamedBytes((e, rows, cols * 2), chunks))
        self.gguf_writer.add_tensor(self.format_tensor_name(key, bid), data, raw_dtype=gguf.GGMLQuantizationType.BF16)

    def _write_engram_tables(self, layer_ids):
        parts, dim = [], None
        for li, bid in enumerate(layer_ids):
            fw, ws = self._hf_slice(f"model.engram_tables.{bid}.weight")
            fs, ss = self._hf_slice(f"model.engram_tables.{bid}.weight_scale_inv")
            rows, d = ws.get_shape()
            if d % 32 or tuple(ss.get_shape()) != (rows, d // 32) or d != (dim or d):
                raise ValueError(f"engram {bid}: expected one scale per 32 elements and the same width per layer")
            if rows != self.hparams["engram_num_embeddings"][li]:
                raise ValueError(f"engram {bid}: {rows} rows, config says {self.hparams['engram_num_embeddings'][li]}")
            dim = d
            parts.append((fw, ws, fs, ss, rows))
        name = self.format_tensor_name(gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD)

        def chunks():
            for _fw, ws, _fs, ss, rows in parts:
                for r0 in range(0, rows, 1 << 18):
                    w = ws[r0:r0 + (1 << 18)].float()
                    s = ss[r0:r0 + (1 << 18)].float().repeat_interleave(32, dim=1)
                    yield (w * s).bfloat16().view(torch.uint8).numpy()
        n_rows = sum(p[4] for p in parts)
        data = cast(np.ndarray, ds41._StreamedBytes((n_rows, dim * 2), chunks))
        self.gguf_writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.BF16)


def _patch_tokenizer():
    """transformers has no config class for deepseek_v41_text: load the fast tokenizer from tokenizer.json."""
    import transformers
    auto = transformers.AutoTokenizer.from_pretrained

    def from_file(path, *a, **kw):
        p = Path(path)
        if (p / "tokenizer.json").is_file() and (p / "config.json").is_file() and \
                json.loads((p / "config.json").read_text()).get("model_type") == "deepseek_v41_text":
            return transformers.PreTrainedTokenizerFast(tokenizer_file=str(p / "tokenizer.json"))
        return auto(path, *a, **kw)
    transformers.AutoTokenizer.from_pretrained = from_file


if __name__ == "__main__":
    patch().main()
