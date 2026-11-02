"""Unit tests for build/tools/convert-ds41-hf.py (tensor-name mapping) and build/tools/fold-imatrix-groups.py (synthetic GGUFs).
No model files, torch or transformers needed.
Run: GGUF_PY=/path/to/llama.cpp/gguf-py python3 -m unittest discover tests"""
import importlib.util, os, tempfile, unittest
import numpy as np

TOOLS = os.path.join(os.path.dirname(__file__), "..", "tools")


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(TOOLS, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conv = load("convert_ds41_hf", "convert-ds41-hf.py")
fold = load("fold_imatrix", "fold-imatrix-groups.py")
gguf = fold.gguf


class Names(unittest.TestCase):
    def test_root_and_layer_names_map_to_native(self):
        self.assertEqual(conv.native_name("model.embed_tokens.weight"), "embed.weight")
        self.assertEqual(conv.native_name("lm_head.weight"), "head.weight")
        self.assertEqual(conv.native_name("model.layers.7.self_attn.o_a_proj.weight"), "layers.7.attn.wo_a.weight")
        self.assertEqual(conv.native_name("model.layers.0.mlp.gate.e_score_correction.bias"), "layers.0.ffn.gate.bias")
        self.assertEqual(conv.native_name("model.layers.3.mlp.shared_experts.up_proj.weight"),
                         "layers.3.ffn.shared_experts.w3.weight")

    def test_stacked_experts_and_engram_are_written_separately(self):
        self.assertIsNone(conv.native_name("model.layers.5.mlp.experts.gate_up_proj"))
        self.assertIsNone(conv.native_name("model.layers.5.mlp.experts.down_proj"))
        self.assertIsNone(conv.native_name("model.engram_tables.1.weight"))

    def test_unknown_names_fail_loudly(self):
        for bad in ("model.layers.0.mlp.mystery.weight", "model.something.weight"):
            with self.assertRaises(ValueError):
                conv.native_name(bad)


def write(path, tensors, kv=None):
    w = gguf.GGUFWriter(path, arch=None)
    w.add_string("general.type", "imatrix")
    for k, v in (kv or {}).items():
        w.add_uint32(k, v)
    for name, arr in tensors.items():
        w.add_tensor(name, np.asarray(arr, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


class Fold(unittest.TestCase):
    def test_grouped_sums_fold_to_ne0_and_everything_else_is_copied(self):
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "m.gguf")
            write(model, {"blk.0.attn_output_a.weight": np.zeros((8, 4)),     # ggml ne0 = 4
                          "blk.0.ffn_up_exps.weight": np.zeros((3, 2, 6))})
            groups = np.arange(16, dtype=np.float32)                          # 4 groups x ne0 4
            exps = np.arange(18, dtype=np.float32).reshape(3, 6)              # 3 experts x ne0 6
            src, dst = os.path.join(d, "i.gguf"), os.path.join(d, "o.gguf")
            write(src, {"blk.0.attn_output_a.weight.in_sum2": groups, "blk.0.attn_output_a.weight.in_sum": groups,
                        "blk.0.attn_output_a.weight.counts": [5],
                        "blk.0.ffn_up_exps.weight.in_sum2": exps, "blk.0.ffn_up_exps.weight.counts": [[1], [2], [3]]},
                  kv={"imatrix.chunk_count": 7})
            fold.main(src, model, dst)
            r = gguf.GGUFReader(dst)
            out = {t.name: (np.array(t.data), [int(x) for x in t.shape]) for t in r.tensors}
            np.testing.assert_allclose(out["blk.0.attn_output_a.weight.in_sum2"][0].ravel(),
                                       groups.reshape(4, 4).mean(axis=0))
            self.assertEqual(out["blk.0.attn_output_a.weight.in_sum"][1], [4])
            np.testing.assert_array_equal(out["blk.0.ffn_up_exps.weight.in_sum2"][0].reshape(3, 6), exps)
            self.assertEqual(out["blk.0.ffn_up_exps.weight.in_sum2"][1], [6, 3])     # ggml shape kept
            self.assertEqual(out["blk.0.ffn_up_exps.weight.counts"][1], [1, 3])
            self.assertEqual(r.fields["imatrix.chunk_count"].contents(), 7)


if __name__ == "__main__":
    unittest.main()
