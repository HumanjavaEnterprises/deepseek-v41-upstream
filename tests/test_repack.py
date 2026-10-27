"""Unit tests for the metadata math in tools/repack-dsv41-ista.py (synthetic tables; no model files needed).
Run: GGUF_PY=/path/to/llama.cpp/gguf-py python3 -m unittest discover tests"""
import importlib.util, os, unittest
import numpy as np

spec = importlib.util.spec_from_file_location(
    "repack", os.path.join(os.path.dirname(__file__), "..", "tools", "repack-dsv41-ista.py"))
repack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repack)
A, VT = repack.A, repack.VT


class T:
    def __init__(self, data):
        self.data = np.asarray(data)


def tables(primes, mult=(3, 5), tmap=(0, 1, 2)):
    primes = np.asarray(primes)
    offs = [np.concatenate([[0], np.cumsum(row)[:-1]]) for row in primes]
    return {"engram_multipliers.weight": T(mult), "engram_primes.weight": T(primes.reshape(-1)),
            "engram_offsets.weight": T(np.concatenate(offs)), "engram_token_map.weight": T(tmap)}


KV = {f"{A}.engram.layer_ids": [1, 5], f"{A}.engram.num_embeddings": [10, 20],
      f"{A}.embedding_length": 4096, f"{A}.hyper_connection.count": 4}


class Derived(unittest.TestCase):
    def test_offsets_become_absolute_over_the_joined_table(self):
        d = repack.derived(tables([[3, 7], [11, 9]]), KV)
        self.assertEqual(d[f"{A}.ple.head_offsets"], ([0, 3, 10, 21], VT.UINT64))
        self.assertEqual(d[f"{A}.ple.head_vocab_sizes"], ([3, 7, 11, 9], VT.UINT64))

    def test_scalars_and_maps(self):
        d = repack.derived(tables([[3, 7], [11, 9]]), KV)
        self.assertEqual(d[f"{A}.embedding_length_out"], (4096 * 4, VT.UINT32))
        self.assertEqual(d[f"{A}.ple.layer_multipliers"], ([3, 5], VT.UINT64))
        self.assertEqual(d[f"{A}.engram.token_map"], ([0, 1, 2], VT.INT32))

    def test_rejects_offsets_that_are_not_prime_prefix_sums(self):
        t = tables([[3, 7], [11, 9]])
        t["engram_offsets.weight"] = T([0, 4, 0, 11])
        with self.assertRaises(ValueError):
            repack.derived(t, KV)

    def test_rejects_prime_sums_that_disagree_with_num_embeddings(self):
        with self.assertRaises(ValueError):
            repack.derived(tables([[3, 7], [11, 8]]), KV)


class Bf16(unittest.TestCase):
    def test_bf16_bytes_decode_exactly(self):
        f = np.array([1.5, -2.0, 0.15625], dtype=np.float32)
        raw = (f.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
        np.testing.assert_array_equal(repack.bf16(T(raw)), f)


class Tables(unittest.TestCase):
    def test_upstream_types_cover_the_known_mismatches(self):
        self.assertEqual(repack.UPSTREAM_TYPES[f"{A}.attention.indexer.candidate_source_layer"], (VT.INT32, None))
        for k in ("attention.kv_source_layers", "attention.indexer.source_layers", "ple.layers"):
            self.assertEqual(repack.UPSTREAM_TYPES[f"{A}.{k}"], (VT.ARRAY, VT.UINT32))

    def test_renames_target_upstream_names_and_never_collide(self):
        self.assertEqual(len(set(repack.RENAME.values())), len(repack.RENAME))
        self.assertTrue(all(v.startswith(f"{A}.") for v in repack.RENAME.values()))


if __name__ == "__main__":
    unittest.main()
