"""Small CPU-only tests; no GPU or previous experiment modules."""
import setup_paths
from benchmark_paths import BENCHMARK, source_path
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

from build_sets import make_manifest, read_rules, compatible_rules


class ManifestTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.data = self.root / "data"
        (self.data / "images").mkdir(parents=True)
        records, splits = [], {}
        for bits in product((0, 1), repeat=3):
            for i in range(30):
                filename = f"scene_{''.join(map(str, bits))}_{i:02d}.png"
                (self.data / "images" / filename).write_bytes(b"test")
                records.append({"filename": filename, "bits": list(bits)})
                splits[filename] = "train" if i < 20 else "validation" if i < 25 else "test"
        (self.data / "metadata.json").write_text(json.dumps({"records": records}))
        self.partitions = self.root / "partitions.json"
        self.partitions.write_text(json.dumps(splits))
        self.rules = source_path("rules.json")

    def make(self, n, count):
        return make_manifest(self.data, self.rules, self.partitions,
                             {"train": count, "validation": 2, "test": 2}, n, 2026)

    def test_ten_images_nested_in_twenty(self):
        small, big = self.make(10, 3), self.make(20, 3)
        for a, b in zip(small["sets"], big["sets"]):
            self.assertEqual(a["set_id"], b["set_id"])
            self.assertEqual(a["image_files"], b["image_files"][:10])
            self.assertEqual(len(set(b["image_files"])), 20)

    def test_smaller_training_budget_nested_in_larger(self):
        small, large = self.make(10, 2), self.make(10, 6)
        by_id = {s["set_id"]: s for s in large["sets"]}
        for s in small["sets"]:
            self.assertEqual(s, by_id[s["set_id"]])

    def test_partition_disjoint_and_rules_identifiable(self):
        manifest = self.make(10, 3)
        rules = read_rules(self.rules)["rules"]
        splits = {}
        for s in manifest["sets"]:
            splits.setdefault(s["split"], set()).update(s["image_files"])
            self.assertEqual(compatible_rules(s["bits"], rules), [s["rule"]])
        for a in splits:
            for b in splits:
                if a != b:
                    self.assertTrue(splits[a].isdisjoint(splits[b]))

    def test_ambiguous_candidates_rejected(self):
        spec = read_rules(self.rules)
        spec["rules"].append({
            "id": "ANY", "expression": "A OR B OR C",
            "true_bits": ["001", "010", "011", "100", "101", "110", "111"]
        })
        rules_file = self.root / "ambiguous.json"
        rules_file.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, "NOT uniquely identifiable"):
            make_manifest(self.data, rules_file, self.partitions,
                          {"train": 1}, 10, 2026)

    def test_incomplete_partitions_rejected(self):
        splits = json.loads(self.partitions.read_text())
        splits.pop(next(iter(splits)))
        self.partitions.write_text(json.dumps(splits))
        with self.assertRaisesRegex(ValueError, "assign every image"):
            self.make(10, 1)


if __name__ == "__main__":
    unittest.main()
