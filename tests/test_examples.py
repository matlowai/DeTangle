import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / "examples" / "creative-scenes"


class ExportTests(unittest.TestCase):
    def test_pairing_scores_summaries_and_illustration_integrity(self):
        load = lambda name: json.loads((ROOT / name).read_text())
        settings, scores = load("settings.json"), load("scores.json")
        records = scores["records"]
        self.assertEqual(len(records), 160)
        self.assertEqual(sum(r["illustration"] is not None for r in records), 64)
        self.assertEqual(len({(r["method"], r["strength"], r["prompt_id"], r["seed"]) for r in records}), 160)
        base = {(r["prompt_id"], r["seed"]): r for r in records if r["method"] == "base"}
        prompts = {p["id"]: p["text"] for p in settings["prompts"]}
        for pair in settings["pairs"]:
            self.assertEqual(prompts[pair["trigger"]], settings["trigger_prefix"] + prompts[pair["generic"]])
        for r in records:
            matched = base[r["prompt_id"], r["seed"]]
            self.assertEqual(r["matched_base_image_sha256"], matched["native_image_sha256"])
            self.assertAlmostEqual(r["excess_above_matched_base"], r["raw_similarity"] - matched["raw_similarity"])
            self.assertEqual(r["face_count"], 1)
        for scene in load("default-summary.json"):
            for method, summary in scene["methods"].items():
                selected = [r for r in records if r["method"] == method and r["strength"] == summary["strength"]]
                triggered = [r["raw_similarity"] for r in selected if r["prompt_id"] == "trigger_" + scene["scene"]]
                generic = [r["excess_above_matched_base"] for r in selected if r["prompt_id"] == "class_" + scene["scene"]]
                self.assertEqual(len(triggered), 2)
                self.assertEqual(len(generic), 2)
                self.assertAlmostEqual(sum(triggered) / 2, summary["triggered_likeness"])
                self.assertAlmostEqual(sum(generic) / 2, summary["generic_excess"])
        for filename, expected in load("provenance.json")["illustrations"]["files"].items():
            self.assertEqual(hashlib.sha256((ROOT / "images" / filename).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
