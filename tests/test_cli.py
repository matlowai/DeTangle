import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from delta_surgery_seed.cli import demo, digest, evaluate, metadata, repair, same_tensor


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.safetensors"
        self.edited = self.root / "edited.safetensors"
        self.name = "transformer.blocks.0.q"
        self.ak = self.name + ".lora_A.weight"
        self.bk = self.name + ".lora_B.weight"
        g = torch.Generator().manual_seed(12)
        self.tensors = {self.ak: torch.randn(2, 6, generator=g, dtype=torch.float16),
                        self.bk: torch.randn(4, 2, generator=g, dtype=torch.float16),
                        "alpha": torch.tensor(2.), "counter": torch.tensor(17)}
        save_file(self.tensors, str(self.source), metadata={"owner": "synthetic", "matlowai_delta_surgery": "original value"})
        self.captures = {}
        for role in ["class", "trigger", "class-heldout", "trigger-heldout"]:
            path = self.root / f"{role}.safetensors"
            save_file({self.name: torch.randn(8, 6, generator=g)}, str(path))
            self.captures[role] = path

    def run_repair(self):
        return repair(self.source, self.captures["class"], self.captures["trigger"], self.edited)

    def run_evaluate(self, **kwargs):
        args = {"source": self.source, "edited": self.edited, "edit_report": self.edited.with_suffix(".json"),
                "class_capture": self.captures["class-heldout"], "trigger_capture": self.captures["trigger-heldout"],
                "out": self.root / "heldout.json"}
        args.update(kwargs)
        return evaluate(**args)

    def test_source_and_all_non_a_tensors_metadata_shapes_dtypes_retained(self):
        original_hash = digest(self.source)
        self.run_repair()
        self.assertEqual(original_hash, digest(self.source))
        new = load_file(self.edited)
        self.assertEqual(set(new), set(self.tensors))
        for key, old in self.tensors.items():
            self.assertEqual(old.dtype, new[key].dtype)
            self.assertEqual(old.shape, new[key].shape)
            if key != self.ak:
                self.assertTrue(same_tensor(old, new[key]))
        for key, value in metadata(self.source).items():
            self.assertEqual(metadata(self.edited)[key], value)
        self.assertIn("matlowai_delta_surgery_new", metadata(self.edited))
        self.run_evaluate()

    def test_no_overwrite_checkpoint_or_report(self):
        self.run_repair()
        old = digest(self.edited)
        with self.assertRaises(FileExistsError):
            self.run_repair()
        self.assertEqual(old, digest(self.edited))
        self.edited.unlink()
        with self.assertRaises(FileExistsError):
            self.run_repair()

    def test_heldout_rejects_either_fitting_capture(self):
        self.run_repair()
        for role in ["class", "trigger"]:
            with self.assertRaises(ValueError):
                self.run_evaluate(class_capture=self.captures[role])

    def test_tampered_b_is_rejected_even_with_updated_report_hash(self):
        self.run_repair()
        new = load_file(self.edited)
        new[self.bk] += 1
        saved_metadata = metadata(self.edited)
        save_file(new, str(self.edited), metadata=saved_metadata)
        report_path = self.edited.with_suffix(".json")
        report = json.loads(report_path.read_text())
        report["output_sha256"] = digest(self.edited)
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "A-only invariant"):
            self.run_evaluate()

    def test_wrong_source_hash_and_missing_b_are_rejected(self):
        self.run_repair()
        changed = {**self.tensors, "counter": torch.tensor(18)}
        save_file(changed, str(self.source))
        with self.assertRaisesRegex(ValueError, "does not belong"):
            self.run_evaluate()
        save_file({self.ak: self.tensors[self.ak]}, str(self.source))
        with self.assertRaisesRegex(ValueError, "Missing B"):
            repair(self.source, self.captures["class"], self.captures["trigger"], self.root / "other.safetensors")
        self.assertFalse((self.root / "other.safetensors").exists())

    def test_demo_distinguishes_uncovered_direction_priors(self):
        result = demo(self.root / "demo")
        self.assertAlmostEqual(result["uncovered_response"]["zero"]["retained"], 0.)
        self.assertAlmostEqual(result["uncovered_response"]["source"]["retained"], 1.)


if __name__ == "__main__":
    unittest.main()
