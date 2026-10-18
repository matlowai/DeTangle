"""Release auditing uses synthetic temporary files, never private source data."""

from io import BytesIO
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, PngImagePlugin

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/audit_release.py"
SPEC = importlib.util.spec_from_file_location("detangle_release_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def picture(format="JPEG", *, metadata=False, color="blue"):
    output = BytesIO()
    image = Image.new("RGB", (8, 8), color)
    kwargs = {}
    if format == "PNG" and metadata:
        info = PngImagePlugin.PngInfo(); info.add_text("workflow", "private fixture")
        kwargs["pnginfo"] = info
    if format == "JPEG" and metadata:
        exif = Image.Exif(); exif[315] = "Synthetic photographer"
        kwargs["exif"] = exif
    image.save(output, format=format, **kwargs)
    return output.getvalue()


class ReleaseTests(unittest.TestCase):
    def test_private_paths_in_docs_data_configs_and_examples(self):
        paths = ["/" + root + "/someone/checkpoint" for root in ["home", "media", "Users", "private", "root"]]
        paths += ["file:" + "///" + "private/checkpoint", "C" + ":\\Users\\someone\\checkpoint", chr(92) * 2 + "server\\share\\checkpoint"]
        for path in paths:
            for name in ["README.md", "examples/data.json", ".github/workflows/test.yml", "src/example.py"]:
                with self.subTest(name=name, path_kind=paths.index(path)):
                    payload = json.dumps({"path": path}).encode() if name.endswith("json") else ("example = " + repr(path)).encode()
                    self.assertTrue(audit.inspect_file(name, payload))
        encoded = json.dumps({"path": "/" + "home/someone/private"}).replace("home", "h\\u006fme")
        self.assertTrue(audit.inspect_file("examples/data.json", encoded.encode()))

    def test_external_urls_relative_paths_and_detection_code_are_allowed(self):
        examples = ["https://example.org/home/project/docs", "http://127.0.0.1:8890/media/image.jpg", "docs/mechanisms.md",
                    "value.startswith(('/', 'file:', '~'))", "route.startswith('/media/')", "r'^[A-Za-z]:[\\\\/]'", "/path/to/model"]
        for text in examples:
            self.assertFalse(audit.inspect_file("src/detect.py", text.encode()), text)
        fixture = "/" + "private/weights/model.safetensors"
        self.assertFalse(audit.inspect_file("lab/tests/test_pipeline.py", repr(fixture).encode()))
        self.assertTrue(audit.inspect_file("src/example.py", repr(fixture).encode()))
        different = "/" + "private/someone/checkpoint"
        self.assertTrue(audit.inspect_file("lab/tests/test_pipeline.py", repr(different).encode()))
        formula = {"code": "At render strength s:\ny = W x"}
        self.assertFalse(audit.inspect_file("dataset.json", json.dumps(formula).encode()))
        self.assertFalse(audit.inspect_file("dataset.js", ("window.DELTASCOPE_DATA = " + json.dumps(formula) + ";").encode()))

    def test_weights_databases_logs_and_build_products_are_rejected(self):
        for name in ["adapter.safetensors", "detector.onnx", "weights.pt", "embedding.npy", "archive.zip", "run.sqlite", "capture.log.1",
                     "lab/build/lib/module.py", "src/package.egg-info/PKG-INFO", "__pycache__/module.pyc", ".adapter-review/scores.sqlite"]:
            with self.subTest(name=name):
                self.assertTrue(audit.inspect_file(name, b"synthetic"))
        self.assertTrue(audit.inspect_file("opaque.txt", b"\x00\x01model"))
        embedded = "data:" + "image/png;base64,AAAA"
        self.assertTrue(audit.inspect_file("data.json", json.dumps({"image": embedded}).encode()))

    def test_only_exact_approved_image_bytes_pass(self):
        payload = picture()
        name = "examples/creative-scenes/images/approved.jpg"
        approved = {name: audit.digest(payload)}
        self.assertFalse(audit.inspect_file(name, payload, approved=approved))
        self.assertTrue(audit.inspect_file("reference.jpg", payload, approved=approved))
        self.assertTrue(audit.inspect_file(name, picture(color="red"), approved=approved))

    def test_metadata_is_rejected_even_if_hash_is_approved(self):
        for format, suffix in [("JPEG", ".jpg"), ("PNG", ".png")]:
            payload = picture(format, metadata=True)
            name = "examples/image" + suffix
            issues = audit.inspect_file(name, payload, approved={name: audit.digest(payload)})
            self.assertTrue(any("metadata" in issue.reason for issue in issues))
        payload = picture("PNG")
        self.assertFalse(audit.image_reason(payload, ".png"))

    def test_default_scope_checks_untracked_and_staged_bytes_but_ignores_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text("build/\n*.egg-info/\n")
            (root / "README.md").write_text("private = " + "/" + "home/someone/project")
            subprocess.run(["git", "-C", str(root), "add", "README.md", ".gitignore"], check=True)
            # An unstaged cleanup must not hide the bad version in the index.
            (root / "README.md").write_text("Clean worktree text")
            (root / "capture.log").write_text("untracked diagnostic")
            (root / "build").mkdir(); (root / "build/adapter.safetensors").write_bytes(b"ignored build fixture")
            findings, counts = audit.audit_repository(root, approved={})
            self.assertTrue(any(f.path == "README.md" and f.source == "index" for f in findings))
            self.assertTrue(any(f.path == "capture.log" and f.source == "worktree" for f in findings))
            self.assertFalse(any(f.path.startswith("build/") for f in findings))
            self.assertEqual(counts["worktree_files"], 3)
            # Force-tracked ignored artifacts are part of the payload and fail.
            subprocess.run(["git", "-C", str(root), "add", "-f", "build/adapter.safetensors"], check=True)
            findings, _ = audit.audit_repository(root, approved={})
            self.assertTrue(any(f.path == "build/adapter.safetensors" for f in findings))

    def test_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "plain.txt").write_text("example")
            (root / "linked.txt").symlink_to("plain.txt")
            findings, _ = audit.audit_repository(root, approved={})
            self.assertTrue(any(f.path == "linked.txt" and "symlink" in f.reason for f in findings))

    def test_public_export_requires_approved_source_derived_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "approved.jpg"; source.write_bytes(picture())
            approved = {"approved.jpg": audit.digest(source.read_bytes())}
            site = root / "export"; (site / "images").mkdir(parents=True)
            for name, text in [("index.html", "<!doctype html>"), ("dataset.json", "{}"), ("dataset.js", "window.DELTASCOPE_DATA={};")]:
                (site / name).write_text(text)
            image_name = "images/" + approved["approved.jpg"] + ".jpg"
            (site / image_name).write_bytes(source.read_bytes())
            findings, _ = audit.audit_public(root, site, approved=approved)
            self.assertFalse(findings)
            (site / image_name).write_bytes(picture(color="red"))
            (site / "cache.sqlite").write_bytes(b"fixture")
            findings, _ = audit.audit_public(root, site, approved=approved)
            self.assertTrue(any(f.reason == "approved-image-bytes-changed" for f in findings))
            self.assertTrue(any(f.path == "cache.sqlite" for f in findings))


if __name__ == "__main__":
    unittest.main()
