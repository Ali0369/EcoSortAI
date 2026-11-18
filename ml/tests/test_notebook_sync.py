"""Pure-Python tests (no TensorFlow needed).  Run from the repo root:  python -m unittest discover -s ml/tests -v

They check that the notebook is valid, that its CBAM layers are byte-identical to the backend's layers.py,
that only the three supported models are referenced, and that the notebook's export logic (SHA-256 ->
metadata -> ZIP -> validation) works and agrees with the BACKEND's artifact validator.
"""
import ast
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "ml" / "notebooks" / "EcoSort_AI_Simple_Training.ipynb"
LAYERS_PATH = ROOT / "ml" / "src" / "ecosort_ml" / "layers.py"
sys.path.insert(0, str(ROOT / "backend"))

NB = json.loads(NB_PATH.read_text())
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
ALL_TEXT = "\n".join("".join(c["source"]) for c in NB["cells"])


def cell_with(marker):
    return next(c for c in CODE if c.startswith(marker))


def helpers_namespace():
    ns = {}
    exec(compile(cell_with("# === ECOSORT HELPERS"), "helpers", "exec"), ns)
    return ns


class TestNotebookStructure(unittest.TestCase):
    def test_every_code_cell_compiles(self):
        for i, src in enumerate(CODE):
            clean = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
            ast.parse(clean, filename=f"code cell {i}")

    def test_outputs_are_cleared(self):
        for c in NB["cells"]:
            if c["cell_type"] == "code":
                self.assertEqual(c["outputs"], []); self.assertIsNone(c["execution_count"])

    def test_layers_cell_identical_to_backend_layers(self):
        cell = cell_with("# === ECOSORT LAYERS")
        self.assertIn(LAYERS_PATH.read_text().strip(), cell)

    def test_only_three_models_referenced(self):
        low = ALL_TEXT.lower()
        for banned in ("mobilenet", "resnet", "abl_channel", "abl_spatial", "channel_only", "spatial_only", "e5_hybrid", "e4_", "e3_efficientnetb0", "e2_mobilenet"):
            self.assertNotIn(banned, low, banned)
        ns = helpers_namespace()
        self.assertEqual(list(ns["MODEL_SPECS"]), ["e1_cnn", "e2_efficientnetb0", "e3_hybrid_cbam"])
        self.assertEqual([v["code"] for v in ns["MODEL_SPECS"].values()], ["E1", "E2", "E3"])
        self.assertEqual(ns["CLASS_NAMES"], ["cardboard", "glass", "metal", "paper", "plastic", "trash"])

    def test_no_nested_sequential_in_cbam_and_backbone_not_named_in_constructor(self):
        layers_src = LAYERS_PATH.read_text()
        self.assertNotIn("Sequential(", layers_src)  # no nested Sequential in the code
        self.assertIn('name="mlp_reduce"', layers_src); self.assertIn('name="mlp_expand"', layers_src)
        build = cell_with("def build_custom_cnn")
        self.assertNotRegex(build, r"EfficientNetB0\([^)]*name\s*=")       # that caused the backbone_notop.h5 403

    def test_versions_pinned_like_backend(self):
        req = (ROOT / "backend" / "requirements.txt").read_text()
        self.assertIn("tensorflow==2.21.0", req); self.assertIn("keras==3.15.1", req)
        self.assertIn('"tensorflow": "2.21.0"', cell_with("# Run this cell FIRST"))
        self.assertIn('"keras": "3.15.1"', cell_with("# Run this cell FIRST"))


class TestHelpers(unittest.TestCase):
    def test_threshold_and_mcnemar(self):
        ns = helpers_namespace()
        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.ones(6), size=200); y = probs.argmax(1)
        t = ns["choose_confidence_threshold"](probs, y)
        self.assertTrue(0.3 <= t["threshold"] <= 0.95); self.assertGreaterEqual(t["val_coverage"], 0.5)
        m = ns["mcnemar_exact"]([1, 1, 0, 0, 1], [1, 0, 1, 1, 1])
        self.assertEqual((m["a_right_b_wrong"], m["a_wrong_b_right"]), (1, 2)); self.assertAlmostEqual(m["p_value"], 1.0)
        self.assertEqual(ns["mcnemar_exact"]([1] * 10, [1] * 10)["p_value"], 1.0)
        # 0 vs 10 discordant pairs -> p = 2 / 2^10
        self.assertAlmostEqual(ns["mcnemar_exact"]([0] * 10, [1] * 10)["p_value"], 2 / 1024)


def fake_artifacts(root: Path, ns, corrupt=None, drop_sha=False, extra_model=False):
    """Write a complete artifact folder with FAKE model bytes, using the notebook's own helper functions."""
    for d in ("models", "metrics", "samples"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / REF_REL).write_bytes(b"jpeg")
    names = ns["CLASS_NAMES"]
    entries, thr, met = [], {}, {}
    for i, mid in enumerate(ns["MODEL_SPECS"]):
        (root / "models" / f"{mid}.keras").write_bytes(f"fake model {mid}".encode() * 50)
        e = ns["build_model_metadata_entry"](mid, root, np.full(6, 1 / 6))
        if drop_sha:
            e.pop("sha256")
        entries.append(e)
        thr[mid] = {"threshold": 0.6}
        met[mid] = {"code": e["code"], "name": e["name"], "test_accuracy": 0.9}
    (root / "class_names.json").write_text(json.dumps(names))
    (root / "model_metadata.json").write_text(json.dumps({"schema_version": 1, "models": entries, "default_model_id": ns["DEFAULT_MODEL_ID"]}))
    (root / "preprocessing_config.json").write_text(json.dumps({"schema_version": 1, "image_size": 224}))
    (root / "confidence_thresholds.json").write_text(json.dumps({"schema_version": 1, "models": thr}))
    (root / "metrics" / "metrics.json").write_text(json.dumps({"schema_version": 1, "class_names": names, "models": met}))
    if corrupt:
        (root / "models" / f"{corrupt}.keras").write_bytes(b"changed after hashing")
    if extra_model:
        (root / "models" / "e5_old.keras").write_bytes(b"stale")


REF_REL = "samples/example_cardboard.jpg"


class TestExportLogic(unittest.TestCase):
    def test_hash_metadata_zip_roundtrip_and_backend_agreement(self):
        from app.core.artifacts import load_bundle
        ns = helpers_namespace()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "art"; root.mkdir(); fake_artifacts(root, ns)
            self.assertEqual(ns["validate_artifact_folder"](root), [])
            meta = json.loads((root / "model_metadata.json").read_text())
            for m in meta["models"]:
                self.assertEqual(len(m["sha256"]), 64)
                self.assertEqual(m["sha256"], ns["sha256_file"](root / m["file"]))
            zp = ns["make_artifact_zip"](root, Path(d) / "a.zip")
            self.assertEqual(ns["verify_artifact_zip"](zp), [])
            names = zipfile.ZipFile(zp).namelist()
            self.assertEqual(sorted(n for n in names if n.endswith(".keras")), sorted(f"models/{i}.keras" for i in ns["MODEL_SPECS"]))
            b = load_bundle(root)                              # the REAL backend loader, hash verification ON
            self.assertEqual([m["id"] for m in b.models], list(ns["MODEL_SPECS"]))
            self.assertEqual(b.default_model_id, "e3_hybrid_cbam")

    def test_failures_detected_by_both_validators(self):
        from app.core.artifacts import ArtifactError, load_bundle
        ns = helpers_namespace()
        for kwargs, needle in [({"corrupt": "e3_hybrid_cbam"}, "sha256"), ({"drop_sha": True}, "sha256"), ({"extra_model": True}, "e5_old")]:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d); fake_artifacts(root, ns, **kwargs)
                nb_problems = ns["validate_artifact_folder"](root)
                self.assertTrue(any(needle in p for p in nb_problems), nb_problems)
                with self.assertRaises(ArtifactError) as cm:
                    load_bundle(root)
                self.assertTrue(any(needle in p for p in cm.exception.problems), cm.exception.problems)


if __name__ == "__main__":
    unittest.main()
