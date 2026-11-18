"""Basic backend tests.  Run from backend/:  python -m unittest discover -s tests -v
(API tests are skipped automatically if FastAPI is not installed; model tests need real artifacts.)"""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.artifacts import ArtifactError, load_bundle  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.services.image_io import ImageValidationError, decode_upload  # noqa: E402

S = Settings()


def png_bytes(size=(64, 48), mode="RGB", fmt="PNG"):
    buf = io.BytesIO()
    Image.new(mode, size, (120, 160, 90) if mode == "RGB" else (120, 160, 90, 128)).save(buf, format=fmt)
    return buf.getvalue()


class TestImageValidation(unittest.TestCase):
    def test_valid_png_and_jpeg(self):
        a = decode_upload(png_bytes(), "image/png", S)
        self.assertEqual(a.shape, (48, 64, 3)); self.assertEqual(a.dtype, np.float32)
        decode_upload(png_bytes(fmt="JPEG"), "image/jpeg", S)

    def test_rgba_is_flattened(self):
        self.assertEqual(decode_upload(png_bytes(mode="RGBA"), "image/png", S).shape[-1], 3)

    def test_rejects_bad_input(self):
        for raw, ctype, code in [(b"", "image/png", "empty_file"), (b"not an image", "image/png", "invalid_image"),
                                 (png_bytes(), "application/pdf", "unsupported_type"),
                                 (b"x" * (S.max_upload_bytes + 1), "image/png", "file_too_large")]:
            with self.assertRaises(ImageValidationError) as cm:
                decode_upload(raw, ctype, S)
            self.assertEqual(cm.exception.code, code)

    def test_gif_rejected_even_with_png_header(self):
        buf = io.BytesIO(); Image.new("P", (20, 20)).save(buf, format="GIF")
        with self.assertRaises(ImageValidationError):
            decode_upload(buf.getvalue(), "image/png", S)


class TestArtifacts(unittest.TestCase):
    def test_missing_folder_and_files_are_reported(self):
        with self.assertRaises(ArtifactError):
            load_bundle("/no/such/folder")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ArtifactError) as cm:
                load_bundle(d)
            self.assertTrue(any("class_names.json" in p for p in cm.exception.problems))

    def test_valid_minimal_bundle_and_missing_model_detected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "metrics").mkdir()
            names = ["a", "b"]
            (root / "class_names.json").write_text(json.dumps(names))
            (root / "model_metadata.json").write_text(json.dumps({"schema_version": 1, "default_model_id": "m", "models": [
                {"id": "m", "file": "models/m.keras", "sha256": "x"}]}))
            (root / "preprocessing_config.json").write_text(json.dumps({"schema_version": 1}))
            (root / "confidence_thresholds.json").write_text(json.dumps({"schema_version": 1, "models": {"m": {"threshold": 0.7}}}))
            (root / "metrics" / "metrics.json").write_text(json.dumps({"schema_version": 1, "class_names": names, "models": {"m": {}}}))
            with self.assertRaises(ArtifactError) as cm:
                load_bundle(root, expected_models={}, expected_class_names=[])
            self.assertTrue(any("model file missing" in p for p in cm.exception.problems))
            (root / "models").mkdir(); (root / "models" / "m.keras").write_bytes(b"fake")
            with self.assertRaises(ArtifactError) as cm:
                load_bundle(root, verify_hashes=True, expected_models={}, expected_class_names=[])
            self.assertTrue(any("sha256" in p for p in cm.exception.problems))
            b = load_bundle(root, verify_hashes=False, expected_models={}, expected_class_names=[])   # structure valid once hash checking is off
            self.assertEqual(b.class_names, names)


    def test_missing_sha256_is_rejected_not_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "metrics").mkdir(); (root / "models").mkdir()
            names = ["a", "b"]
            (root / "models" / "m.keras").write_bytes(b"x")
            (root / "class_names.json").write_text(json.dumps(names))
            (root / "model_metadata.json").write_text(json.dumps({"schema_version": 1, "default_model_id": "m", "models": [{"id": "m", "file": "models/m.keras"}]}))
            (root / "preprocessing_config.json").write_text(json.dumps({"schema_version": 1}))
            (root / "confidence_thresholds.json").write_text(json.dumps({"schema_version": 1, "models": {"m": {"threshold": 0.7}}}))
            (root / "metrics" / "metrics.json").write_text(json.dumps({"schema_version": 1, "class_names": names, "models": {"m": {}}}))
            with self.assertRaises(ArtifactError) as cm:
                load_bundle(root, expected_models={}, expected_class_names=[])
            self.assertTrue(any("no sha256 recorded" in p for p in cm.exception.problems))

    def test_old_seven_model_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "metrics").mkdir()
            names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
            (root / "class_names.json").write_text(json.dumps(names))
            (root / "model_metadata.json").write_text(json.dumps({"schema_version": 1, "default_model_id": "e5_hybrid_cbam", "models": [
                {"id": "e5_hybrid_cbam", "code": "E5", "file": "models/e5_hybrid_cbam.keras"}]}))
            (root / "preprocessing_config.json").write_text(json.dumps({"schema_version": 1}))
            (root / "confidence_thresholds.json").write_text(json.dumps({"schema_version": 1, "models": {}}))
            (root / "metrics" / "metrics.json").write_text(json.dumps({"schema_version": 1, "class_names": names, "models": {}}))
            with self.assertRaises(ArtifactError) as cm:
                load_bundle(root)
            self.assertTrue(any("supported models" in p for p in cm.exception.problems))


try:
    from fastapi.testclient import TestClient
    from app.main import create_app
    HAVE_API = True
except Exception:  # noqa: BLE001
    HAVE_API = False


@unittest.skipUnless(HAVE_API, "fastapi/httpx not installed")
class TestApiWithoutArtifacts(unittest.TestCase):
    """The API must start and report a clear degraded state when artifacts are missing."""

    def test_degraded_mode(self):
        with tempfile.TemporaryDirectory() as d:
            settings = Settings(artifacts_dir=Path(d) / "none")
            with TestClient(create_app(settings, load_models=False)) as c:
                h = c.get("/api/health").json()
                self.assertEqual(h["status"], "degraded"); self.assertTrue(h["problems"])
                r = c.post("/api/predict", files={"file": ("a.png", png_bytes(), "image/png")})
                self.assertEqual(r.status_code, 503); self.assertEqual(r.json()["error"]["code"], "models_unavailable")
                self.assertEqual(c.post("/api/predict").status_code, 422)


REAL = Settings().artifacts_dir / "model_metadata.json"


@unittest.skipUnless(HAVE_API and REAL.exists(), "needs FastAPI + real trained artifacts in ml/artifacts")
class TestApiWithRealModels(unittest.TestCase):
    def test_predict_compare_and_invalid(self):
        with TestClient(create_app()) as c:
            self.assertEqual(c.get("/api/health").json()["status"], "ok")
            img = ("a.png", png_bytes((300, 200)), "image/png")
            p = c.post("/api/predict", files={"file": img}).json()
            names = json.loads((Settings().artifacts_dir / "class_names.json").read_text())
            self.assertIn(p["predicted_class"], names); self.assertEqual(len(p["top3"]), 3)
            self.assertAlmostEqual(sum(p["probabilities"].values()), 1.0, places=3)
            cmp_ = c.post("/api/predict/compare", files={"file": img}).json()
            self.assertEqual(len(cmp_["predictions"]), 3)
            self.assertEqual([m["id"] for m in c.get("/api/models").json()], ["e1_cnn", "e2_efficientnetb0", "e3_hybrid_cbam"])
            self.assertEqual(c.post("/api/predict", files={"file": ("x.txt", b"hello", "text/plain")}).status_code, 415)
            self.assertEqual(c.post("/api/predict", files={"file": img}, data={"model_id": "nope"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
