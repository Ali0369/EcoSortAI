"""Loads every exported model ONCE at startup and runs real inference.

TensorFlow is imported lazily so the API can still start (in a clearly reported degraded mode)
when TensorFlow or the model files are missing.
"""
from __future__ import annotations

import base64
import logging
import sys
import threading
import time

import numpy as np

from ..core.artifacts import ArtifactBundle
from ..core.config import Settings

log = logging.getLogger("ecosort.registry")


class ModelsUnavailable(Exception):
    pass


class UnknownModel(Exception):
    pass


class ModelRegistry:
    def __init__(self, settings: Settings, bundle: ArtifactBundle | None, bundle_problems: list[str] | None = None):
        self.settings = settings
        self.bundle = bundle
        self.problems: list[str] = list(bundle_problems or [])
        self.warnings: list[str] = []
        self.load_reports: dict[str, dict] = {}     # model_id -> what was verified at load time
        self.models: dict[str, object] = {}
        self._locks: dict[str, threading.Lock] = {}
        self.tf_version: str | None = None

    # ------------------------------------------------------------------ loading
    def load_all(self) -> None:
        """Load all models once. Any failure is recorded in ``self.problems`` (never silently ignored)."""
        if self.bundle is None:
            return
        try:
            if str(self.settings.ml_src) not in sys.path:
                sys.path.insert(0, str(self.settings.ml_src))
            import tensorflow as tf
            import keras
            from ecosort_ml import layers  # noqa: F401  registers ecosort>* custom layers
            self.tf_version = f"TensorFlow {tf.__version__} / Keras {keras.__version__}"
            for pkg, have in (("tensorflow", tf.__version__), ("keras", keras.__version__)):
                want = self.bundle.training_environment.get(pkg)
                if want and want != have:
                    self.warnings.append(f"{pkg} {have} is installed but the models were trained/exported with {pkg} {want}; "
                                         "install the same version if a model fails to load")
        except Exception as exc:  # noqa: BLE001
            self.problems.append(f"TensorFlow/Keras or ecosort_ml could not be imported: {exc!r}")
            return
        for meta in self.bundle.models:
            path = self.bundle.root / meta["file"]
            try:
                model = keras.models.load_model(path)
                out_dim = int(model.output_shape[-1])
                if out_dim != len(self.bundle.class_names):
                    raise ValueError(f"output size {out_dim} != {len(self.bundle.class_names)} classes")
                report = self._verify_loaded(meta, model)
                self.models[meta["id"]] = model
                self._locks[meta["id"]] = threading.Lock()
                self.load_reports[meta["id"]] = report
                log.info("loaded model %s", meta["id"])
            except Exception as exc:  # noqa: BLE001
                self.problems.append(f"model {meta['id']} failed to load: {exc!r}")

    def _verify_loaded(self, meta: dict, model) -> dict:
        """Post-load checks. Raises if the restored model is not the one the notebook exported."""
        report: dict = {"cbam_layers": {}}
        # 1) CBAM sub-layers must hold restored variables (Dense kernel+bias, conv kernel+bias)
        if meta.get("attention") == "cbam":
            ca, sa = model.get_layer("channel_attention"), model.get_layer("spatial_attention")
            for label, layer in (("mlp_reduce", ca.fc1), ("mlp_expand", ca.fc2), ("spatial_conv", sa.conv)):
                ws = layer.get_weights()
                if len(ws) != 2 or not all(np.isfinite(w).all() for w in ws):
                    raise ValueError(f"CBAM layer {label} has {len(ws)} restored variables (expected 2 finite arrays)")
                report["cbam_layers"][label] = [list(w.shape) for w in ws]
        # 2) warm-up + reference parity with the notebook's reloaded model on a bundled sample image
        ref = meta.get("reference_check")
        if ref and self.settings.verify_reference:
            from PIL import Image
            img = np.asarray(Image.open(self.bundle.root / ref["image"]).convert("RGB"), dtype="float32")
            got = np.asarray(model(img[None], training=False))[0].astype("float64")
            want = np.asarray(ref["probabilities"], dtype="float64")
            diff = float(np.abs(got - want).max())
            report["reference_max_abs_diff"] = diff
            if diff > float(ref.get("tolerance", 0.02)):
                raise ValueError(f"output on the reference image differs from the notebook's by {diff:.4f} "
                                 f"(tolerance {ref.get('tolerance', 0.02)}); weights were not restored correctly or the "
                                 "TensorFlow/Keras version differs")
        else:
            model(np.zeros((1, 224, 224, 3), dtype="float32"), training=False)   # warm-up only
        return report

    # ------------------------------------------------------------------ status
    @property
    def ready(self) -> bool:
        return self.bundle is not None and bool(self.models)

    def require_ready(self) -> None:
        if not self.ready:
            detail = "; ".join(self.problems) or "no models loaded"
            raise ModelsUnavailable(f"Models are not available: {detail}")

    def resolve(self, model_id: str | None) -> str:
        self.require_ready()
        mid = model_id or self.bundle.default_model_id
        if mid not in self.models:
            raise UnknownModel(f"Unknown or unavailable model '{mid}'. Available: {sorted(self.models)}")
        return mid

    # ------------------------------------------------------------------ inference
    def predict(self, model_id: str | None, image: np.ndarray) -> dict:
        mid = self.resolve(model_id)
        model, names = self.models[mid], self.bundle.class_names
        t0 = time.perf_counter()
        with self._locks[mid]:
            probs = np.asarray(model(image[None], training=False))[0].astype("float64")
        ms = (time.perf_counter() - t0) * 1000
        order = np.argsort(-probs)
        thr = float(self.bundle.thresholds[mid]["threshold"])
        conf = float(probs[order[0]])
        return {
            "model_id": mid,
            "predicted_class": names[int(order[0])],
            "confidence": conf,
            "is_uncertain": conf < thr,
            "uncertainty_threshold": thr,
            "top3": [{"class_name": names[int(i)], "probability": float(probs[i])} for i in order[:3]],
            "probabilities": {n: float(p) for n, p in zip(names, probs)},
            "inference_ms": round(ms, 2),
        }

    def compare(self, model_ids: list[str] | None, image: np.ndarray) -> list[dict]:
        self.require_ready()
        ids = model_ids or [m["id"] for m in self.bundle.models if m["id"] in self.models]
        return [self.predict(i, image) for i in ids]

    # ------------------------------------------------------------------ explainability
    def explain(self, model_id: str | None, image: np.ndarray) -> dict:
        from ecosort_ml import gradcam as gc  # local import (needs TF)
        from PIL import Image

        mid = self.resolve(model_id)
        meta = self.bundle.model_meta(mid)
        with self._locks[mid]:
            cam, probs, cls = gc.gradcam(self.models[mid], image[None], meta["feature_layer"])
        # keep the payload small: show the image at max 512 px on the long side
        h, w = image.shape[:2]
        s = min(1.0, 512 / max(h, w))
        base = np.asarray(Image.fromarray(image.astype("uint8")).resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR))
        heat = gc.colorize(cam, (base.shape[1], base.shape[0]))
        over = gc.overlay(base, cam)
        ys, xs = np.unravel_index(int(cam.argmax()), cam.shape)
        vert = "upper" if ys < cam.shape[0] / 3 else "lower" if ys >= 2 * cam.shape[0] / 3 else "middle"
        horiz = "left" if xs < cam.shape[1] / 3 else "right" if xs >= 2 * cam.shape[1] / 3 else "centre"
        region = "centre" if (vert, horiz) == ("middle", "centre") else f"{vert} {horiz}".replace("middle ", "")
        b64 = lambda a: base64.b64encode(gc.to_png_bytes(a)).decode()  # noqa: E731
        names = self.bundle.class_names
        return {
            "model_id": mid, "explained_class": names[cls], "confidence": float(probs[cls]),
            "original_png_b64": b64(base), "heatmap_png_b64": b64(heat), "overlay_png_b64": b64(over),
            "peak_region": region, "high_attention_area_fraction": round(float((cam >= 0.5).mean()), 3),
            "feature_map_size": list(cam.shape),
        }
