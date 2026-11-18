"""Loads and VALIDATES the artifact folder produced by the training notebook.

Pure Python (no TensorFlow). The backend refuses to serve predictions unless every check passes,
and it lists every problem it finds so the fix is obvious.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SCHEMA = 1
# The project supports exactly these three models, in this order (id -> code) and this class order.
EXPECTED_MODELS = {"e1_cnn": "E1", "e2_efficientnetb0": "E2", "e3_hybrid_cbam": "E3"}
EXPECTED_CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
REQUIRED_FILES = [
    "class_names.json", "model_metadata.json", "preprocessing_config.json",
    "confidence_thresholds.json", "metrics/metrics.json",
]


class ArtifactError(RuntimeError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass
class ArtifactBundle:
    root: Path
    class_names: list[str]
    models: list[dict]                 # model_metadata.json -> models
    default_model_id: str
    preprocessing: dict
    thresholds: dict                   # model_id -> {"threshold": float, ...}
    metrics: dict                      # full metrics.json
    dataset_summary: dict | None = None
    samples: list[dict] = field(default_factory=list)
    training_environment: dict = field(default_factory=dict)   # python/tensorflow/keras used by the notebook

    @property
    def smoke_test(self) -> bool:
        return bool(self.metrics.get("smoke_test"))

    def model_meta(self, model_id: str) -> dict | None:
        return next((m for m in self.models if m["id"] == model_id), None)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path, problems: list[str]):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        problems.append(f"missing file: {path.name if path.parent.name == 'artifacts' else path.parent.name + '/' + path.name}")
    except json.JSONDecodeError as exc:
        problems.append(f"invalid JSON in {path.name}: {exc}")
    return None


def load_bundle(root: str | Path, verify_hashes: bool = True, expected_models: dict | None = None,
                expected_class_names: list | None = None) -> ArtifactBundle:
    """Validate and load the artifact folder.

    ``expected_models`` / ``expected_class_names`` default to the project's fixed three models and six classes;
    pass ``{}`` / ``[]`` (falsy) only in unit tests that use toy bundles.
    """
    expected_models = EXPECTED_MODELS if expected_models is None else expected_models
    expected_class_names = EXPECTED_CLASS_NAMES if expected_class_names is None else expected_class_names
    root = Path(root)
    problems: list[str] = []
    if not root.is_dir():
        raise ArtifactError([f"artifacts folder not found: {root} (run the notebook and unzip its artifacts there)"])

    class_names = _read_json(root / "class_names.json", problems)
    meta = _read_json(root / "model_metadata.json", problems)
    pre = _read_json(root / "preprocessing_config.json", problems)
    thr = _read_json(root / "confidence_thresholds.json", problems)
    met = _read_json(root / "metrics" / "metrics.json", problems)
    if problems:
        raise ArtifactError(problems)

    for name, obj in (("model_metadata", meta), ("preprocessing_config", pre), ("confidence_thresholds", thr), ("metrics", met)):
        if obj.get("schema_version") != SUPPORTED_SCHEMA:
            problems.append(f"{name}: schema_version {obj.get('schema_version')!r} not supported (expected {SUPPORTED_SCHEMA})")
    if not isinstance(class_names, list) or len(class_names) < 2:
        problems.append("class_names.json must be a list with at least 2 classes")
    if expected_class_names and class_names != expected_class_names:
        problems.append(f"class_names.json {class_names} != the project's class order {expected_class_names}")
    if (pre.get("image_size") if isinstance(pre, dict) else None) is not None and pre["image_size"] != 224:
        problems.append("preprocessing_config.json: image_size must be 224")
    if met.get("class_names") != class_names:
        problems.append("class order in metrics.json differs from class_names.json")

    models = meta.get("models", [])
    if not models:
        problems.append("model_metadata.json lists no models")
    ids = [m["id"] for m in models]
    if len(set(ids)) != len(ids):
        problems.append("duplicate model ids in model_metadata.json")
    if expected_models:
        if ids != list(expected_models):
            problems.append(f"model ids {ids} != the supported models {list(expected_models)} "
                            "(old artifacts from a different project version? re-run the notebook)")
        for m in models:
            if m["id"] in expected_models and m.get("code") != expected_models[m["id"]]:
                problems.append(f"model {m['id']}: code {m.get('code')!r} != {expected_models[m['id']]!r}")
    for m in models:
        f = root / m["file"]
        if not f.is_file():
            problems.append(f"model file missing: {m['file']}")
        elif verify_hashes:
            if not m.get("sha256"):
                problems.append(f"model {m['id']}: no sha256 recorded in model_metadata.json (re-run the notebook export)")
            elif _sha256(f) != m["sha256"]:
                problems.append(f"model file {m['file']} does not match its recorded sha256 (corrupted or replaced)")
        mm = met.get("models", {}).get(m["id"])
        if mm and expected_models and (mm.get("code"), mm.get("name")) != (m.get("code"), m.get("name")):
            problems.append(f"model {m['id']}: code/name differ between metrics.json and model_metadata.json")
        if m["id"] not in thr.get("models", {}):
            problems.append(f"no confidence threshold for model {m['id']}")
        if m["id"] not in met.get("models", {}):
            problems.append(f"no metrics for model {m['id']}")
    if verify_hashes and (root / "models").is_dir():
        listed = {m["file"] for m in models}
        for extra in sorted((root / "models").iterdir()):
            if f"models/{extra.name}" not in listed:
                problems.append(f"unlisted file in models/: {extra.name} (not part of model_metadata.json)")
    default_id = meta.get("default_model_id")
    if default_id not in ids:
        problems.append(f"default_model_id {default_id!r} is not an exported model")
    if problems:
        raise ArtifactError(problems)

    dataset_summary = None
    if (root / "dataset_summary.json").is_file():
        dataset_summary = json.loads((root / "dataset_summary.json").read_text())
    samples = []
    if (root / "samples" / "samples.json").is_file():
        samples = json.loads((root / "samples" / "samples.json").read_text())
    return ArtifactBundle(root, class_names, models, default_id, pre, thr["models"], met, dataset_summary, samples,
                          meta.get("training_environment") or {})
