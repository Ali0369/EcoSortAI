"""Validate ml/artifacts WITHOUT starting the server.

    python check_artifacts.py [path] [--skip-models]

1. Always: structure, class order, thresholds/metrics for all three models, SHA-256 of every .keras file.
2. If TensorFlow is installed (default): load every model with the backend's real loader, confirm the CBAM
   Dense layers have restored weights, compare outputs with the notebook's reference outputs, and run one
   prediction per model on a bundled sample image.  Use --skip-models to skip this part.
Exit code 0 only if everything that was run passed.
"""
import sys
from pathlib import Path

from app.core.artifacts import ArtifactError, load_bundle
from app.core.config import Settings

args = [a for a in sys.argv[1:] if not a.startswith("--")]
skip_models = "--skip-models" in sys.argv
settings = Settings()
root = Path(args[0]) if args else settings.artifacts_dir
try:
    b = load_bundle(root, verify_hashes=settings.verify_hashes)
except ArtifactError as exc:
    print("ARTIFACTS INVALID:")
    for p in exc.problems:
        print(" -", p)
    sys.exit(1)
print(f"OK (files): {len(b.models)} models {[m['id'] for m in b.models]}, classes={b.class_names}, default={b.default_model_id}"
      + ("  [SMOKE TEST ARTIFACTS]" if b.smoke_test else ""))
print(f"SHA-256 verification: {'ON - all hashes match' if settings.verify_hashes else 'OFF (ECOSORT_VERIFY_HASHES=false)'}")
if skip_models:
    print("Model loading: SKIPPED (--skip-models)")
    sys.exit(0)

try:
    import tensorflow  # noqa: F401
except Exception as exc:  # noqa: BLE001
    print(f"Model loading: NOT CHECKED - TensorFlow could not be imported ({exc!r}). Install backend/requirements.txt.")
    sys.exit(0)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.services.model_registry import ModelRegistry  # noqa: E402

reg = ModelRegistry(settings, b)
reg.load_all()
print("Runtime:", reg.tf_version)
for w in reg.warnings:
    print(" WARNING:", w)
for p in reg.problems:
    print(" PROBLEM:", p)
sample = next(iter(sorted((b.root / "samples").glob("*.jpg"))), None) if (b.root / "samples").is_dir() else None
image = np.asarray(Image.open(sample).convert("RGB"), dtype="float32") if sample else np.full((224, 224, 3), 128, "float32")
ok = not reg.problems and len(reg.models) == len(b.models)
for m in b.models:
    if m["id"] not in reg.models:
        print(f" FAIL  {m['id']}: not loaded"); continue
    r = reg.load_reports[m["id"]]
    pred = reg.predict(m["id"], image)
    n = len(pred["probabilities"])
    cbam = f", CBAM layers restored: {r['cbam_layers']}" if r["cbam_layers"] else ""
    print(f" PASS  {m['id']}: {n} class probabilities (sum={sum(pred['probabilities'].values()):.4f}), "
          f"reference diff={r.get('reference_max_abs_diff', 'n/a')}, predicted {pred['predicted_class']} {pred['confidence']:.2f}{cbam}")
    ok = ok and n == len(b.class_names)
print("RESULT:", "ALL MODELS LOADED AND PREDICT" if ok else "FAILED")
sys.exit(0 if ok else 1)
