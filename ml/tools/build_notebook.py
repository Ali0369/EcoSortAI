"""Builds ml/notebooks/EcoSort_AI_Simple_Training.ipynb.

The layers cell is copied verbatim from ml/src/ecosort_ml/layers.py so the notebook and the backend can
never register different implementations of the same layer name.   Run:  python ml/tools/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYERS_SRC = (ROOT / "src" / "ecosort_ml" / "layers.py").read_text()
OUT = ROOT / "notebooks" / "EcoSort_AI_Simple_Training.ipynb"
HELPERS_MARK = "# === ECOSORT HELPERS (pure Python, unit-tested) ==="
LAYERS_MARK = "# === ECOSORT LAYERS (identical to ml/src/ecosort_ml/layers.py) ==="

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md(r'''
# EcoSort AI — Simple Model Training Notebook

**Project:** EcoSort AI — An Intelligent Waste Classification and Interactive Sorting System Using Hybrid Deep Learning

This notebook trains **exactly three models** on TrashNet and exports the artifact ZIP that the EcoSort backend loads.

| ID (file) | Code | Model |
|---|---|---|
| `e1_cnn` | E1 | Custom CNN (baseline, from scratch) |
| `e2_efficientnetb0` | E2 | EfficientNetB0 (ImageNet transfer-learning baseline) |
| `e3_hybrid_cbam` | E3 | EfficientNetB0 + CBAM (proposed hybrid) |

### How to use
1. Open in Google Colab, **Runtime → Change runtime type → T4 GPU**.
2. Run the cells from top to bottom. **Cell 1 pins TensorFlow 2.21.0 / Keras 3.15.1** (the backend versions). If it installs anything it tells you to *Runtime → Restart session*; then run again from the top.
3. At the end `ecosort_artifacts.zip` is created and downloaded. Unzip it **into** `ecosort-ai/ml/artifacts/` (so that `ml/artifacts/model_metadata.json` and `ml/artifacts/models/*.keras` exist).
4. In `ecosort-ai/backend` run `python check_artifacts.py`.

Colab storage is temporary: download the ZIP before closing the session. All numbers come from this run; nothing is hard-coded.

**What changed versus the earlier simplified notebook** (see README "Changes"): the EfficientNet backbone no longer breaks the ImageNet download, CBAM layers are identical to the backend's, model IDs match the E1/E2/E3 codes, SHA-256 hashes are computed from the final files, and the export is validated the same way the backend validates it.
''')

md("## 1. Pin the library versions (must match the backend)")
code(r'''
# Run this cell FIRST in a fresh session. The backend uses TensorFlow 2.21.0 + Keras 3.15.1, so the
# .keras files must be produced with the same versions.
import importlib.metadata as md, subprocess, sys

PINNED = {"tensorflow": "2.21.0", "keras": "3.15.1"}

def _installed(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None

mismatch = {p: (_installed(p), v) for p, v in PINNED.items() if _installed(p) != v}
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "scikit-learn", "pandas", "matplotlib", "pillow"], check=True)
if mismatch:
    print("Version mismatch (installed, wanted):", mismatch)
    subprocess.run([sys.executable, "-m", "pip", "-q", "install"] + [f"{p}=={v}" for p, v in PINNED.items()], check=True)
    print("\n>>> Pinned versions installed. Now choose  Runtime -> Restart session , then run the notebook again from the top. <<<")
else:
    print("TensorFlow / Keras already match the backend:", {p: _installed(p) for p in PINNED})
''')

md("## 2. Imports, settings and helper functions")
code(r'''
import os, io, json, time, random, shutil, hashlib, zipfile, subprocess, sys, math, tempfile, datetime, platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import keras
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32

# Modest defaults for a student project. CNN_EPOCHS was 12 in the earlier run, where validation loss was
# still improving at epoch 11, so it is now 20 (early stopping + best-weight restore keep it safe).
CNN_EPOCHS = 20
TRANSFER_HEAD_EPOCHS = 6
TRANSFER_FINE_TUNE_EPOCHS = 5

ALLOW_VERSION_MISMATCH = False   # keep False so the exported files are guaranteed to load in the backend

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__, "| Keras:", keras.__version__, "| Python:", platform.python_version())
print("GPU devices:", tf.config.list_physical_devices("GPU"))
if not ALLOW_VERSION_MISMATCH:
    assert tf.__version__ == "2.21.0" and keras.__version__ == "3.15.1", (
        "Versions differ from the backend (TensorFlow 2.21.0 / Keras 3.15.1). Run cell 1, restart the session, run again.")
''')

code(HELPERS_MARK + r'''
import json, hashlib, math, zipfile, tempfile, shutil
from pathlib import Path
import numpy as np

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]   # ONE class order for the whole project

MODEL_SPECS = {
    "e1_cnn": dict(code="E1", name="Custom CNN", role="baseline", attention="none", feature_layer="last_conv_act",
                   is_proposed=False, description="Custom CNN baseline trained from scratch."),
    "e2_efficientnetb0": dict(code="E2", name="EfficientNetB0", role="baseline", attention="none", feature_layer="backbone",
                              is_proposed=False, description="ImageNet-pretrained EfficientNetB0 transfer-learning baseline (same head as E3, no attention)."),
    "e3_hybrid_cbam": dict(code="E3", name="EfficientNetB0 + CBAM", role="proposed", attention="cbam", feature_layer="backbone",
                           is_proposed=True, description="Proposed EfficientNetB0 feature extractor with CBAM channel and spatial attention."),
}
DEFAULT_MODEL_ID = "e3_hybrid_cbam"
REFERENCE_SAMPLE = "samples/example_cardboard.jpg"   # image used for the backend's load-parity check
REQUIRED_FILES = ["class_names.json", "model_metadata.json", "preprocessing_config.json",
                  "confidence_thresholds.json", "metrics/metrics.json"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_confidence_threshold(probabilities, true_labels):
    """Pick the threshold on VALIDATION data: best harmonic mean of coverage and selective accuracy."""
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    best_threshold, best_score = 0.50, -1.0
    for threshold in np.arange(0.30, 0.96, 0.01):
        accepted = confidence >= threshold
        coverage = accepted.mean()
        if accepted.sum() == 0 or coverage < 0.50:
            continue
        selective_accuracy = (predictions[accepted] == true_labels[accepted]).mean()
        score = (2 * coverage * selective_accuracy) / (coverage + selective_accuracy)
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    accepted = confidence >= best_threshold
    selective = float((predictions[accepted] == true_labels[accepted]).mean()) if accepted.any() else None
    return {"threshold": round(best_threshold, 4), "val_coverage": round(float(accepted.mean()), 4),
            "val_selective_accuracy": round(selective, 4) if selective is not None else None,
            "rule": "Selected using validation data; balances coverage and selective accuracy."}


def apply_threshold(probabilities, true_labels, threshold):
    """Coverage / selective accuracy of a FIXED threshold on another split (used for the test split)."""
    accepted = probabilities.max(axis=1) >= threshold
    pred = probabilities.argmax(axis=1)
    return {"coverage": round(float(accepted.mean()), 4),
            "selective_accuracy": round(float((pred[accepted] == true_labels[accepted]).mean()), 4) if accepted.any() else None}


def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar test on paired correctness arrays (a = baseline, b = proposed)."""
    correct_a, correct_b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    a_right_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_right = int(np.sum(~correct_a & correct_b))
    n = a_right_b_wrong + a_wrong_b_right
    if n == 0:
        p = 1.0
    else:
        k = min(a_right_b_wrong, a_wrong_b_right)
        p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    return {"a_right_b_wrong": a_right_b_wrong, "a_wrong_b_right": a_wrong_b_right, "p_value": float(p)}


def build_model_metadata_entry(model_id, artifacts_dir, reference_probabilities=None):
    """Metadata for one model. The SHA-256 is computed HERE from the final .keras file on disk."""
    spec = MODEL_SPECS[model_id]
    path = Path(artifacts_dir) / "models" / f"{model_id}.keras"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist - save the model before writing metadata")
    entry = {
        "id": model_id, "code": spec["code"], "name": spec["name"], "role": spec["role"],
        "attention": spec["attention"], "description": spec["description"],
        "file": f"models/{model_id}.keras",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "input": {"dtype": "float32", "range": "0-255", "channels": "RGB", "shape": [None, None, 3]},
        "feature_layer": spec["feature_layer"], "is_proposed": spec["is_proposed"],
    }
    if reference_probabilities is not None:
        entry["reference_check"] = {"image": REFERENCE_SAMPLE, "tolerance": 0.02,
                                    "probabilities": [float(p) for p in reference_probabilities]}
    return entry


def validate_artifact_folder(root):
    """Mirror of the backend's checks (backend/app/core/artifacts.py). Returns a list of problems (empty = OK)."""
    root = Path(root)
    problems = []
    def read(rel):
        try:
            return json.loads((root / rel).read_text())
        except FileNotFoundError:
            problems.append(f"missing file: {rel}")
        except json.JSONDecodeError as exc:
            problems.append(f"invalid JSON in {rel}: {exc}")
    class_names, meta = read("class_names.json"), read("model_metadata.json")
    pre, thr, met = read("preprocessing_config.json"), read("confidence_thresholds.json"), read("metrics/metrics.json")
    if problems:
        return problems
    for name, obj in (("model_metadata", meta), ("preprocessing_config", pre), ("confidence_thresholds", thr), ("metrics", met)):
        if obj.get("schema_version") != 1:
            problems.append(f"{name}: schema_version must be 1")
    if class_names != CLASS_NAMES:
        problems.append(f"class_names.json {class_names} != expected {CLASS_NAMES}")
    if met.get("class_names") != class_names:
        problems.append("class order in metrics.json differs from class_names.json")
    if pre.get("image_size") != 224:
        problems.append("preprocessing_config image_size must be 224")
    models = meta.get("models", [])
    ids = [m["id"] for m in models]
    if sorted(ids) != sorted(MODEL_SPECS):
        problems.append(f"model ids {ids} != expected {list(MODEL_SPECS)}")
    for m in models:
        mid = m["id"]
        if m.get("file") != f"models/{mid}.keras":
            problems.append(f"{mid}: file must be models/{mid}.keras")
        f = root / m.get("file", "")
        if not f.is_file():
            problems.append(f"model file missing: {m.get('file')}")
        elif not m.get("sha256"):
            problems.append(f"{mid}: no sha256 recorded in model_metadata.json")
        elif sha256_file(f) != m["sha256"]:
            problems.append(f"{mid}: sha256 mismatch (file changed after hashing)")
        if mid not in thr.get("models", {}):
            problems.append(f"no confidence threshold for {mid}")
        if mid not in met.get("models", {}):
            problems.append(f"no metrics for {mid}")
        elif (met["models"][mid].get("code"), met["models"][mid].get("name")) != (m.get("code"), m.get("name")):
            problems.append(f"{mid}: code/name differ between metrics.json and model_metadata.json")
        if not m.get("reference_check"):
            problems.append(f"{mid}: reference_check missing")
    listed = {m.get("file") for m in models}
    for extra in sorted((root / "models").glob("*")) if (root / "models").is_dir() else []:
        if f"models/{extra.name}" not in listed:
            problems.append(f"unlisted file in models/: {extra.name}")
    if meta.get("default_model_id") not in ids:
        problems.append("default_model_id is not an exported model")
    return problems


def make_artifact_zip(artifacts_dir, zip_path):
    """ZIP the artifact folder (paths relative to it). Contains exactly what is on disk, hashed files included."""
    artifacts_dir = Path(artifacts_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(artifacts_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(artifacts_dir).as_posix())
    return Path(zip_path)


def verify_artifact_zip(zip_path):
    """Extract the ZIP into a temp folder and run the full validation on what a user will actually get."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad:
                return [f"corrupt member in zip: {bad}"]
            zf.extractall(tmp)
        return validate_artifact_folder(tmp)
''')

md('''
## 3. Download and locate TrashNet
''')
code(r'''
DATA_DIR = Path("/content/trashnet")
if not DATA_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/garythung/trashnet.git", str(DATA_DIR)], check=True)

# Some copies of the repository keep the images inside dataset-resized.zip.
for zip_path in DATA_DIR.rglob("dataset-resized.zip"):
    extract_to = zip_path.parent / "dataset-resized"
    if not extract_to.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(zip_path.parent)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_dataset_folder(search_folder):
    search_folder = Path(search_folder)
    for folder in [search_folder] + [p for p in search_folder.rglob("*") if p.is_dir()]:
        subfolders = {p.name.lower(): p for p in folder.iterdir() if p.is_dir()}
        if all(name in subfolders for name in CLASS_NAMES):
            if all(any(f.suffix.lower() in IMAGE_EXTENSIONS for f in subfolders[name].iterdir() if f.is_file()) for name in CLASS_NAMES):
                return folder
    raise FileNotFoundError("Could not locate the six TrashNet class folders. Check the dataset download.")

DATASET_ROOT = find_dataset_folder(DATA_DIR)
print("Dataset folder:", DATASET_ROOT)
''')

md("## 4. Read images and inspect the dataset\nThe class order (`CLASS_NAMES`) is the same in training, export and the backend.")
code(r'''
records = []
for label_index, class_name in enumerate(CLASS_NAMES):
    class_folder = next(p for p in DATASET_ROOT.iterdir() if p.is_dir() and p.name.lower() == class_name)
    for image_path in sorted(class_folder.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            records.append({"path": str(image_path), "label": class_name, "label_idx": label_index})

data = pd.DataFrame(records)
if data.empty:
    raise ValueError("No images were found in the TrashNet class folders.")

print("Total images:", len(data))
display(data.groupby("label").size().reindex(CLASS_NAMES).rename("images").to_frame())
data.groupby("label").size().reindex(CLASS_NAMES).plot(kind="bar", title="TrashNet images per class", ylabel="Number of images")
plt.xticks(rotation=30); plt.show()
''')

md("## 5. Basic image and duplicate checks\nChecks that images open and removes exact duplicate files (it does not detect near-duplicates). An identical image filed under two different classes has an ambiguous label, so all its copies are removed.")
code(r'''
valid_rows, bad_files = [], []
for row in data.itertuples():
    try:
        with Image.open(row.path) as im:
            im.verify()
        valid_rows.append(row.Index)
    except Exception as exc:
        bad_files.append((row.path, str(exc)))

data = data.loc[valid_rows].reset_index(drop=True)
print("Unreadable images:", len(bad_files))

# Hash every file, then handle exact duplicates
data["digest"] = [sha256_file(p) for p in data["path"]]

# 1) Same image filed under DIFFERENT classes -> its label is ambiguous, so drop ALL copies
labels_per_digest = data.groupby("digest")["label"].nunique()
conflicting_digests = set(labels_per_digest[labels_per_digest > 1].index)
conflict_mask = data["digest"].isin(conflicting_digests)
conflicting_label_removed = int(conflict_mask.sum())
if conflicting_label_removed:
    print(f"Images with identical content but different class labels (all copies removed): {conflicting_label_removed}")
    display(data[conflict_mask].sort_values("digest")[["path", "label"]])
data = data[~conflict_mask]

# 2) Same image repeated inside the SAME class -> keep the first copy only
before = len(data)
data = data.drop_duplicates(subset="digest", keep="first")
duplicate_count = before - len(data)

data = data.drop(columns="digest").reset_index(drop=True)
print("Exact duplicate files removed (same class):", duplicate_count)
print("Images remaining:", len(data))
''')

md("## 6. Split into training, validation and test sets (70 / 15 / 15, stratified)")
code(r'''
train_df, temp_df = train_test_split(data, test_size=0.30, random_state=SEED, stratify=data["label_idx"])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label_idx"])
train_df, val_df, test_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    print(f"\n{split_name}: {len(split_df)} images")
    display(split_df["label"].value_counts().reindex(CLASS_NAMES).rename("images").to_frame())
''')

md("## 7. Create TensorFlow datasets\nImages are resized to 224×224; augmentation is applied to the training set only. Pixel values stay in 0–255 (each model rescales internally).")
code(r'''
def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(tf.cast(image, tf.float32), (IMG_SIZE, IMG_SIZE))
    return image, label

def make_dataset(frame, training=False):
    paths = frame["path"].astype(str).to_numpy()
    labels = frame["label_idx"].astype("int32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(len(frame), seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    if training:
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"), layers.RandomRotation(0.08),
            layers.RandomZoom(0.10), layers.RandomContrast(0.10)])
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmentation(x, training=True), 0, 255), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_df, training=True)
train_eval_ds = make_dataset(train_df)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

images, labels = next(iter(train_eval_ds))
plt.figure(figsize=(10, 5))
for i in range(min(8, len(images))):
    plt.subplot(2, 4, i + 1); plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(CLASS_NAMES[int(labels[i])]); plt.axis("off")
plt.tight_layout(); plt.show()
''')

md('''
## 8. CBAM attention layers (identical to the backend)
This cell is a verbatim copy of `ml/src/ecosort_ml/layers.py`. **Do not edit it here**: the backend imports that same file, and both register the layers as `ecosort>ResizeTo`, `ecosort>ChannelAttention`, `ecosort>SpatialAttention`.

`ChannelAttention` uses two explicit, named Dense layers (`mlp_reduce`, `mlp_expand`) built inside `build()` instead of a nested `keras.Sequential`. The earlier notebook saved a nested Sequential while the backend expected named Dense layers; Keras 3 then found no weights for `mlp_reduce` / `mlp_expand` and raised *"expected 2 variables, but received 0 variables"*.
''')
code(LAYERS_MARK + "\n" + LAYERS_SRC + '\nprint("CBAM layers are ready.")')

md("## 9. Build the three models\nAll models take RGB pixels in 0–255 of any size and resize inside the model, so the backend does no extra scaling.")
code(r'''
def build_custom_cnn(number_of_classes):
    inputs = keras.Input(shape=(None, None, 3), name="image")
    x = ResizeTo(IMG_SIZE, name="resize")(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)
    for block, filters in enumerate([32, 64, 128, 256], start=1):
        if filters == 256:
            x = layers.Conv2D(filters, 3, padding="same", name=f"conv{block}")(x)
            x = layers.Activation("relu", name="last_conv_act")(x)     # feature layer used by Grad-CAM
        else:
            x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"conv{block}")(x)
        x = layers.MaxPooling2D(name=f"pool{block}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(number_of_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")


def build_efficientnet_backbone():
    # IMPORTANT: do NOT pass name="backbone" to EfficientNetB0. Keras builds the weight-file name from it and
    # tries to download ".../backbone_notop.h5" (HTTP 403). Build with the default name, then wrap/rename.
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    return keras.Model(base.input, base.output, name="backbone")


def build_efficientnet(number_of_classes, use_cbam=False):
    inputs = keras.Input(shape=(None, None, 3), name="image")
    x = ResizeTo(IMG_SIZE, name="resize")(inputs)
    backbone = build_efficientnet_backbone()
    backbone.trainable = False
    x = backbone(x)                      # EfficientNetB0 rescales/normalises 0-255 pixels itself
    if use_cbam:
        x = ChannelAttention(name="channel_attention")(x)
        x = SpatialAttention(name="spatial_attention")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(number_of_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="e3_hybrid_cbam" if use_cbam else "e2_efficientnetb0")

print("Model builders are ready.")
''')

md("## 10. Training helper\nTransfer-learning models train in two stages: new head first, then the last 20 backbone layers (BatchNorm stays frozen). Best-validation weights are always restored at the end.")
code(r'''
class RestoreBestWeights(keras.callbacks.Callback):
    """Keep the weights of the epoch with the lowest val_loss and put them back when training ends.
    (Keras' own EarlyStopping only restores them when it actually triggers.)"""
    def __init__(self, baseline_ds=None):
        super().__init__(); self.baseline_ds = baseline_ds
    def on_train_begin(self, logs=None):
        self.best, self.best_weights, self.best_epoch = float("inf"), None, None
        if self.baseline_ds is not None:   # stage 2: the stage-1 result is the one to beat
            self.best = float(self.model.evaluate(self.baseline_ds, verbose=0, return_dict=True)["loss"])
            self.best_weights, self.best_epoch = self.model.get_weights(), 0
    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get("val_loss")
        if value is not None and value < self.best:
            self.best, self.best_weights, self.best_epoch = float(value), self.model.get_weights(), epoch + 1
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"Restored weights from epoch {self.best_epoch} (val_loss={self.best:.4f})")


def make_callbacks(patience, baseline_ds=None):
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
            RestoreBestWeights(baseline_ds)]


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def train_model(model, transfer_learning=False):
    start_time, histories = time.time(), []
    if transfer_learning:
        model.get_layer("backbone").trainable = False
        compile_model(model, 1e-3)
        histories.append(model.fit(train_ds, validation_data=val_ds, epochs=TRANSFER_HEAD_EPOCHS,
                                   callbacks=make_callbacks(3), verbose=1).history)
        backbone = model.get_layer("backbone")
        backbone.trainable = True
        for layer in backbone.layers[:-20]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        compile_model(model, 1e-5)
        histories.append(model.fit(train_ds, validation_data=val_ds, epochs=TRANSFER_FINE_TUNE_EPOCHS,
                                   callbacks=make_callbacks(3, baseline_ds=val_ds), verbose=1).history)
    else:
        compile_model(model, 1e-3)
        histories.append(model.fit(train_ds, validation_data=val_ds, epochs=CNN_EPOCHS,
                                   callbacks=make_callbacks(5), verbose=1).history)
    return model, histories, time.time() - start_time


def plot_history(histories, title):
    plt.figure(figsize=(7, 4))
    for n, h in enumerate(histories, start=1):
        plt.plot(h["accuracy"], label=f"Stage {n} train"); plt.plot(h["val_accuracy"], linestyle="--", label=f"Stage {n} validation")
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True); plt.show()
''')

md("## 11. Train E1 — Custom CNN")
code(r'''
custom_cnn = build_custom_cnn(len(CLASS_NAMES))
custom_cnn.summary()
custom_cnn, cnn_history, cnn_seconds = train_model(custom_cnn, transfer_learning=False)
plot_history(cnn_history, "E1 Custom CNN training history")
''')
md("## 12. Train E2 — EfficientNetB0")
code(r'''
efficientnet = build_efficientnet(len(CLASS_NAMES), use_cbam=False)
efficientnet.summary()
efficientnet, efficientnet_history, efficientnet_seconds = train_model(efficientnet, transfer_learning=True)
plot_history(efficientnet_history, "E2 EfficientNetB0 training history")
''')
md("## 13. Train E3 — Proposed EfficientNetB0 + CBAM")
code(r'''
hybrid_model = build_efficientnet(len(CLASS_NAMES), use_cbam=True)
hybrid_model.summary()
hybrid_model, hybrid_history, hybrid_seconds = train_model(hybrid_model, transfer_learning=True)
plot_history(hybrid_history, "E3 EfficientNetB0 + CBAM training history")
''')

md("## 14. Evaluate the models\nThe test set is used only here. Macro precision/recall/F1 are reported because TrashNet is imbalanced.")
code(r'''
TRAINED = {
    "e1_cnn": dict(model=custom_cnn, history=cnn_history, seconds=cnn_seconds),
    "e2_efficientnetb0": dict(model=efficientnet, history=efficientnet_history, seconds=efficientnet_seconds),
    "e3_hybrid_cbam": dict(model=hybrid_model, history=hybrid_history, seconds=hybrid_seconds),
}
assert list(TRAINED) == list(MODEL_SPECS)

def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(precision),
        "macro_recall": float(recall), "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
        "per_class": {n: {"precision": float(report[n]["precision"]), "recall": float(report[n]["recall"]),
                          "f1-score": float(report[n]["f1-score"]), "support": int(report[n]["support"])} for n in CLASS_NAMES},
    }

y_train, y_val, y_test = (d["label_idx"].to_numpy() for d in (train_df, val_df, test_df))

RESULTS = {}
for model_id, info in TRAINED.items():
    model = info["model"]
    train_probs = model.predict(train_eval_ds, verbose=0)
    val_probs = model.predict(val_ds, verbose=0)
    test_probs = model.predict(test_ds, verbose=0)
    assert test_probs.shape == (len(test_df), len(CLASS_NAMES))
    RESULTS[model_id] = {"train": calculate_metrics(y_train, train_probs.argmax(1)), "val": calculate_metrics(y_val, val_probs.argmax(1)),
                         "test": calculate_metrics(y_test, test_probs.argmax(1)),
                         "val_probabilities": val_probs, "test_probabilities": test_probs}
    print("\n", MODEL_SPECS[model_id]["code"], MODEL_SPECS[model_id]["name"])
    print("Test accuracy:", round(RESULTS[model_id]["test"]["accuracy"], 4), "| Macro F1:", round(RESULTS[model_id]["test"]["macro_f1"], 4))
    print(classification_report(y_test, test_probs.argmax(1), target_names=CLASS_NAMES, zero_division=0))
''')

md("## 15. Compare model performance and confusion matrices")
code(r'''
rows = [{"Model": f'{MODEL_SPECS[i]["code"]} {MODEL_SPECS[i]["name"]}', "Train accuracy": r["train"]["accuracy"], "Validation accuracy": r["val"]["accuracy"],
         "Test accuracy": r["test"]["accuracy"], "Macro precision": r["test"]["macro_precision"], "Macro recall": r["test"]["macro_recall"],
         "Macro F1": r["test"]["macro_f1"]} for i, r in RESULTS.items()]
comparison_df = pd.DataFrame(rows)
display(comparison_df.round(4))
comparison_df.set_index("Model")[["Test accuracy", "Macro F1"]].plot(kind="bar", ylim=(0, 1), title="Test performance comparison")
plt.ylabel("Score"); plt.xticks(rotation=20); plt.tight_layout(); plt.show()

for model_id in TRAINED:
    cm = np.array(RESULTS[model_id]["test"]["confusion_matrix"])
    plt.figure(figsize=(6, 5)); plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {MODEL_SPECS[model_id]['code']} {MODEL_SPECS[model_id]['name']}")
    plt.xlabel("Predicted class"); plt.ylabel("Actual class")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right"); plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for r_ in range(cm.shape[0]):
        for c_ in range(cm.shape[1]):
            plt.text(c_, r_, cm[r_, c_], ha="center", va="center")
    plt.tight_layout(); plt.show()
''')

md("## 16. Uncertainty thresholds (validation only) and E2-vs-E3 significance test\nThresholds are chosen on validation predictions; the test split only reports what the chosen threshold does. McNemar's exact test compares E2 and E3 on the same test images.")
code(r'''
THRESHOLDS = {}
for model_id in TRAINED:
    t = choose_confidence_threshold(RESULTS[model_id]["val_probabilities"], y_val)
    t["test"] = apply_threshold(RESULTS[model_id]["test_probabilities"], y_test, t["threshold"])
    THRESHOLDS[model_id] = t
display(pd.DataFrame({k: {"threshold": v["threshold"], "val_coverage": v["val_coverage"], "val_selective_accuracy": v["val_selective_accuracy"],
                          "test_coverage": v["test"]["coverage"], "test_selective_accuracy": v["test"]["selective_accuracy"]} for k, v in THRESHOLDS.items()}).T)

correct = {i: RESULTS[i]["test_probabilities"].argmax(1) == y_test for i in TRAINED}
STATISTICS = {"mcnemar_e2_vs_e3": mcnemar_exact(correct["e2_efficientnetb0"], correct["e3_hybrid_cbam"])}
print("McNemar exact test, E2 vs E3:", STATISTICS["mcnemar_e2_vs_e3"])
''')

md("""
## 17. Export step 1 - sample images and final `.keras` files
The artifact folder is wiped first, so nothing stale (for example an old model file) can end up in the ZIP. **Each model is saved exactly once; the SHA-256 is computed from these files afterwards.**
""")
code(r'''
ARTIFACTS = Path("/content/ecosort_artifacts")
shutil.rmtree(ARTIFACTS, ignore_errors=True)
for folder in ["models", "metrics", "training_history", "confusion_matrices", "samples"]:
    (ARTIFACTS / folder).mkdir(parents=True)

# Real example images from the held-out test split (examples, not predictions).
sample_rows = []
for class_name in CLASS_NAMES:
    row = test_df[test_df["label"] == class_name].iloc[0]
    filename = f"example_{class_name}.jpg"
    shutil.copy(row["path"], ARTIFACTS / "samples" / filename)
    sample_rows.append({"file": filename, "kind": "example", "true_label": class_name, "note": "Example image from the held-out test split."})
(ARTIFACTS / "samples" / "samples.json").write_text(json.dumps(sample_rows, indent=2))
assert (ARTIFACTS / REFERENCE_SAMPLE).is_file()

for model_id, info in TRAINED.items():
    info["model"].save(ARTIFACTS / "models" / f"{model_id}.keras")
    print("saved", model_id, round((ARTIFACTS / "models" / f"{model_id}.keras").stat().st_size / 1e6, 2), "MB")
''')

md("""
## 18. Export step 2 - reload every saved model and prove the weights were restored
Each `.keras` file is loaded again **on the CPU**, exactly like the backend does. For every layer (including E3's CBAM `mlp_reduce` / `mlp_expand` Dense layers and the spatial conv) the reloaded weights must be identical to the trained ones, and predictions must match. This cell also measures CPU latency and records the reference probabilities the backend re-checks at startup. If anything fails, do **not** use the artifacts.
""")
code(r'''
REFERENCE_PROBS, CPU_MS, VERIFICATION = {}, {}, {}
reference_image = np.asarray(Image.open(ARTIFACTS / REFERENCE_SAMPLE).convert("RGB"), dtype="float32")[None, ...]
parity_batch = np.concatenate([np.asarray(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype="float32")[None] for p in test_df["path"].iloc[:16]])

for model_id, info in TRAINED.items():
    original = info["model"]
    with tf.device("/CPU:0"):
        reloaded = keras.models.load_model(ARTIFACTS / "models" / f"{model_id}.keras")

        # 1) every layer's weights identical (nested backbone included)
        checked_layers, checked_arrays = 0, 0
        for layer in original.layers:
            a, b = layer.get_weights(), reloaded.get_layer(layer.name).get_weights()
            assert len(a) == len(b), f"{model_id}/{layer.name}: {len(a)} weights saved vs {len(b)} restored"
            for x, y in zip(a, b):
                assert np.array_equal(x, y), f"{model_id}/{layer.name}: restored weights differ from trained weights"
            checked_layers += 1; checked_arrays += len(a)

        # 2) CBAM Dense layers explicitly present with trained weights
        cbam_report = None
        if MODEL_SPECS[model_id]["attention"] == "cbam":
            ca = reloaded.get_layer("channel_attention")
            assert [ca.fc1.name, ca.fc2.name] == ["mlp_reduce", "mlp_expand"]
            assert len(ca.fc1.get_weights()) == 2 and len(ca.fc2.get_weights()) == 2, "CBAM Dense layers have no restored variables"
            sa = reloaded.get_layer("spatial_attention")
            assert len(sa.conv.get_weights()) == 2
            cbam_report = {"mlp_reduce": [list(w.shape) for w in ca.fc1.get_weights()], "mlp_expand": [list(w.shape) for w in ca.fc2.get_weights()],
                           "spatial_conv": [list(w.shape) for w in sa.conv.get_weights()]}

        # 3) predictions agree, output is a 6-way probability vector
        p_orig, p_re = original.predict(parity_batch, verbose=0), reloaded.predict(parity_batch, verbose=0)
        max_diff = float(np.abs(p_orig - p_re).max())
        assert max_diff < 2e-2, f"{model_id}: reloaded predictions differ from the trained model (max probability difference {max_diff})"
        ref = reloaded(reference_image, training=False).numpy()[0]
        assert ref.shape == (len(CLASS_NAMES),) and np.isfinite(ref).all() and abs(float(ref.sum()) - 1.0) < 1e-3
        REFERENCE_PROBS[model_id] = ref

        # 4) CPU latency (batch 1, 512x384 sample, median of 20 runs after warm-up)
        reloaded(reference_image, training=False)
        times = []
        for _ in range(20):
            t0 = time.perf_counter(); reloaded(reference_image, training=False).numpy(); times.append((time.perf_counter() - t0) * 1000)
        CPU_MS[model_id] = float(np.median(times))

    VERIFICATION[model_id] = {"layers_checked": checked_layers, "weight_arrays_identical": checked_arrays, "max_prob_diff_vs_trained": max_diff,
                              "reference_prediction": CLASS_NAMES[int(ref.argmax())], "reference_confidence": float(ref.max()),
                              "cpu_ms_median": CPU_MS[model_id], "cbam_weights": cbam_report}
    print(model_id, "OK ->", VERIFICATION[model_id])
print("\nAll three exported models reload with identical trained weights.")
''')

md("## 19. Export step 3 - metadata (with SHA-256), metrics, thresholds, environment")
code(r'''
# --- model_metadata.json: the SHA-256 of every FINAL .keras file is computed here, after saving ---
model_metadata = [build_model_metadata_entry(i, ARTIFACTS, REFERENCE_PROBS[i]) for i in TRAINED]
for m in model_metadata:
    print(m["id"], m["sha256"])

versions = {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__}
(ARTIFACTS / "model_metadata.json").write_text(json.dumps({
    "schema_version": 1, "models": model_metadata, "default_model_id": DEFAULT_MODEL_ID, "training_environment": versions}, indent=2))
(ARTIFACTS / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2))
(ARTIFACTS / "preprocessing_config.json").write_text(json.dumps({
    "schema_version": 1, "image_size": IMG_SIZE, "color_mode": "RGB", "input_range": "0-255 float32",
    "resize": "bilinear (tf.image.resize), performed inside every model by the ResizeTo layer",
    "preprocessing": "baked into each model (Rescaling for E1, EfficientNet's built-in normalisation for E2/E3)",
    "backend_must": "decode image to RGB float32 array 0-255 and call the model without extra scaling"}, indent=2))
(ARTIFACTS / "confidence_thresholds.json").write_text(json.dumps({
    "schema_version": 1, "metric": "max softmax probability", "chosen_on": "validation split only", "models": THRESHOLDS}, indent=2))

metrics_output = {}
for model_id, info in TRAINED.items():
    model, test_m, path = info["model"], RESULTS[model_id]["test"], ARTIFACTS / "models" / f"{model_id}.keras"
    spec = MODEL_SPECS[model_id]
    metrics_output[model_id] = {
        "code": spec["code"], "name": spec["name"], "seed": SEED,
        "train_accuracy": RESULTS[model_id]["train"]["accuracy"], "val_accuracy": RESULTS[model_id]["val"]["accuracy"],
        "test_accuracy": test_m["accuracy"], "macro_precision": test_m["macro_precision"], "macro_recall": test_m["macro_recall"],
        "macro_f1": test_m["macro_f1"], "confusion_matrix": test_m["confusion_matrix"], "per_class": test_m["per_class"],
        "training_seconds": float(info["seconds"]), "epochs_run": int(sum(len(s.get("loss", [])) for s in info["history"])),
        "inference_ms_cpu": CPU_MS[model_id], "inference_device": "Colab CPU, batch 1, median of 20 runs",
        "model_size_mb": round(path.stat().st_size / 1e6, 3), "params_total": int(model.count_params()),
        "params_trainable_at_end": int(sum(int(np.prod(v.shape)) for v in model.trainable_weights)),
        "uncertainty": THRESHOLDS[model_id]}
    (ARTIFACTS / "training_history" / f"{model_id}.json").write_text(json.dumps(info["history"], indent=2))
    (ARTIFACTS / "confusion_matrices" / f"{model_id}.json").write_text(json.dumps({"class_names": CLASS_NAMES, "matrix": test_m["confusion_matrix"]}, indent=2))

dataset_summary = {
    "name": "TrashNet", "source": "https://github.com/garythung/trashnet", "license": "MIT (see the TrashNet repository)",
    "classes": CLASS_NAMES, "class_counts": {n: int((data["label"] == n).sum()) for n in CLASS_NAMES},
    "total_images": int(len(data)),
    "splits": {"train": {"n": int(len(train_df)), "fraction": round(len(train_df) / len(data), 4)},
               "validation": {"n": int(len(val_df)), "fraction": round(len(val_df) / len(data), 4)},
               "test": {"n": int(len(test_df)), "fraction": round(len(test_df) / len(data), 4)}},
    "exact_duplicates_removed": int(duplicate_count), "conflicting_label_images_removed": int(conflicting_label_removed), "unreadable_images_removed": int(len(bad_files))}
(ARTIFACTS / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))

(ARTIFACTS / "metrics" / "metrics.json").write_text(json.dumps({
    "schema_version": 1, "smoke_test": False, "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "protocol": "stratified 70/15/15 split, held-out test set, single run (seed 42)",
    "class_names": CLASS_NAMES,
    "train_config": {"image_size": IMG_SIZE, "batch_size": BATCH_SIZE, "seed": SEED, "cnn_epochs": CNN_EPOCHS,
                     "transfer_head_epochs": TRANSFER_HEAD_EPOCHS, "transfer_fine_tune_epochs": TRANSFER_FINE_TUNE_EPOCHS},
    "dataset": dataset_summary, "models": metrics_output, "statistics": STATISTICS}, indent=2))

freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
(ARTIFACTS / "environment.txt").write_text(f"# EcoSort AI training environment\n# python=={versions['python']}\n# tensorflow=={versions['tensorflow']}\n# keras=={versions['keras']}\n" + freeze)
(ARTIFACTS / "verification_report.json").write_text(json.dumps({"status": "passed", "environment": versions, "models": VERIFICATION}, indent=2))
print("\nArtifact files written.")
''')

md("## 20. Validate the folder the way the backend does, then build and re-verify the ZIP")
code(r'''
problems = validate_artifact_folder(ARTIFACTS)
assert not problems, "Artifact validation FAILED:\n - " + "\n - ".join(problems)
print("Folder validation passed (all SHA-256 hashes match the files on disk).")

zip_path = make_artifact_zip(ARTIFACTS, "/content/ecosort_artifacts.zip")
zip_problems = verify_artifact_zip(zip_path)
assert not zip_problems, "ZIP validation FAILED:\n - " + "\n - ".join(zip_problems)
with zipfile.ZipFile(zip_path) as zf:
    print("\nZIP contents:"); [print("  ", n) for n in zf.namelist()]
print("\nZIP verified by extracting it and re-checking every hash:", zip_path, "|", round(zip_path.stat().st_size / 1e6, 1), "MB")
print("Recorded SHA-256 values:"); [print("  ", m["id"], m["sha256"]) for m in model_metadata]
''')

md("## 21. Download\nExtract `ecosort_artifacts.zip` **into** `ecosort-ai/ml/artifacts/`, then run `python check_artifacts.py` in `ecosort-ai/backend`.")
code(r'''
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as exc:
    print("Automatic download unavailable. The ZIP is at:", zip_path)
''')

md('''
## Notes for your presentation
- **E1 Custom CNN** is trained from scratch; **E2 EfficientNetB0** uses ImageNet transfer learning; **E3** adds CBAM (channel then spatial attention) to E2 with an identical head, so the E2-vs-E3 difference isolates the attention module.
- All three use the same split, same data pipeline and macro-averaged test metrics; the test set is used only for the final evaluation; thresholds are chosen on validation data.
- McNemar's exact test (section 16) tells you whether E3 vs E2 differs beyond chance on this single split. Results come from one run and may be within noise.

**Limitations:** TrashNet is small with controlled backgrounds; exact-duplicate removal does not remove near-duplicate photos of the same object, so test results may be optimistic and may not transfer to real-world photos.
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT, len(cells), "cells")
