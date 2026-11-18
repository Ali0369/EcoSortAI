# Artifact contract: notebook → backend

`ml/notebooks/EcoSort_AI_Simple_Training.ipynb` writes `ecosort_artifacts.zip`. **Unzip it into `ecosort-ai/ml/artifacts/`.** The backend reads only this folder.

```
ml/artifacts/
├── class_names.json            REQUIRED  ["cardboard","glass","metal","paper","plastic","trash"] - the ONE class order
├── model_metadata.json         REQUIRED  schema_version, default_model_id, training_environment, models[]:
│                                         id, code, name, role, attention, description, file, sha256, size_bytes, input,
│                                         feature_layer, is_proposed, reference_check{image, tolerance, probabilities}
├── preprocessing_config.json   REQUIRED  image_size 224, RGB, 0-255 float32; resize + preprocessing are INSIDE each model
├── confidence_thresholds.json  REQUIRED  per-model threshold (max softmax prob) chosen on the VALIDATION split only
├── metrics/metrics.json        REQUIRED  smoke_test=false, generated_utc, protocol, class_names, train_config, dataset, models{}, statistics{mcnemar_e2_vs_e3}
├── models/                     REQUIRED  exactly: e1_cnn.keras, e2_efficientnetb0.keras, e3_hybrid_cbam.keras
├── dataset_summary.json        name, source, license, classes, class_counts, total_images, splits{train,validation,test:{n,fraction}}
├── samples/                    demo images + samples.json (example_<class>.jpg; example_cardboard.jpg is the reference image)
├── training_history/  confusion_matrices/
├── environment.txt             pip freeze + python/tensorflow/keras lines of the training run
└── verification_report.json    per-layer weight-restoration and CPU-latency results of the notebook's reload check
```

## The three models (fixed)
| id | code | role | attention | feature_layer (Grad-CAM) |
|---|---|---|---|---|
| `e1_cnn` | E1 | baseline | none | `last_conv_act` |
| `e2_efficientnetb0` | E2 | baseline | none | `backbone` |
| `e3_hybrid_cbam` | E3 | proposed (default) | cbam | `backbone` |

## Rules the backend enforces at startup
* every required file exists and has `schema_version == 1`; `class_names.json` equals the six classes in the order above; `image_size == 224`;
* `model_metadata.json` lists **exactly** the three ids above with the codes above; each model file exists, **has a recorded `sha256`, and matches it** (`ECOSORT_VERIFY_HASHES=true`, default); no unlisted file is in `models/`;
* every model has a threshold and metrics; `code`/`name` agree between metadata and metrics; class order in `metrics.json` equals `class_names.json`;
* every model reloads (custom layers registered as `ecosort>ResizeTo`, `ecosort>ChannelAttention`, `ecosort>SpatialAttention` from `ml/src/ecosort_ml/layers.py`), outputs 6 probabilities, E3's CBAM sub-layers (`mlp_reduce`, `mlp_expand`, `spatial_conv`) hold restored variables, and the output on `reference_check.image` matches the notebook's recorded probabilities within `tolerance` (0.02; disable only for debugging with `ECOSORT_VERIFY_REFERENCE=false`).

If anything fails the API still starts, `GET /api/health` returns `"status": "degraded"` with the exact problems, and prediction endpoints answer **503** - it never fabricates a prediction. A TensorFlow/Keras version different from `training_environment` is reported under `warnings`.

## Model input/output
Input: float32 RGB pixels 0..255, any size, shape `(1, H, W, 3)`. Output: softmax over `class_names`. Resizing to 224×224 (`tf.image.resize`, bilinear) and model-specific scaling are layers inside the model.

## Layer implementation (why it is shared)
`ml/src/ecosort_ml/layers.py` is the only definition of the custom layers. `ml/tools/build_notebook.py` copies it verbatim into the notebook and `ml/tests/test_notebook_sync.py` fails if they differ. `ChannelAttention` holds two named Dense layers (`mlp_reduce`, `mlp_expand`) built in `build()`; `SpatialAttention` holds one Conv2D (`spatial_conv`). Changing a layer's structure means the saved files must be re-exported.
