# API (FastAPI, prefix `/api`) - interactive docs at http://127.0.0.1:8000/docs

| Method & path | Purpose |
|---|---|
| `GET /api/health` | `ok` or `degraded` + `problems`, `warnings` (e.g. TF/Keras version differs from training), loaded model count (3 expected), `smoke_test` flag |
| `GET /api/models` | the three models (`e1_cnn`, `e2_efficientnetb0`, `e3_hybrid_cbam`) with id, code, name, description, threshold, `is_default`, `loaded` |
| `GET /api/models/compare` | the real `metrics.json` (accuracy, macro P/R/F1, confusion matrices, times, sizes, McNemar E2 vs E3) |
| `GET /api/categories` | six categories with guidance, bin order |
| `GET /api/dataset` | dataset summary from the notebook |
| `GET /api/samples`, `GET /api/samples/{file}` | demo images exported by the notebook |
| `POST /api/predict` | multipart `file` (+ optional `model_id`) → predicted class, confidence, top3, probabilities, model id, inference ms, `is_uncertain` |
| `POST /api/predict/compare` | multipart `file` (+ optional comma-separated `model_ids`) → one prediction per model |
| `POST /api/explain` | multipart `file` (+ `model_id`) → Grad-CAM heatmap/overlay (base64 PNG) + peak region |

Errors always look like `{"error": {"code": "...", "message": "..."}}`:
`empty_file` 400 · `invalid_image` 400 · `unsupported_type` 415 · `file_too_large` 413 · `unknown_model` 404 · `models_unavailable` 503 · `bad_request` 422.
Limits: JPG/PNG/WebP, 10 MB (`ECOSORT_MAX_UPLOAD_MB`), 40 megapixels; images larger than 2048 px are downscaled before inference.
