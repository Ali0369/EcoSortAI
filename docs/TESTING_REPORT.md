# Testing report (what was actually executed)

Build environment: Python 3.12, numpy, Pillow, scikit-learn, Node 22 with a global TypeScript compiler.
**No internet, no TensorFlow/Keras, no FastAPI/pydantic/httpx, no npm packages, no GPU, no TrashNet data.**

## PASSED (executed)
| # | Check | How |
|---|---|---|
| 1 | Notebook is valid JSON, 45 cells, every code cell compiles, outputs cleared | `ml/tests/test_notebook_sync.py` |
| 2 | Notebook layers cell is byte-identical to `ml/src/ecosort_ml/layers.py` (the file the backend imports); no nested `Sequential`; `EfficientNetB0(...)` not given a `name=` | same |
| 3 | Only E1/E2/E3 (`e1_cnn`, `e2_efficientnetb0`, `e3_hybrid_cbam`) referenced in the notebook; no MobileNet/ResNet/ablation/E4/E5 strings in notebook, backend, frontend source | same + grep |
| 4 | Notebook helper cell executed: threshold calibration, exact McNemar test (checked against hand-computed p = 2/1024), SHA-256 → `model_metadata.json` → folder validation → ZIP → re-extract → validation, using **fake model bytes** | same |
| 5 | The same artifact folders accepted/rejected identically by the notebook validator and by the real backend `load_bundle` (hash ON): valid bundle accepted; corrupted model, missing `sha256`, stale extra `.keras`, old 7-model bundle all rejected | same + `backend/tests` |
| 6 | `python check_artifacts.py` on a fake valid bundle → OK (hashes match); on a corrupted file → exit 1 with the hash problem; on the empty `ml/artifacts` → exit 1 with the missing files | executed manually |
| 7 | Backend image-validation and artifact-validation unit tests | `cd backend && python -m unittest discover -s tests` (8 pass, 3 skipped) |
| 8 | All TS/TSX files have no syntax errors | `tsc --noEmit` filtered to TS1xxx syntax errors |

## FAILED
Nothing that was run failed in its final state (the first version of one structural test was too strict about the word "Sequential" in a docstring and was corrected).

## COULD NOT BE RUN here - you must run these
| Item | Why not | How to verify |
|---|---|---|
| Training E1/E2/E3, ImageNet weight download, TrashNet download | no internet/GPU/TensorFlow | run the notebook |
| **Fix of the CBAM `mlp_reduce`/`mlp_expand` load error** | needs TensorFlow 2.21/Keras 3.15. The root cause is established from the code (two different `ecosort>ChannelAttention` classes), the fix is to use one shared class, but I have **not** executed a save/load | notebook section 18 (per-layer weight equality), `python backend/check_artifacts.py`, `python -m unittest tests.test_cbam_roundtrip` in `backend/` |
| Loading the 3 real models with the backend loader; one prediction per model; 6 probabilities each | no models exist yet (your attached run never saved any) | `python check_artifacts.py` |
| Real hashes of real model files | no model files | notebook sections 19-20 print them; `check_artifacts.py` re-checks |
| Starting the API, `/api/predict`, `/api/predict/compare`, `/api/explain` (Grad-CAM), FastAPI tests | no FastAPI/TensorFlow | `uvicorn app.main:app`, then `/api/health` |
| `npm install` / `npm run build` (TypeScript type-check, Vite) | no npm registry access; only syntax was checked | `cd frontend && npm install && npm run build` |
| Browser check that the three models appear in Scanner dropdown / comparison page | needs backend + build | open http://localhost:5173 after steps above |
| TF 2.21.0 + Keras 3.15.1 resolving in a fresh Colab | no internet | notebook cell 1/2 |

## Known risks
* Because TensorFlow was unavailable, the notebook's TensorFlow cells (model builders, training, export reload) have never been executed by me. The notebook's assertions are designed to stop with a clear message rather than export bad files; if one fails, send me the message.
* The backend's reference-output check (tolerance 0.02) compares your CPU to the notebook's CPU result; if it reports a mismatch on a legitimately different machine, you can set `ECOSORT_VERIFY_REFERENCE=false` for diagnosis (hash verification stays on).
