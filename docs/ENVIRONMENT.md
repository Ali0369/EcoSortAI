# Environment and version compatibility

| Component | Version | Where it is pinned |
|---|---|---|
| TensorFlow | **2.21.0** | notebook cell 1 (`PINNED`) + `backend/requirements.txt` |
| Keras | **3.15.1** | notebook cell 1 + `backend/requirements.txt` |
| Python (backend) | 3.10-3.12 | |
| Node (frontend) | 18+ | `frontend/package.json` |

* Your Colab run printed `TensorFlow: 2.21.0`; its Keras version was not in the output you attached. Cell 1 compares the installed versions with the pins, installs the pins if they differ, and asks for a session restart; cell 2 then asserts them (`ALLOW_VERSION_MISMATCH = False`). If pip has to replace TensorFlow itself, Colab may lose GPU support for that session - in that case set the GPU runtime again or use a fresh runtime.
* Exports record the real versions in `model_metadata.json → training_environment` and `environment.txt` (pip freeze). The backend compares them with what is installed and shows a warning in `/api/health` if they differ.
* `.keras` files are version-sensitive. If you ever upgrade TensorFlow/Keras in the backend, re-run `python check_artifacts.py`; if it fails, re-export with matching versions.
* The custom layers use `tf.image.resize` and `keras.ops`; the TensorFlow backend of Keras is required (default with `pip install tensorflow`).
* **Not verified by me:** that TensorFlow 2.21.0 and Keras 3.15.1 resolve together in a fresh Colab pip install (you reported them installed together in your backend, which is the basis for the pins).
