"""Grad-CAM for the EcoSort models (TensorFlow backend).

All three models are strictly linear chains of layers (image -> resize -> ... -> feature layer -> ... -> softmax),
so Grad-CAM simply runs the layers one after another inside ONE GradientTape, keeps the output of the feature layer,
and differentiates the class score with respect to it.  No sub-model has to be built from nested graph tensors.
"""
import io

import numpy as np


def _tf():
    import tensorflow as tf  # local import: keeps this module importable without TF
    return tf


def gradcam(model, image_batch: np.ndarray, feature_layer: str, class_index: int | None = None, check: bool = False):
    """Return (heatmap in 0..1 shaped like the feature map, class probabilities, class used)."""
    import keras

    tf = _tf()
    chain = [layer for layer in model.layers if not isinstance(layer, keras.layers.InputLayer)]
    if feature_layer not in [layer.name for layer in chain]:
        raise ValueError(f"feature layer {feature_layer!r} not found in model (layers: {[l.name for l in chain]})")

    x = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)                      # needed: a frozen backbone has no watched variables, so nothing would be recorded
        h, fmap = x, None
        for layer in chain:
            h = layer(h)
            if layer.name == feature_layer:
                fmap = h
        probs = h
        if fmap is None:
            raise RuntimeError("feature map was not produced")
        if class_index is None:
            class_index = int(tf.argmax(probs[0]))
        score = probs[:, class_index]
    grads = tape.gradient(score, fmap)
    if grads is None:
        raise RuntimeError(f"no gradient reached feature layer {feature_layer!r}")
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * fmap, axis=-1))[0].numpy()
    if check:  # sanity: the manual chain must reproduce the real model output
        ref = model(x, training=False).numpy()
        if not np.allclose(ref, probs.numpy(), atol=1e-4):
            raise RuntimeError("Grad-CAM chain does not reproduce the model output")
    cam = cam - cam.min()
    peak = cam.max()
    cam = cam / peak if peak > 0 else cam
    return cam, probs.numpy()[0], class_index


def _jet(values: np.ndarray) -> np.ndarray:
    """values 0..1 -> RGB float 0..1 (classic 'jet' colour map, no matplotlib needed)."""
    r = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def colorize(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Heatmap (0..1) -> RGB uint8 image of ``size`` (w, h) using a blue->red colormap."""
    from PIL import Image

    img = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8)).resize(size, Image.Resampling.BICUBIC)
    return (_jet(np.asarray(img) / 255.0) * 255).astype(np.uint8)


def overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    heat = colorize(heatmap, (w, h)).astype(np.float32)
    out = (1 - alpha) * image_rgb.astype(np.float32) + alpha * heat
    return np.clip(out, 0, 255).astype(np.uint8)


def to_png_bytes(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()