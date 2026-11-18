"""Regression test for "Layer 'mlp_reduce' expected 2 variables, but received 0 variables during loading".

Needs TensorFlow 2.21 / Keras 3.15 (skipped otherwise).  It builds a tiny model with the REAL ecosort_ml
CBAM layers (the ones the notebook embeds), randomises their weights, saves a .keras file, loads it again
and requires every weight - especially mlp_reduce / mlp_expand - to be restored exactly.
Run from backend/:  python -m unittest tests.test_cbam_roundtrip -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml" / "src"))
try:
    import keras
    from ecosort_ml.layers import ChannelAttention, ResizeTo, SpatialAttention  # noqa: F401  (registers ecosort>*)
    HAVE_TF = True
except Exception:  # noqa: BLE001
    HAVE_TF = False


@unittest.skipUnless(HAVE_TF, "TensorFlow/Keras not installed")
class TestCbamRoundTrip(unittest.TestCase):
    def test_cbam_weights_survive_save_and_load(self):
        inp = keras.Input(shape=(None, None, 3), name="image")
        x = ResizeTo(32, name="resize")(inp)
        x = keras.layers.Conv2D(32, 3, padding="same", name="conv")(x)
        x = ChannelAttention(name="channel_attention")(x)
        x = SpatialAttention(name="spatial_attention")(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        out = keras.layers.Dense(6, activation="softmax")(x)
        model = keras.Model(inp, out)
        rng = np.random.default_rng(0)
        for layer in model.layers:                                   # make every weight non-default
            layer.set_weights([rng.normal(size=w.shape).astype("float32") for w in layer.get_weights()])
        ca = model.get_layer("channel_attention")
        self.assertEqual([ca.fc1.name, ca.fc2.name], ["mlp_reduce", "mlp_expand"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.keras"
            model.save(path)
            loaded = keras.models.load_model(path)                   # must NOT raise "expected 2 variables, but received 0"
        for layer in model.layers:
            for a, b in zip(layer.get_weights(), loaded.get_layer(layer.name).get_weights()):
                np.testing.assert_array_equal(a, b)
        lca = loaded.get_layer("channel_attention")
        self.assertEqual(len(lca.fc1.get_weights()), 2); self.assertEqual(len(lca.fc2.get_weights()), 2)
        img = rng.uniform(0, 255, size=(1, 40, 52, 3)).astype("float32")
        np.testing.assert_allclose(model(img, training=False).numpy(), loaded(img, training=False).numpy(), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
