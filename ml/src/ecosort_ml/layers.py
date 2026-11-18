"""Custom Keras layers used by EcoSort models (SINGLE SOURCE OF TRUTH).

The training notebook embeds the exact source of this file (ml/tools/build_notebook.py copies it into
the layers cell, and ml/tests/test_notebook_sync.py fails if the two ever differ). The backend imports
this module before calling ``keras.models.load_model``. Because both sides register the SAME class under
the SAME name ("ecosort>ChannelAttention", ...), a saved .keras file always deserializes into an
identical layer structure, so every trained weight can be restored.

Why this matters: a ``.keras`` file stores the weights of a layer under the attribute names of its
sub-layers. If training used one ChannelAttention implementation (for example a nested Sequential) and
the backend registered another (named Dense layers ``mlp_reduce`` / ``mlp_expand``), Keras 3 raises
"Layer 'mlp_reduce' expected 2 variables, but received 0 variables during loading".
"""
import keras
import tensorflow as tf
from keras import layers, ops

PACKAGE = "ecosort"


@keras.saving.register_keras_serializable(package=PACKAGE)
class ResizeTo(layers.Layer):
    """Bilinear resize to a fixed square size (first layer of every model)."""

    def __init__(self, size=224, **kwargs):
        super().__init__(**kwargs)
        self.size = int(size)

    def call(self, inputs):
        return tf.image.resize(inputs, (self.size, self.size), method="bilinear")

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.size, self.size, input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"size": self.size})
        return config


@keras.saving.register_keras_serializable(package=PACKAGE)
class ChannelAttention(layers.Layer):
    """CBAM channel attention: shared MLP (mlp_reduce -> mlp_expand) on avg- and max-pooled descriptors."""

    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = int(reduction_ratio)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden = max(channels // self.reduction_ratio, 1)
        # Two explicit, named Dense sub-layers (NOT a nested keras.Sequential) and both are built here,
        # so their variables exist before Keras restores the saved weights.
        self.fc1 = layers.Dense(hidden, activation="relu", use_bias=True, name="mlp_reduce")
        self.fc2 = layers.Dense(channels, use_bias=True, name="mlp_expand")
        self.fc1.build((None, channels))
        self.fc2.build((None, hidden))
        super().build(input_shape)

    def call(self, x):
        avg = ops.mean(x, axis=(1, 2))
        mx = ops.max(x, axis=(1, 2))
        attn = ops.sigmoid(self.fc2(self.fc1(avg)) + self.fc2(self.fc1(mx)))
        return x * attn[:, None, None, :]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


@keras.saving.register_keras_serializable(package=PACKAGE)
class SpatialAttention(layers.Layer):
    """CBAM spatial attention: channel-wise mean and max maps -> 7x7 conv -> sigmoid."""

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, self.kernel_size, padding="same", use_bias=True, name="spatial_conv")
        self.conv.build((None, input_shape[1], input_shape[2], 2))
        super().build(input_shape)

    def call(self, x):
        avg = ops.mean(x, axis=-1, keepdims=True)
        mx = ops.max(x, axis=-1, keepdims=True)
        attn = ops.sigmoid(self.conv(ops.concatenate([avg, mx], axis=-1)))
        return x * attn

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


CUSTOM_LAYERS = {
    "ResizeTo": ResizeTo,
    "ChannelAttention": ChannelAttention,
    "SpatialAttention": SpatialAttention,
}
