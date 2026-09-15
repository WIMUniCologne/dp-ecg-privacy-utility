"""
1D-ResNet re-identification attacker (second architecture).

Why a second architecture
-------------------------
The primary attacker (`reid_cnn.build_reid_model`) is a deliberately generic,
untuned 1D CNN, so that the leakage we measure is not an artefact of an
identifier engineered to find it. That argument is only as good as the claim
that a stronger identifier would do at least as well. This module supplies the
stronger identifier so the claim can be tested rather than asserted.

Architecture: the ResNet baseline for time-series classification of
Wang, Yan & Oates (IJCNN 2017) -- the acknowledged stronger sibling of the FCN
design the primary attacker follows. Three residual blocks (64, 128, 128
filters) with kernels 8/5/3, batch normalisation and ReLU, a shortcut that is
1x1-convolved when the channel count changes, then global average pooling and a
dense identity head.

Consumes the same input as the primary attacker (raw z-normalised windows) and
produces the same output (logits over enrolled identities), so results are
directly comparable at matched (mechanism, epsilon, seed).
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
    Reshape,
)


def _residual_block(x, filters: int, kernels=(8, 5, 3)):
    """One residual block: 3 conv/BN/ReLU stages plus a shortcut."""
    shortcut = x
    for i, k in enumerate(kernels):
        x = Conv1D(filters, kernel_size=k, padding="same")(x)
        x = BatchNormalization()(x)
        if i < len(kernels) - 1:
            x = Activation("relu")(x)

    # Match channels on the shortcut when they differ, else just normalise.
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding="same")(shortcut)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    return Activation("relu")(x)


def build_reid_resnet(
    window_samples: int,
    n_patients: int,
    filters=(64, 128, 128),
) -> tf.keras.Model:
    """
    Build the 1D-ResNet identifier.

    Args:
        window_samples: length of each input window.
        n_patients: number of identities to classify.
        filters: per-block filter counts (Wang et al. default: 64, 128, 128).
    """
    inputs = tf.keras.Input(shape=(window_samples,), name="window_input")
    x = Reshape((window_samples, 1))(inputs)

    for f in filters:
        x = _residual_block(x, f)

    x = GlobalAveragePooling1D()(x)
    logits = Dense(n_patients, name="reid_logits")(x)

    return tf.keras.Model(inputs, logits, name="reid_resnet")
