"""
1D-CNN re-identification model.

A deliberately generic 1D CNN that classifies short ECG windows by identity.
The design was not adopted from a specific published source. Structurally it
sits in the FCN family of time-series classification baselines (Wang, Yan &
Oates, IJCNN 2017): Conv1D + BatchNorm + ReLU blocks feeding a
GlobalAveragePooling layer rather than a Flatten. It departs from FCN proper
in two ways -- a VGG-style filter pyramid (32/64/128, kernel 7) with
max-pooling after each block, where FCN deliberately has no local pooling,
and a dropout + dense head.

The model is NOT a reimplementation of any published ECG-biometrics system.
Published identifiers (Donida Labati et al., PRL 126, 2019; Zhang et al.,
PRL 125, 2019; da Silva Luz et al., IEEE TIFS 13, 2018) are stronger: they
segment on fiducial points and use metric embeddings or template matching.
Using an untuned generic model is deliberate -- the measured leakage is then
a lower bound on attacker capability rather than an artefact of an identifier
engineered to find it. `reid_resnet.py` supplies the stronger second
architecture used to check that claim.

Architecture:
    Input (window_samples,)
    -> Reshape to (window_samples, 1)
    -> Conv1D(32, k=7) + BN + ReLU + MaxPool(2)
    -> Conv1D(64, k=7) + BN + ReLU + MaxPool(2)
    -> Conv1D(128, k=7) + BN + ReLU + MaxPool(2)
    -> GlobalAveragePooling1D
    -> Dropout
    -> Dense(128) + ReLU + Dropout
    -> Dense(n_patients) + Softmax (via logits + sparse CCE)
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    MaxPooling1D,
    Reshape,
)


def build_reid_model(
    window_samples: int,
    n_patients: int,
    conv_filters=(32, 64, 128),
    conv_kernel: int = 7,
    pool_size: int = 2,
    dense_units: int = 128,
    dropout: float = 0.3,
) -> tf.keras.Model:
    """
    Build the 1D-CNN classifier.

    Args:
        window_samples: length of each input window.
        n_patients: number of identities to classify.
    """
    inputs = tf.keras.Input(shape=(window_samples,), name="window_input")
    x = Reshape((window_samples, 1))(inputs)

    for filters in conv_filters:
        x = Conv1D(filters, kernel_size=conv_kernel, padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = MaxPooling1D(pool_size=pool_size, padding="same")(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)
    logits = Dense(n_patients, name="reid_logits")(x)

    return tf.keras.Model(inputs, logits, name="reid_cnn")
