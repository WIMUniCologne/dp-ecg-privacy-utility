"""
1D denoising autoencoder -- the optional, stronger temporal-correlation attacker.

The low-pass denoisers in ``src/dp/denoise.py`` are fixed, hand-tuned filters.
A denoising autoencoder is a *learned* low-pass front end: a small 1D conv
encoder-decoder trained to map DP-perturbed windows back to their clean
originals, exploiting the same fact (the signal is temporally correlated, the
i.i.d. DP noise is not), but adapting the filter to the data and the noise
level instead of fixing it a priori.

Usage (see ``scripts/train_reid_denoise.py`` for the full protocol):

    ae = train_denoising_ae(clean_train, noised_train, epochs=30, seed=42)
    denoised_train = ae_denoise(ae, noised_train)
    denoised_test  = ae_denoise(ae, noised_test)
    # then train + evaluate the re-ID CNN on the denoised representation.

Threat-model note
-----------------
The autoencoder is trained only on the attacker's *training* windows (clean and
their perturbed counterparts). The test windows are never used to fit the
denoiser, so there is no train/test leakage through the AE. As with the low-pass
attacker, the released data is unchanged: the AE is purely an adversary-side
post-processing step.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    Conv1DTranspose,
    Cropping1D,
    Input,
    Reshape,
    ZeroPadding1D,
)


def build_denoising_ae(
    window_samples: int,
    base_filters: int = 32,
    depth: int = 2,
    kernel_size: int = 9,
) -> tf.keras.Model:
    """
    Build a small symmetric 1D conv denoising autoencoder.

    The encoder halves the temporal resolution ``depth`` times (stride-2
    convolutions); the decoder mirrors it with transposed convolutions. Input
    and output are both ``(window_samples,)`` -- a single z-normalized channel.

    Length handling: stride-2 down/up-sampling only round-trips cleanly when the
    length is divisible by ``2**depth``. The model pads the signal up to the
    next such multiple on input and crops back to ``window_samples`` on output,
    so any window length is accepted.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")

    factor = 2 ** depth
    pad_total = (-window_samples) % factor  # pad up to a multiple of `factor`
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    inp = Input(shape=(window_samples,), name="noised_window")
    x = Reshape((window_samples, 1))(inp)
    if pad_total:
        x = ZeroPadding1D((pad_left, pad_right))(x)

    # Encoder: stride-2 conv blocks.
    filters = base_filters
    for _ in range(depth):
        x = Conv1D(filters, kernel_size, strides=2, padding="same",
                   activation="relu")(x)
        filters *= 2

    # Bottleneck.
    x = Conv1D(filters, kernel_size, padding="same", activation="relu")(x)

    # Decoder: mirror with transposed convs.
    for _ in range(depth):
        filters //= 2
        x = Conv1DTranspose(filters, kernel_size, strides=2, padding="same",
                            activation="relu")(x)

    # Project back to one channel (linear: z-normalized signal is unbounded).
    x = Conv1D(1, kernel_size, padding="same", activation="linear")(x)
    if pad_total:
        x = Cropping1D((pad_left, pad_right))(x)
    out = Reshape((window_samples,), name="denoised_window")(x)

    return tf.keras.Model(inp, out, name="denoising_ae")


def train_denoising_ae(
    clean_X: np.ndarray,
    noised_X: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    base_filters: int = 32,
    depth: int = 2,
    kernel_size: int = 9,
    val_frac: float = 0.1,
    seed: int = 42,
    verbose: int = 0,
    logger=None,
    log_prefix: str = "",
) -> tf.keras.Model:
    """
    Fit a denoising autoencoder to map ``noised_X -> clean_X``.

    ``clean_X`` and ``noised_X`` must be aligned row-for-row (same windows,
    one clean and one DP-perturbed) and shaped ``(N, window_samples)``.
    """
    if clean_X.shape != noised_X.shape:
        raise ValueError(
            f"clean_X {clean_X.shape} and noised_X {noised_X.shape} must match."
        )

    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)

    window_samples = clean_X.shape[1]
    model = build_denoising_ae(
        window_samples, base_filters=base_filters, depth=depth,
        kernel_size=kernel_size,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mse",
    )

    # Deterministic shuffle + held-out validation split for early stopping.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(clean_X))
    clean_X = clean_X[perm]
    noised_X = noised_X[perm]
    n_val = max(1, int(len(clean_X) * val_frac)) if val_frac > 0 else 0

    if n_val > 0:
        x_val, y_val = noised_X[:n_val], clean_X[:n_val]
        x_tr, y_tr = noised_X[n_val:], clean_X[n_val:]
        validation_data = (x_val, y_val)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True,
            )
        ]
    else:
        x_tr, y_tr = noised_X, clean_X
        validation_data = None
        callbacks = []

    hist = model.fit(
        x_tr, y_tr,
        validation_data=validation_data,
        epochs=epochs, batch_size=batch_size,
        shuffle=True, verbose=verbose, callbacks=callbacks,
    )
    if logger is not None:
        final_loss = float(hist.history["loss"][-1])
        msg = f"{log_prefix}denoising-AE trained: final train mse={final_loss:.4f}"
        if validation_data is not None:
            msg += f", val mse={float(hist.history['val_loss'][-1]):.4f}"
        logger.info(msg)
    return model


def ae_denoise(
    model: tf.keras.Model,
    X: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Apply a trained denoising autoencoder to a batch of windows."""
    if X is None or np.size(X) == 0:
        return np.asarray(X, dtype=np.float32)
    out = model.predict(X, batch_size=batch_size, verbose=0)
    return out.reshape(X.shape).astype(np.float32)
