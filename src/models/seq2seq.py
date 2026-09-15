"""
Seq2Seq classifier (Mousavi & Afghah 2019 architecture).

This module implements the CNN-LSTM seq2seq classifier with the architectural
choices from Mousavi's reference TensorFlow code (via Ricarda's TF2 port):

  - Per-beat CNN: Reshape(10, 28) → Conv1D(32) → MaxPool → Conv1D(64) → MaxPool
                  → Conv1D(128) → Flatten = 384-dim feature per beat
  - TimeDistributed over 10 beats → (None, 10, 384)
  - LSTM encoder (128 units) → returns final (c, h) state
  - LSTM decoder (128 units, return_sequences=True) initialized with [c, h]
    (note: this is c → h0 and h → c0; not the Keras default [h, c] order)
  - Dense(vocab_size) output logits

The state-order choice — passing the encoder's cell state to the decoder's
hidden state initialization — combined with pure-class sequence training
(see `pipeline.read_mitbih_*` functions) gives the strong inter-patient
performance reported by Mousavi.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Embedding,
    Flatten,
    LSTM,
    MaxPooling1D,
    Reshape,
    TimeDistributed,
)


def build_seq2seq_model(
    n_channels: int = 10,
    input_depth: int = 280,
    max_time: int = 10,
    num_units: int = 128,
    vocab_size: int = 4,
    embed_size: int = 10,
    bidirectional: bool = False,
) -> tf.keras.Model:
    """
    Build the CNN-LSTM seq2seq classifier.

    Args:
        n_channels: number of CNN input channels (typically 10).
        input_depth: samples per beat (280 = Mousavi default).
        max_time: number of beats per sequence (10 = Mousavi default).
        num_units: LSTM hidden units (128 = Mousavi default).
        vocab_size: number of output classes including <GO> token
                    (e.g. 4 for N/S/V/<GO>).
        embed_size: decoder token embedding dimension (10).
        bidirectional: if True, use Bidirectional LSTM encoder. Default False
            (matches Mousavi's "best" configuration).

    Returns:
        Compiled Keras Model with two inputs:
          - encoder_input: (None, max_time, input_depth) float32
          - decoder_input: (None, max_time) int32 (with <GO> prefix)
        and one output:
          - logits: (None, max_time, vocab_size)
    """
    encoder_input = tf.keras.Input(
        shape=(max_time, input_depth), name="encoder_input"
    )
    decoder_input = tf.keras.Input(
        shape=(max_time,), dtype=tf.int32, name="decoder_input"
    )

    # Per-beat CNN: each (280,) beat → (10, 28) → 3x Conv1D+Pool → flatten = 384
    beat_in = tf.keras.Input(shape=(input_depth,))
    x = Reshape((n_channels, input_depth // n_channels))(beat_in)
    x = Conv1D(32, kernel_size=2, strides=1, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding="same")(x)
    x = Conv1D(64, kernel_size=2, strides=1, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding="same")(x)
    x = Conv1D(128, kernel_size=2, strides=1, padding="same", activation="relu")(x)
    beat_features = Flatten()(x)
    beat_cnn = tf.keras.Model(beat_in, beat_features, name="beat_cnn")

    # Apply CNN to each of the 10 beats independently
    data_input_embed = TimeDistributed(beat_cnn, name="time_distributed_cnn")(encoder_input)

    # Encoder
    if not bidirectional:
        lstm_enc = LSTM(num_units, return_state=True, name="encoder_lstm")
        _, last_h, last_c = lstm_enc(data_input_embed)
        initial_state = [last_c, last_h]  # NOTE: c → h0, h → c0 (Mousavi quirk)
    else:
        lstm_enc = Bidirectional(LSTM(num_units, return_state=True), name="encoder_lstm")
        _, fw_h, fw_c, bw_h, bw_c = lstm_enc(data_input_embed)
        last_c = tf.concat([fw_c, bw_c], axis=-1)
        last_h = tf.concat([fw_h, bw_h], axis=-1)
        initial_state = [last_c, last_h]

    # Decoder
    decoder_embed = Embedding(vocab_size, embed_size, name="dec_embedding")(decoder_input)
    dec_units = num_units if not bidirectional else 2 * num_units
    lstm_dec = LSTM(dec_units, return_sequences=True, name="decoder_lstm")
    dec_outputs = lstm_dec(decoder_embed, initial_state=initial_state)

    logits = Dense(vocab_size, name="output_logits")(dec_outputs)

    return tf.keras.Model(
        [encoder_input, decoder_input], logits, name="mousavi_seq2seq",
    )


# ---------------------------------------------------------------------------
# Greedy decoding for evaluation
# ---------------------------------------------------------------------------
def _build_encoder_state_model(
    full_model: tf.keras.Model,
    num_units: int,
    bidirectional: bool,
    max_time: int = 10,
    input_depth: int = 280,
) -> tf.keras.Model:
    """Extract a sub-model that returns the (c, h) initial state for the decoder."""
    td_layer = full_model.get_layer("time_distributed_cnn")
    enc_lstm_layer = full_model.get_layer("encoder_lstm")

    new_input = tf.keras.Input(shape=(max_time, input_depth))
    x = td_layer(new_input)
    if not bidirectional:
        _, h, c = enc_lstm_layer(x)
    else:
        _, fw_h, fw_c, bw_h, bw_c = enc_lstm_layer(x)
        c = tf.concat([fw_c, bw_c], axis=-1)
        h = tf.concat([fw_h, bw_h], axis=-1)
    return tf.keras.Model(new_input, [c, h], name="encoder_state")


def _build_decoder_step_model(
    full_model: tf.keras.Model,
    vocab_size: int,
    num_units: int,
    bidirectional: bool,
) -> tf.keras.Model:
    """Build a stateful one-step decoder that shares weights with the full model."""
    embed_layer = full_model.get_layer("dec_embedding")
    dec_lstm_layer = full_model.get_layer("decoder_lstm")
    out_layer = full_model.get_layer("output_logits")

    dec_units = num_units if not bidirectional else 2 * num_units

    token_in = tf.keras.Input(shape=(1,), dtype=tf.int32)
    h_in = tf.keras.Input(shape=(dec_units,))
    c_in = tf.keras.Input(shape=(dec_units,))

    emb = embed_layer(token_in)
    one_step = LSTM(dec_units, return_sequences=True, return_state=True)
    _ = one_step(emb, initial_state=[h_in, c_in])  # build
    one_step.set_weights(dec_lstm_layer.get_weights())

    out_seq, new_h, new_c = one_step(emb, initial_state=[h_in, c_in])
    logits = out_layer(out_seq)
    return tf.keras.Model([token_in, h_in, c_in], [logits, new_h, new_c])


@tf.function(reduce_retracing=True)
def _greedy_decode_batch(
    encoder_state_model: tf.keras.Model,
    decoder_step_model: tf.keras.Model,
    source_batch: tf.Tensor,
    go_id: tf.Tensor,
    y_seq_length: tf.Tensor,
) -> tf.Tensor:
    """Greedy decode one batch, fully tf.function compiled."""
    init_c, init_h = encoder_state_model(source_batch, training=False)
    # Mousavi/Ricarda order: decoder's h0 = encoder's c, decoder's c0 = encoder's h
    h_state = init_c
    c_state = init_h
    bs = tf.shape(source_batch)[0]

    last_token = tf.fill([bs, 1], go_id)
    preds_ta = tf.TensorArray(dtype=tf.int32, size=y_seq_length)

    for i in tf.range(y_seq_length):
        logits, h_state, c_state = decoder_step_model(
            [last_token, h_state, c_state], training=False,
        )
        pred = tf.argmax(logits[:, 0, :], axis=-1, output_type=tf.int32)
        preds_ta = preds_ta.write(i, pred)
        last_token = tf.expand_dims(pred, axis=1)

    return tf.transpose(preds_ta.stack(), [1, 0])


class GreedyDecoder:
    """
    Fast greedy decoder for evaluation. Caches sub-models so the @tf.function
    is compiled only once per training run.

    Usage:
        decoder = GreedyDecoder(model, num_units=128, vocab_size=4, ...)
        cm = decoder.evaluate(X_test, y_test, batch_size=128)
    """
    def __init__(
        self,
        model: tf.keras.Model,
        num_units: int = 128,
        vocab_size: int = 4,
        go_id: int = 3,
        bidirectional: bool = False,
        max_time: int = 10,
        input_depth: int = 280,
    ):
        self.model = model
        self.num_units = num_units
        self.vocab_size = vocab_size
        self.go_id = go_id
        self.bidirectional = bidirectional
        self.max_time = max_time

        self.encoder_state_model = _build_encoder_state_model(
            model, num_units, bidirectional, max_time, input_depth,
        )
        self.decoder_step_model = _build_decoder_step_model(
            model, vocab_size, num_units, bidirectional,
        )

    def sync_weights(self) -> None:
        """Refresh decoder weights from the main model (call after training updates)."""
        dec_lstm_full = self.model.get_layer("decoder_lstm")
        for layer in self.decoder_step_model.layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                layer.set_weights(dec_lstm_full.get_weights())
                break

    def decode(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Greedy-decode an array of sequences. Returns (n, max_time) int32 predictions."""
        self.sync_weights()
        go_t = tf.constant(self.go_id, dtype=tf.int32)
        len_t = tf.constant(self.max_time, dtype=tf.int32)

        all_preds = []
        n = X.shape[0]
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            preds = _greedy_decode_batch(
                self.encoder_state_model, self.decoder_step_model,
                tf.constant(X[s:e]), go_t, len_t,
            )
            all_preds.append(preds.numpy())
        return np.concatenate(all_preds, axis=0)
