"""Seq2Seq classifier (utility) and Re-ID CNN (attacker)."""
from src.models.seq2seq import build_seq2seq_model, GreedyDecoder
from src.models.pipeline import (
    apply_smote,
    build_decoder_inputs,
    build_vocab,
    confusion_matrix_per_class,
    labels_to_targets,
    metrics_from_cm,
    read_mitbih_mat,
    read_mitbih_wfdb,
)
from src.models.reid_cnn import build_reid_model
from src.models.reid_resnet import build_reid_resnet
from src.models.denoise_ae import (
    build_denoising_ae,
    train_denoising_ae,
    ae_denoise,
)
from src.models.ecg_features import extract_beat_features, feature_names

__all__ = [
    # Utility classifier
    "build_seq2seq_model",
    "GreedyDecoder",
    "apply_smote",
    "build_decoder_inputs",
    "build_vocab",
    "confusion_matrix_per_class",
    "labels_to_targets",
    "metrics_from_cm",
    "read_mitbih_mat",
    "read_mitbih_wfdb",
    # Re-ID attacker
    "build_reid_model",
    "build_reid_resnet",
    # Denoising-autoencoder attacker
    "build_denoising_ae",
    "train_denoising_ae",
    "ae_denoise",
    # Feature-based (RF) utility model
    "extract_beat_features",
    "feature_names",
]
