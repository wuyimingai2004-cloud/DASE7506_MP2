"""CPU-only MLP inference. No reference answers or PyTorch are loaded here."""
from pathlib import Path
import json

import numpy as np


def load_bundle(checkpoint):
    """Load exported weights and normalization fitted on training data."""
    if checkpoint is None:
        raise ValueError("Pass --checkpoint pointing to an exported MLP directory.")
    root = Path(checkpoint)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != "mp2-mlp-numpy-v1":
        raise ValueError("Unsupported checkpoint format.")
    with np.load(root / "weights.npz", allow_pickle=False) as data:
        weights = {name: data[name].copy() for name in data.files}
    dims = config["layer_dims"]
    if len(dims) < 2 or dims[0] != 1024 or dims[-1] != 1:
        raise ValueError("Expected an MLP from 1024 features to one score.")
    if config.get("activation") != "relu":
        raise ValueError("Only ReLU checkpoints are supported.")
    for name in ("mean", "scale"):
        if weights[name].shape != (1024,):
            raise ValueError(f"Invalid {name} shape.")
    if np.any(weights["scale"] <= 0):
        raise ValueError("Standardization scales must be positive.")
    for i, (din, dout) in enumerate(zip(dims, dims[1:])):
        if weights[f"weight_{i}"].shape != (din, dout):
            raise ValueError(f"Invalid weight shape at layer {i}.")
        if weights[f"bias_{i}"].shape != (dout,):
            raise ValueError(f"Invalid bias shape at layer {i}.")
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in weights.values()):
        raise ValueError("Checkpoint arrays must be finite float32 values.")
    return config, weights


def normalize(embeddings, mean, scale):
    # [N, 4, 1024] -> [N, 4, 1024], without modifying the input array.
    return (np.asarray(embeddings, dtype=np.float32) - mean) / scale


def score_embeddings(embeddings, config, weights, batch_size=256):
    """Return one logit per candidate; higher scores are preferred."""
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (4, 1024):
        raise ValueError("Expected embeddings with shape [N, 4, 1024].")
    if not np.isfinite(x).all() or batch_size < 1:
        raise ValueError("Inputs must be finite and batch_size positive.")
    result = np.empty((len(x), 4), dtype=np.float32)
    layers = len(config["layer_dims"]) - 1
    for start in range(0, len(x), batch_size):
        hidden = normalize(x[start:start + batch_size], weights["mean"], weights["scale"])
        hidden = hidden.reshape(-1, 1024)
        for i in range(layers):
            hidden = hidden @ weights[f"weight_{i}"] + weights[f"bias_{i}"]
            if i + 1 < layers:
                np.maximum(hidden, 0, out=hidden)
        result[start:start + batch_size] = hidden.reshape(-1, 4)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite MLP output.")
    return result


def predict(bank, arrays, checkpoint):
    """Official interface: return one Python int in 0..3 for each question."""
    embeddings = arrays["embeddings"]
    if len(bank) != len(embeddings):
        raise ValueError("Bank and embeddings have different question counts.")
    config, weights = load_bundle(checkpoint)
    scores = score_embeddings(embeddings, config, weights)
    # Share one network across candidates without question-ID features.
    return scores.argmax(axis=1).tolist()
