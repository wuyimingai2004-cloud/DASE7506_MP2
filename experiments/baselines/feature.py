"""CPU NumPy inference for feature-aware MLPs and a small seed ensemble."""
import json
from pathlib import Path

import numpy as np

from mp2.features import FEATURE_NAMES, build_features, mask_unparseable


def load_member(root):
    config = json.loads((root / "config.json").read_text())
    if config.get("format") != "mp2-feature-mlp-v1":
        raise ValueError("Unsupported checkpoint format.")
    if config["metadata_features"] != FEATURE_NAMES:
        raise ValueError("Checkpoint feature schema mismatch.")
    with np.load(root / "weights.npz", allow_pickle=False) as archive:
        weights = {name: archive[name].copy() for name in archive.files}
    dims = config["layer_dims"]
    expected = {"metadata": 1024 + len(FEATURE_NAMES),
                "context": 2048 + len(FEATURE_NAMES)}[config["feature_mode"]]
    if dims[0] != expected or dims[-1] != 1:
        raise ValueError("Invalid network dimensions.")
    if weights["mean"].shape != (dims[0],) or weights["scale"].shape != (dims[0],):
        raise ValueError("Invalid normalization shapes.")
    if (weights["scale"] <= 0).any():
        raise ValueError("Scales must be positive.")
    for i, (din, dout) in enumerate(zip(dims, dims[1:])):
        if weights[f"weight_{i}"].shape != (din, dout) or weights[f"bias_{i}"].shape != (dout,):
            raise ValueError("Invalid layer weight shape.")
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in weights.values()):
        raise ValueError("Expected finite float32 checkpoint arrays.")
    return config, weights


def member_logits(features, config, weights, batch_size=256):
    scores = np.empty(features.shape[:2], dtype=np.float32)
    layers = len(config["layer_dims"]) - 1
    for start in range(0, len(features), batch_size):
        hidden = ((features[start:start + batch_size] - weights["mean"])
                  / weights["scale"]).reshape(-1, config["layer_dims"][0])
        for i in range(layers):
            hidden = hidden @ weights[f"weight_{i}"] + weights[f"bias_{i}"]
            if i + 1 < layers:
                np.maximum(hidden, 0, out=hidden)
        scores[start:start + batch_size] = hidden.reshape(-1, 4)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores.")
    return scores


def score_bank(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("A feature-MLP checkpoint is required.")
    root = Path(checkpoint)
    if (root / "ensemble.json").is_file():
        manifest = json.loads((root / "ensemble.json").read_text())
        if manifest.get("format") != "mp2-feature-ensemble-v1":
            raise ValueError("Unsupported ensemble format.")
        members = manifest["members"]
        if not members or any(Path(name).name != name or name in (".", "..") for name in members):
            raise ValueError("Invalid ensemble members.")
        roots = [root / name for name in members]
    else:
        roots = [root]
    total = np.zeros((len(bank), 4), dtype=np.float32)
    cache = {}
    for member in roots:
        config, weights = load_member(member)
        mode = config["feature_mode"]
        if mode not in cache:
            cache[mode] = build_features(bank, arrays["embeddings"], mode)
        features, valid = cache[mode]
        total += member_logits(features, config, weights)
    return mask_unparseable(total / len(roots), valid)


def predict(bank, arrays, checkpoint):
    return score_bank(bank, arrays, checkpoint).argmax(axis=1).tolist()
