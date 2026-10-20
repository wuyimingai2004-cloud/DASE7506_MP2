"""NumPy-only inference for the selected TabM + adaptive MoB ensemble."""
import json
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, build_features
from .aggregation import aggregate_heads, select_answers


def tabm_heads(features, config, weights, batch_size=128):
    """Return [N, K, 4], matching the official TabM package in eval mode."""
    n, k = len(features), config["k"]
    scores = np.empty((n, k, 4), dtype=np.float32)
    for start in range(0, n, batch_size):
        batch = features[start:start + batch_size]
        hidden = ((batch - weights["mean"]) / weights["scale"]).reshape(-1, features.shape[-1])
        hidden = hidden[:, None, :]
        for layer in range(config["n_blocks"]):
            hidden = ((hidden * weights[f"r_{layer}"]) @ weights[f"weight_{layer}"])
            hidden = hidden * weights[f"s_{layer}"] + weights[f"bias_{layer}"]
            np.maximum(hidden, 0, out=hidden)
        output = np.einsum("bkd,kdo->bko", hidden, weights["output_weight"], optimize=True)
        output += weights["output_bias"]
        scores[start:start + len(batch)] = output.reshape(len(batch), 4, k).transpose(0, 2, 1)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite TabM output.")
    return scores


def load_tabm(root):
    config = json.loads((root / "config.json").read_text())
    if config.get("format") != "mp2-official-tabm-v1" or config["metadata_features"] != FEATURE_NAMES:
        raise ValueError("Incompatible TabM checkpoint.")
    with np.load(root / "weights.npz", allow_pickle=False) as archive:
        weights = {name: archive[name].copy() for name in archive.files}
    k, width, dim = config["k"], config["width"], config["input_dim"]
    if k < 1 or config["n_blocks"] < 1 or config["feature_mode"] not in ("metadata", "context"):
        raise ValueError("Invalid TabM architecture.")
    expected_dim = 1024 + len(FEATURE_NAMES) if config["feature_mode"] == "metadata" else 2048 + len(FEATURE_NAMES)
    if dim != expected_dim:
        raise ValueError("Feature dimension mismatch.")
    expected = {"mean": (dim,), "scale": (dim,),
                "output_weight": (k, width, 1), "output_bias": (k, 1)}
    for i in range(config["n_blocks"]):
        din = dim if i == 0 else width
        expected.update({f"weight_{i}": (din, width), f"r_{i}": (k, din),
                         f"s_{i}": (k, width), f"bias_{i}": (k, width)})
    for name, shape in expected.items():
        value = weights[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint array: {name}")
    if (weights["scale"] <= 0).any():
        raise ValueError("Nonpositive normalization scale.")
    return config, weights


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("Pass the checkpoint folder.")
    root = Path(checkpoint)
    manifest = json.loads((root / "ensemble.json").read_text())
    if manifest.get("format") != "mp2-research-ensemble-v1":
        raise ValueError("Incompatible ensemble manifest.")
    cache, all_heads = {}, []
    if not manifest["members"]:
        raise ValueError("No ensemble members.")
    for member in manifest["members"]:
        if Path(member).name != member or member in (".", ".."):
            raise ValueError("Unsafe member path.")
        folder = root / member
        header = json.loads((folder / "config.json").read_text())
        mode = header["feature_mode"]
        if mode not in cache:
            cache[mode] = build_features(bank, arrays["embeddings"], mode)
        features, valid = cache[mode]
        if header["format"] == "mp2-official-tabm-v1":
            config, weights = load_tabm(folder)
            heads = tabm_heads(features, config, weights)
        else:
            raise ValueError("Unsupported member.")
        all_heads.append(heads)
    scores = aggregate_heads(np.concatenate(all_heads, axis=1).transpose(1, 0, 2),
                             valid, manifest["aggregation"])
    return select_answers(bank, scores, manifest["selection"])
