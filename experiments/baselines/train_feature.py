"""Bounded validation-only experiment: two feature sets, three fixed seeds."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "4"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
import torch
from torch import nn

from mp2.features import FEATURE_NAMES, build_features, mask_unparseable
from common import DATA, PROTOCOL
from mp2.training_utils import load_supervised, selection_loss
from .embedding import load_bundle, score_embeddings
from .feature import load_member, member_logits, predict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class FeatureMLP(nn.Module):
    def __init__(self, dims, dropout):
        super().__init__()
        layers = []
        for i, (din, dout) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(din, dout))
            if i + 2 < len(dims):
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def count_correct(scores, labels, valid):
    chosen = mask_unparseable(scores, valid).argmax(axis=1)
    return int(labels[np.arange(len(labels)), chosen].sum())


def torch_logits(model, features):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(chunk) for chunk in features.split(256)]).numpy()


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_member(model, mean, scale, mode, dims, folder):
    folder.mkdir(parents=True)
    values = {"mean": mean, "scale": scale}
    i = 0
    for layer in model.network:
        if isinstance(layer, nn.Linear):
            values[f"weight_{i}"] = layer.weight.detach().numpy().T.copy()
            values[f"bias_{i}"] = layer.bias.detach().numpy().copy()
            i += 1
    np.savez(folder / "weights.npz", **values)
    (folder / "config.json").write_text(json.dumps({
        "format": "mp2-feature-mlp-v1", "feature_mode": mode,
        "layer_dims": dims, "metadata_features": FEATURE_NAMES,
        "activation": "relu",
    }, indent=2) + "\n")


def train_one(x_train, y_train, x_val, y_val, valid, mode, seed, args, folder):
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mean = x_train.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(x_train.std(axis=(0, 1), dtype=np.float64), 1e-6).astype(np.float32)
    train = torch.from_numpy((x_train - mean) / scale)
    val = torch.from_numpy((x_val - mean) / scale)
    labels = torch.from_numpy(y_train)
    val_labels = torch.from_numpy(y_val)
    dims = [x_train.shape[-1], *args.hidden, 1]
    model = FeatureMLP(dims, args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_key, best_state, best_epoch, stale = None, None, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train))
        total_loss = 0.0
        for indices in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train[indices])
            loss = selection_loss(logits, labels[indices], args.ranking_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        scores = torch_logits(model, val)
        correct = count_correct(scores, y_val, valid)
        val_loss = float(selection_loss(torch.from_numpy(scores), val_labels, args.ranking_weight))
        key = (correct, -val_loss)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train),
                        "validation_correct": correct, "validation_loss": val_loss})
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{mode} seed={seed} epoch={epoch} val={correct}/{len(y_val)} "
                  f"best={best_key[0]}/{len(y_val)}", flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    export_member(model, mean, scale, mode, dims, folder / "checkpoint")
    config, weights = load_member(folder / "checkpoint")
    scores = member_logits(x_val, config, weights)
    expected = torch_logits(model, val)
    np.testing.assert_allclose(scores, expected, atol=3e-5, rtol=3e-5)
    np.testing.assert_array_equal(mask_unparseable(scores, valid).argmax(1),
                                  mask_unparseable(expected, valid).argmax(1))
    np.save(folder / "validation_logits.npy", scores)
    record = {
        "mode": mode, "seed": seed, "best_epoch": best_epoch,
        "epochs_completed": len(history), "validation_correct": count_correct(scores, y_val, valid),
        "validation_questions": len(y_val), "total_training_seconds": time.perf_counter() - started,
        "parameters": sum(p.numel() for p in model.parameters()), "layer_dims": dims,
        "asset_bytes": sum(p.stat().st_size for p in (folder / "checkpoint").iterdir()),
    }
    (folder / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (folder / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return scores, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("metadata", "context"), default=["metadata", "context"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--hidden", nargs="+", type=int, default=[128, 32])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ranking-weight", type=float, default=1)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new, empty output directory.")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes):
        parser.error("Seeds and modes must be unique.")
    if min(*args.hidden, args.batch_size, args.epochs, args.patience) < 1:
        parser.error("Positive sizes required.")
    if not 0 <= args.dropout < 1 or args.lr <= 0 or args.weight_decay < 0 or args.ranking_weight < 0:
        parser.error("Invalid optimization parameters.")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Save the fixed plan first; load train and validation data only.
    plan = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    plan["selection_rule"] = (
        "Highest validation correct count among each feature-mode fixed-seed mean-logit "
        "ensemble and the all-member mean-logit ensemble. Ties prefer fewer asset bytes. "
        "Compare with masked old baseline before recommending replacement. "
        "Single-seed scores are diagnostics only."
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    train_bank, train_embeddings, y_train = load_supervised("train", args.data_dir)
    val_bank, val_embeddings, y_val = load_supervised("validation", args.data_dir)
    scores_by_name, records, valid = {}, [], None
    for mode in args.modes:
        train_features, _ = build_features(train_bank, train_embeddings, mode)
        val_features, valid = build_features(val_bank, val_embeddings, mode)
        for seed in args.seeds:
            name = f"{mode}_seed{seed}"
            scores, record = train_one(train_features, y_train, val_features, y_val, valid,
                                       mode, seed, args, args.output / name)
            scores_by_name[name] = scores
            records.append({"name": name, **record})
    candidates = []
    for mode in args.modes:
        names = [name for name in scores_by_name if name.startswith(f"{mode}_")]
        averaged = np.mean([scores_by_name[name] for name in names], axis=0)
        candidates.append({
            "name": f"{mode}_ensemble", "members": names,
            "validation_correct": count_correct(averaged, y_val, valid),
        })
    if len(args.modes) > 1:
        candidates.append({
            "name": "all_ensemble", "members": list(scores_by_name),
            "validation_correct": count_correct(np.mean(list(scores_by_name.values()), axis=0),
                                                y_val, valid),
        })
    for item in candidates:
        item["asset_bytes"] = sum(record["asset_bytes"] for record in records
                                 if record["name"] in item["members"])
        item["accuracy"] = 100 * item["validation_correct"] / len(y_val)
    old_config, old_weights = load_bundle(args.baseline_checkpoint)
    old_logits = score_embeddings(val_embeddings, old_config, old_weights)
    old_correct = count_correct(old_logits, y_val, valid)
    best = max(candidates, key=lambda row: (row["validation_correct"], -row["asset_bytes"]))
    selected = args.output / "checkpoint"
    selected.mkdir()
    for index, name in enumerate(best["members"]):
        shutil.copytree(args.output / name / "checkpoint", selected / f"member_{index}")
    (selected / "ensemble.json").write_text(json.dumps({
        "format": "mp2-feature-ensemble-v1",
        "aggregation": "mean_logits",
        "members": [f"member_{index}" for index in range(len(best["members"]))],
    }, indent=2) + "\n")
    asset_bytes = sum(p.stat().st_size for p in selected.rglob("*") if p.is_file())
    if asset_bytes > 16 * 2**20:
        raise ValueError("Ensemble exceeds the inference asset limit.")
    choices = predict(val_bank, {"embeddings": val_embeddings}, selected)
    assert int(y_val[np.arange(len(y_val)), choices].sum()) == best["validation_correct"]
    result = {
        "protocol": PROTOCOL, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": plan, "runs": records, "ensemble_candidates": candidates,
        "old_mlp_with_validity_mask_correct": old_correct,
        "selected": best, "improves_old_masked_baseline": best["validation_correct"] > old_correct,
        "validation_questions": len(y_val), "asset_bytes": asset_bytes,
        "total_experiment_seconds": time.perf_counter() - started,
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "torch": torch.__version__, "device": "cpu", "threads": 4,
        "source_sha256": {name: hash_file(ROOT / name) for name in (
            "code/mp2/features.py", "code/mp2/training_utils.py", "code/common.py",
            "experiments/baselines/feature.py", "experiments/baselines/train_feature.py",
            "experiments/baselines/embedding.py")},
        "asset_sha256": {str(p.relative_to(selected)): hash_file(p)
                         for p in sorted(selected.rglob("*")) if p.is_file()},
        "test_loaded_by_this_experiment": False,
        "run_purpose": "Fixed-setting report comparison reproduction",
    }
    (args.output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
