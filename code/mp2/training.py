"""Train the selected TabM architecture, or reproduce its validation ablation."""
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
import shutil
import time

import numpy as np
import tabm
import torch

from .features import FEATURE_NAMES, build_features
from .aggregation import aggregate_heads, select_answers
from common import DATA, PROTOCOL
from .model import load_tabm, tabm_heads, predict
from .training_utils import load_supervised, selection_loss

HERE = Path(__file__).resolve().parent.parent


def correct_count(bank, heads, valid, labels, aggregation, selection):
    scores = aggregate_heads(heads, valid, aggregation)
    choices = select_answers(bank, scores, selection)
    return int(labels[np.arange(len(labels)), choices].sum())


def forward_heads(model, features):
    n = len(features)
    # The official model receives one row per candidate and outputs [B*4, K, 1].
    return model(features.reshape(-1, features.shape[-1])).reshape(n, 4, model.k).transpose(1, 2)


def evaluate_heads(model, features):
    model.eval()
    with torch.no_grad():
        return torch.cat([forward_heads(model, chunk) for chunk in features.split(128)]).numpy()


def head_loss(logits, labels):
    # [B,K,4] -> [B*K,4]. Average individual branch losses, not predictions.
    repeated = labels[:, None, :].expand_as(logits)
    return selection_loss(logits.reshape(-1, 4), repeated.reshape(-1, 4), 1.0)


def export_tabm(model, mean, scale, args, folder):
    folder.mkdir(parents=True)
    values = {"mean": mean, "scale": scale}
    for i, block in enumerate(model.backbone.blocks):
        layer = block[0]
        values[f"weight_{i}"] = layer.weight.detach().numpy().T.copy()
        for key in ("r", "s", "bias"):
            values[f"{key}_{i}"] = getattr(layer, key).detach().numpy().copy()
    values["output_weight"] = model.output.weight.detach().numpy().copy()
    values["output_bias"] = model.output.bias.detach().numpy().copy()
    np.savez(folder / "weights.npz", **values)
    (folder / "config.json").write_text(json.dumps({
        "format": "mp2-official-tabm-v1", "feature_mode": "context",
        "input_dim": len(mean), "k": args.k, "width": args.width,
        "n_blocks": 2, "metadata_features": FEATURE_NAMES, "tabm_version": tabm.__version__,
    }, indent=2) + "\n")


def train_one(x_train, labels, x_val, y_val, val_bank, valid, seed, args):
    started = time.perf_counter()
    torch.manual_seed(seed)
    mean = x_train.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(x_train.std(axis=(0, 1), dtype=np.float64), 1e-6).astype(np.float32)
    train = torch.from_numpy((x_train - mean) / scale)
    val = torch.from_numpy((x_val - mean) / scale)
    targets = torch.from_numpy(labels)
    val_targets = torch.from_numpy(y_val)
    model = tabm.TabM.make(n_num_features=x_train.shape[-1], d_out=1, n_blocks=2,
                          d_block=args.width, dropout=0.15, k=args.k, arch_type="tabm")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)
    best_key, best_state, best_epoch, stale = None, None, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for indices in torch.randperm(len(train)).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = head_loss(forward_heads(model, train[indices]), targets[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        heads = evaluate_heads(model, val)
        correct = correct_count(val_bank, heads.transpose(1, 0, 2), valid, y_val,
                                "mean_sigmoid", "best")
        loss = float(head_loss(torch.from_numpy(heads), val_targets))
        key = (correct, -loss)
        history.append({"epoch": epoch, "train_loss": total / len(train),
                        "validation_correct": correct, "validation_loss": loss})
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"TabM seed={seed} epoch={epoch} val={correct}/{len(y_val)} "
                  f"best={best_key[0]}/{len(y_val)}",
                  flush=True)
        if stale >= 12:
            break
    model.load_state_dict(best_state)
    model.eval()
    folder = args.output / f"tabm_seed{seed}"
    export_tabm(model, mean, scale, args, folder / "checkpoint")
    config, weights = load_tabm(folder / "checkpoint")
    actual = tabm_heads(x_val, config, weights)
    expected = evaluate_heads(model, val)
    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5)
    actual_choices = select_answers(val_bank, aggregate_heads(actual.transpose(1, 0, 2), valid,
                                   "mean_sigmoid"), "best")
    expected_choices = select_answers(val_bank, aggregate_heads(expected.transpose(1, 0, 2), valid,
                                     "mean_sigmoid"), "best")
    assert actual_choices == expected_choices
    summary = {
        "seed": seed, "best_epoch": best_epoch, "epochs": len(history),
        "validation_correct": best_key[0], "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": time.perf_counter() - started,
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (folder / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return actual, summary



STRATEGIES = [
    ("mean_logits", "best"), ("mean_sigmoid", "best"), ("mean_softmax", "best"),
    ("mean_sigmoid", "answer_sum"), ("mean_softmax", "answer_sum"),
    ("mean_logits", "mob_poly"), ("mean_logits", "mob_adaptive"),
]


def validation_ablation(bank, heads, valid, labels):
    return [
        {"aggregation": aggregation, "selection": selection,
         "correct": (count := correct_count(
             bank, heads, valid, labels, aggregation, selection)),
         "n": len(labels), "accuracy": 100 * count / len(labels)}
        for aggregation, selection in STRATEGIES
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory; existing results are never overwritten.")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--ablate-checkpoint", type=Path,
                        help="Evaluate seven validation aggregation rules without retraining.")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new empty output directory.")
    if min(args.k, args.width, args.epochs) < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Require positive sizes and unique seeds.")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan = {
        "seeds": args.seeds, "k": args.k, "width": args.width, "blocks": 2,
        "epochs": args.epochs, "patience": 12, "dropout": 0.15, "batch_size": 128,
        "lr": 0.002, "weight_decay": 0.0003, "head_loss": "BCE + multi-positive ranking",
        "early_stopping": "mean-sigmoid best-candidate validation count, then loss",
        "final_aggregation": "mean_logits", "final_selection": "mob_adaptive",
        "note": "Final settings are fixed from the recorded validation comparison.",
        "test_loaded": False,
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    bank, embeddings, labels = load_supervised("validation", args.data_dir)
    xv, valid = build_features(bank, embeddings, "context")
    if args.ablate_checkpoint:
        manifest = json.loads((args.ablate_checkpoint / "ensemble.json").read_text())
        members = []
        for member in manifest["members"]:
            if Path(member).name != member or member in (".", ".."):
                raise ValueError("Unsafe member path.")
            config, weights = load_tabm(args.ablate_checkpoint / member)
            members.append(tabm_heads(xv, config, weights))
        heads = np.concatenate(members, axis=1).transpose(1, 0, 2)
        result = {"split": "validation", "test_loaded": False,
                  "candidates": validation_ablation(bank, heads, valid, labels)}
        (args.output / "ablation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    bt, et, yt = load_supervised("train", args.data_dir)
    xt, _ = build_features(bt, et, "context")
    trained, summaries = [], []
    for seed in args.seeds:
        heads, record = train_one(xt, yt, xv, labels, bank, valid, seed, args)
        trained.append(heads)
        summaries.append(record)
    heads = np.concatenate(trained, axis=1).transpose(1, 0, 2)
    checkpoint = args.output / "checkpoint"
    checkpoint.mkdir()
    for i, seed in enumerate(args.seeds):
        shutil.move(str(args.output / f"tabm_seed{seed}/checkpoint"),
                    str(checkpoint / f"member_{i}"))
    (checkpoint / "ensemble.json").write_text(json.dumps({
        "format": "mp2-research-ensemble-v1",
        "members": [f"member_{i}" for i in range(len(args.seeds))],
        "aggregation": "mean_logits", "selection": "mob_adaptive",
    }, indent=2) + "\n")
    asset_bytes = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
    if asset_bytes > 16 * 2**20:
        raise ValueError("Checkpoint exceeds 16 MiB.")
    choices = predict(bank, {"embeddings": embeddings}, checkpoint)
    count = int(labels[np.arange(len(labels)), choices].sum())
    result = {
        "protocol": PROTOCOL, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": plan, "runs": summaries,
        "validation": {"correct": count, "n": len(labels), "accuracy": 100 * count / len(labels)},
        "validation_ablation": validation_ablation(bank, heads, valid, labels),
        "asset_bytes": asset_bytes, "total_seconds": time.perf_counter() - started,
        "numpy": np.__version__, "torch": torch.__version__, "tabm": tabm.__version__,
        "platform": platform.platform(), "device": "cpu", "threads": 4, "test_loaded": False,
        "source_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("mp2/training.py", "mp2/training_utils.py", "mp2/model.py", "mp2/features.py",
                         "mp2/aggregation.py", "mp2/third_party/mob_algorithms.py")},
        "asset_sha256": {
            str(p.relative_to(checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(checkpoint.rglob("*")) if p.is_file()},
    }
    (args.output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
