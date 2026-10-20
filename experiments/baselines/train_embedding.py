"""Train a shared candidate-scoring MLP; export a compact NumPy checkpoint."""
import os

# Limit threads before numerical imports; use CPU training by default.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn

from common import DATA, PROTOCOL, parse_answer
from mp2.training_utils import load_supervised, selection_loss
from .majority import predict as majority_predict
from .embedding import load_bundle, normalize, score_embeddings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class CandidateMLP(nn.Module):
    """[B, 4, 1024] -> [B, 4], with parameters shared across candidates."""

    def __init__(self, hidden=(128, 32), dropout=0.15):
        super().__init__()
        dims = [1024, *hidden, 1]
        layers = []
        for i, (din, dout) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(din, dout))
            if i + 2 < len(dims):
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def selection_accuracy(logits, labels):
    choices = logits.argmax(axis=1)
    return float(100 * labels[np.arange(len(labels)), choices].mean())


def validation_metrics(logits, labels, informative, ranking_weight):
    return {
        "accuracy": selection_accuracy(logits, labels),
        "informative_accuracy": (
            selection_accuracy(logits[informative], labels[informative])
            if informative.any() else None
        ),
        "loss": float(selection_loss(
            torch.from_numpy(logits), torch.from_numpy(labels), ranking_weight)),
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_bundle(model, mean, scale, hidden, output):
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}
    linear_layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        # Export PyTorch x @ W.T weights for NumPy x @ W.
        arrays[f"weight_{i}"] = layer.weight.detach().cpu().numpy().T.copy()
        arrays[f"bias_{i}"] = layer.bias.detach().cpu().numpy().copy()
    np.savez(output / "weights.npz", **arrays)
    config = {
        "format": "mp2-mlp-numpy-v1",
        "layer_dims": [1024, *hidden, 1],
        "activation": "relu",
        "normalization": "train-candidate per-feature mean/std; std floor 1e-6",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if size > 16 * 2**20:
        raise ValueError("Inference assets exceed the 16 MiB limit.")
    return size


def evaluate_model(model, features, device, batch_size=256):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            outputs.append(model(features[start:start + batch_size].to(device)).cpu())
    return torch.cat(outputs).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 32])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128, help="Questions per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ranking-weight", type=float, default=1.0,
                        help="Use 0 for the BCE-only ablation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(*args.hidden, args.epochs, args.patience, args.batch_size) < 1:
        parser.error("Sizes, epochs and patience must be positive.")
    if not 0 <= args.dropout < 1 or args.lr <= 0 or args.weight_decay < 0 or args.ranking_weight < 0:
        parser.error("Invalid dropout, learning rate, weight decay or ranking weight.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is not empty. Choose a new --output to preserve previous runs.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu.")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    train_bank, x_train_raw, y_train = load_supervised("train", args.data_dir)
    val_bank, x_val_raw, y_val = load_supervised("validation", args.data_dir)
    # Fit normalization on training candidates only.
    flat = x_train_raw.reshape(-1, 1024)
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(flat.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    x_train = torch.from_numpy(normalize(x_train_raw, mean, scale))
    x_val = torch.from_numpy(normalize(x_val_raw, mean, scale))
    targets = torch.from_numpy(y_train)
    informative = np.array([
        len({a for c in row["candidates"] if (a := parse_answer(c["text"])) is not None}) >= 2
        for row in val_bank
    ])
    majority_choices = majority_predict(val_bank, {}, None)
    first_accuracy = float(100 * y_val[:, 0].mean())
    majority_accuracy = float(100 * y_val[np.arange(len(y_val)), majority_choices].mean())
    print(f"Train: {len(train_bank)} questions; validation: {len(val_bank)}", flush=True)
    print(f"Validation baselines: first={first_accuracy:.2f}%, "
          f"majority={majority_accuracy:.2f}%", flush=True)

    model = CandidateMLP(args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_metrics, best_epoch, stale = None, None, 0, 0
    history = []
    train_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(x_train))
        loss_sum = 0.0
        for indices in order.split(args.batch_size):
            xb, yb = x_train[indices].to(device), targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = selection_loss(model(xb), yb, args.ranking_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        val_logits = evaluate_model(model, x_val, device)
        metrics = validation_metrics(val_logits, y_val, informative, args.ranking_weight)
        record = {
            "epoch": epoch, "train_loss": loss_sum / len(x_train),
            "validation": metrics, "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        # Select by validation accuracy, breaking ties by lower validation loss.
        improved = (best_metrics is None or metrics["accuracy"] > best_metrics["accuracy"]
                    or (metrics["accuracy"] == best_metrics["accuracy"]
                        and metrics["loss"] < best_metrics["loss"]))
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics, best_epoch, stale = metrics, epoch, 0
        else:
            stale += 1
        print(f"epoch={epoch:03d} train_loss={record['train_loss']:.4f} "
              f"val={metrics['accuracy']:.2f}% val_loss={metrics['loss']:.4f}"
              f"{' *' if improved else ''}", flush=True)
        if stale >= args.patience:
            print(f"Early stopping after {stale} epochs without improvement.", flush=True)
            break
    training_seconds = time.perf_counter() - train_started
    model.load_state_dict(best_state)
    model.to("cpu").eval()
    bundle = args.output / "checkpoint"
    asset_bytes = export_bundle(model, mean, scale, args.hidden, bundle)
    # Verify NumPy export against the training network.
    config, weights = load_bundle(bundle)
    numpy_logits = score_embeddings(x_val_raw, config, weights)
    torch_logits = evaluate_model(model, x_val, torch.device("cpu"))
    np.testing.assert_allclose(numpy_logits, torch_logits, atol=2e-5, rtol=2e-5)
    if not np.array_equal(numpy_logits.argmax(1), torch_logits.argmax(1)):
        raise RuntimeError("NumPy export changed the selected candidates.")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                         capture_output=True, text=True, check=False)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    summary = {
        "protocol": PROTOCOL,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "architecture": config["layer_dims"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device), "platform": platform.platform(),
        "processor": platform.processor(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "threads": 4,
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "validation": validation_metrics(numpy_logits, y_val, informative, args.ranking_weight),
        "validation_informative_questions": int(informative.sum()),
        "validation_first_accuracy": first_accuracy,
        "validation_majority_accuracy": majority_accuracy,
        "training_seconds": training_seconds, "total_seconds": time.perf_counter() - started,
        "asset_bytes": asset_bytes, "asset_mib": asset_bytes / 2**20,
        "asset_sha256": {p.name: sha256(p) for p in sorted(bundle.iterdir())},
        "source_sha256": {name: sha256(ROOT / name) for name in (
            "experiments/baselines/train_embedding.py", "experiments/baselines/embedding.py",
            "experiments/baselines/majority.py", "code/mp2/training_utils.py", "code/common.py")},
        "upstream_commit": git.stdout.strip() if git.returncode == 0 else None,
        "training_data_sha256": {k: v for k, v in manifest["sha256"].items()
                                if "train" in k or "validation" in k},
        "test_used_for_training_or_selection": False,
    }
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
