"""Run fixed comparison settings and collect a validation-only report table."""
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "4"

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from common import DATA
from mp2.features import build_features, mask_unparseable
from mp2.model import predict as final_predict
from mp2.training_utils import load_supervised
from .embedding import load_bundle, score_embeddings
from .majority import predict as majority_predict

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "majority": 688,
    "embedding_mlp": 721,
    "embedding_mlp_masked": 725,
    "feature_mlp_metadata": 741,
    "feature_mlp_context": 746,
    "tabm_mob": 749,
}


def train_fixed_baselines(data_dir, output):
    """Use separate processes so every run has a fresh numerical runtime."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "code"), str(ROOT / "experiments")])
    commands = [
        [sys.executable, "-m", "baselines.train_embedding",
         "--data-dir", str(data_dir), "--output", str(output / "embedding")],
        [sys.executable, "-m", "baselines.train_feature",
         "--data-dir", str(data_dir), "--output", str(output / "features"),
         "--baseline-checkpoint", str(output / "embedding/checkpoint")],
    ]
    for name, command in zip(("embedding", "features"), commands):
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)


def validation_comparisons(data_dir, output, checkpoint):
    bank, embeddings, labels = load_supervised("validation", data_dir)
    _, valid = build_features(bank, embeddings, "metadata")
    config, weights = load_bundle(output / "embedding/checkpoint")
    logits = score_embeddings(embeddings, config, weights)
    choices = {
        "majority": majority_predict(bank, {}, None),
        "embedding_mlp": logits.argmax(axis=1),
        "embedding_mlp_masked": mask_unparseable(logits, valid).argmax(axis=1),
    }
    for mode in ("metadata", "context"):
        scores = [np.load(output / f"features/{mode}_seed{seed}/validation_logits.npy",
                          allow_pickle=False) for seed in (42, 123, 2026)]
        choices[f"feature_mlp_{mode}"] = mask_unparseable(
            np.mean(scores, axis=0), valid).argmax(axis=1)
    choices["tabm_mob"] = final_predict(bank, {"embeddings": embeddings}, checkpoint)
    rows = []
    for name, indices in choices.items():
        correct = int(labels[np.arange(len(bank)), indices].sum())
        rows.append({
            "method": name, "correct": correct, "n": len(bank),
            "accuracy": 100 * correct / len(bank),
            "reported_correct": EXPECTED[name],
            "matches_report": correct == EXPECTED[name],
        })
    (output / "validation_choices.json").write_text(json.dumps(
        {name: np.asarray(indices).tolist() for name, indices in choices.items()}) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/tabm_mob")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new, empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("Replaying one embedding MLP and six feature MLP runs.", flush=True)
    train_fixed_baselines(args.data_dir.resolve(), output)
    rows = validation_comparisons(args.data_dir.resolve(), output, args.checkpoint.resolve())
    result = {
        "purpose": "Fixed-setting reproduction of the report's validation comparisons",
        "training_runs": 7,
        "split": "validation",
        "rows": rows,
        "all_counts_match_report": all(row["matches_report"] for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
        "test_loaded": False,
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
