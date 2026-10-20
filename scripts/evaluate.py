"""Time fresh official prediction + scoring processes with a four-thread limit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "code"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=Path, default=HERE / "submission.py")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints/tabm_mob")
    parser.add_argument("--split", choices=("validation", "test", "audit"), default="validation")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predict-only", action="store_true",
                        help="For an audit bank without references; no accuracy is produced.")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / f"{args.split}-predictions.json"
    score = args.output_dir / f"{args.split}-score.json"
    if predictions.exists() or score.exists():
        parser.error("Evaluation output already exists. Choose a new --output-dir.")
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "4"
    command = [
        sys.executable, str(HERE / "predict.py"), "--method", str(args.method.resolve()),
        "--split", args.split, "--data-dir", str(args.data_dir.resolve()),
        "--output", str(predictions),
    ]
    assets = {}
    if args.checkpoint is not None:
        root = args.checkpoint.resolve()
        if not root.exists():
            parser.error("Checkpoint does not exist.")
        command.extend(["--checkpoint", str(root)])
        paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        assets = {str(p.relative_to(root) if root.is_dir() else p.name): {
            "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()
        } for p in paths}
    asset_bytes = sum(item["bytes"] for item in assets.values())
    if asset_bytes > 16 * 2**20:
        parser.error("Checkpoint files exceed 16 MiB.")
    commands = [command]
    started = time.perf_counter()
    subprocess.run(command, cwd=HERE, env=env, check=True)
    prediction_wall = time.perf_counter() - started
    scoring_wall = None
    if not args.predict_only:
        command = [
            sys.executable, str(HERE / "score_predictions.py"), "--split", args.split,
            "--data-dir", str(args.data_dir.resolve()), "--predictions", str(predictions),
            "--output", str(score),
        ]
        commands.append(command)
        scoring_started = time.perf_counter()
        subprocess.run(command, cwd=HERE, env=env, check=True)
        scoring_wall = time.perf_counter() - scoring_started
    total = time.perf_counter() - started
    result = {
        "split": args.split, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "python": platform.python_version(), "threads": 4,
        "prediction_process_wall_seconds": prediction_wall,
        "scoring_process_wall_seconds": scoring_wall,
        "total_wall_seconds": total, "asset_bytes": asset_bytes,
        "asset_mib": asset_bytes / 2**20, "assets": assets, "commands": commands,
        "local_under_30_seconds": total <= 30 if not args.predict_only else None,
        "reference_cpu_eligibility_verified": False,
        "accuracy": json.loads(score.read_text())["accuracy"] if not args.predict_only else None,
    }
    (args.output_dir / f"{args.split}-resources.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
