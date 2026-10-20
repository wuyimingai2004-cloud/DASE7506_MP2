# Report comparisons

This directory reproduces the baseline and feature-ablation rows in the report.
The final inference implementation remains in `code/mp2/`, with its single
selected ensemble in `checkpoints/tabm_mob/`.

## Run

From the project root, install the root `requirements.txt`, then run:

```bash
python experiments/reproduce.py --output results/reproduce_comparisons
```

Use a new output directory. The command trains the embedding MLP with seed 42
and both feature MLP variants with seeds 42, 123 and 2026, then evaluates all
comparison rows on validation. It uses the supplied final checkpoint for the
TabM row. No test inputs or references are loaded.

| Method | Expected validation correct / 798 |
|---|---:|
| Answer majority | 688 |
| Embedding MLP | 721 |
| Embedding MLP with parseability mask | 725 |
| Feature MLP without relative embedding | 741 |
| Feature MLP with relative embedding | 746 |
| Final TabM + adaptive MoB | 749 |

The output contains `comparison.json`, validation choices, per-run checkpoints,
training histories, plans and logs. The original fixed settings use two hidden
layers of width 128 and 32, dropout 0.15, AdamW with learning rate 0.001 and weight
decay 0.001, batch size 128, at most 80 epochs and patience 12. Early stopping
uses validation accuracy, with validation loss breaking ties. The feature
ensembles average logits across the three seeds. Normalization uses training
data only; the BCE plus multi-positive ranking objective matches the original.
Numerical differences on another platform may affect retraining.

For the final model's aggregation ablation, hold its weights fixed:

```bash
python scripts/train.py --ablate-checkpoint checkpoints/tabm_mob \
  --output results/ablation_validation
```

## Layout and records

- `reproduce.py`: thin command-line entry point.
- `baselines/comparison.py`: fixed-run orchestration and validation table.
- `baselines/majority.py`: answer-majority control.
- `baselines/embedding.py` and `train_embedding.py`: initial MLP inference and training.
- `baselines/feature.py` and `train_feature.py`: feature MLP inference and training.

Feature extraction and supervision helpers are shared with `code/mp2/`.
The restored training and inference computations match the original code;
imports, paths, record metadata and comments were adapted to this layout.

`results/comparisons/` contains the original comparison summaries.
`results/history/original_records.zip` preserves all 267 original experiment JSON
files byte for byte, including histories, plans, configuration and evaluation
records from rejected search stages. `results/history/manifest.json` lists their
original paths and hashes and maps restored implementations to original source
hashes. Historical paths describe the original runs.

The archive is evidence, not a second runnable project. The maintained replay
covers the report table and final aggregation ablation; rejected search
implementations and their weights are not installed as active models.
