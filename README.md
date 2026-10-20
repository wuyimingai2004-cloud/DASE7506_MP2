# Selecting Reliable LLM Reasoning

A compact **TabM + adaptive Majority-of-the-Bests** selector for DASE7506
Mini-Project 2. Given four candidate solutions, the model compares their fixed
embeddings, text features and final answers, then returns one candidate index.

| Model | Validation | Public test | Inference assets |
|---|---:|---:|---:|
| TabM + adaptive MoB | **93.86%** | **81.80%** | **3.55 MiB** |

The retained model had the highest recorded validation accuracy.
See the [report](docs/report.pdf) for comparisons, ablations and limitations.

## Setup

Clone the repository and run commands with **Python 3.12**:

```bash
git clone https://github.com/wuyimingai2004-cloud/DASE7506_MP2.git
cd DASE7506_MP2
git checkout v1.0.0
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/check_data.py
```

The repository includes the official data in `code/data/` and the selected
checkpoint in `checkpoints/tabm_mob/`. No separate data or model download is
required. The reproducible release is tagged `v1.0.0`; see
[reproduction details](docs/reproduction.md).

## Evaluate

The selected checkpoint is included in `checkpoints/tabm_mob/`.
No retraining is needed.

```bash
python scripts/evaluate.py --split validation \
  --output-dir results/reproduce_validation
```

For final test scoring of the frozen model:

```bash
python scripts/evaluate.py --split test \
  --output-dir results/reproduce_test
```

The script runs the official predictor and scorer with four numerical threads.
It saves predictions, accuracy and resource measurements in the output directory.
Use a new directory for each run. Expected test score: **382/467**.

## Train

```bash
python scripts/train.py --output results/retrained
```

Defaults reproduce the selected architecture: seeds `42, 123, 2026`, eight
branches per seed and two hidden layers of width 128. Training uses the supplied
training bank and validation-based early stopping.

To reproduce the aggregation ablation without retraining:

```bash
python scripts/train.py --ablate-checkpoint checkpoints/tabm_mob \
  --output results/ablation_validation
```

The report's baseline and feature-ablation implementations are in
[`experiments/`](experiments/README.md), with one command to reproduce their
validation comparison table.

## Project structure

```text
MP2/
├── README.md
├── requirements.txt
├── code/
│   ├── mp2/               # Model, features, aggregation and training
│   ├── submission.py      # Thin adapter for the official evaluator
│   ├── data/              # Supplied course data
│   └── ...                # Unchanged official parser and evaluation tools
├── scripts/               # train.py and evaluate.py command-line entry points
├── experiments/           # Reproducible report baselines and feature comparisons
├── checkpoints/tabm_mob/   # The selected ensemble and normalization statistics
├── results/               # Scores, training logs, comparisons and asset hashes
├── tests/                 # Focused implementation tests
├── docs/                  # Report, reproduction details and disclosures
└── guide/                 # Original assignment guide
```

For the implementation, start with [`model.py`](code/mp2/model.py),
[`features.py`](code/mp2/features.py) and [`training.py`](code/mp2/training.py).
The three checkpoint members form one ensemble and must be kept together.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Documentation

- [Report](docs/report.pdf) and [editable source](docs/report.md)
- [Detailed reproduction and resource accounting](docs/reproduction.md)
- [Submission checklist](docs/submission.md)
- [References and licenses](docs/references.md)
- [AI assistance and evaluation provenance](docs/ai_disclosure.md)
