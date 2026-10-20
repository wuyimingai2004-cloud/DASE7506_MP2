# Reproduction

Run every command below from the project root. Use Python 3.12 and a new output
directory for each run. The model, data and official scoring protocol are fixed.

## Data

This repository includes the complete official course data in `code/data/`
and all selected inference assets in `checkpoints/tabm_mob/`.
Clone the frozen release:

```bash
git clone https://github.com/wuyimingai2004-cloud/DASE7506_MP2.git
cd DASE7506_MP2
git checkout v1.0.0
```

Alternatively, download the [v1.0.0 source ZIP](https://github.com/wuyimingai2004-cloud/DASE7506_MP2/archive/refs/tags/v1.0.0.zip)
and extract it. Both routes include the data, report, records and checkpoints.
Then follow the setup commands in the root [README](../README.md).
On Windows PowerShell, use `.venv\Scripts\Activate.ps1` to activate the environment.

The supplied data and benchmark tools come from official course revision
`bb801f4f2e7629d270df765ab12c333b2dc616bb` and retain their original license,
notice and file hashes. The upstream guide and code README are retained
unchanged; use the root README for this project's entry points.

The course protocol is `dase7506-mp2-open-hard-verifier-v5`. The train, validation
and test banks contain 4,755, 798 and 467 questions, respectively.

## Dependencies

The root `requirements.txt` pins the full training environment: NumPy 2.5.3,
Torch 2.8.0, TabM 0.0.3, rtdl-num-embeddings 0.0.12 and their recorded dependencies.
For evaluation only, the original `code/requirements.txt` installs NumPy alone:

```bash
python -m pip install -r code/requirements.txt
```

Prediction imports neither Torch nor TabM. It works offline without an API key
or any model download.

## Frozen model

The course adapter is `code/submission.py`; the implementation is
`code/mp2/model.py`. All inference assets live in `checkpoints/tabm_mob/`.
Keep its `ensemble.json` and all three `member_*` directories.

```bash
python scripts/evaluate.py --split validation \
  --output-dir results/reproduce_validation
python scripts/evaluate.py --split test \
  --output-dir results/reproduce_test
```

The evaluator defaults to the selected method and checkpoint. It launches the
unchanged official predictor and independent scorer as separate sequential
processes, with numerical-library thread limits set to four.

Expected official score fields:

| Split | `correct` | `n` | `accuracy` |
|---|---:|---:|---:|
| Validation | 749 | 798 | 93.85964912280701 |
| Test | 382 | 467 | 81.79871520342613 |

The leaderboard value is **81.79871520342613**.

To invoke the official tools directly:

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 BLIS_NUM_THREADS=4
python code/predict.py --method code/submission.py \
  --checkpoint checkpoints/tabm_mob --split test \
  --output results/reproduce_direct/test-predictions.json
python code/score_predictions.py --split test \
  --predictions results/reproduce_direct/test-predictions.json \
  --output results/reproduce_direct/test-score.json
```

For an audit bank without references:

```bash
python scripts/evaluate.py --split audit \
  --data-dir /path/to/audit_inputs --predict-only \
  --output-dir results/reproduce_audit
```

`predict()` uses the bank and embeddings it receives. It does not read reference
answers or use question IDs as features.

## Training

```bash
python scripts/train.py --seeds 42 123 2026 --k 8 --width 128 --epochs 80 \
  --output results/retrained
```

Weights and feature normalization are fitted on the training bank only.
Validation controls early stopping. Each seed trains eight shared-parameter TabM
branches with two 128-wide hidden layers, dropout 0.15 and a loss combining binary
cross-entropy with multi-positive ranking. AdamW uses a learning rate of 0.002,
weight decay of 0.0003, batches of 128 questions and patience 12.

The final mean-logit aggregation and adaptive-MoB selection are fixed from the
recorded validation comparison. The training script reports seven validation
aggregation ablations without changing those final settings or loading test
references. Development should use training and validation; freeze the method
before final test scoring.

A fixed-setting local reproduction produced byte-identical checkpoint files.
Numerical differences across platforms can affect retraining; use the supplied
checkpoint to reproduce the recorded score.

## Validation ablation and tests

```bash
python scripts/train.py --ablate-checkpoint checkpoints/tabm_mob \
  --output results/ablation_validation
python -m unittest discover -s tests -v
```

Tests cover export parity, exact MoB resampling probabilities, independent branch
losses, candidate permutations, safe arithmetic, invalid-answer fallbacks and
rejection of test/audit references in the training loader.

## Report comparisons

```bash
python experiments/reproduce.py --output results/reproduce_comparisons
```

This replays the initial MLP and the two feature MLP ensembles, then compares
their validation scores with answer majority and the frozen final model.
See [experiments/README.md](../experiments/README.md) for settings, expected
counts, file roles and the scope of the retained historical evidence.

## Resources and records

Inference assets total **3,723,503 bytes = 3.5510 MiB**, including all weights,
normalization statistics and configurations. The recorded complete local test
evaluation took **0.3183 seconds** on an Apple M5 with 16 GB RAM and four numerical
threads. This measurement predates the directory-only reorganization and includes
startup, imports, loading, preprocessing, all ensemble members and scoring.
The reference Xeon Platinum 8457C has not been measured.

| Location | Contents |
|---|---|
| `results/evaluation/` | Official scores, predictions and recorded process timings |
| `results/training/` | Original histories and summaries for the selected seeds |
| `results/comparisons/` | Original validation plans and comparison measurements |
| `results/history/` | Unmodified original experiment JSON archive and hash manifest |
| `results/reproduction/` | Fixed-setting comparison replay checks |
| `results/search_costs.json` | Recorded neural search cost, including rejected ideas |
| `results/verification.json` | Reproduction and directory-refactor checks |
| `results/publication_verification.json` | Public clone, fresh environment and official test reproduction |
| `results/release.json` | Current source version, source hashes and checkpoint sizes/SHA-256 hashes |

Historical records preserve the original paths and source hashes from their runs.
Use `results/release.json` for the current implementation. The original course
manifest is retained at `code/PACKAGE_MANIFEST.json`; it describes the starter
package, not the student's added files.

The final model, report baselines and final aggregation ablation are reproducible
with the commands above. Comparison checkpoints are generated into the chosen
run directory; only the final inference checkpoint is distributed.
See the [report](report.pdf), [reuse references](references.md) and
[AI/evaluation disclosure](ai_disclosure.md) for interpretation and provenance.

Public-repository verification on 20 October 2026 used a new Python 3.12.14
virtual environment and the pinned dependencies. The anonymous clone passed
data and asset-hash checks and reproduced 382/467 test answers in 0.3227 seconds
for complete local prediction plus scoring. See `results/publication_verification.json`.
