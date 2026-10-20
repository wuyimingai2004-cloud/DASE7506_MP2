# Third-party code and methodological references

## Official course package

- Repository: <https://github.com/xudongwu-0/DASE7506-MP2>
- Downloaded commit: `bb801f4f2e7629d270df765ab12c333b2dc616bb`.
- The supplied dataset, numeric parser, prediction runner, data checker and scorer remain unchanged.
- Supplied `code/data/LICENSE.txt` and `code/data/NOTICE.txt` are retained.

## TabM

- Repository: <https://github.com/yandex-research/tabm>
- Paper: <https://arxiv.org/abs/2410.24210>
- Training dependency: `tabm==0.0.3`, Apache-2.0.
- Its dependency `rtdl-num-embeddings==0.0.12` is pinned; numerical feature embeddings
  from that package are not enabled in this experiment.
- We instantiate the official TabM architecture with random weights.
- Our code adapts its output to groups of four candidates and a multi-positive ranking
  objective, then exports its learned weights for NumPy CPU inference.
- No pretrained weights or external datasets from this project are used.

## Majority-of-the-Bests

- Repository: <https://github.com/arakhsha/mob>
- Inspected commit: `7925cdd66c409f03c47d7bc02cd8121ebab5dc0d`.
- Original path: `src/mob/algorithms.py`.
- Vendored without edits as `code/mp2/third_party/mob_algorithms.py`.
- License: MIT, copyright (c) 2025 Autodesk Inc.
- Original license retained as `code/mp2/third_party/MOB_LICENSE.txt`.
- Our wrapper excludes candidates rejected by the course numeric parser,
  calls the original `mob_poly_m` / `mob_adaptive_m`, and maps the selected index back.
- No MoB datasets, generation models or reward model weights are used.

## LEVER

- Repository: <https://github.com/niansong1996/lever>
- Inspected commit: `8173b747e6e1351176fe8ae6e3885db1bc63ab04`.
- Paper: <https://arxiv.org/abs/2302.08468>.
- Inspected method: `aggregate_exec_results` in
  `finetuning/lightning_modules/models/seq2seq_verification_model.py`.
- Methodological reference only: the validation-ablation `answer_sum` implementation groups answers
  using the official course parser and sums verifier probabilities.
- Generation probabilities and program execution results are not available in this
  coursework, so this is a restricted adaptation, not a full reproduction of LEVER.
- No LEVER source code is vendored, and no datasets or model weights are used.
