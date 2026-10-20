"""Training-only reference loading and the multi-positive objective."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from common import check_data, load_inputs, parse_answer, read_jsonl


def correctness_labels(bank, references):
    """Build [N, 4] multi-positive labels with the official parser."""
    if len(bank) != len(references):
        raise ValueError("Reference count mismatch.")
    labels = np.zeros((len(bank), 4), dtype=np.float32)
    for i, (row, reference) in enumerate(zip(bank, references)):
        if row["id"] != reference["id"] or row["question"] != reference["question"]:
            raise ValueError("Reference alignment mismatch.")
        gold = parse_answer(reference["answer"])
        if gold is None:
            raise ValueError("Unparseable reference answer.")
        for j, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            labels[i, j] = float(answer is not None and answer == gold)
    if np.any(labels.sum(axis=1) == 0):
        raise ValueError("Every training/validation question must have a correct candidate.")
    return labels


def load_supervised(split, data_dir):
    # Disallow test and audit references in the training entry point.
    if split not in ("train", "validation"):
        raise ValueError("Only train/validation are allowed in training.")
    check_data(data_dir, (split,))
    bank, arrays = load_inputs(split, data_dir)
    refs = read_jsonl(Path(data_dir) / f"gsm8k_{split}.jsonl")
    return bank, arrays["embeddings"], correctness_labels(bank, refs)


def selection_loss(logits, labels, ranking_weight=1.0):
    # BCE scores correctness; the ranking term rewards any correct candidate.
    # Other correct answers are not negatives; all-correct groups have zero ranking loss.
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    correct_logits = logits.masked_fill(labels < 0.5, -torch.inf)
    ranking = (torch.logsumexp(logits, dim=1)
               - torch.logsumexp(correct_logits, dim=1)).mean()
    return bce + ranking_weight * ranking


