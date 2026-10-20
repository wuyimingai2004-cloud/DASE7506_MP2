"""Answer aggregation inspired by TabM, LEVER, and the official MoB algorithms."""
import numpy as np

from common import parse_answer
from .third_party.mob_algorithms import mob_adaptive_m, mob_poly_m


def aggregate_heads(heads, valid, rule):
    """heads: [members, N, 4]. Each member makes an independent prediction."""
    if rule == "mean_logits":
        scores = heads.mean(axis=0)
    elif rule == "mean_sigmoid":
        scores = (1 / (1 + np.exp(-np.clip(heads, -60, 60)))).mean(axis=0)
    elif rule == "mean_softmax":
        usable = valid | ~valid.any(axis=1, keepdims=True)
        masked = np.where(usable[None], heads, -1e9)
        exp = np.exp(masked - masked.max(axis=-1, keepdims=True))
        scores = (exp / exp.sum(axis=-1, keepdims=True)).mean(axis=0)
    else:
        raise ValueError(f"Unknown head aggregation: {rule}")
    return scores


def select_answers(bank, scores, method):
    choices = []
    for row, values in zip(bank, scores):
        answers = [parse_answer(c["text"]) for c in row["candidates"]]
        indices = [i for i, answer in enumerate(answers) if answer is not None]
        if not indices:
            choices.append(int(np.argmax(values)))
            continue
        valid_scores = np.array([values[i] for i in indices])
        valid_answers = np.array([answers[i] for i in indices])
        if method == "best":
            local = int(valid_scores.argmax())
        elif method == "answer_sum":
            # LEVER-style marginalization over equivalent results.
            # Generation probabilities are unavailable; sum nonnegative verifier scores.
            if (valid_scores < 0).any():
                raise ValueError("answer_sum requires nonnegative probabilities.")
            totals = {answer: float(valid_scores[valid_answers == answer].sum())
                      for answer in valid_answers}
            local = max(range(len(indices)),
                        key=lambda j: (totals[valid_answers[j]], valid_scores[j], -j))
        elif method == "mob_poly":
            local = int(mob_poly_m(valid_answers, valid_scores))
        elif method == "mob_adaptive":
            local = int(mob_adaptive_m(valid_answers, valid_scores))
        else:
            raise ValueError(f"Unknown answer selection method: {method}")
        choices.append(indices[local])
    return choices
