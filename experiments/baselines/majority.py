"""Training-free majority-answer baseline using the unchanged official parser."""
from collections import Counter

from common import parse_answer


def predict(bank, arrays, checkpoint):
    choices = []
    for row in bank:
        answers = [parse_answer(c["text"]) for c in row["candidates"]]
        counts = Counter(a for a in answers if a is not None)
        # Exclude unparseable answers; break ties by original candidate order.
        votes = [counts[a] if a is not None else 0 for a in answers]
        choices.append(max(range(4), key=lambda i: votes[i]))
    return choices
