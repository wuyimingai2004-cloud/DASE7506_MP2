"""Label-free candidate and group features; shared by training and inference."""
import ast
from collections import Counter
from fractions import Fraction
import math
import re

import numpy as np

from common import parse_answer

NUMBER = r"[-+]?\d+(?:\.\d+)?"
NUMBERS = re.compile(NUMBER)
# Check numeric arithmetic only; this does not verify natural-language reasoning.
EQUATION = re.compile(
    rf"(?<![\w.])({NUMBER}(?:\s*[+\-*/]\s*{NUMBER})+)\s*=\s*({NUMBER})(?![\w.])"
)
FEATURE_NAMES = [
    "valid_answer", "same_answer_count", "distinct_answers", "largest_answer_count",
    "unique_plurality", "is_plurality", "answer_is_integer", "answer_is_negative",
    "answer_is_zero", "answer_log_magnitude", "log_characters", "log_words",
    "log_lines", "log_number_mentions", "log_question_words", "log_question_numbers",
    "question_number_overlap", "answer_mentioned_in_question", "boxed_count",
    "hash_answer_count", "question_mark_count", "uncertainty_markers",
    "correction_markers", "step_markers", "log_checked_equations",
    "equation_correct_fraction", "log_incorrect_equations", "has_incorrect_equation",
    "relative_word_length", "relative_number_count",
]


def numeric_expression(expression):
    """Small, bounded AST evaluator allowing only constants and + - * /."""
    if len(expression) > 128:
        raise ValueError("Expression too long.")
    node = ast.parse(expression, mode="eval")
    if len(list(ast.walk(node))) > 64:
        raise ValueError("Expression too complex.")

    def walk(item):
        if isinstance(item, ast.Expression):
            return walk(item.body)
        if isinstance(item, ast.Constant) and type(item.value) in (int, float):
            return Fraction(str(item.value))
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.UAdd, ast.USub)):
            value = walk(item.operand)
            return value if isinstance(item.op, ast.UAdd) else -value
        if isinstance(item, ast.BinOp):
            a, b = walk(item.left), walk(item.right)
            if isinstance(item.op, ast.Add):
                return a + b
            if isinstance(item.op, ast.Sub):
                return a - b
            if isinstance(item.op, ast.Mult):
                return a * b
            if isinstance(item.op, ast.Div):
                return a / b
        raise ValueError("Unsupported arithmetic.")

    return walk(node)


def equation_statistics(text):
    # Normalize simple LaTeX. Ignore variables, complex brackets and powers.
    text = text.replace(",", "").replace("\\times", "*").replace("\\cdot", "*")
    text = text.replace("×", "*").replace("\\div", "/").replace("÷", "/")
    text = re.sub(r"\\frac\{([-+]?\d+(?:\.\d+)?)\}\{([-+]?\d+(?:\.\d+)?)\}",
                  r"\1/\2", text)
    good = bad = 0
    for expression, rhs in EQUATION.findall(text):
        try:
            actual, expected = numeric_expression(expression), Fraction(rhs)
            # Allow rounding error; use this as a feature, never as a rejection rule.
            match = abs(float(actual - expected)) <= 1e-3 * max(1.0, abs(float(expected)))
            good += int(match)
            bad += int(not match)
        except (ValueError, ZeroDivisionError, SyntaxError, OverflowError, RecursionError):
            continue
    return good, bad


def parsed_answers(bank):
    return [[parse_answer(candidate["text"]) for candidate in row["candidates"]]
            for row in bank]


def metadata_features(bank):
    features = np.zeros((len(bank), 4, len(FEATURE_NAMES)), dtype=np.float32)
    valid = np.zeros((len(bank), 4), dtype=bool)
    for i, row in enumerate(bank):
        if len(row["candidates"]) != 4:
            raise ValueError("Exactly four candidates are required.")
        question = row["question"]
        q_numbers = set(NUMBERS.findall(question.replace(",", "")))
        q_words = len(question.split())
        answers = [parse_answer(c["text"]) for c in row["candidates"]]
        counts = Counter(answer for answer in answers if answer is not None)
        largest = max(counts.values(), default=0)
        unique_plurality = sum(value == largest for value in counts.values()) == 1
        lengths = [len(c["text"].split()) for c in row["candidates"]]
        number_lists = [NUMBERS.findall(c["text"].replace(",", "")) for c in row["candidates"]]
        average_length = max(float(np.mean(lengths)), 1)
        average_numbers = max(float(np.mean([len(nums) for nums in number_lists])), 1)
        for j, candidate in enumerate(row["candidates"]):
            text, answer = candidate["text"], answers[j]
            lower = text.lower()
            is_valid = answer is not None
            valid[i, j] = is_valid
            value = Fraction(answer) if is_valid else Fraction(0)
            try:
                magnitude = min(math.log1p(abs(float(value))), 30)
            except OverflowError:
                magnitude = 30.0
            numbers = set(number_lists[j])
            good, bad = equation_statistics(text)
            count = counts[answer] if is_valid else 0
            features[i, j] = [
                is_valid, count / 4, len(counts) / 4, largest / 4,
                unique_plurality, is_valid and count == largest,
                is_valid and value.denominator == 1, value < 0,
                is_valid and value == 0, magnitude,
                math.log1p(len(text)), math.log1p(lengths[j]),
                math.log1p(text.count("\n")), math.log1p(len(number_lists[j])),
                math.log1p(q_words), math.log1p(len(q_numbers)),
                len(numbers & q_numbers) / max(len(q_numbers), 1),
                is_valid and str(value) in q_numbers,
                min(text.count("\\boxed"), 5), min(text.count("####"), 5),
                min(text.count("?"), 10),
                sum(lower.count(word) for word in ("maybe", "perhaps", "assuming", "unclear")),
                sum(lower.count(word) for word in ("however", "instead", "wait", "mistake")),
                min(lower.count("step"), 20), math.log1p(good + bad),
                good / max(good + bad, 1), math.log1p(bad), bad > 0,
                min(lengths[j] / average_length, 4),
                min(len(number_lists[j]) / average_numbers, 4),
            ]
    return features, valid


def build_features(bank, embeddings, mode):
    x = np.asarray(embeddings, dtype=np.float32)
    if x.shape != (len(bank), 4, 1024):
        raise ValueError("Expected bank-aligned [N, 4, 1024] embeddings.")
    metadata, valid = metadata_features(bank)
    if mode == "metadata":
        result = np.concatenate([x, metadata], axis=-1)
    elif mode == "context":
        # Compare against the group mean; features follow candidate permutations.
        relative = x - x.mean(axis=1, keepdims=True)
        result = np.concatenate([x, relative, metadata], axis=-1)
    else:
        raise ValueError(f"Unknown feature mode: {mode}")
    if not np.isfinite(result).all():
        raise ValueError("Non-finite features.")
    return result, valid


def mask_unparseable(scores, valid):
    # Unparseable answers cannot score; preserve scores if all answers are invalid.
    return np.where(valid | ~valid.any(axis=1, keepdims=True), scores, -1e9)
