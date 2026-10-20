"""Check official TabM export parity and MoB bootstrap probabilities."""
from itertools import product
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import numpy as np
import tabm
import torch

from mp2.features import build_features
from mp2.aggregation import aggregate_heads, select_answers
from mp2.model import load_tabm, predict, tabm_heads
from mp2.third_party.mob_algorithms import _bootstrap_bestofn_dist
from mp2.training import export_tabm, forward_heads, head_loss


class ModelTests(unittest.TestCase):
    def test_mob_matches_exact_bootstrap_enumeration(self):
        answers = np.array(["A", "B", "B", "C"])
        rewards = np.array([0.1, 0.2, 0.3, 0.4])
        distribution, values = _bootstrap_bestofn_dist(answers, rewards, 2)
        counts = {answer: 0 for answer in values}
        for sampled in product(range(4), repeat=2):
            best = max(sampled, key=lambda i: rewards[i])
            counts[answers[best]] += 1
        np.testing.assert_allclose(distribution, [counts[a] / 16 for a in values])

    def test_group_aggregation_and_invalid_fallback(self):
        bank = [{"candidates": [{"text": "#### 1/2"}, {"text": "#### 0.5"},
                                {"text": "#### 3"}, {"text": "invalid"}]}]
        scores = np.array([[0.4, 0.4, 0.7, 0.9]], dtype=np.float32)
        self.assertEqual(select_answers(bank, scores, "best"), [2])
        self.assertEqual(select_answers(bank, scores, "answer_sum"), [0])
        for method in ("best", "answer_sum", "mob_poly", "mob_adaptive"):
            invalid = [{"candidates": [{"text": "invalid"} for _ in range(4)]}]
            self.assertEqual(select_answers(invalid, scores, method), [3])

    def test_independent_heads_have_independent_losses(self):
        y = torch.tensor([[1., 0., 1., 0.]])
        heads = torch.tensor([[[5., -1., 0., 0.], [-1., 5., 0., 0.]]], requires_grad=True)
        loss = head_loss(heads, y)
        from mp2.training_utils import selection_loss
        expected = (selection_loss(heads[:, 0], y) + selection_loss(heads[:, 1], y)) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(heads.grad).all())

    def test_tabm_export_and_empty_bank(self):
        torch.manual_seed(7)
        bank = [{"question": "2 + 3?", "candidates": [
            {"text": "#### 5"}, {"text": "#### 4"}, {"text": "invalid"},
            {"text": r"\boxed{5}"}]}]
        x = np.random.default_rng(7).normal(size=(1, 4, 1024)).astype(np.float32)
        features, valid = build_features(bank, x, "context")
        dim = features.shape[-1]
        model = tabm.TabM.make(n_num_features=dim, d_out=1, n_blocks=2,
                              d_block=8, dropout=0.15, k=2, arch_type="tabm").eval()
        mean = np.zeros(dim, np.float32)
        scale = np.ones(dim, np.float32)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export_tabm(model, mean, scale, SimpleNamespace(k=2, width=8), root / "member_0")
            config, weights = load_tabm(root / "member_0")
            scores = tabm_heads(features, config, weights)
            with torch.no_grad():
                expected = forward_heads(model, torch.from_numpy(features)).numpy()
            np.testing.assert_allclose(scores, expected, atol=3e-5, rtol=3e-5)
            (root / "ensemble.json").write_text(json.dumps({
                "format": "mp2-research-ensemble-v1", "members": ["member_0"],
                "aggregation": "mean_sigmoid", "selection": "best",
            }))
            choices = predict(bank, {"embeddings": x}, root)
            self.assertEqual(choices, select_answers(
                bank, aggregate_heads(expected.transpose(1, 0, 2), valid, "mean_sigmoid"), "best"))
            self.assertEqual(predict([], {"embeddings": x[:0]}, root), [])


    def test_supervision_rejects_test_and_audit(self):
        from mp2.training_utils import load_supervised
        for split in ("test", "audit"):
            with self.assertRaises(ValueError):
                load_supervised(split, Path("/not-read"))

    def test_bounded_arithmetic(self):
        from fractions import Fraction
        from mp2.features import numeric_expression, equation_statistics
        self.assertEqual(numeric_expression("1 / 2 + 3 * 2"), Fraction("6.5"))
        self.assertEqual(equation_statistics("2 + 3 = 5; 2 * 3 = 9"), (1, 1))
        for value in ("__import__('os').system('pwd')", "2 ** 1000000", "[1][0]", "True"):
            with self.assertRaises((ValueError, SyntaxError)):
                numeric_expression(value)

    def test_features_follow_candidate_permutations(self):
        bank = [{"id": "new", "question": "2 + 3?",
                 "candidates": [{"text": f"#### {n}"} for n in (5, 4, 6, 5)]}]
        x = np.random.default_rng(9).normal(size=(1, 4, 1024)).astype(np.float32)
        order = [2, 0, 3, 1]
        first, valid = build_features(bank, x, "context")
        changed = [{**bank[0], "id": "another",
                    "candidates": [bank[0]["candidates"][j] for j in order]}]
        second, other_valid = build_features(changed, x[:, order], "context")
        np.testing.assert_allclose(second, first[:, order], atol=1e-6)
        np.testing.assert_array_equal(other_valid, valid[:, order])


if __name__ == "__main__":
    unittest.main()
