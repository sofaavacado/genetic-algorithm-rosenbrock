import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import (
    Config, DIM, LOWER, UPPER, ga, rosenbrock, mutate,
    blx_alpha_crossover, paired_t_comparison, tournament,
)


class TestAlgorithm(unittest.TestCase):
    def test_global_optimum(self):
        self.assertAlmostEqual(float(rosenbrock(np.ones(DIM))[0]), 0.0)

    def test_nonnegativity(self):
        x = np.random.default_rng(3).uniform(-5, 10, size=(100, DIM))
        self.assertTrue(np.all(rosenbrock(x) >= 0))

    def test_bounds_after_mutation(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            a = rng.uniform(LOWER, UPPER, DIM)
            b = rng.uniform(LOWER, UPPER, DIM)
            child = blx_alpha_crossover(a, b, 0.5, rng)
            result = mutate(child, 1.0, 10.0, rng)
            self.assertTrue(np.all(result >= LOWER) and np.all(result <= UPPER))

    def test_batch_operators_match_shapes_and_bounds(self):
        """Проверяем реальные пакетные операторы, которые вызывает ga()."""
        rng = np.random.default_rng(12)
        first = rng.uniform(LOWER, UPPER, (28, DIM))
        second = rng.uniform(LOWER, UPPER, (28, DIM))
        crossed = blx_alpha_crossover(first, second, 0.5, rng)
        children = mutate(crossed, 0.4, 0.7, rng)
        self.assertEqual(children.shape, (28, DIM))
        self.assertTrue(np.all((children >= LOWER) & (children <= UPPER)))
        self.assertTrue(np.array_equal(first.shape, children.shape))

    def test_tournament_batch(self):
        fitness = np.array([3.0, 10.0, 0.0, 5.0])
        rng1 = np.random.default_rng(17)
        actual = tournament(fitness, rng1, size=3, count=20)
        rng2 = np.random.default_rng(17)
        candidates = rng2.integers(len(fitness), size=(20, 3))
        expected = candidates[np.arange(20), np.argmin(fitness[candidates], axis=1)]
        np.testing.assert_array_equal(actual, expected)
        self.assertIsInstance(tournament(fitness, np.random.default_rng(17), size=3), int)

    def test_ga_calls_tested_operators(self):
        cfg = Config(population=30, evaluations=90)
        with (patch('main.tournament', wraps=tournament) as select,
              patch('main.blx_alpha_crossover', wraps=blx_alpha_crossover) as cross,
              patch('main.mutate', wraps=mutate) as mutation):
            ga(cfg, 10)
        self.assertEqual(select.call_count, 4)
        self.assertEqual(cross.call_count, 2)
        self.assertEqual(mutation.call_count, 2)

    def test_paired_t_statistic(self):
        data = paired_t_comparison([1, 3, 2, 4, 5], [2, 2, 4, 5, 6])
        self.assertEqual(data['degrees_of_freedom'], 4)
        self.assertLess(data['p_value'], 1.0)
        self.assertGreaterEqual(data['p_value'], 0.0)

    def test_budget_seed_reproducibility(self):
        cfg = Config(population=30, evaluations=300)
        a = ga(cfg, 5)
        b = ga(cfg, 5)
        self.assertEqual(a['evaluations'], 300)
        self.assertEqual(a['best_f'], b['best_f'])
        self.assertEqual(a['history'], b['history'])
        self.assertEqual(len(a['history']), 10)
        self.assertTrue(np.all(np.diff(a['history']) <= 1e-12))


if __name__ == '__main__':
    unittest.main()
