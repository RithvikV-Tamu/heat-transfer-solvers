import unittest

import numpy as np

from bar_problem import make_regions, benchmark_cases, exact_temperature, region_alpha, T0, TL
from convergence import METHODS, solve_with_method, convergence_study, richardson_orders


class TestExactSolution(unittest.TestCase):

    def test_single_material_matches_sinh_formula(self):
        regions = make_regions([], [1.0])
        alpha = region_alpha(regions[0])
        x = np.linspace(0.0, 1.0, 11)
        expected = TL * np.sinh(alpha * x) / np.sinh(alpha)
        np.testing.assert_allclose(exact_temperature(regions, x), expected, rtol=1e-12, atol=1e-12)


class TestSolvers(unittest.TestCase):

    def test_penalty_method_enforces_end_temperatures(self):
        regions = benchmark_cases()['Two interfaces']
        for method_name in METHODS:
            x, elements, T, Q = solve_with_method(regions, 4, method_name)
            self.assertAlmostEqual(T[0], T0, places=8)
            self.assertAlmostEqual(T[-1], TL, places=8)

    def test_nodal_temperatures_close_to_exact(self):
        regions = benchmark_cases()['Three interfaces']
        for method_name in METHODS:
            x, elements, T, Q = solve_with_method(regions, 16, method_name)
            max_error = np.max(np.abs(T - exact_temperature(regions, x)))
            self.assertLess(max_error, 1e-2, method_name)

    def test_observed_order_matches_expected(self):
        regions = benchmark_cases()['Case 2 (interface at 2/pi)']
        for method_name, (_, _, expected_order) in METHODS.items():
            rows = convergence_study(regions, method_name, [4, 8, 16, 32])
            self.assertAlmostEqual(rows[-1]['order_from_error'], expected_order, delta=0.1, msg=method_name)
            self.assertAlmostEqual(rows[-1]['richardson_order'], expected_order, delta=0.1, msg=method_name)


class TestRichardson(unittest.TestCase):

    def test_recovers_known_order(self):
        h = np.array([0.4, 0.2, 0.1, 0.05])
        values = 3.0 + 0.7 * h**3
        orders = richardson_orders(list(values))
        np.testing.assert_allclose(orders[2:], [3.0, 3.0], rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
