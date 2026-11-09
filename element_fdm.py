"""Element version of the finite difference method (AERO 430 Assignment 4).

Each interval is a two-node element. The local matrices come from Taylor
expansions of the end heat rates, truncated at second or fourth order.
"""
import numpy as np

from bar_problem import A, P


def element_matrix_second_order(k, h, dx):
    return np.array([
        [k * A / dx + h * P * dx / 2.0, -k * A / dx],
        [-k * A / dx, k * A / dx + h * P * dx / 2.0]
    ])


def element_matrix_fourth_order(k, h, dx):
    alpha2 = h * P / (k * A)
    diagonal = (1.0 + alpha2 * dx**2 / 2.0 + alpha2**2 * dx**4 / 24.0) / (dx + alpha2 * dx**3 / 6.0)
    off_diagonal = 1.0 / (dx + alpha2 * dx**3 / 6.0)
    return np.array([
        [k * A * diagonal, -k * A * off_diagonal],
        [-k * A * off_diagonal, k * A * diagonal]
    ])
