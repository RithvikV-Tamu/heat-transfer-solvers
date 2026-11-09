"""Finite element method with linear (p=1) and quadratic (p=2) elements
(AERO 430 Assignment 5). Element matrices come from the weak form with
exact integration of the shape-function products.
"""
import numpy as np

from bar_problem import A, P


def element_matrix_linear(k, h, L):
    diffusion = k * A / L * np.array([
        [1.0, -1.0],
        [-1.0, 1.0]
    ])
    reaction = h * P * L / 6.0 * np.array([
        [2.0, 1.0],
        [1.0, 2.0]
    ])
    return diffusion + reaction


def element_matrix_quadratic(k, h, L):
    diffusion = k * A / (3.0 * L) * np.array([
        [7.0, -8.0, 1.0],
        [-8.0, 16.0, -8.0],
        [1.0, -8.0, 7.0]
    ])
    reaction = h * P * L / 30.0 * np.array([
        [4.0, 2.0, -1.0],
        [2.0, 16.0, 2.0],
        [-1.0, 2.0, 4.0]
    ])
    return diffusion + reaction


def evaluate_solution(x, elements, T, x_query):
    # Interpolate the FEM solution with its own shape functions.
    values = np.zeros(len(x_query))
    for m, xq in enumerate(x_query):
        e = len(elements) - 1
        for candidate, conn in enumerate(elements):
            if x[conn[0]] - 1e-14 <= xq <= x[conn[-1]] + 1e-14:
                e = candidate
                break
        conn = elements[e]
        xa = x[conn[0]]
        xb = x[conn[-1]]
        xi = 2.0 * (xq - xa) / (xb - xa) - 1.0
        if len(conn) == 2:
            N = np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0])
        else:
            N = np.array([xi * (xi - 1.0) / 2.0, 1.0 - xi**2, xi * (xi + 1.0) / 2.0])
        values[m] = N @ T[conn]
    return values
