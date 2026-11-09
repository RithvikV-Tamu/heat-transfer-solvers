"""Mesh refinement study for the hot-end heat loss with Richardson extrapolation."""
import numpy as np

from assembly import solve_bar, heat_loss_hot_end
from bar_problem import exact_heat_loss_hot_end
from element_fdm import element_matrix_second_order, element_matrix_fourth_order
from fem import element_matrix_linear, element_matrix_quadratic

# name: (element matrix function, nodes per element, expected order of the heat loss)
METHODS = {
    'FDM 2nd order': (element_matrix_second_order, 2, 2),
    'FDM 4th order': (element_matrix_fourth_order, 2, 4),
    'FEM p=1 (linear)': (element_matrix_linear, 2, 2),
    'FEM p=2 (quadratic)': (element_matrix_quadratic, 3, 4),
}


def solve_with_method(regions, n_each, method_name):
    element_matrix, nodes_per_element, _ = METHODS[method_name]
    x, elements, elem_regions, T = solve_bar(regions, n_each, element_matrix, nodes_per_element)
    Q = heat_loss_hot_end(x, elements, elem_regions, regions, T, element_matrix)
    return x, elements, T, Q


def richardson_orders(values):
    # Observed order from three meshes h, h/2, h/4. Needs no exact solution.
    orders = [np.nan, np.nan]
    for i in range(2, len(values)):
        coarse_change = values[i - 2] - values[i - 1]
        fine_change = values[i - 1] - values[i]
        orders.append(np.log2(abs(coarse_change / fine_change)))
    return orders


def richardson_extrapolate(values, orders):
    extrapolated = [np.nan, np.nan]
    for i in range(2, len(values)):
        p = orders[i]
        extrapolated.append(values[i] + (values[i] - values[i - 1]) / (2.0**p - 1.0))
    return extrapolated


def orders_from_exact_error(errors):
    orders = [np.nan]
    for i in range(1, len(errors)):
        orders.append(np.log2(errors[i - 1] / errors[i]))
    return orders


def convergence_study(regions, method_name, n_list):
    Q_exact = exact_heat_loss_hot_end(regions)
    longest_region = max(region['x1'] - region['x0'] for region in regions)
    h_values = [longest_region / n_each for n_each in n_list]
    Q_values = [solve_with_method(regions, n_each, method_name)[3] for n_each in n_list]
    errors = [abs(Q - Q_exact) for Q in Q_values]
    error_orders = orders_from_exact_error(errors)
    rich_orders = richardson_orders(Q_values)
    rich_values = richardson_extrapolate(Q_values, rich_orders)

    rows = []
    for i, n_each in enumerate(n_list):
        rows.append({
            'n_each': n_each,
            'h_max': h_values[i],
            'Q': Q_values[i],
            'Q_exact': Q_exact,
            'abs_error': errors[i],
            'order_from_error': error_orders[i],
            'richardson_order': rich_orders[i],
            'richardson_Q': rich_values[i],
        })
    return rows
