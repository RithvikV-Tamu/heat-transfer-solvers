"""Mesh, global assembly, penalty boundary conditions and heat-loss extraction.

Shared by the element FDM and the FEM. Each method only supplies its own
element matrix function element_matrix(k, h, element_length).
"""
import numpy as np

from bar_problem import T0, TL


def build_mesh(regions, n_each, nodes_per_element):
    # n_each equal elements per material region, so every interface is a node.
    x = [regions[0]['x0']]
    elements = []
    elem_regions = []
    for j, region in enumerate(regions):
        x_ends = np.linspace(region['x0'], region['x1'], n_each + 1)
        for e in range(n_each):
            xa = x_ends[e]
            xb = x_ends[e + 1]
            left = len(x) - 1
            for m in range(1, nodes_per_element):
                x.append(xa + (xb - xa) * m / (nodes_per_element - 1))
            elements.append(list(range(left, left + nodes_per_element)))
            elem_regions.append(j)
    return np.array(x), elements, elem_regions


def assemble_system(x, elements, elem_regions, regions, element_matrix):
    n_nodes = len(x)
    K = np.zeros((n_nodes, n_nodes))
    f = np.zeros(n_nodes)
    for e, conn in enumerate(elements):
        region = regions[elem_regions[e]]
        element_length = x[conn[-1]] - x[conn[0]]
        ke = element_matrix(region['k'], region['h'], element_length)
        K[np.ix_(conn, conn)] += ke
    return K, f


def apply_penalty_bc(K, f, left_value, right_value, penalty_factor=1.0e12):
    penalty = penalty_factor * np.max(np.diag(K))
    K[0, 0] += penalty
    K[-1, -1] += penalty
    f[0] += penalty * left_value
    f[-1] += penalty * right_value
    return K, f


def solve_bar(regions, n_each, element_matrix, nodes_per_element):
    x, elements, elem_regions = build_mesh(regions, n_each, nodes_per_element)
    K, f = assemble_system(x, elements, elem_regions, regions, element_matrix)
    K, f = apply_penalty_bc(K, f, T0, TL)
    T = np.linalg.solve(K, f)
    return x, elements, elem_regions, T


def heat_loss_hot_end(x, elements, elem_regions, regions, T, element_matrix):
    # The last element's equation at node N gives -kappa A dT/dx at x = ell
    # without differentiating the numerical solution.
    conn = elements[-1]
    region = regions[elem_regions[-1]]
    element_length = x[conn[-1]] - x[conn[0]]
    ke = element_matrix(region['k'], region['h'], element_length)
    return -(ke @ T[conn])[-1]
