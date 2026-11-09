"""Physical setup and exact solution for a bar with Newton heat losses.

Governing equation inside each material region:
    -d/dx( kappa A dT/dx ) + h P T = 0,   T(0) = T0,  T(ell) = TL
"""
import numpy as np

ell = 1.0
r = 0.1
A = np.pi * r**2
P = 2 * np.pi * r
T0 = 0.0
TL = 100.0
h_default = 1.0


def make_regions(x_breaks, k_values, h_values=None):
    if h_values is None:
        h_values = [h_default] * len(k_values)
    regions = []
    x_left = 0.0
    for x_right, k, h in zip(list(x_breaks) + [ell], k_values, h_values):
        regions.append({'x0': x_left, 'x1': float(x_right), 'k': float(k), 'h': float(h)})
        x_left = float(x_right)
    return regions


def benchmark_cases():
    return {
        'Case 1 (interface at 0.5)': make_regions([0.5], [1.0, 4.0]),
        'Case 2 (interface at 2/pi)': make_regions([2.0 / np.pi], [1.0, 4.0]),
        'Two interfaces': make_regions([0.30, 0.72], [1.0, 4.0, 0.5]),
        'Three interfaces': make_regions([0.20, 0.55, 0.83], [1.0, 8.0, 2.0, 16.0]),
    }


def region_alpha(region):
    return np.sqrt(region['h'] * P / (region['k'] * A))


def region_transfer_matrix(region):
    # Maps [T, kappa A dT/dx] at the left end of a region to the right end.
    k = region['k']
    alpha = region_alpha(region)
    length = region['x1'] - region['x0']
    c = np.cosh(alpha * length)
    s = np.sinh(alpha * length)
    return np.array([
        [c, s / (k * A * alpha)],
        [k * A * alpha * s, c]
    ])


def exact_region_states(regions):
    total = np.eye(2)
    for region in regions:
        total = region_transfer_matrix(region) @ total
    heat_rate_at_left = (TL - total[0, 0] * T0) / total[0, 1]
    state = np.array([T0, heat_rate_at_left])
    states = [state.copy()]
    for region in regions:
        state = region_transfer_matrix(region) @ state
        states.append(state.copy())
    return states


def exact_temperature(regions, x_points):
    states = exact_region_states(regions)
    x_points = np.asarray(x_points, dtype=float)
    T = np.zeros_like(x_points)
    for j, region in enumerate(regions):
        in_region = (x_points >= region['x0'] - 1e-14) & (x_points <= region['x1'] + 1e-14)
        local_x = x_points[in_region] - region['x0']
        k = region['k']
        alpha = region_alpha(region)
        T_left, heat_rate_left = states[j]
        T[in_region] = (np.cosh(alpha * local_x) * T_left
                        + np.sinh(alpha * local_x) / (k * A * alpha) * heat_rate_left)
    return T


def exact_heat_loss_hot_end(regions):
    # Q(ell) = -kappa A dT/dx at x = ell
    states = exact_region_states(regions)
    return -states[-1][1]
