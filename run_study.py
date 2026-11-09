"""Run every method on every benchmark case and save tables and plots to results/."""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from bar_problem import benchmark_cases, exact_temperature, ell
from convergence import METHODS, convergence_study, solve_with_method
from fem import evaluate_solution

N_LIST = [2, 4, 8, 16, 32, 64]
RESULTS_DIR = 'results'
MAIN_CASE = 'Case 2 (interface at 2/pi)'


def format_number(value, fmt):
    return '-' if np.isnan(value) else format(value, fmt)


def write_csv(all_rows, filename):
    columns = ['case', 'method', 'n_each', 'h_max', 'Q', 'Q_exact', 'abs_error',
               'order_from_error', 'richardson_order', 'richardson_Q']
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def markdown_detail_table(rows_by_method):
    method_names = list(rows_by_method.keys())
    header = '| n per region | h max | ' + ' | '.join(f'{m} error | order' for m in method_names) + ' |'
    lines = [header, '|---' * (2 + 2 * len(method_names)) + '|']
    first_rows = rows_by_method[method_names[0]]
    for i, row in enumerate(first_rows):
        cells = [str(row['n_each']), f"{row['h_max']:.5f}"]
        for method_name in method_names:
            method_row = rows_by_method[method_name][i]
            cells.append(f"{method_row['abs_error']:.3e}")
            cells.append(format_number(method_row['order_from_error'], '.3f'))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def markdown_richardson_table(rows_by_method):
    lines = ['| Method | n per region | Q(ell) | Richardson order | Richardson extrapolated Q |',
             '|---|---|---|---|---|']
    for method_name, rows in rows_by_method.items():
        for row in rows[2:]:
            lines.append(f"| {method_name} | {row['n_each']} | {row['Q']:.10f} | "
                         f"{row['richardson_order']:.3f} | {row['richardson_Q']:.10f} |")
    return '\n'.join(lines)


def markdown_summary_table(results):
    method_names = list(METHODS.keys())
    header = '| Case | ' + ' | '.join(method_names) + ' |'
    expected = '| Expected order | ' + ' | '.join(str(METHODS[m][2]) for m in method_names) + ' |'
    lines = [header, '|---' * (len(method_names) + 1) + '|', expected]
    for case_name, rows_by_method in results.items():
        cells = []
        for method_name in method_names:
            last = rows_by_method[method_name][-1]
            cells.append(f"{last['order_from_error']:.3f} / {last['richardson_order']:.3f}")
        lines.append(f'| {case_name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def plot_convergence(results, filename):
    markers = ['o', 's', '^', 'D']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (case_name, rows_by_method) in zip(axes.flat, results.items()):
        for marker, (method_name, rows) in zip(markers, rows_by_method.items()):
            h = [row['h_max'] for row in rows]
            err = [row['abs_error'] for row in rows]
            ax.loglog(h, err, marker + '-', linewidth=1.2, markersize=5, label=method_name)
        h = np.array([row['h_max'] for row in rows_by_method['FDM 2nd order']])
        e2 = rows_by_method['FDM 2nd order'][0]['abs_error']
        e4 = rows_by_method['FEM p=2 (quadratic)'][0]['abs_error']
        ax.loglog(h, e2 * 2.0 * (h / h[0])**2, 'k:', linewidth=1.0, label='slope 2 reference')
        ax.loglog(h, e4 * 0.3 * (h / h[0])**4, 'k--', linewidth=1.0, label='slope 4 reference')
        ax.invert_xaxis()
        ax.set_title(case_name)
        ax.set_xlabel('max element length h')
        ax.set_ylabel('|Q(ell) - Q_exact|')
        ax.grid(True, which='major', alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    plt.close(fig)


def plot_temperature(regions, case_name, n_each, filename):
    x_fine = np.linspace(0.0, ell, 800)
    plt.figure(figsize=(7, 4.5))
    plt.plot(x_fine, exact_temperature(regions, x_fine), 'k', linewidth=1.8, label='Exact')
    styles = {'FDM 2nd order': 'o', 'FDM 4th order': 's',
              'FEM p=1 (linear)': '--', 'FEM p=2 (quadratic)': '-.'}
    for method_name in METHODS:
        x, elements, T, _ = solve_with_method(regions, n_each, method_name)
        if method_name.startswith('FEM'):
            T_plot = evaluate_solution(x, elements, T, x_fine)
            plt.plot(x_fine, T_plot, styles[method_name], linewidth=1.2, label=method_name)
        else:
            plt.plot(x, T, styles[method_name], markersize=5, fillstyle='none', label=method_name + ' (nodes)')
    for region in regions[:-1]:
        plt.axvline(region['x1'], color='gray', linestyle=':', linewidth=0.9)
    plt.title(f'{case_name}, {n_each} elements per region')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(filename, dpi=140)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cases = benchmark_cases()
    results = {}
    all_rows = []
    for case_name, regions in cases.items():
        results[case_name] = {}
        for method_name in METHODS:
            rows = convergence_study(regions, method_name, N_LIST)
            results[case_name][method_name] = rows
            for row in rows:
                all_rows.append({'case': case_name, 'method': method_name, **row})

    write_csv(all_rows, os.path.join(RESULTS_DIR, 'convergence_all_cases.csv'))
    plot_convergence(results, os.path.join(RESULTS_DIR, 'heat_loss_convergence.png'))
    plot_temperature(cases['Three interfaces'], 'Three interfaces', 2,
                     os.path.join(RESULTS_DIR, 'temperature_three_interfaces.png'))

    Q_exact = results[MAIN_CASE]['FDM 2nd order'][0]['Q_exact']
    detail = markdown_detail_table(results[MAIN_CASE])
    richardson = markdown_richardson_table(results[MAIN_CASE])
    summary = markdown_summary_table(results)
    with open(os.path.join(RESULTS_DIR, 'convergence_tables.md'), 'w') as file:
        file.write(f'## {MAIN_CASE}, exact Q(ell) = {Q_exact:.10f}\n\n')
        file.write('Error in Q(ell) against the exact solution\n\n' + detail + '\n\n')
        file.write('Richardson extrapolation (no exact solution used)\n\n' + richardson + '\n\n')
        file.write('## Observed order on the finest mesh pair (from exact error / Richardson)\n\n')
        file.write(summary + '\n')

    print(f'Main case: {MAIN_CASE}, exact Q(ell) = {Q_exact:.10f}')
    print(detail)
    print()
    print(richardson)
    print()
    print(summary)
    print(f'\nSaved tables and plots to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
