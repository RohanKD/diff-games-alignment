#!/usr/bin/env python3
"""
Generate publication-quality figures for the differential games alignment paper.

Produces 4 PDF figures:
  1. phase_portraits.pdf   - Stable spiral vs limit cycle
  2. bifurcation.pdf       - Bifurcation diagram (amplitude vs beta)
  3. regret_comparison.pdf - Cumulative regret for 5 beta schedules
  4. eigenvalue_tracking.pdf - Re(lambda) vs beta for the 4D model
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)
from src.nonlinear_game import (
    simulate_2d, simulate_4d,
    compute_beta_c_2d, compute_beta_c_4d,
    limit_cycle_radius, compute_bifurcation_diagram,
    jacobian_4d, DEFAULT_PARAMS_2D, DEFAULT_PARAMS_4D,
)

FIGDIR = os.path.join(_PROJECT_DIR, 'paper', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    try:
        plt.style.use('seaborn-paper')
    except OSError:
        pass

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

# Shared simulation parameters
z0_2d = np.array([1.0, 0.5])
T = 100.0
dt = 0.005
beta_c = compute_beta_c_2d()  # 0.5
omega_0 = DEFAULT_PARAMS_2D['omega_0']
a = DEFAULT_PARAMS_2D['a']
mu = DEFAULT_PARAMS_2D['mu']


def make_colored_trajectory(ax, x, y, t, cmap='viridis', lw=0.8, alpha=0.9):
    """Draw a trajectory colored by time using LineCollection."""
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=t.min(), vmax=t.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=lw, alpha=alpha)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    return lc


# ================================================================== #
# Figure 1: Phase Portraits
# ================================================================== #
def fig_phase_portraits():
    print("Generating phase_portraits.pdf ...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    cases = [
        (2.0 * beta_c, r'$\beta = 2\beta_c$ (stable spiral)'),
        (0.5 * beta_c, r'$\beta = 0.5\beta_c$ (limit cycle)'),
    ]

    for ax, (beta_val, title) in zip(axes, cases):
        res = simulate_2d(z0_2d, T, dt, beta=beta_val)
        z1, z2, t = res['x1'], res['x2'], res['t']

        lc = make_colored_trajectory(ax, z1, z2, t)

        # Auto-scale
        margin = 0.15
        xmax = max(abs(z1.max()), abs(z1.min())) * (1 + margin)
        ymax = max(abs(z2.max()), abs(z2.min())) * (1 + margin)
        lim = max(xmax, ymax, 0.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        # Origin
        ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=1.5, zorder=5)

        # Limit cycle circle
        r_lc = limit_cycle_radius(beta_val)
        if r_lc > 0:
            theta = np.linspace(0, 2 * np.pi, 300)
            ax.plot(r_lc * np.cos(theta), r_lc * np.sin(theta),
                    'r--', linewidth=1.2, label=f'$r_{{eq}}={r_lc:.2f}$')
            ax.legend(loc='upper right', fontsize=10)

        ax.set_xlabel(r'$z_1$')
        ax.set_ylabel(r'$z_2$')
        ax.set_title(title)
        ax.set_aspect('equal')

    # Shared colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, T), cmap='viridis'),
        ax=axes, orientation='vertical', fraction=0.025, pad=0.02,
    )
    cbar.set_label('Time $t$')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'phase_portraits.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  -> done")


# ================================================================== #
# Figure 2: Bifurcation Diagram
# ================================================================== #
def fig_bifurcation():
    print("Generating bifurcation.pdf ...")
    bif = compute_bifurcation_diagram('2d', beta_range=(0.01, 1.0), n_points=100)

    betas = bif['betas']
    amps = bif['amplitudes']

    # Theoretical curve: amplitude of x1 oscillation = r_eq = sqrt((a-beta)/mu)
    theory_beta = np.linspace(0.01, 1.0, 500)
    theory_amp = np.where(
        theory_beta < a,
        np.sqrt((a - theory_beta) / mu),
        0.0,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(betas / beta_c, amps, s=18, color='C0', zorder=3,
               label='Simulation', edgecolors='none', alpha=0.8)
    ax.plot(theory_beta / beta_c, theory_amp, 'C3-', linewidth=1.5,
            label=r'Theory: $\sqrt{(a-\beta)/\mu}$')
    ax.axvline(1.0, color='grey', linestyle='--', linewidth=1.0,
               label=r'$\beta_c$')

    ax.set_xlabel(r'$\beta / \beta_c$')
    ax.set_ylabel('Asymptotic oscillation amplitude')
    ax.set_title('Supercritical Hopf Bifurcation')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 2.05)
    ax.set_ylim(bottom=-0.05)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'bifurcation.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  -> done")


# ================================================================== #
# Figure 3: Regret Comparison
# ================================================================== #
def fig_regret_comparison():
    print("Generating regret_comparison.pdf ...")

    from src.pmp_schedule import (
        constant_schedule, linear_decay_schedule, cosine_schedule, analytical_schedule,
    )
    schedules = {
        r'Constant $\beta_c$': constant_schedule(beta_c),
        r'Constant $1.5\beta_c$': constant_schedule(1.5 * beta_c),
        'Linear decay': linear_decay_schedule(2 * beta_c, 0.5 * beta_c, T),
        'Cosine anneal': cosine_schedule(2 * beta_c, 0.5 * beta_c, T),
        'PMP (Prop. 1)': analytical_schedule(beta_c, alpha=1.5, eta=0.5, mu_s=0.03, omega=omega_0),
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    final_regrets = {}
    for (name, sched), color in zip(schedules.items(), colors):
        res = simulate_2d(z0_2d, T, dt, beta_schedule=sched)
        r = res['r']
        times = res['t']
        # Cumulative trapezoidal integration
        integrand = r**2
        dt_actual = np.diff(times)
        increments = 0.5 * (integrand[:-1] + integrand[1:]) * dt_actual
        cumulative_regret = np.concatenate([[0], np.cumsum(increments)])
        ax.plot(times, cumulative_regret, color=color, linewidth=1.4, label=name)
        final_regrets[name] = cumulative_regret[-1]

    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Cumulative regret $\\int_0^t r^2\\,ds$')
    ax.set_title('Cumulative Regret Under Different $\\beta$ Schedules')
    ax.legend(fontsize=9, loc='upper left')

    # Print final values
    for name, val in final_regrets.items():
        print(f"    {name}: {val:.3f}")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'regret_comparison.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  -> done")


# ================================================================== #
# Figure 4: Eigenvalue Tracking (4D model)
# ================================================================== #
def fig_eigenvalue_tracking():
    print("Generating eigenvalue_tracking.pdf ...")

    beta_range = np.linspace(0.1, 1.5, 300)
    beta_c_4d = compute_beta_c_4d()
    print(f"    beta_c_4d = {beta_c_4d:.4f}")

    # Collect eigenvalue real parts
    all_re = np.zeros((len(beta_range), 4))
    for i, beta in enumerate(beta_range):
        J = jacobian_4d(beta)
        eigs = np.linalg.eigvals(J)
        # Sort by real part (descending) for consistent coloring
        idx = np.argsort(-eigs.real)
        all_re[i] = eigs.real[idx]

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']
    colors = ['C0', 'C1', 'C2', 'C3']
    for j in range(4):
        ax.plot(beta_range, all_re[:, j], color=colors[j], linewidth=1.4,
                label=labels[j])

    ax.axhline(0, color='k', linewidth=0.6, linestyle='-')
    ax.axvline(beta_c_4d, color='grey', linestyle='--', linewidth=1.0,
               label=rf'$\beta_c^{{4D}} \approx {beta_c_4d:.3f}$')

    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\mathrm{Re}(\lambda)$')
    ax.set_title('Eigenvalues of 4D Jacobian at Origin')
    ax.legend(fontsize=9, loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'eigenvalue_tracking.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  -> done")


# ================================================================== #
# Figure 5: Softmax Alignment Game Bifurcation
# ================================================================== #
def fig_softmax_bifurcation():
    print("Generating softmax_bifurcation.pdf ...")

    from src.softmax_bandit import (
        find_beta_c, simulate, DEFAULT_PARAMS, bifurcation_sweep,
    )

    beta_c = find_beta_c()
    print(f"    beta_c = {beta_c:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # --- Panel (a): Time series ---
    ax = axes[0]
    theta0 = np.array([0.3, 0.1])
    phi0 = DEFAULT_PARAMS['phi_star'] + np.array([0.05, 0.02])

    for frac, color, ls in [(0.7, 'C3', '-'), (0.9, 'C1', '-'), (1.3, 'C0', '-')]:
        beta = frac * beta_c
        traj = simulate(theta0, phi0, T=200, beta=beta, dt=0.01)
        label = rf'$\beta = {frac}\beta_c$'
        ax.plot(traj['t'], traj['true_reward'], color=color, linewidth=1.0,
                label=label, alpha=0.85)

    ax.set_xlabel('Time $t$')
    ax.set_ylabel('True reward $E_\\pi[r^*]$')
    ax.set_title('Alignment Cycling in Softmax Game')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, 200)

    # --- Panel (b): Bifurcation diagram ---
    ax = axes[1]
    betas, amplitudes, _ = bifurcation_sweep(n_betas=35, T=400)

    ax.scatter(betas / beta_c, amplitudes, s=18, c='C0', zorder=3, label='Simulation')
    ax.axvline(1.0, color='grey', linestyle='--', linewidth=1.0,
               label=rf'$\beta_c \approx {beta_c:.3f}$')
    ax.set_xlabel(r'$\beta / \beta_c$')
    ax.set_ylabel('Oscillation amplitude')
    ax.set_title('Bifurcation Diagram (Softmax Game)')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=-0.02)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'softmax_bifurcation.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  -> done")


# ================================================================== #
# Main
# ================================================================== #
if __name__ == '__main__':
    fig_phase_portraits()
    fig_bifurcation()
    fig_regret_comparison()
    fig_eigenvalue_tracking()
    fig_softmax_bifurcation()
    print(f"\nAll figures saved to {FIGDIR}/")
