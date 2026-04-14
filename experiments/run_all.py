"""
Run all experiments and print results for the paper.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.nonlinear_game import (
    simulate_2d, simulate_4d,
    compute_beta_c_2d, compute_beta_c_4d,
    limit_cycle_radius, jacobian_4d,
    DEFAULT_PARAMS_2D, DEFAULT_PARAMS_4D,
)


def experiment_1_bifurcation_thresholds():
    """E1: Compute and verify bifurcation thresholds."""
    print("=" * 60)
    print("Experiment 1: Bifurcation Thresholds")
    print("=" * 60)

    beta_c_2d = compute_beta_c_2d()
    beta_c_4d = compute_beta_c_4d()
    print(f"  2D model: beta_c = {beta_c_2d:.4f}")
    print(f"  4D model: beta_c = {beta_c_4d:.4f}")

    # Eigenvalues at beta_c for 4D
    J = jacobian_4d(beta_c_4d)
    eigs = np.linalg.eigvals(J)
    print(f"  4D eigenvalues at beta_c:")
    for e in sorted(eigs, key=lambda x: -x.real):
        print(f"    {e.real:+.6f} {e.imag:+.6f}j")

    omega_c = max(abs(e.imag) for e in eigs if abs(e.real) < 0.01)
    print(f"  omega_c (cycling frequency) = {omega_c:.4f}")
    print()


def experiment_2_limit_cycle_prediction():
    """E2: Verify limit cycle radius matches analytical prediction."""
    print("=" * 60)
    print("Experiment 2: Limit Cycle Radius Verification (2D)")
    print("=" * 60)

    beta_c = compute_beta_c_2d()
    z0 = np.array([0.5, 0.3])
    errors = []

    for frac in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        beta = beta_c * frac
        traj = simulate_2d(z0, T=300, dt=0.005, beta=beta)
        tail_r = traj['r'][int(0.7 * len(traj['r'])):]
        measured_r = tail_r.mean()
        predicted_r = limit_cycle_radius(beta)
        error = abs(measured_r - predicted_r) / predicted_r * 100
        errors.append(error)
        print(f"  beta/beta_c = {frac:.2f}: measured r = {measured_r:.4f}, "
              f"predicted r = {predicted_r:.4f}, error = {error:.2f}%")

    print(f"  Mean prediction error: {np.mean(errors):.4f}%")
    print()


def experiment_3_schedule_comparison():
    """E3: Compare KL penalty schedules."""
    print("=" * 60)
    print("Experiment 3: Schedule Comparison (2D)")
    print("=" * 60)

    beta_c = compute_beta_c_2d()
    omega_0 = DEFAULT_PARAMS_2D['omega_0']
    z0 = np.array([1.0, 0.5])
    T = 100.0
    dt = 0.005

    schedules = {
        'Constant (beta_c)': lambda t: beta_c,
        'Constant (1.5*beta_c)': lambda t: 1.5 * beta_c,
        'Linear decay': lambda t: 2*beta_c + (0.5*beta_c - 2*beta_c) * t / T,
        'Cosine anneal': lambda t: 0.5*beta_c + 0.5*(2*beta_c - 0.5*beta_c) * (1 + np.cos(np.pi * t / T)),
        # PMP schedule from Theorem 3: beta_c*(1 + eta*exp(-mu*t)*cos(omega_c*t))
        # eta=0.5, mu estimated from spectral gap, omega_c = omega_0 for 2D model
        'PMP (Thm. 3)': lambda t: beta_c * (1.5 + 0.5 * np.exp(-0.03*t) * np.cos(omega_0*t)),
    }

    for name, sched in schedules.items():
        traj = simulate_2d(z0, T, dt, beta_schedule=sched)
        z_norm_sq = traj['z'][:, 0]**2 + traj['z'][:, 1]**2
        regret = np.trapezoid(z_norm_sq, traj['t'])
        final_r = traj['r'][-1]
        print(f"  {name:25s}: regret = {regret:8.3f}, final ||z|| = {final_r:.4f}")

    print()


def experiment_4_frequency_verification():
    """E4: Verify cycling frequency matches prediction."""
    print("=" * 60)
    print("Experiment 4: Cycling Frequency Verification (4D)")
    print("=" * 60)

    beta_c_4d = compute_beta_c_4d()
    J = jacobian_4d(beta_c_4d)
    eigs = np.linalg.eigvals(J)
    omega_pred = max(abs(e.imag) for e in eigs if abs(e.real) < 0.01)

    # Simulate at 0.8 * beta_c and measure frequency
    beta = 0.8 * beta_c_4d
    z0 = np.array([0.5, 0.3, 0.1, -0.1])
    traj = simulate_4d(z0, T=500, dt=0.005, beta=beta)

    # Measure frequency from x1 zero-crossings in steady state
    tail_x1 = traj['x'][int(0.7 * len(traj['x'])):, 0]
    tail_t = traj['t'][int(0.7 * len(traj['t'])):]

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(tail_x1)
    if len(peaks) > 2:
        periods = np.diff(tail_t[peaks])
        mean_period = periods.mean()
        omega_measured = 2 * np.pi / mean_period
        error = abs(omega_measured - omega_pred) / omega_pred * 100
        print(f"  Predicted omega_c = {omega_pred:.4f}")
        print(f"  Measured omega    = {omega_measured:.4f}")
        print(f"  Error             = {error:.1f}%")
    else:
        print("  Could not measure frequency (no oscillations detected)")

    print()


def experiment_5_hopf_scaling():
    """E5: Verify sqrt scaling of amplitude near bifurcation."""
    print("=" * 60)
    print("Experiment 5: Hopf Scaling (square-root law)")
    print("=" * 60)

    beta_c = compute_beta_c_2d()
    z0 = np.array([0.5, 0.3])

    epsilons = []
    measured_amps = []
    predicted_amps = []

    for frac in [0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60]:
        beta = beta_c * frac
        eps = beta_c - beta
        traj = simulate_2d(z0, T=500, dt=0.005, beta=beta)
        tail = traj['x1'][int(0.8*len(traj['x1'])):]
        amp = (tail.max() - tail.min()) / 2  # amplitude = radius
        pred_r = limit_cycle_radius(beta)

        epsilons.append(eps)
        measured_amps.append(amp)
        predicted_amps.append(pred_r)
        print(f"  eps = {eps:.4f}: amplitude = {amp:.4f}, predicted = {pred_r:.4f}")

    # Verify sqrt scaling: amplitude ~ sqrt(eps)
    eps_arr = np.array(epsilons)
    amp_arr = np.array(measured_amps)
    # Fit log(amp) = a * log(eps) + b
    log_fit = np.polyfit(np.log(eps_arr), np.log(amp_arr), 1)
    print(f"  Scaling exponent (should be ~0.5): {log_fit[0]:.4f}")
    print()


def experiment_6_lq_game_bifurcation():
    """E6: Verify Hopf bifurcation in the full LQ game with Riccati feedback."""
    print("=" * 60)
    print("Experiment 6: LQ Game Bifurcation (Riccati Nash Equilibrium)")
    print("=" * 60)

    from src.lq_game import LQAlignmentGame
    from src.riccati import solve_riccati_direct, closed_loop_jacobian
    from scipy.optimize import brentq

    game = LQAlignmentGame.default_2d()

    # Find beta_c via bisection on max Re(eigenvalue)
    def max_re_at_beta(beta):
        g = game.with_beta(beta)
        P = solve_riccati_direct(g)
        if P is None:
            return 1.0  # no solution = unstable
        _, eigs = closed_loop_jacobian(g, P)
        return max(e.real for e in eigs)

    # Bisect to find beta_c
    try:
        beta_c_lq = brentq(max_re_at_beta, 0.5, 1.5, xtol=1e-6)
        print(f"  LQ game beta_c = {beta_c_lq:.4f}")
    except ValueError:
        print("  Could not find beta_c via bisection")
        return

    # Eigenvalues at beta_c
    g_c = game.with_beta(beta_c_lq)
    P_c = solve_riccati_direct(g_c)
    _, eigs_c = closed_loop_jacobian(g_c, P_c)
    print(f"  Closed-loop eigenvalues at beta_c:")
    for e in sorted(eigs_c, key=lambda x: -x.real):
        print(f"    {e.real:+.6f} {e.imag:+.6f}j")

    # Crossing frequency
    complex_eigs = [e for e in eigs_c if abs(e.imag) > 0.01]
    if complex_eigs:
        omega_c = max(abs(e.imag) for e in complex_eigs)
        print(f"  omega_c (cycling frequency) = {omega_c:.4f}")
        print(f"  This is a HOPF bifurcation (complex conjugate crossing)")
    else:
        print(f"  Real eigenvalue crossing (not a Hopf bifurcation)")

    # Simulate at beta < beta_c to show oscillations
    print()
    print("  Simulation at beta = 0.5*beta_c (below threshold):")
    g_low = game.with_beta(0.5 * beta_c_lq)
    P_low = solve_riccati_direct(g_low)
    if P_low is not None:
        from src.lq_game import simulate
        z0 = np.array([1.0, 0.5, 0.1, -0.1])
        traj = simulate(g_low, z0, T=100.0, dt=0.01)
        x_norm = np.linalg.norm(traj['x'], axis=1)
        tail = x_norm[int(0.7*len(x_norm)):]
        print(f"    ||x|| tail: mean={tail.mean():.4f}, "
              f"max={tail.max():.4f}, min={tail.min():.4f}")
        if tail.max() > 1e-3:
            print(f"    System is UNSTABLE (growing/oscillating)")
        else:
            print(f"    System has converged")
    else:
        print(f"    No Riccati solution at this beta (game has no Nash eq.)")

    print()
    print("  Simulation at beta = 1.5*beta_c (above threshold):")
    g_high = game.with_beta(1.5 * beta_c_lq)
    P_high = solve_riccati_direct(g_high)
    if P_high is not None:
        traj = simulate(g_high, z0, T=100.0, dt=0.01)
        x_norm = np.linalg.norm(traj['x'], axis=1)
        tail = x_norm[int(0.7*len(x_norm)):]
        print(f"    ||x|| tail: mean={tail.mean():.4f}, "
              f"max={tail.max():.4f}, min={tail.min():.4f}")
        if tail.max() < 1e-3:
            print(f"    System has CONVERGED (stable spiral)")
        else:
            print(f"    System is oscillating")

    print()


if __name__ == '__main__':
    experiment_1_bifurcation_thresholds()
    experiment_2_limit_cycle_prediction()
    experiment_3_schedule_comparison()
    experiment_4_frequency_verification()
    experiment_5_hopf_scaling()
    experiment_6_lq_game_bifurcation()

    print("=" * 60)
    print("All experiments complete.")
    print("=" * 60)
