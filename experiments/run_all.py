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

    from src.pmp_schedule import (
        constant_schedule, linear_decay_schedule, cosine_schedule, analytical_schedule,
    )
    schedules = {
        'Constant (beta_c)': constant_schedule(beta_c),
        'Constant (1.5*beta_c)': constant_schedule(1.5 * beta_c),
        'Linear decay': linear_decay_schedule(2 * beta_c, 0.5 * beta_c, T),
        'Cosine anneal': cosine_schedule(2 * beta_c, 0.5 * beta_c, T),
        'PMP (Prop. 3)': analytical_schedule(beta_c, alpha=1.5, eta=0.5, mu_s=0.03, omega=omega_0),
    }

    for name, sched in schedules.items():
        traj = simulate_2d(z0, T, dt, beta_schedule=sched)
        z_norm_sq = traj['z'][:, 0]**2 + traj['z'][:, 1]**2
        _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        regret = _trapz(z_norm_sq, traj['t'])
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
    from src.riccati import solve_riccati_direct, closed_loop_jacobian, open_loop_jacobian
    from scipy.optimize import brentq

    game = LQAlignmentGame.default_2d()

    # --- Open-loop (gradient play) bifurcation ---
    def max_re_open_loop(beta):
        g = game.with_beta(beta)
        _, eigs = open_loop_jacobian(g)
        return max(e.real for e in eigs)

    try:
        beta_c_ol = brentq(max_re_open_loop, 0.01, 1.5, xtol=1e-6)
        print(f"  Open-loop (gradient play) beta_c = {beta_c_ol:.4f}")
    except ValueError:
        print("  Could not find open-loop beta_c via bisection")
        beta_c_ol = None

    if beta_c_ol is not None:
        g_c = game.with_beta(beta_c_ol)
        _, eigs_ol = open_loop_jacobian(g_c)
        print(f"  Open-loop eigenvalues at beta_c:")
        for e in sorted(eigs_ol, key=lambda x: -x.real):
            print(f"    {e.real:+.6f} {e.imag:+.6f}j")
        complex_eigs = [e for e in eigs_ol if abs(e.imag) > 0.01]
        if complex_eigs:
            omega_c = max(abs(e.imag) for e in complex_eigs)
            print(f"  omega_c (cycling frequency) = {omega_c:.4f}")
            print(f"  This is a HOPF bifurcation (complex conjugate crossing)")

    # --- Nash closed-loop (A - SP): always stable ---
    print()
    print("  Nash closed-loop (A - SP) stability:")
    from src.lq_game import simulate
    z0 = np.array([1.0, 0.5, 0.1, -0.1])
    for beta_mult, label in [(0.5, "0.5*beta_c"), (1.0, "beta_c"), (1.5, "1.5*beta_c")]:
        beta_val = beta_mult * (beta_c_ol if beta_c_ol else 0.13)
        g = game.with_beta(beta_val)
        P = solve_riccati_direct(g)
        if P is not None:
            _, eigs = closed_loop_jacobian(g, P)
            max_re = max(e.real for e in eigs)
            traj = simulate(g, z0, T=100.0, dt=0.01)
            x_norm = np.linalg.norm(traj['x'], axis=1)
            tail = x_norm[int(0.7*len(x_norm)):]
            status = "CONVERGED" if tail.max() < 1e-3 else "DIVERGED"
            print(f"    beta = {label} ({beta_val:.4f}): max Re(A-SP) = {max_re:.6f}, "
                  f"||x(T)|| = {tail[-1]:.2e} [{status}]")
        else:
            print(f"    beta = {label}: No Riccati solution")

    print()


def experiment_7_pmp_bvp_solver():
    """E7: Run the PMP BVP solver on the LQ game."""
    print("=" * 60)
    print("Experiment 7: PMP BVP Solver (LQ Game)")
    print("=" * 60)

    from src.lq_game import LQAlignmentGame
    from src.pmp_schedule import solve_pmp_schedule

    game = LQAlignmentGame.default_2d(beta=1.0)
    result = solve_pmp_schedule(game, T=50.0, dt=0.05, n_mesh=100, max_iter=50)

    print(f"  Converged: {result['converged']}")
    print(f"  Time grid: {len(result['t'])} points, t in [{result['t'][0]:.1f}, {result['t'][-1]:.1f}]")
    print(f"  beta range: [{result['beta'].min():.4f}, {result['beta'].max():.4f}]")
    print(f"  beta(0) = {result['beta'][0]:.4f}, beta(T) = {result['beta'][-1]:.4f}")
    print()


def experiment_8_lyapunov_coefficients():
    """E8: Compute first Lyapunov coefficients to verify supercriticality."""
    print("=" * 60)
    print("Experiment 8: First Lyapunov Coefficients")
    print("=" * 60)

    from src.lyapunov import lyapunov_coefficient_2d, lyapunov_coefficient_4d

    # 2D model (exact — system IS the normal form)
    res_2d = lyapunov_coefficient_2d()
    print(f"  2D model:")
    print(f"    l_1 = {res_2d['l1']:.4f} ({res_2d['method']})")
    print(f"    Supercritical: {res_2d['supercritical']}")

    # 4D model (center manifold reduction)
    res_4d = lyapunov_coefficient_4d()
    print(f"  4D model:")
    print(f"    l_1 = {res_4d['l1']:.4f} ({res_4d['method']})")
    print(f"    beta_c = {res_4d['beta_c']:.4f}, omega_c = {res_4d['omega_c']:.4f}")
    print(f"    Supercritical: {res_4d['supercritical']}")

    assert res_2d['supercritical'], "2D model should be supercritical!"
    assert res_4d['supercritical'], "4D model should be supercritical!"
    print(f"\n  Both models confirmed SUPERCRITICAL (l_1 < 0)")
    print()


def experiment_9_softmax_bandit():
    """E9: Hopf bifurcation in a nonlinear softmax alignment game."""
    print("=" * 60)
    print("Experiment 9: Softmax Alignment Game (Feature-Based)")
    print("=" * 60)

    from src.softmax_bandit import (
        find_beta_c, linearize_at_equilibrium, simulate, DEFAULT_PARAMS,
    )

    beta_c = find_beta_c()
    print(f"  beta_c = {beta_c:.4f}")

    _, eigs = linearize_at_equilibrium(beta_c)
    complex_eigs = [e for e in eigs if abs(e.imag) > 0.01]
    if complex_eigs:
        omega_c = max(abs(e.imag) for e in complex_eigs)
        print(f"  omega_c = {omega_c:.4f}")
        print(f"  Eigenvalues at beta_c:")
        for e in sorted(eigs, key=lambda x: -x.real):
            print(f"    {e.real:+.6f} {e.imag:+.6f}j")

    # Verify oscillation below beta_c, convergence above
    theta0 = np.array([0.3, 0.1])
    phi0 = DEFAULT_PARAMS['phi_star'] + np.array([0.05, 0.02])

    print(f"\n  Oscillation test:")
    for frac in [0.5, 0.8, 0.95, 1.0, 1.2, 1.5]:
        beta = frac * beta_c
        traj = simulate(theta0, phi0, T=400, beta=beta, dt=0.01)
        tail = traj['true_reward'][int(0.6 * len(traj['true_reward'])):]
        amp = (tail.max() - tail.min()) / 2
        status = "CYCLING" if amp > 0.01 else "CONVERGED"
        print(f"    beta/beta_c = {frac:.2f}: amplitude = {amp:.4f} [{status}]")

    # Scaling exponent
    epsilons = []
    amps = []
    for frac in [0.97, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60]:
        beta = frac * beta_c
        traj = simulate(theta0, phi0, T=500, beta=beta, dt=0.01)
        tail = traj['true_reward'][int(0.6 * len(traj['true_reward'])):]
        amp = (tail.max() - tail.min()) / 2
        if amp > 0.01:
            epsilons.append(beta_c - beta)
            amps.append(amp)

    if len(epsilons) > 2:
        fit = np.polyfit(np.log(epsilons), np.log(amps), 1)
        print(f"\n  Scaling exponent (should be ~0.5): {fit[0]:.4f}")

    print()


if __name__ == '__main__':
    experiment_1_bifurcation_thresholds()
    experiment_2_limit_cycle_prediction()
    experiment_3_schedule_comparison()
    experiment_4_frequency_verification()
    experiment_5_hopf_scaling()
    experiment_6_lq_game_bifurcation()
    experiment_7_pmp_bvp_solver()
    experiment_8_lyapunov_coefficients()
    experiment_9_softmax_bandit()

    print("=" * 60)
    print("All experiments complete.")
    print("=" * 60)
