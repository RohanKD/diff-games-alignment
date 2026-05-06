"""
Feature-based alignment game with distribution-shift reward corruption.

Demonstrates that alignment cycling (Hopf bifurcation) extends beyond
the LQ/normal-form framework to nonlinear softmax policy dynamics.

Model:
  K actions with feature vectors f_k uniformly on the unit circle in R^2.
  Policy pi_theta(k) = softmax(theta^T f_k), theta in R^2.
  Proxy reward: r_phi(k) = phi^T f_k, phi in R^2.
  True reward: r*(k) = phi*^T f_k, phi* in R^2 (fixed).

Gradient play dynamics:
  Policy (maximizer):
    dtheta/dt = nabla_theta [E_pi[r_phi] - beta * KL(pi || pi_ref)]

  Reward model (distribution-shift corruption):
    dphi/dt = -delta*(phi - phi*) + c * J * mu_shift
    where mu_shift = E_pi[f] - E_pi_ref[f] is the feature-space distribution shift,
    and J = [[0,-1],[1,0]] is a 90-degree rotation.

The rotation models a generic high-dimensional phenomenon: when the policy
shifts the data distribution in direction d, the reward model loses calibration
in orthogonal directions (less data there), creating bias that rotates the
effective reward landscape.  This is the "misaligned degradation" mechanism
that drives alignment cycling.
"""

import numpy as np
from scipy.integrate import solve_ivp


def _make_features(K):
    """K feature vectors uniformly on the unit circle in R^2."""
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def softmax(logits):
    """Numerically stable softmax."""
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()


DEFAULT_PARAMS = {
    'K': 16,
    'phi_star': np.array([0.5, 0.0]),
    'delta': 0.4,
    'c': 1.5,          # distribution-shift coupling strength
}


def _rhs(t, z, K, features, beta, delta, c, phi_star, pi_ref_logits):
    """RHS of gradient play dynamics."""
    theta = z[:2]
    phi = z[2:]

    # Policy
    logits = features @ theta
    pi = softmax(logits)

    # Expected feature under policy and reference
    mu_pi = features.T @ pi          # 2-vector
    mu_ref = features.T @ (np.ones(K) / K)  # = 0 for symmetric features

    # Policy reward gradient: nabla_theta E_pi[r_phi]
    r_phi = features @ phi            # K-vector of proxy rewards
    E_r = pi @ r_phi
    reward_grad = features.T @ (pi * (r_phi - E_r))  # 2-vector

    # KL gradient: nabla_theta KL(pi || pi_ref)
    log_pi = np.log(np.clip(pi, 1e-15, None))
    log_pi_ref = pi_ref_logits - np.log(np.sum(np.exp(pi_ref_logits)))
    log_ratio = log_pi - log_pi_ref
    kl_val = pi @ log_ratio
    kl_grad = features.T @ (pi * (log_ratio - kl_val))  # 2-vector

    dtheta = reward_grad - beta * kl_grad

    # Reward model: distribution-shift corruption
    J90 = np.array([[0.0, -1.0], [1.0, 0.0]])
    mu_shift = mu_pi - mu_ref
    dphi = -delta * (phi - phi_star) + c * J90 @ mu_shift

    return np.concatenate([dtheta, dphi])


def simulate(theta0, phi0, T, beta, params=None, dt=0.01):
    """Simulate gradient play in the feature-based alignment game."""
    p = params or DEFAULT_PARAMS
    K = p['K']
    features = _make_features(K)
    pi_ref_logits = np.zeros(K)  # uniform reference

    z0 = np.concatenate([theta0, phi0])

    sol = solve_ivp(
        lambda t, z: _rhs(t, z, K, features, beta, p['delta'], p['c'],
                          p['phi_star'], pi_ref_logits),
        [0, T],
        z0,
        method='RK45',
        max_step=0.05,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
    )

    t_eval = np.arange(0, T, dt)
    z_eval = sol.sol(t_eval).T

    theta_traj = z_eval[:, :2]
    phi_traj = z_eval[:, 2:]
    pi_traj = np.array([softmax(features @ th) for th in theta_traj])
    true_reward = np.array([pi @ (features @ p['phi_star']) for pi in pi_traj])
    proxy_reward = np.array([pi @ (features @ ph)
                             for pi, ph in zip(pi_traj, phi_traj)])

    return {
        't': t_eval,
        'theta': theta_traj,
        'phi': phi_traj,
        'pi': pi_traj,
        'true_reward': true_reward,
        'proxy_reward': proxy_reward,
    }


def linearize_at_equilibrium(beta, params=None):
    """
    Compute Jacobian eigenvalues at the equilibrium (theta=0, phi=phi*).

    At equilibrium with symmetric features, pi = uniform, mu = 0.
    """
    p = params or DEFAULT_PARAMS
    K = p['K']
    features = _make_features(K)
    pi_ref_logits = np.zeros(K)

    # Equilibrium: theta=0 (uniform policy), phi=phi*
    theta_eq = np.zeros(2)
    phi_eq = p['phi_star'].copy()
    z_eq = np.concatenate([theta_eq, phi_eq])

    # Numerical Jacobian
    n = 4
    eps = 1e-7
    J = np.zeros((n, n))
    f0 = _rhs(0, z_eq, K, features, beta, p['delta'], p['c'],
              p['phi_star'], pi_ref_logits)
    for j in range(n):
        z_pert = z_eq.copy()
        z_pert[j] += eps
        f_pert = _rhs(0, z_pert, K, features, beta, p['delta'], p['c'],
                      p['phi_star'], pi_ref_logits)
        J[:, j] = (f_pert - f0) / eps

    eigs = np.linalg.eigvals(J)
    return J, eigs


def find_beta_c(params=None, beta_range=(0.01, 5.0), tol=1e-5):
    """Find critical beta_c where max Re(eigenvalue) crosses zero."""
    p = params or DEFAULT_PARAMS

    def max_real(beta):
        _, eigs = linearize_at_equilibrium(beta, p)
        return max(eigs.real)

    lo, hi = beta_range
    f_lo, f_hi = max_real(lo), max_real(hi)
    if f_lo <= 0 or f_hi > 0:
        return None

    for _ in range(100):
        mid = (lo + hi) / 2
        if max_real(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    return (lo + hi) / 2


def bifurcation_sweep(params=None, beta_range=None, n_betas=40, T=300, dt=0.01):
    """Sweep beta and measure asymptotic oscillation amplitude."""
    p = params or DEFAULT_PARAMS

    beta_c = find_beta_c(p)
    if beta_c is None:
        raise ValueError("Could not find beta_c")

    if beta_range is None:
        beta_range = (0.3 * beta_c, 2.0 * beta_c)

    betas = np.linspace(beta_range[0], beta_range[1], n_betas)
    amplitudes = np.zeros(n_betas)

    np.random.seed(42)
    for i, beta in enumerate(betas):
        theta0 = np.random.randn(2) * 0.2
        phi0 = p['phi_star'] + np.random.randn(2) * 0.05
        traj = simulate(theta0, phi0, T, beta, p, dt)

        tail = traj['true_reward'][int(0.6 * len(traj['true_reward'])):]
        amplitudes[i] = (tail.max() - tail.min()) / 2

    return betas, amplitudes, beta_c
