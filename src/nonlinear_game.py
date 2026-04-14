"""
Nonlinear alignment game simulation.

We provide two models:
1. A 2D reduced model (Hopf normal form) for the projected dynamics
   on the critical eigenspace — clean bifurcation for figures
2. A 4D model with the full policy-reward interaction

The 2D model captures the essential phenomenon: as beta decreases
through beta_c, the equilibrium loses stability and a stable limit
cycle (reward hacking oscillation) is born.
"""

import numpy as np
from typing import Optional, Callable, Dict
from scipy.optimize import brentq


# ================================================================== #
#  2D Reduced Model (Hopf Normal Form)                                #
# ================================================================== #

DEFAULT_PARAMS_2D = {
    'a': 0.5,          # natural growth rate (instability)
    'omega_0': 1.0,    # natural frequency
    'mu': 0.2,         # cubic saturation
    'kappa': 0.1,      # cubic frequency correction
}


def dynamics_2d(z, beta, params=None):
    """
    2D Hopf normal form for the alignment game.

    In polar (r, theta): dr/dt = (a - beta)*r - mu*r^3
                          dtheta/dt = omega_0 + kappa*r^2

    In Cartesian (z1, z2):
        dz1/dt = (a - beta)*z1 - omega_0*z2 - mu*(z1^2+z2^2)*z1 + kappa*(z1^2+z2^2)*(-z2)
        dz2/dt = omega_0*z1 + (a - beta)*z2 - mu*(z1^2+z2^2)*z2 + kappa*(z1^2+z2^2)*(z1)

    Bifurcation at beta_c = a. For beta < a: stable limit cycle with
    r_eq = sqrt((a - beta) / mu). For beta > a: stable origin.
    """
    p = params or DEFAULT_PARAMS_2D
    z1, z2 = z
    a, omega_0, mu, kappa = p['a'], p['omega_0'], p['mu'], p['kappa']

    r2 = z1**2 + z2**2

    dz1 = (a - beta) * z1 - omega_0 * z2 - mu * r2 * z1 + kappa * r2 * (-z2)
    dz2 = omega_0 * z1 + (a - beta) * z2 - mu * r2 * z2 + kappa * r2 * z1

    return np.array([dz1, dz2])


def compute_beta_c_2d(params=None):
    """Analytical bifurcation threshold for 2D model."""
    p = params or DEFAULT_PARAMS_2D
    return p['a']


def limit_cycle_radius(beta, params=None):
    """Predicted limit cycle radius for beta < beta_c."""
    p = params or DEFAULT_PARAMS_2D
    a, mu = p['a'], p['mu']
    if beta >= a:
        return 0.0
    return np.sqrt((a - beta) / mu)


def simulate_2d(
    z0: np.ndarray,
    T: float,
    dt: float,
    beta: float = 1.0,
    beta_schedule: Optional[Callable[[float], float]] = None,
    params: dict = None,
) -> Dict[str, np.ndarray]:
    """Simulate the 2D reduced alignment game via RK4."""
    p = params or DEFAULT_PARAMS_2D
    n_steps = int(np.ceil(T / dt)) + 1
    times = np.linspace(0, T, n_steps)
    z_traj = np.zeros((n_steps, 2))
    z_traj[0] = z0

    for k in range(n_steps - 1):
        t = times[k]
        z = z_traj[k]
        bt = beta_schedule(t) if beta_schedule is not None else beta

        k1 = dynamics_2d(z, bt, p)
        k2 = dynamics_2d(z + 0.5*dt*k1, bt, p)
        k3 = dynamics_2d(z + 0.5*dt*k2, bt, p)
        k4 = dynamics_2d(z + dt*k3, bt, p)
        z_traj[k+1] = z + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    return {
        't': times,
        'z': z_traj,
        'x1': z_traj[:, 0],
        'x2': z_traj[:, 1],
        'r': np.sqrt(z_traj[:, 0]**2 + z_traj[:, 1]**2),
    }


# ================================================================== #
#  4D Full Model                                                       #
# ================================================================== #

DEFAULT_PARAMS_4D = {
    'a': 0.5,         # policy instability
    'omega_0': 1.0,   # oscillation frequency
    'c': 0.6,         # reward -> policy coupling
    'd': 0.5,         # policy -> reward coupling
    'sigma': 0.5,     # reward damping
    'mu': 0.15,       # policy cubic saturation
    'nu': 0.15,       # reward cubic saturation
}


def dynamics_4d(z, beta, params=None):
    """
    4D alignment game with policy-reward coupling.
    Exhibits Hopf bifurcation at a critical beta_c determined
    numerically (approximately 0.6957 for default parameters).
    """
    p = params or DEFAULT_PARAMS_4D
    x1, x2, y1, y2 = z
    a, omega_0 = p['a'], p['omega_0']
    c, d, sigma = p['c'], p['d'], p['sigma']
    mu, nu = p['mu'], p['nu']

    rx2 = x1**2 + x2**2
    ry2 = y1**2 + y2**2

    # Policy dynamics: rotational + coupling + cubic
    dx1 = (a - beta)*x1 - omega_0*x2 + c*y1 - mu*rx2*x1
    dx2 = omega_0*x1 + (a - beta)*x2 + c*y2 - mu*rx2*x2

    # Reward dynamics: damped + driven by policy + cubic
    dy1 = d*x1 - sigma*y1 - nu*ry2*y1
    dy2 = d*x2 - sigma*y2 - nu*ry2*y2

    return np.array([dx1, dx2, dy1, dy2])


def jacobian_4d(beta, params=None):
    """Jacobian at the origin for the 4D model."""
    p = params or DEFAULT_PARAMS_4D
    a, omega_0, c, d, sigma = p['a'], p['omega_0'], p['c'], p['d'], p['sigma']
    return np.array([
        [a - beta, -omega_0, c, 0],
        [omega_0, a - beta, 0, c],
        [d, 0, -sigma, 0],
        [0, d, 0, -sigma],
    ])


def compute_beta_c_4d(params=None):
    """Find beta_c numerically for the 4D model."""
    p = params or DEFAULT_PARAMS_4D
    def max_re(beta):
        J = jacobian_4d(beta, p)
        return max(e.real for e in np.linalg.eigvals(J))
    try:
        return brentq(max_re, 0.01, 5.0)
    except ValueError:
        return None


def simulate_4d(
    z0: np.ndarray,
    T: float,
    dt: float,
    beta: float = 1.0,
    beta_schedule: Optional[Callable[[float], float]] = None,
    params: dict = None,
) -> Dict[str, np.ndarray]:
    """Simulate the 4D alignment game via RK4."""
    p = params or DEFAULT_PARAMS_4D
    n_steps = int(np.ceil(T / dt)) + 1
    times = np.linspace(0, T, n_steps)
    z_traj = np.zeros((n_steps, 4))
    z_traj[0] = z0

    for k in range(n_steps - 1):
        t = times[k]
        z = z_traj[k]
        bt = beta_schedule(t) if beta_schedule is not None else beta

        k1 = dynamics_4d(z, bt, p)
        k2 = dynamics_4d(z + 0.5*dt*k1, bt, p)
        k3 = dynamics_4d(z + 0.5*dt*k2, bt, p)
        k4 = dynamics_4d(z + dt*k3, bt, p)
        z_traj[k+1] = z + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    return {
        't': times,
        'z': z_traj,
        'x': z_traj[:, :2],
        'y': z_traj[:, 2:],
        'rx': np.sqrt(z_traj[:, 0]**2 + z_traj[:, 1]**2),
        'ry': np.sqrt(z_traj[:, 2]**2 + z_traj[:, 3]**2),
    }


def compute_bifurcation_diagram(
    model='2d',
    beta_range=(0.01, 1.5),
    n_points=150,
    T=300.0,
    dt=0.005,
    params=None,
):
    """
    Construct bifurcation diagram: asymptotic oscillation amplitude vs beta.
    """
    betas = np.linspace(beta_range[0], beta_range[1], n_points)
    amplitudes = np.zeros(n_points)

    if model == '2d':
        z0 = np.array([0.5, 0.3])
        sim_fn = simulate_2d
        p = params or DEFAULT_PARAMS_2D
        beta_c = compute_beta_c_2d(p)
    else:
        z0 = np.array([0.5, 0.3, 0.1, -0.1])
        sim_fn = simulate_4d
        p = params or DEFAULT_PARAMS_4D
        beta_c = compute_beta_c_4d(p)

    for i, beta in enumerate(betas):
        traj = sim_fn(z0, T, dt, beta=beta, params=p)
        n = len(traj['z'])
        tail = traj['z'][int(0.7 * n):]
        # Use a single Cartesian component to measure oscillation amplitude;
        # the radius is constant on a limit cycle, so max-min of ||z|| ≈ 0.
        x1_tail = tail[:, 0]
        amplitudes[i] = (x1_tail.max() - x1_tail.min()) / 2.0

    return {
        'betas': betas,
        'amplitudes': amplitudes,
        'beta_c': beta_c,
    }
