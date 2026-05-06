"""
Pontryagin's Maximum Principle (PMP) for optimal beta(t) schedules.

The outer optimisation problem is:

    min_{beta(t)}  integral_0^T  L(z, beta) dt

where z evolves under the closed-loop Nash dynamics parameterised by
beta(t), and L penalises alignment regret, reward hacking, and
excessive KL cost.

We also provide several closed-form schedule families used as baselines
or analytical approximations.
"""

from __future__ import annotations
from typing import Optional, Callable, Dict, TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_bvp

from . import riccati as _riccati
from . import lq_game as _lq

if TYPE_CHECKING:
    from .lq_game import LQAlignmentGame


# ====================================================================== #
#  Analytical / closed-form schedules                                      #
# ====================================================================== #

def constant_schedule(beta_val: float) -> Callable[[float], float]:
    """Return a callable that gives constant beta for all t."""
    def _schedule(t):
        return beta_val
    return _schedule


def linear_decay_schedule(
    beta_start: float,
    beta_end: float,
    T: float,
) -> Callable[[float], float]:
    """Linear interpolation from beta_start to beta_end over [0, T]."""
    def _schedule(t):
        frac = np.clip(t / T, 0.0, 1.0)
        return beta_start + (beta_end - beta_start) * frac
    return _schedule


def cosine_schedule(
    beta_start: float,
    beta_end: float,
    T: float,
) -> Callable[[float], float]:
    """Cosine annealing from beta_start to beta_end over [0, T]."""
    def _schedule(t):
        frac = np.clip(t / T, 0.0, 1.0)
        return beta_end + 0.5 * (beta_start - beta_end) * (1.0 + np.cos(np.pi * frac))
    return _schedule


def analytical_schedule(
    beta_c: float,
    alpha: float = 1.5,
    eta: float = 0.5,
    mu_s: float = 0.03,
    omega: float = 1.0,
    psi: float = 0.0,
) -> Callable[[float], float]:
    """
    PMP-motivated schedule (Proposition 3):

        beta*(t) = beta_c * (alpha + eta * exp(-mu_s*t) * cos(omega*t + psi))

    Starts above beta_c and decays toward alpha*beta_c > beta_c,
    maintaining a margin above the bifurcation threshold at all times.

    Parameters
    ----------
    beta_c : float
        Critical beta at the Hopf bifurcation.
    eta : float
        Initial fractional overshoot above beta_c.
    mu_s : float
        Exponential decay rate (spectral gap of the stable manifold).
    omega : float
        Oscillation frequency (ideally matches omega_c from the
        bifurcation analysis).
    psi : float
        Phase offset.
    """
    def _schedule(t):
        return beta_c * (alpha + eta * np.exp(-mu_s * t) * np.cos(omega * t + psi))
    return _schedule


# ====================================================================== #
#  PMP-based optimal schedule                                              #
# ====================================================================== #

def solve_pmp_schedule(
    game: "LQAlignmentGame",
    T: float = 50.0,
    dt: float = 0.01,
    z0: Optional[np.ndarray] = None,
    beta_bounds: tuple = (0.01, 5.0),
    n_mesh: int = 200,
    max_iter: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Solve for the optimal beta(t) schedule via Pontryagin's Maximum
    Principle, formulated as a two-point boundary value problem (BVP).

    The state equation (closed-loop Nash dynamics with a given beta(t)):
        dz/dt = f(z, beta)  =  (A(beta) - M P(beta)) z
    with  z(0) = z0.

    The running cost is:
        L(z, beta) = z^T Q_tilde z + alpha * (beta - beta_ref)^2
    where alpha regularises the schedule and beta_ref is a reference.

    The adjoint (costate) equation:
        d lambda/dt = - dL/dz  -  (df/dz)^T lambda
                    = -2 Q_tilde z  -  J_cl(beta)^T lambda
    with terminal condition  lambda(T) = 0.

    The optimality condition:
        dL/d beta + lambda^T (df/d beta) z = 0
    yields beta*(t) at each t.

    We approximate df/d beta by finite differences on the closed-loop
    Jacobian and use scipy.integrate.solve_bvp.

    Parameters
    ----------
    game : LQAlignmentGame
    T : float
    dt : float
    z0 : ndarray or None  (defaults to ones)
    beta_bounds : (float, float)
    n_mesh : int
        Initial mesh points for solve_bvp.
    max_iter : int

    Returns
    -------
    result : dict
        't'         : ndarray (N,)
        'beta'      : ndarray (N,)      -- optimal beta(t)
        'z'         : ndarray (N, nz)   -- state trajectory
        'lambda'    : ndarray (N, nz)   -- costate trajectory
        'converged' : bool
    """
    nz = game.nz
    nx = game.nx

    if z0 is None:
        z0 = np.ones(nz)

    alpha = 0.1          # regularisation weight on beta deviation
    beta_ref = 1.0       # reference beta
    eps_fd = 1e-4        # finite-difference step for df/dbeta

    Q_t = game.Q_tilde

    def _get_Jcl(beta_val):
        """Closed-loop Jacobian at a given beta."""
        g = game.with_beta(beta_val)
        P = _riccati.solve_riccati_direct(g)
        if P is None:
            return None
        J_cl, _ = _riccati.closed_loop_jacobian(g, P)
        return J_cl

    def _ode(t_norm, Y):
        """
        ODE for the augmented system [z; lambda; integral_state].
        t_norm in [0, 1], rescaled so that t = t_norm * T.

        Y shape: (2*nz + 1, n_points)  -- solve_bvp passes multiple
        collocation points at once.
        """
        n_pts = Y.shape[1]
        dYdt = np.zeros_like(Y)

        for j in range(n_pts):
            z_j = Y[:nz, j]
            lam_j = Y[nz:2*nz, j]

            # Determine beta from optimality condition (simplified):
            # beta* = beta_ref - (1/(2*alpha)) * lam^T (dJ_cl/dbeta) z  * z
            # Approximate with current iterate -- use a heuristic projection.
            # For the BVP we parameterise beta as part of the state to make
            # the system self-consistent.

            # Heuristic: beta from a closed-form expression
            # dJ_cl/dbeta ~ -I_{xx block}  (since A11_eff = A11 - beta I)
            # So df/dbeta z ~ -[I 0; 0 0] z  => first nx components of z negated.
            dfdb_z = np.zeros(nz)
            dfdb_z[:nx] = -z_j[:nx]
            beta_opt = beta_ref - (1.0 / (2.0 * alpha)) * lam_j @ dfdb_z
            beta_opt = np.clip(beta_opt, beta_bounds[0], beta_bounds[1])

            J_cl = _get_Jcl(beta_opt)
            if J_cl is None:
                J_cl = _get_Jcl(beta_ref)
                if J_cl is None:
                    # Fallback: use open-loop A
                    J_cl = game.A

            # State dynamics (rescale by T)
            dz = T * (J_cl @ z_j)

            # Costate dynamics
            dlam = T * (-2.0 * Q_t @ z_j - J_cl.T @ lam_j)

            # Running cost (for monitoring)
            L = z_j @ Q_t @ z_j + alpha * (beta_opt - beta_ref) ** 2

            dYdt[:nz, j] = dz
            dYdt[nz:2*nz, j] = dlam
            dYdt[2*nz, j] = T * L

        return dYdt

    def _bc(Ya, Yb):
        """Boundary conditions: z(0)=z0, lambda(T)=0, cost(0)=0."""
        res = np.zeros(2 * nz + 1)
        res[:nz] = Ya[:nz] - z0               # z(0) = z0
        res[nz:2*nz] = Yb[nz:2*nz]            # lambda(T) = 0
        res[2*nz] = Ya[2*nz]                   # cost(0) = 0
        return res

    # Initial mesh and guess
    t_mesh = np.linspace(0, 1, n_mesh)
    Y_guess = np.zeros((2 * nz + 1, n_mesh))
    # Guess: z decays linearly from z0 to 0, lambda = 0
    for i in range(n_mesh):
        Y_guess[:nz, i] = z0 * (1.0 - t_mesh[i])

    try:
        sol = solve_bvp(_ode, _bc, t_mesh, Y_guess,
                        max_nodes=5000, tol=1e-4, verbose=0)
        converged = sol.success
    except Exception as exc:
        # If solve_bvp fails, fall back to a forward simulation with
        # an analytical schedule
        import warnings
        warnings.warn(f"solve_bvp failed ({exc}); returning analytical fallback.")

        # Use constant schedule as fallback
        t_grid = np.linspace(0, T, int(T / dt) + 1)
        return {
            't':         t_grid,
            'beta':      np.full_like(t_grid, beta_ref),
            'z':         np.zeros((len(t_grid), nz)),
            'lambda':    np.zeros((len(t_grid), nz)),
            'converged': False,
        }

    # Evaluate solution on a fine grid
    t_fine = np.linspace(0, 1, int(T / dt) + 1)
    Y_fine = sol.sol(t_fine)
    t_phys = t_fine * T

    z_sol = Y_fine[:nz, :].T
    lam_sol = Y_fine[nz:2*nz, :].T

    # Reconstruct beta(t) from optimality condition
    beta_sol = np.zeros(len(t_phys))
    for j in range(len(t_phys)):
        z_j = z_sol[j]
        lam_j = lam_sol[j]
        dfdb_z = np.zeros(nz)
        dfdb_z[:nx] = -z_j[:nx]
        beta_opt = beta_ref - (1.0 / (2.0 * alpha)) * lam_j @ dfdb_z
        beta_sol[j] = np.clip(beta_opt, beta_bounds[0], beta_bounds[1])

    return {
        't':         t_phys,
        'beta':      beta_sol,
        'z':         z_sol,
        'lambda':    lam_sol,
        'converged': converged,
    }
