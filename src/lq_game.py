"""
Linear-Quadratic differential game model for RLHF alignment.

Models RLHF as a two-player zero-sum differential game where:
  - Player 1 (policy trainer) controls u to minimize the cost
  - Player 2 (adversarial reward model) controls v to maximize the cost
  - State z = (x, y) tracks deviations from optimal policy/reward parameters

Dynamics:
    dx/dt = A11 x + A12 y + B1 u
    dy/dt = A21 x + A22 y + B2 v

Payoff (Player 1 minimizes, Player 2 maximizes):
    J = int_0^T { x^T Q x + y^T R y + 2 x^T S y
                   - (gamma/2)||u||^2 + (delta/2)||v||^2 } dt
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict

from . import riccati as _riccati_mod  # deferred to avoid circular at module level


@dataclass
class LQAlignmentGame:
    """
    Stores the matrices defining a linear-quadratic alignment game.

    Parameters
    ----------
    A11, A12, A21, A22 : ndarray
        Blocks of the joint drift matrix A = [[A11, A12], [A21, A22]].
    B1, B2 : ndarray
        Control-input matrices for player 1 (policy) and player 2 (reward).
    Q : ndarray
        State cost on policy deviation x.
    R : ndarray
        State cost on reward deviation y.
    S : ndarray
        Cross-coupling cost between x and y.
    gamma : float
        Control cost coefficient for player 1 (appears as -gamma/2 ||u||^2).
    delta : float
        Control cost coefficient for player 2 (appears as +delta/2 ||v||^2).
    beta : float
        KL-penalty strength; modifies effective A11 as A11 - beta * I.
    """
    A11: np.ndarray
    A12: np.ndarray
    A21: np.ndarray
    A22: np.ndarray
    B1: np.ndarray
    B2: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    S: np.ndarray
    gamma: float
    delta: float
    beta: float = 0.0

    # ------------------------------------------------------------------ #
    #  Derived properties                                                  #
    # ------------------------------------------------------------------ #

    @property
    def nx(self) -> int:
        """Dimension of the policy-deviation state x."""
        return self.A11.shape[0]

    @property
    def ny(self) -> int:
        """Dimension of the reward-deviation state y."""
        return self.A22.shape[0]

    @property
    def nz(self) -> int:
        """Total state dimension."""
        return self.nx + self.ny

    @property
    def A11_eff(self) -> np.ndarray:
        """Effective A11 including KL regularisation: A11 - beta * I."""
        return self.A11 - self.beta * np.eye(self.nx)

    @property
    def A(self) -> np.ndarray:
        """Full drift matrix using effective A11."""
        return np.block([
            [self.A11_eff, self.A12],
            [self.A21,     self.A22],
        ])

    @property
    def B(self) -> np.ndarray:
        """Stacked control matrix [B1; B2] (for reference only)."""
        nz = self.nz
        nu = self.B1.shape[1]
        nv = self.B2.shape[1]
        out = np.zeros((nz, nu + nv))
        out[:self.nx, :nu] = self.B1
        out[self.nx:, nu:] = self.B2
        return out

    @property
    def Q_tilde(self) -> np.ndarray:
        """Block cost matrix [[Q, S], [S^T, R]]."""
        return np.block([
            [self.Q,      self.S],
            [self.S.T,    self.R],
        ])

    # ------------------------------------------------------------------ #
    #  Factory methods                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def default_2d(cls, beta: float = 0.0) -> "LQAlignmentGame":
        """
        Canonical 2-D example (x in R^2, y in R^2).

        Uses rotation-like cross-coupling (A12, A21 have skew-symmetric
        components) to produce complex eigenvalues, enabling a Hopf
        bifurcation as beta varies. The adversary cost delta=10.0 ensures
        the zero-sum ARE admits a stabilizing solution over a wide range
        of beta values.

        Parameters
        ----------
        beta : float
            KL penalty strength.
        """
        A11 = np.array([[-0.3, 0.0],
                        [ 0.0,-0.3]])
        A12 = np.array([[ 0.5, 0.8],
                        [-0.8, 0.5]])
        A21 = np.array([[ 0.4, 0.3],
                        [-0.3, 0.4]])
        A22 = np.array([[-0.5, 0.0],
                        [ 0.0,-0.5]])
        B1 = np.eye(2)
        B2 = np.eye(2)
        Q  = np.eye(2)
        R  = 0.5 * np.eye(2)
        S  = 0.1 * np.eye(2)
        gamma = 1.0
        delta = 10.0
        return cls(A11=A11, A12=A12, A21=A21, A22=A22,
                   B1=B1, B2=B2, Q=Q, R=R, S=S,
                   gamma=gamma, delta=delta, beta=beta)

    def with_beta(self, beta: float) -> "LQAlignmentGame":
        """Return a copy of this game with a different beta value."""
        return LQAlignmentGame(
            A11=self.A11.copy(), A12=self.A12.copy(),
            A21=self.A21.copy(), A22=self.A22.copy(),
            B1=self.B1.copy(), B2=self.B2.copy(),
            Q=self.Q.copy(), R=self.R.copy(), S=self.S.copy(),
            gamma=self.gamma, delta=self.delta, beta=beta,
        )


# ====================================================================== #
#  Simulation                                                              #
# ====================================================================== #

def simulate(
    game: LQAlignmentGame,
    z0: np.ndarray,
    T: float,
    dt: float,
    beta_schedule: Optional[Callable[[float], float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Simulate the closed-loop Nash equilibrium dynamics via Euler integration.

    At each time step the Nash feedback controls are:
        u*(t) = (1/gamma) B1^T P z(t)
        v*(t) = -(1/delta) B2^T P z(t)
    where P is the solution of the algebraic Riccati equation for the
    current (possibly time-varying) beta.

    Parameters
    ----------
    game : LQAlignmentGame
        The game definition (beta is used as default if beta_schedule is None).
    z0 : ndarray, shape (nz,)
        Initial joint state [x0; y0].
    T : float
        Simulation horizon.
    dt : float
        Euler step size.
    beta_schedule : callable or None
        If provided, a function beta(t) -> float giving the KL penalty at
        time t.  Otherwise the constant game.beta is used.

    Returns
    -------
    result : dict
        't'      : (N,)     time grid
        'z'      : (N, nz)  full state trajectory
        'x'      : (N, nx)  policy deviation trajectory
        'y'      : (N, ny)  reward deviation trajectory
        'u'      : (N, nu)  player-1 control trajectory
        'v'      : (N, nv)  player-2 control trajectory
        'reward' : (N,)     instantaneous alignment reward  -x^T Q x
        'kl'     : (N,)     instantaneous KL penalty  beta * ||x||^2
    """
    n_steps = int(np.ceil(T / dt)) + 1
    times = np.linspace(0.0, T, n_steps)
    nx, ny = game.nx, game.ny
    nz = game.nz
    nu = game.B1.shape[1]
    nv = game.B2.shape[1]

    z_traj = np.zeros((n_steps, nz))
    u_traj = np.zeros((n_steps, nu))
    v_traj = np.zeros((n_steps, nv))
    reward = np.zeros(n_steps)
    kl     = np.zeros(n_steps)

    z_traj[0] = z0

    # Pre-compute Riccati solution for constant-beta case
    if beta_schedule is None:
        P_const = _riccati_mod.solve_riccati_direct(game)
        if P_const is None:
            raise RuntimeError(
                "Riccati equation has no solution at beta={:.4f} "
                "(finite-escape / reward-hacking blowup).".format(game.beta)
            )

    # Cache for time-varying beta (avoid re-solving at every step if beta
    # hasn't changed appreciably)
    _P_cache: dict = {}

    def _get_P(t: float) -> np.ndarray:
        if beta_schedule is None:
            return P_const
        beta_t = beta_schedule(t)
        # Quantise beta to avoid excessive Riccati solves
        beta_key = round(beta_t, 6)
        if beta_key not in _P_cache:
            g = game.with_beta(beta_t)
            P = _riccati_mod.solve_riccati_direct(g)
            if P is None:
                raise RuntimeError(
                    "Riccati equation has no solution at t={:.4f}, "
                    "beta={:.6f}".format(t, beta_t)
                )
            _P_cache[beta_key] = P
        return _P_cache[beta_key]

    for k in range(n_steps):
        t = times[k]
        z = z_traj[k]
        x = z[:nx]
        y = z[nx:]

        beta_t = beta_schedule(t) if beta_schedule is not None else game.beta

        # Riccati gain
        P = _get_P(t)

        # Nash equilibrium controls
        u = (1.0 / game.gamma) * game.B1.T @ P[:nx, :] @ z  # B1^T is (nu, nx)
        # For the full P, u depends on P applied to full z but B1 only couples
        # to the first nx rows.  Correct: B1^T @ P_top @ z where P_top = P[:nx,:].
        # Actually B1^T is (nu, nx), P is (nz, nz). We need B1_full^T P z.
        # B1_full = [[B1],[0]], so B1_full^T P z = B1^T @ P[:nx, :] @ z.  Correct.
        v = -(1.0 / game.delta) * game.B2.T @ P[nx:, :] @ z

        u_traj[k] = u
        v_traj[k] = v

        # Observables
        reward[k] = -x @ game.Q @ x
        kl[k] = beta_t * np.dot(x, x)

        # Euler step (skip for last point)
        if k < n_steps - 1:
            # Build effective A for this beta
            A_eff = np.block([
                [game.A11 - beta_t * np.eye(nx), game.A12],
                [game.A21,                        game.A22],
            ])
            dz = A_eff @ z
            dz[:nx] += game.B1 @ u
            dz[nx:] += game.B2 @ v
            z_traj[k + 1] = z + dt * dz

    return {
        't':      times,
        'z':      z_traj,
        'x':      z_traj[:, :nx],
        'y':      z_traj[:, nx:],
        'u':      u_traj,
        'v':      v_traj,
        'reward': reward,
        'kl':     kl,
    }
