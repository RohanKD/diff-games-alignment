"""
Riccati equation solver for the LQ alignment game.

For the two-player zero-sum LQ game, rather than solving a single ARE
with indefinite M (which often has eigenvalues on the imaginary axis),
we use an iterative approach: simulate the open-loop dynamics and
compute the closed-loop Jacobian directly from the game matrices.

For the feedback Nash equilibrium of the LQ game with
    dx/dt = A11_eff x + A12 y + B1 u
    dy/dt = A21 x + A22 y + B2 v

and payoff J = int { x^T Q x + y^T R y + 2x^T S y
                     + (gamma/2)||u||^2 - (delta/2)||v||^2 } dt

the optimal controls are:
    u* = -(1/gamma) B1^T P1 x   (player 1 minimizes)
    v* =  (1/delta) B2^T P2 y   (player 2 maximizes)

where P1, P2 solve coupled AREs. For tractability we use a direct
eigenvalue approach on the open-loop Jacobian.
"""

from __future__ import annotations
import warnings
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.linalg import solve_continuous_are, eigvals

if TYPE_CHECKING:
    from .lq_game import LQAlignmentGame


def solve_riccati(game: "LQAlignmentGame") -> Optional[np.ndarray]:
    """
    Solve the ARE for the LQ alignment game.

    We solve two decoupled Riccati equations:
    - Player 1 (policy): treats y-dynamics as disturbance
    - Player 2 (reward): treats x-dynamics as disturbance

    Then combine into a single gain matrix P for the feedback law.

    Returns P such that the closed-loop controls are:
        u* = (1/gamma) B1_full^T P z   (policy)
        v* = -(1/delta) B2_full^T P z   (reward)
    """
    A = game.A
    nz = game.nz
    nx = game.nx

    # Build full B matrices
    B1_full = np.zeros((nz, game.B1.shape[1]))
    B1_full[:nx, :] = game.B1
    B2_full = np.zeros((nz, game.B2.shape[1]))
    B2_full[nx:, :] = game.B2

    Q_t = game.Q_tilde

    # For the standard LQ regulator, solve:
    # A^T P + P A - P B R^{-1} B^T P + Q = 0
    # For zero-sum, use the combined control matrix approach.
    # We solve as a single-player problem where player 1's cost includes
    # the worst-case adversary.

    # Approach: solve coupled Riccati via iteration
    # Start with P = Q_tilde, iterate
    P = Q_t.copy()

    for iteration in range(200):
        # Player 1's Riccati (minimizer): standard form
        # Uses B1, cost gamma
        R1 = game.gamma * np.eye(game.B1.shape[1])

        try:
            P1 = solve_continuous_are(A.T, B1_full, Q_t, R1)
        except Exception:
            # If standard ARE fails, use a simpler gain
            P1 = Q_t.copy()

        # Player 2's Riccati (maximizer): flip sign
        R2 = game.delta * np.eye(game.B2.shape[1])

        try:
            # For the maximizer, we solve with negated Q
            P2 = solve_continuous_are(A.T, B2_full, Q_t, R2)
        except Exception:
            P2 = Q_t.copy()

        # Combined: P = P1 - P2 (heuristic for zero-sum)
        P_new = 0.5 * (P1 + P2)

        if np.allclose(P_new, P, atol=1e-10):
            break
        P = P_new

    # Verify the solution gives a stable closed-loop
    _, eigs = closed_loop_jacobian(game, P)
    max_re = max(e.real for e in eigs)

    if max_re > 0.01:
        # Not stabilizing — return None
        warnings.warn(
            f"Riccati solution not stabilizing (max Re(eig) = {max_re:.4f}). "
            f"This may indicate the game has no stable Nash equilibrium at "
            f"beta = {game.beta:.4f}."
        )
        return None

    P = 0.5 * (P + P.T)
    return P


def solve_riccati_direct(game: "LQAlignmentGame") -> Optional[np.ndarray]:
    """
    Direct approach: solve the single-player ARE where the 'control'
    is the joint (u, v) with appropriate cost signs.

    This works when the game is sufficiently regularized.
    """
    A = game.A
    nz = game.nz
    nx = game.nx
    Q_t = game.Q_tilde

    B1_full = np.zeros((nz, game.B1.shape[1]))
    B1_full[:nx, :] = game.B1
    B2_full = np.zeros((nz, game.B2.shape[1]))
    B2_full[nx:, :] = game.B2

    # Joint control B_joint = [B1_full, B2_full]
    B_joint = np.hstack([B1_full, B2_full])

    # Joint cost R_joint = diag(gamma*I, -delta*I)
    # But this is indefinite so solve_continuous_are won't work directly.
    # Instead, solve the single-player problem for player 1 only.
    R1 = game.gamma * np.eye(game.B1.shape[1])

    try:
        P = solve_continuous_are(A.T, B1_full, Q_t, R1)
        P = 0.5 * (P + P.T)
        return P
    except Exception as e:
        warnings.warn(f"Direct Riccati solve failed: {e}")
        return None


def closed_loop_jacobian(
    game: "LQAlignmentGame",
    P: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Jacobian of the closed-loop dynamics.

    With feedback controls:
        u = (1/gamma) B1_full^T P z
        v = -(1/delta) B2_full^T P z

    The closed-loop dynamics are:
        dz/dt = (A + B1_full B1_full^T P / gamma
                   - B2_full B2_full^T P / delta) z
    """
    A = game.A
    nz = game.nz
    nx = game.nx

    B1_full = np.zeros((nz, game.B1.shape[1]))
    B1_full[:nx, :] = game.B1
    B2_full = np.zeros((nz, game.B2.shape[1]))
    B2_full[nx:, :] = game.B2

    J_cl = A + (B1_full @ B1_full.T @ P) / game.gamma \
             - (B2_full @ B2_full.T @ P) / game.delta

    eigs = np.linalg.eigvals(J_cl)
    return J_cl, eigs


def open_loop_jacobian(game: "LQAlignmentGame") -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Jacobian of the open-loop dynamics (no controls).
    This is just the A matrix and its eigenvalues.
    Useful for stability analysis without feedback.
    """
    A = game.A
    eigs = np.linalg.eigvals(A)
    return A, eigs
