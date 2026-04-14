"""
Riccati equation solver for the LQ alignment game.

For the two-player zero-sum LQ game, the Nash equilibrium is
characterized by the ARE:
    A^T P + P A + Q_tilde - P S P = 0
where S = B1_full B1_full^T / gamma - B2_full B2_full^T / delta.

Since S is indefinite (the game is zero-sum), scipy's
solve_continuous_are cannot be used directly. We solve via the
Hamiltonian eigenvalue decomposition: form the 2n x 2n Hamiltonian
    H = [[A, -S], [-Q, -A^T]]
and extract the stable invariant subspace.
"""

from __future__ import annotations
import warnings
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.linalg import eigvals

if TYPE_CHECKING:
    from .lq_game import LQAlignmentGame


def _build_full_B(game: "LQAlignmentGame"):
    """Build full-state B matrices for each player."""
    nz, nx = game.nz, game.nx
    B1_full = np.zeros((nz, game.B1.shape[1]))
    B1_full[:nx, :] = game.B1
    B2_full = np.zeros((nz, game.B2.shape[1]))
    B2_full[nx:, :] = game.B2
    return B1_full, B2_full


def _compute_S(game: "LQAlignmentGame") -> np.ndarray:
    """Compute S = B1_full B1_full^T / gamma - B2_full B2_full^T / delta."""
    B1_full, B2_full = _build_full_B(game)
    return (B1_full @ B1_full.T) / game.gamma - (B2_full @ B2_full.T) / game.delta


def solve_riccati_hamiltonian(game: "LQAlignmentGame") -> Optional[np.ndarray]:
    """
    Solve the zero-sum game ARE via the Hamiltonian eigenvalue approach.

    The ARE is: A^T P + P A + Q - P S P = 0
    where S = B1 B1^T / gamma - B2 B2^T / delta (indefinite).

    We form the Hamiltonian matrix:
        H = [[ A,   -S  ],
             [-Q,  -A^T ]]
    and find the n-dimensional stable invariant subspace. If [X1; X2]
    spans this subspace with X1 invertible, then P = X2 @ X1^{-1}.

    Returns None if the Hamiltonian has eigenvalues on the imaginary axis
    (no stabilizing solution exists).
    """
    A = game.A
    nz = game.nz
    Q_t = game.Q_tilde
    S = _compute_S(game)

    # Build 2n x 2n Hamiltonian
    H = np.block([
        [A,     -S],
        [-Q_t, -A.T],
    ])

    # Eigendecomposition of the Hamiltonian
    eig_vals, eig_vecs = np.linalg.eig(H)

    # Select eigenvectors with Re(lambda) < 0
    stable_idx = np.where(eig_vals.real < -1e-10)[0]

    if len(stable_idx) != nz:
        warnings.warn(
            f"Hamiltonian has {len(stable_idx)} stable eigenvalues (expected {nz}). "
            f"No stabilizing solution exists at beta = {game.beta:.4f}."
        )
        return None

    # Build stable subspace from selected eigenvectors
    V_stable = eig_vecs[:, stable_idx]

    # Extract blocks
    X1 = V_stable[:nz, :]
    X2 = V_stable[nz:, :]

    # Check X1 is invertible
    cond = np.linalg.cond(X1)
    if cond > 1e12:
        warnings.warn(
            f"X1 is nearly singular (cond = {cond:.2e}). "
            f"Riccati solution may be unreliable at beta = {game.beta:.4f}."
        )
        return None

    P = X2 @ np.linalg.inv(X1)
    P = np.real(P)  # Discard numerical imaginary residual
    P = 0.5 * (P + P.T)  # Symmetrize

    # Check if the solution is stabilizing
    S = _compute_S(game)
    J_cl = A + S @ P
    max_re = max(np.linalg.eigvals(J_cl).real)
    if max_re > 1e-6:
        warnings.warn(
            f"Riccati solution is NOT stabilizing at beta = {game.beta:.4f} "
            f"(max Re(eig) = {max_re:.4f}). The system is beyond the "
            f"bifurcation threshold."
        )

    return P


def solve_riccati_direct(game: "LQAlignmentGame") -> Optional[np.ndarray]:
    """
    Solve the zero-sum game ARE. This is the primary solver.

    Uses the Hamiltonian eigenvalue approach which correctly handles
    the indefinite S matrix arising in zero-sum games.
    """
    return solve_riccati_hamiltonian(game)


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
        dz/dt = (A + S P) z
    where S = B1 B1^T / gamma - B2 B2^T / delta.
    """
    A = game.A
    S = _compute_S(game)

    J_cl = A + S @ P

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
