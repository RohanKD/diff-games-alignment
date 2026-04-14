"""
Hopf bifurcation analysis for the LQ alignment game.

As the KL penalty beta varies, the closed-loop eigenvalues of the Nash
equilibrium can cross the imaginary axis, triggering a Hopf bifurcation
that manifests as sustained oscillations in the RLHF training dynamics
(the "alignment cycling" phenomenon).
"""

from __future__ import annotations
from typing import Tuple, Optional, TYPE_CHECKING

import numpy as np

from . import riccati as _riccati
from . import lq_game as _lq

if TYPE_CHECKING:
    from .lq_game import LQAlignmentGame


def compute_bifurcation_threshold(
    game: "LQAlignmentGame",
    beta_range: Tuple[float, float] = (0.01, 5.0),
    n_points: int = 500,
) -> dict:
    """
    Sweep beta and locate the Hopf bifurcation threshold beta_c.

    For each beta, we solve the ARE, compute the closed-loop eigenvalues,
    and track the maximum real part.  The critical beta_c is where
    max Re(lambda) crosses zero from below.

    Parameters
    ----------
    game : LQAlignmentGame
        Base game (beta value is overridden during sweep).
    beta_range : (float, float)
        Sweep interval for beta.
    n_points : int
        Number of sample points.

    Returns
    -------
    result : dict
        'beta_c'      : float or None  -- critical beta (None if no crossing)
        'omega_c'     : float or None  -- imaginary part at the crossing
        'betas'       : ndarray (n_points,)
        'max_real'    : ndarray (n_points,)  -- max Re(eigenvalue) for each beta
        'eigenvalues' : list of ndarrays     -- full eigenvalue arrays
        'solved'      : ndarray (n_points,) bool -- whether ARE had a solution
    """
    betas = np.linspace(beta_range[0], beta_range[1], n_points)
    max_real = np.full(n_points, np.nan)
    all_eigs = [None] * n_points
    solved = np.zeros(n_points, dtype=bool)

    for i, beta in enumerate(betas):
        g = game.with_beta(beta)
        P = _riccati.solve_riccati_direct(g)
        if P is None:
            continue
        _, eigs = _riccati.closed_loop_jacobian(g, P)
        all_eigs[i] = eigs
        max_real[i] = np.max(eigs.real)
        solved[i] = True

    # Find zero-crossing of max_real (from negative to positive)
    beta_c = None
    omega_c = None

    valid = np.where(solved)[0]
    if len(valid) > 1:
        for j in range(len(valid) - 1):
            i0, i1 = valid[j], valid[j + 1]
            if max_real[i0] >= 0 and max_real[i1] < 0:
                # Linear interpolation for beta_c
                r0, r1 = max_real[i0], max_real[i1]
                frac = r0 / (r0 - r1)  # fraction from i0 to i1
                beta_c = betas[i0] + frac * (betas[i1] - betas[i0])

                # omega_c: imaginary part of the eigenvalue closest to the
                # imaginary axis at the crossing (interpolate between i0, i1)
                eigs0 = all_eigs[i0]
                eigs1 = all_eigs[i1]
                # Pick the eigenvalue with largest real part at i1
                idx1 = np.argmax(eigs1.real)
                omega_c = np.abs(eigs1[idx1].imag)
                break

    return {
        'beta_c':      beta_c,
        'omega_c':     omega_c,
        'betas':       betas,
        'max_real':    max_real,
        'eigenvalues': all_eigs,
        'solved':      solved,
    }


def bifurcation_diagram(
    game: "LQAlignmentGame",
    beta_range: Tuple[float, float] = (0.01, 5.0),
    n_points: int = 100,
    T: float = 200.0,
    dt: float = 0.01,
    z0: Optional[np.ndarray] = None,
) -> dict:
    """
    Construct a bifurcation diagram: asymptotic oscillation amplitude vs beta.

    For each beta value, simulate the Nash dynamics for time T and measure
    the oscillation amplitude in the last 20% of the trajectory.

    Parameters
    ----------
    game : LQAlignmentGame
    beta_range : (float, float)
    n_points : int
    T : float
        Simulation horizon (should be long enough for transients to decay).
    dt : float
    z0 : ndarray or None
        Initial condition; defaults to ones vector.

    Returns
    -------
    result : dict
        'betas'      : ndarray (n_points,)
        'amplitudes' : ndarray (n_points,)  -- oscillation amplitudes
        'converged'  : ndarray (n_points,) bool
    """
    betas = np.linspace(beta_range[0], beta_range[1], n_points)
    amplitudes = np.full(n_points, np.nan)
    converged = np.zeros(n_points, dtype=bool)

    if z0 is None:
        z0 = np.ones(game.nz)

    for i, beta in enumerate(betas):
        g = game.with_beta(beta)
        try:
            traj = _lq.simulate(g, z0, T, dt)
        except RuntimeError:
            # Riccati solver failed -- no equilibrium at this beta
            continue

        z = traj['z']
        n = len(z)
        tail = z[int(0.8 * n):]
        norms = np.linalg.norm(tail, axis=1)
        amplitudes[i] = norms.max() - norms.min()
        converged[i] = True

    return {
        'betas':      betas,
        'amplitudes': amplitudes,
        'converged':  converged,
    }
