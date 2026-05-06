"""
Hopf bifurcation analysis for the LQ alignment game.

As the KL penalty beta varies, the open-loop (gradient-play) eigenvalues
of A(beta) can cross the imaginary axis, triggering a Hopf bifurcation
that manifests as sustained oscillations in the RLHF training dynamics
(the "alignment cycling" phenomenon).

Note: The Nash equilibrium closed-loop A - SP is always Hurwitz (stable)
by construction of the Hamiltonian eigenvalue solver.  The bifurcation
characterizes gradient play, not equilibrium play.
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
    Sweep beta and locate the Hopf bifurcation threshold beta_c
    in the open-loop (gradient-play) dynamics A(beta).

    For each beta, we compute the eigenvalues of the open-loop Jacobian
    A(beta) and track the maximum real part.  The critical beta_c is
    where max Re(lambda) crosses zero.

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
    """
    betas = np.linspace(beta_range[0], beta_range[1], n_points)
    max_real = np.full(n_points, np.nan)
    all_eigs = [None] * n_points

    for i, beta in enumerate(betas):
        g = game.with_beta(beta)
        _, eigs = _riccati.open_loop_jacobian(g)
        all_eigs[i] = eigs
        max_real[i] = np.max(eigs.real)

    # Find zero-crossing of max_real
    beta_c = None
    omega_c = None

    for j in range(n_points - 1):
        if max_real[j] >= 0 and max_real[j + 1] < 0:
            # Linear interpolation for beta_c
            r0, r1 = max_real[j], max_real[j + 1]
            frac = r0 / (r0 - r1)
            beta_c = betas[j] + frac * (betas[j + 1] - betas[j])

            # omega_c: imaginary part of the eigenvalue closest to
            # the imaginary axis at the crossing
            eigs1 = all_eigs[j + 1]
            idx1 = np.argmax(eigs1.real)
            omega_c = np.abs(eigs1[idx1].imag)
            break

    return {
        'beta_c':      beta_c,
        'omega_c':     omega_c,
        'betas':       betas,
        'max_real':    max_real,
        'eigenvalues': all_eigs,
    }
