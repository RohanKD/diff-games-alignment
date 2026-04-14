"""
Utility functions for analysis and visualisation of the LQ alignment game.
"""

from typing import Dict, Optional

import numpy as np


def compute_alignment_regret(
    trajectory: Dict[str, np.ndarray],
    z_star: Optional[np.ndarray] = None,
) -> float:
    """
    Integrated alignment regret: integral of ||z(t) - z*||^2 dt.

    Parameters
    ----------
    trajectory : dict
        Must contain 't' (time array) and 'z' (state array).
    z_star : ndarray or None
        Target state; defaults to zero (perfect alignment).

    Returns
    -------
    regret : float
    """
    t = trajectory['t']
    z = trajectory['z']
    if z_star is None:
        z_star = np.zeros(z.shape[1])
    diff = z - z_star[np.newaxis, :]
    sq_norms = np.sum(diff ** 2, axis=1)
    # Trapezoidal integration (np.trapezoid in numpy>=2.0, np.trapz before)
    _trapz = getattr(np, 'trapezoid', np.trapz)
    regret = float(_trapz(sq_norms, t))
    return regret


def compute_reward_hacking_metric(
    trajectory: Dict[str, np.ndarray],
) -> float:
    """
    Measure how much the reward model is being exploited.

    Defined as the average exponential growth rate of ||y(t)||:
        metric = (1/T) * log(||y(T)|| / ||y(0)||)

    A positive value indicates reward hacking (the reward-model
    deviation is growing).

    Parameters
    ----------
    trajectory : dict
        Must contain 't' and 'y'.

    Returns
    -------
    growth_rate : float
        Average exponential growth rate.  Returns 0 if ||y(0)|| is
        negligible.
    """
    t = trajectory['t']
    y = trajectory['y']
    T = t[-1] - t[0]
    if T <= 0:
        return 0.0
    norm_0 = np.linalg.norm(y[0])
    norm_T = np.linalg.norm(y[-1])
    if norm_0 < 1e-15:
        # Cannot compute growth rate from zero initial condition
        return 0.0
    ratio = max(norm_T / norm_0, 1e-30)
    return float(np.log(ratio) / T)


def compute_oscillation_amplitude(
    trajectory: Dict[str, np.ndarray],
    fraction: float = 0.2,
) -> float:
    """
    Amplitude of oscillations in the tail of the trajectory.

    Computed as max(||z||) - min(||z||) in the last `fraction` of the
    time series.

    Parameters
    ----------
    trajectory : dict
        Must contain 'z'.
    fraction : float
        Fraction of the trajectory to use (from the end).

    Returns
    -------
    amplitude : float
    """
    z = trajectory['z']
    n = len(z)
    start = int((1.0 - fraction) * n)
    tail = z[start:]
    norms = np.linalg.norm(tail, axis=1)
    return float(norms.max() - norms.min())


def set_plot_style():
    """
    Configure matplotlib for paper-quality figures.

    Sets serif fonts, appropriate sizes for NeurIPS column width, and
    a clean style.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Use seaborn style if available; fall back gracefully
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass  # use default matplotlib style

    mpl.rcParams.update({
        # Fonts
        'font.family':      'serif',
        'font.serif':       ['Computer Modern Roman', 'Times New Roman',
                             'DejaVu Serif'],
        'font.size':        12,
        'axes.labelsize':   13,
        'axes.titlesize':   14,
        'xtick.labelsize':  11,
        'ytick.labelsize':  11,
        'legend.fontsize':  11,

        # Figure size (NeurIPS single column ~ 5.5 in)
        'figure.figsize':   (5.5, 4.0),
        'figure.dpi':       150,
        'savefig.dpi':      300,
        'savefig.bbox':     'tight',

        # Lines
        'lines.linewidth':  1.5,
        'lines.markersize': 5,

        # Axes
        'axes.linewidth':   0.8,
        'axes.grid':        True,
        'grid.alpha':       0.3,

        # LaTeX (use mathtext to avoid requiring a full LaTeX install)
        'text.usetex':      False,
        'mathtext.fontset':  'cm',
    })
