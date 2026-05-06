"""
Analyze RLHF experiment results: compute oscillation metrics and generate figures.

Usage:
    python analyze_results.py --results_dir results/rlhf
"""

import argparse
import json
import os
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def load_all_logs(results_dir):
    """Load all experiment logs from the results directory."""
    logs = []
    for log_path in sorted(glob(os.path.join(results_dir, "*/log.json"))):
        with open(log_path) as f:
            log = json.load(f)
        logs.append(log)
    return logs


def compute_oscillation_metrics(gold_rewards, tail_fraction=0.4):
    """
    Compute oscillation metrics on the tail of the gold reward time series.

    Returns:
        amplitude: (max - min) / 2 in the tail
        n_peaks: number of local maxima in the tail
        autocorr_peak: lag of the first autocorrelation peak (periodicity estimate)
    """
    n = len(gold_rewards)
    tail_start = int((1 - tail_fraction) * n)
    tail = np.array(gold_rewards[tail_start:])

    if len(tail) < 5:
        return {"amplitude": 0.0, "n_peaks": 0, "autocorr_peak_lag": None}

    amplitude = (tail.max() - tail.min()) / 2.0

    # Count peaks
    detrended = tail - np.mean(tail)
    peaks, _ = find_peaks(detrended, distance=2)
    n_peaks = len(peaks)

    # Autocorrelation to detect periodicity
    autocorr_peak_lag = None
    if len(detrended) > 10:
        autocorr = np.correlate(detrended, detrended, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]  # take positive lags
        autocorr = autocorr / (autocorr[0] + 1e-12)  # normalize
        # Find first peak after lag 0
        ac_peaks, _ = find_peaks(autocorr, distance=2)
        if len(ac_peaks) > 0:
            autocorr_peak_lag = ac_peaks[0]

    return {
        "amplitude": amplitude,
        "n_peaks": n_peaks,
        "autocorr_peak_lag": autocorr_peak_lag,
    }


def aggregate_by_condition(logs):
    """Group logs by (beta, condition) and compute statistics."""
    groups = {}
    for log in logs:
        cfg = log["config"]
        beta = cfg["beta"]
        condition = "online" if cfg["online_rm"] else "frozen"
        key = (beta, condition)
        if key not in groups:
            groups[key] = []
        groups[key].append(log)

    results = {}
    for (beta, condition), group_logs in sorted(groups.items()):
        metrics_list = []
        for log in group_logs:
            m = compute_oscillation_metrics(log["gold_reward_mean"])
            m["peak_gold"] = max(log["gold_reward_mean"])
            m["final_gold"] = log["gold_reward_mean"][-1]
            metrics_list.append(m)

        results[(beta, condition)] = {
            "n_seeds": len(group_logs),
            "amplitude_mean": np.mean([m["amplitude"] for m in metrics_list]),
            "amplitude_std": np.std([m["amplitude"] for m in metrics_list]),
            "n_peaks_mean": np.mean([m["n_peaks"] for m in metrics_list]),
            "peak_gold_mean": np.mean([m["peak_gold"] for m in metrics_list]),
            "final_gold_mean": np.mean([m["final_gold"] for m in metrics_list]),
            "logs": group_logs,
        }

    return results


# ---------------------------------------------------------------------------
# Figure 1: Frozen vs. Online RM time series (2×3 grid)
# ---------------------------------------------------------------------------
def fig_time_series(results, output_dir, betas_to_show=None):
    """2-row figure: top = frozen, bottom = online, columns = different beta."""
    if betas_to_show is None:
        all_betas = sorted(set(b for b, _ in results.keys()))
        # Pick 3 representative betas: low, medium, high
        if len(all_betas) >= 3:
            betas_to_show = [all_betas[0], all_betas[len(all_betas) // 2], all_betas[-1]]
        else:
            betas_to_show = all_betas

    n_cols = len(betas_to_show)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 6), sharey=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, beta in enumerate(betas_to_show):
        for row, condition in enumerate(["frozen", "online"]):
            ax = axes[row, col]
            key = (beta, condition)
            if key not in results:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            for log in results[key]["logs"]:
                steps = log["step"]
                gold = log["gold_reward_mean"]
                ax.plot(steps, gold, alpha=0.5, linewidth=1.0)

                # EMA overlay
                if len(gold) > 5:
                    ema = _ema(gold, alpha=0.15)
                    ax.plot(steps, ema, linewidth=2.0, color="black", alpha=0.7)

            if row == 0:
                ax.set_title(rf"$\beta = {beta}$", fontsize=12)
            if col == 0:
                label = "Frozen RM" if condition == "frozen" else "Online RM"
                ax.set_ylabel(f"{label}\nGold reward", fontsize=10)
            if row == 1:
                ax.set_xlabel("PPO step")

    fig.suptitle("Gold Reward: Frozen RM (top) vs. Online RM (bottom)", fontsize=13)
    fig.tight_layout()

    path = os.path.join(output_dir, "frozen_vs_online_timeseries.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def _ema(data, alpha=0.1):
    """Exponential moving average."""
    ema = [data[0]]
    for x in data[1:]:
        ema.append(alpha * x + (1 - alpha) * ema[-1])
    return ema


# ---------------------------------------------------------------------------
# Figure 2: Bifurcation diagram (amplitude vs beta)
# ---------------------------------------------------------------------------
def fig_bifurcation(results, output_dir):
    """Oscillation amplitude vs. beta for both conditions."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for condition, marker, color in [("frozen", "s", "C0"), ("online", "o", "C3")]:
        betas = []
        amps = []
        errs = []
        for (beta, cond), stats in sorted(results.items()):
            if cond != condition:
                continue
            betas.append(beta)
            amps.append(stats["amplitude_mean"])
            errs.append(stats["amplitude_std"])

        ax.errorbar(
            betas, amps, yerr=errs, fmt=f"{marker}-", color=color,
            label=f"{condition.capitalize()} RM", capsize=3, markersize=6,
        )

    ax.set_xlabel(r"KL penalty $\beta$", fontsize=11)
    ax.set_ylabel("Oscillation amplitude (gold reward)", fontsize=11)
    ax.set_title("Bifurcation Diagram: Frozen vs. Online RM")
    ax.legend(fontsize=10)
    ax.set_xscale("log")
    ax.set_ylim(bottom=-0.02)

    fig.tight_layout()
    path = os.path.join(output_dir, "bifurcation_frozen_vs_online.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Table: summary statistics
# ---------------------------------------------------------------------------
def print_table(results):
    """Print a summary table."""
    print(f"\n{'Condition':<12} {'Beta':>6} {'Amp':>8} {'Peaks':>6} "
          f"{'Peak Gold':>10} {'Final Gold':>11}")
    print("-" * 60)
    for (beta, condition), stats in sorted(results.items()):
        print(f"{condition:<12} {beta:>6.3f} {stats['amplitude_mean']:>8.4f} "
              f"{stats['n_peaks_mean']:>6.1f} {stats['peak_gold_mean']:>10.4f} "
              f"{stats['final_gold_mean']:>11.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/rlhf")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save figures (default: results_dir)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    logs = load_all_logs(args.results_dir)
    if not logs:
        print(f"No logs found in {args.results_dir}")
        return

    print(f"Loaded {len(logs)} experiment logs")
    results = aggregate_by_condition(logs)

    print_table(results)
    fig_time_series(results, output_dir)
    fig_bifurcation(results, output_dir)


if __name__ == "__main__":
    main()
