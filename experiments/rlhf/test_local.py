"""
Quick local test of the RLHF pipeline logic (CPU, tiny config).
Verifies that all components work before submitting to PACE.

Usage: pip install trl datasets accelerate && python test_local.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig
from run_ppo import run_experiment


def main():
    cfg = ExperimentConfig()
    # Override for fast local test
    cfg.total_steps = 5
    cfg.batch_size = 4
    cfg.mini_batch_size = 2
    cfg.eval_freq = 2
    cfg.eval_samples = 4
    cfg.rm_initial_dataset_size = 32
    cfg.rm_initial_epochs = 1
    cfg.rm_update_freq = 3
    cfg.rm_update_samples = 16
    cfg.rm_update_steps = 2
    cfg.max_new_tokens = 16
    cfg.output_dir = "/tmp/rlhf_test"

    print("=" * 60)
    print("TEST: Frozen RM")
    print("=" * 60)
    log_frozen = run_experiment(beta=0.1, online_rm=False, seed=42, cfg=cfg)
    print(f"  Steps logged: {len(log_frozen['step'])}")
    print(f"  Gold rewards: {log_frozen['gold_reward_mean']}")

    print("\n" + "=" * 60)
    print("TEST: Online RM")
    print("=" * 60)
    log_online = run_experiment(beta=0.1, online_rm=True, seed=42, cfg=cfg)
    print(f"  Steps logged: {len(log_online['step'])}")
    print(f"  Gold rewards: {log_online['gold_reward_mean']}")

    print("\n" + "=" * 60)
    print("All tests passed! Pipeline is functional.")
    print("=" * 60)


if __name__ == "__main__":
    main()
