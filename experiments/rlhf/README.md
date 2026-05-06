# RLHF Experiment: Frozen vs. Online Reward Model

This experiment tests the paper's core prediction: **oscillatory reward hacking (Hopf bifurcation) emerges when both policy and reward model update (two-player game), but not with a frozen reward model (one-player optimization).**

## Setup

### Requirements

- Python 3.10+
- 1-2 NVIDIA GPUs with >= 12GB VRAM (tested on Quadro P5000 16GB and A100)
- ~10GB disk for model caches

### Install

```bash
pip install -r requirements_rlhf.txt
```

**Note:** TRL must be version 0.11.4 (the PPOTrainer was removed in TRL 1.0+).

### Verify

```bash
python test_local.py
```

This runs a tiny 5-step test on CPU to confirm all components work (~2 min).

## Running the Experiment

### Local machine (1-2 GPUs)

```bash
# Full sweep: 36 runs (6 betas x 3 seeds x 2 conditions), 2 GPUs in parallel
bash run_sweep_local.sh

# Quick test: 100 steps per run instead of 500
bash run_sweep_local.sh --quick
```

The script auto-detects 2 GPUs via `CUDA_VISIBLE_DEVICES` and dispatches one job per GPU. Edit `run_sweep_local.sh` if you have more or fewer GPUs.

**Estimated time:**
- 500 steps/run on A100: ~20-30 min/run, ~9 hours total
- 500 steps/run on P5000/V100: ~45-60 min/run, ~18 hours total
- `--quick` mode (100 steps): ~1/5 of above

### SLURM cluster (e.g., PACE)

```bash
bash submit_sweep.sh
```

Submits 36 independent SLURM jobs (1 A100 each, 2hr walltime). Edit the script to change module loads or partition names for your cluster.

### Single run

```bash
python run_ppo.py --beta 0.1 --online_rm --seed 42 --total_steps 500 --output_dir results/rlhf
```

Key flags:
- `--beta`: KL penalty coefficient (try 0.01-0.5)
- `--online_rm`: enable online RM updates (two-player game); omit for frozen RM
- `--seed`: random seed
- `--total_steps`: number of PPO steps

## Analysis

After runs complete:

```bash
python analyze_results.py --results_dir results/rlhf
```

This generates:
- `frozen_vs_online_timeseries.pdf` — gold reward traces for both conditions
- `bifurcation_frozen_vs_online.pdf` — oscillation amplitude vs. beta
- Summary table printed to stdout

## Experiment Design

| Parameter | Value |
|-----------|-------|
| Policy model | GPT-2 124M |
| Gold RM | distilbert-base-uncased-finetuned-sst-2-english (frozen) |
| Proxy RM | 2-layer MLP on GPT-2 embeddings (768 -> 128 -> 1) |
| Task | IMDB sentiment generation |
| Beta sweep | 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 |
| Seeds | 42, 123, 456 |
| PPO steps | 500 |
| Online RM update freq | every 50 steps |

### Two conditions

1. **Frozen RM** (standard RLHF): proxy RM trained once, then fixed. Expected: monotone over-optimization (Gao et al. 2023 replication).

2. **Online RM** (two-player game): proxy RM retrained every 50 PPO steps on fresh preference pairs from the current policy (labeled by gold RM). Expected: oscillatory gold reward for low beta, convergence for high beta.

## Troubleshooting

- **`ImportError: cannot import name 'PPOTrainer'`**: You have TRL >= 1.0. Downgrade: `pip install trl==0.11.4`
- **NumPy/numexpr errors**: Run `pip install --upgrade numexpr bottleneck`
- **OOM on GPU**: Reduce `batch_size` in `config.py` (default 32). GPT-2 124M needs ~4-5GB VRAM total.
- **CUDA not found**: Ensure PyTorch is installed with CUDA support for your GPU architecture. P5000 (Pascal) needs CUDA 11.x or 12.x with `sm_61` support.
