"""
Experiment configuration for GPT-2 RLHF: Frozen vs. Online Reward Model.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    # Models
    policy_model: str = "gpt2"
    gold_rm_model: str = "lvwerra/distilbert-imdb"

    # PPO hyperparameters
    ppo_epochs: int = 4
    batch_size: int = 64
    mini_batch_size: int = 16
    learning_rate: float = 1.41e-5
    max_new_tokens: int = 48
    total_steps: int = 500  # total PPO steps

    # KL penalty sweep
    betas: List[float] = field(
        default_factory=lambda: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    )

    # Online RM settings
    rm_update_freq: int = 50        # update RM every N PPO steps
    rm_update_samples: int = 256    # preference pairs per RM update
    rm_update_lr: float = 1e-4      # RM fine-tuning learning rate
    rm_update_steps: int = 10       # gradient steps per RM update
    rm_hidden_dim: int = 128        # proxy RM MLP hidden dimension

    # Proxy RM initial training
    rm_initial_dataset_size: int = 2000  # preference pairs for initial RM training
    rm_initial_epochs: int = 3

    # Evaluation
    eval_freq: int = 10             # compute gold reward every N steps
    eval_samples: int = 64          # samples for gold reward evaluation

    # Experiment
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    output_dir: str = "results/rlhf"
    use_wandb: bool = False
