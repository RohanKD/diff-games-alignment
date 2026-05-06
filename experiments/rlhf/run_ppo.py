"""
GPT-2 RLHF experiment: Frozen vs. Online Reward Model.

Tests the paper's prediction: oscillatory reward hacking (Hopf bifurcation)
emerges when both policy and reward model update (two-player game), but not
when the reward model is frozen (one-player optimization).

Usage:
    python run_ppo.py --beta 0.05 --online_rm --seed 42
    python run_ppo.py --beta 0.05 --seed 42  # frozen RM baseline
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline,
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from config import ExperimentConfig


# ---------------------------------------------------------------------------
# Proxy Reward Model: lightweight MLP on top of frozen GPT-2 embeddings
# ---------------------------------------------------------------------------
class ProxyRewardModel(nn.Module):
    """
    Small MLP reward model that operates on GPT-2's last hidden state.
    Much cheaper to retrain online than a full transformer RM.
    """

    def __init__(self, embed_dim=768, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states):
        """hidden_states: (batch, embed_dim) or (batch, seq_len, embed_dim) → scores: (batch,)"""
        if hidden_states.dim() == 3:
            pooled = hidden_states[:, -1, :]
        else:
            pooled = hidden_states
        return self.net(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Gold reward: use a pretrained sentiment classifier
# ---------------------------------------------------------------------------
def make_gold_scorer(model_name="lvwerra/distilbert-imdb", device="cuda"):
    """Returns a function that scores a list of strings → list of floats."""
    sent_pipe = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
        truncation=True,
        max_length=512,
        batch_size=32,
    )

    def score(texts):
        results = sent_pipe(texts)
        # POSITIVE → +1, NEGATIVE → -1, scaled by confidence
        scores = []
        for r in results:
            s = r["score"] if r["label"] == "POSITIVE" else -r["score"]
            scores.append(s)
        return scores

    return score


# ---------------------------------------------------------------------------
# Dataset: IMDB prompts
# ---------------------------------------------------------------------------
def load_prompts(tokenizer, max_prompt_length=64, n=5000):
    """Load IMDB dataset and extract prompt prefixes."""
    dataset = load_dataset("imdb", split="train")
    prompts = []
    for text in dataset["text"][:n]:
        tokens = tokenizer.encode(text, truncation=True, max_length=max_prompt_length)
        # Use first ~half as prompt
        prompt_len = max(8, len(tokens) // 2)
        prompt_tokens = tokens[:prompt_len]
        prompts.append(tokenizer.decode(prompt_tokens))
    return prompts


# ---------------------------------------------------------------------------
# Proxy RM: initial training from gold RM preferences
# ---------------------------------------------------------------------------
def train_proxy_rm_initial(
    proxy_rm, encoder_model, tokenizer, gold_scorer, prompts, policy_model, cfg, device
):
    """
    Train the proxy RM on preference pairs labeled by the gold RM.
    Generates responses from the current policy, scores with gold RM,
    creates preference pairs, trains the proxy MLP.
    """
    print("  Training initial proxy RM...")
    proxy_rm.train()
    optimizer = Adam(proxy_rm.parameters(), lr=cfg.rm_update_lr)

    # Generate responses and score them
    texts, gold_scores = _generate_and_score(
        policy_model, tokenizer, gold_scorer, prompts,
        n_samples=cfg.rm_initial_dataset_size, max_new_tokens=cfg.max_new_tokens,
        device=device,
    )

    # Get embeddings from the encoder
    embeddings = _get_embeddings(encoder_model, tokenizer, texts, device)

    # Train on pairwise ranking loss
    n = len(texts)
    for epoch in range(cfg.rm_initial_epochs):
        perm = np.random.permutation(n)
        total_loss = 0.0
        n_pairs = 0
        for i in range(0, n - 1, 2):
            idx_a, idx_b = perm[i], perm[i + 1]
            score_a, score_b = gold_scores[idx_a], gold_scores[idx_b]
            emb_a = embeddings[idx_a].unsqueeze(0)
            emb_b = embeddings[idx_b].unsqueeze(0)

            pred_a = proxy_rm(emb_a)
            pred_b = proxy_rm(emb_b)

            # Bradley-Terry loss: -log sigma(r_chosen - r_rejected)
            if score_a > score_b:
                loss = -torch.log(torch.sigmoid(pred_a - pred_b) + 1e-8)
            else:
                loss = -torch.log(torch.sigmoid(pred_b - pred_a) + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_pairs += 1

        print(f"    Epoch {epoch + 1}/{cfg.rm_initial_epochs}: "
              f"loss = {total_loss / max(n_pairs, 1):.4f}")

    proxy_rm.eval()
    return proxy_rm


def update_proxy_rm_online(
    proxy_rm, encoder_model, tokenizer, gold_scorer, prompts, policy_model, cfg, device
):
    """
    Online RM update: generate new responses from current policy,
    score with gold RM, fine-tune proxy RM on fresh preference pairs.
    """
    proxy_rm.train()
    optimizer = Adam(proxy_rm.parameters(), lr=cfg.rm_update_lr)

    texts, gold_scores = _generate_and_score(
        policy_model, tokenizer, gold_scorer, prompts,
        n_samples=cfg.rm_update_samples, max_new_tokens=cfg.max_new_tokens,
        device=device,
    )
    embeddings = _get_embeddings(encoder_model, tokenizer, texts, device)

    n = len(texts)
    for step in range(cfg.rm_update_steps):
        perm = np.random.permutation(n)
        total_loss = 0.0
        n_pairs = 0
        for i in range(0, n - 1, 2):
            idx_a, idx_b = perm[i], perm[i + 1]
            score_a, score_b = gold_scores[idx_a], gold_scores[idx_b]
            emb_a = embeddings[idx_a].unsqueeze(0)
            emb_b = embeddings[idx_b].unsqueeze(0)

            pred_a = proxy_rm(emb_a)
            pred_b = proxy_rm(emb_b)

            if score_a > score_b:
                loss = -torch.log(torch.sigmoid(pred_a - pred_b) + 1e-8)
            else:
                loss = -torch.log(torch.sigmoid(pred_b - pred_a) + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_pairs += 1

    proxy_rm.eval()


# ---------------------------------------------------------------------------
# Helper: generate responses and score with gold RM
# ---------------------------------------------------------------------------
@torch.no_grad()
def _generate_and_score(policy_model, tokenizer, gold_scorer, prompts,
                        n_samples=256, max_new_tokens=48, device="cuda"):
    """Generate responses from policy, score with gold RM."""
    policy_model.eval()
    selected_prompts = [prompts[i % len(prompts)] for i in range(n_samples)]

    texts = []
    batch_size = 32
    for start in range(0, n_samples, batch_size):
        batch_prompts = selected_prompts[start:start + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=64,
        ).to(device)
        outputs = policy_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
        for seq in outputs:
            texts.append(tokenizer.decode(seq, skip_special_tokens=True))

    gold_scores = gold_scorer(texts)
    policy_model.train()
    return texts, gold_scores


@torch.no_grad()
def _get_embeddings(encoder_model, tokenizer, texts, device, batch_size=32):
    """Get last hidden state embeddings from the encoder model."""
    encoder_model.eval()
    all_embs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(device)
        outputs = encoder_model(**inputs, output_hidden_states=True)
        # Last layer, last token
        hidden = outputs.hidden_states[-1]  # (batch, seq, dim)
        # Gather last non-padding positions
        lengths = inputs["attention_mask"].sum(dim=1) - 1
        emb = hidden[torch.arange(len(batch)), lengths]
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0).to(device)


# ---------------------------------------------------------------------------
# Proxy RM scoring function for PPOTrainer
# ---------------------------------------------------------------------------
def make_proxy_scorer(proxy_rm, encoder_model, tokenizer, device):
    """
    Returns a function compatible with PPOTrainer's reward computation.
    Scores a batch of response strings → tensor of scalar rewards.
    """
    @torch.no_grad()
    def score(texts):
        embeddings = _get_embeddings(encoder_model, tokenizer, texts, device)
        scores = proxy_rm(embeddings)
        return scores.cpu().tolist()

    return score


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_experiment(beta, online_rm, seed, cfg):
    """Run a single PPO experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    condition = "online" if online_rm else "frozen"
    run_name = f"beta{beta}_rm{condition}_seed{seed}"
    output_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"  beta={beta}, online_rm={online_rm}, seed={seed}, device={device}")
    print(f"{'='*60}")

    # --- Load models ---
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.policy_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.policy_model)
    model.to(device)

    # Frozen encoder for proxy RM embeddings (shared with policy base)
    encoder_model = AutoModelForCausalLM.from_pretrained(cfg.policy_model)
    encoder_model.to(device)
    encoder_model.eval()
    for p in encoder_model.parameters():
        p.requires_grad = False

    # Gold RM
    gold_scorer = make_gold_scorer(cfg.gold_rm_model, device=device)

    # Proxy RM (small MLP)
    embed_dim = encoder_model.config.n_embd
    proxy_rm = ProxyRewardModel(embed_dim=embed_dim, hidden_dim=cfg.rm_hidden_dim)
    proxy_rm.to(device)

    # Load prompts
    print("Loading prompts...")
    prompts = load_prompts(tokenizer)

    # Train initial proxy RM
    train_proxy_rm_initial(
        proxy_rm, encoder_model, tokenizer, gold_scorer, prompts,
        encoder_model, cfg, device,  # use encoder for initial generation
    )

    # --- PPO Config ---
    ppo_config = PPOConfig(
        model_name=cfg.policy_model,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        ppo_epochs=cfg.ppo_epochs,
        init_kl_coef=beta,
        adap_kl_ctrl=False,  # fixed KL coefficient, no adaptive
        seed=seed,
        log_with="wandb" if cfg.use_wandb else None,
    )

    # --- Dataset for PPO ---
    # Create a simple dataset of prompt tensors
    from datasets import Dataset as HFDataset

    prompt_texts = prompts[:cfg.batch_size * (cfg.total_steps + 10)]
    # Tokenize prompts
    tokenized = tokenizer(
        prompt_texts, padding=True, truncation=True, max_length=64,
        return_tensors="pt",
    )
    ppo_dataset = HFDataset.from_dict({
        "input_ids": tokenized["input_ids"].tolist(),
        "attention_mask": tokenized["attention_mask"].tolist(),
    })

    # --- Training loop ---
    print("Starting PPO training...")
    log = {
        "step": [],
        "proxy_reward_mean": [],
        "gold_reward_mean": [],
        "kl_mean": [],
        "config": {
            "beta": beta,
            "online_rm": online_rm,
            "seed": seed,
        },
    }

    # Manual PPO loop (since TRL's new API doesn't expose step-by-step control easily)
    from trl import PPOTrainer as LegacyPPOTrainer

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.policy_model)
    ref_model.to(device)

    ppo_trainer = LegacyPPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    prompt_idx = 0
    for step in range(cfg.total_steps):
        # Get batch of prompts
        batch_prompts = []
        query_tensors = []
        for _ in range(cfg.batch_size):
            p = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1
            enc = tokenizer.encode(p, truncation=True, max_length=64)
            query_tensors.append(torch.tensor(enc).to(device))
            batch_prompts.append(p)

        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode responses
        response_texts = [tokenizer.decode(r, skip_special_tokens=True)
                          for r in response_tensors]

        # Compute proxy rewards
        with torch.no_grad():
            embeddings = _get_embeddings(
                encoder_model, tokenizer, response_texts, device
            )
            proxy_scores = proxy_rm(embeddings)
            rewards = [torch.tensor(s.item()) for s in proxy_scores]

        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # Log proxy reward
        proxy_mean = np.mean([r.item() for r in rewards])

        # Periodically compute gold reward
        if step % cfg.eval_freq == 0:
            gold_scores = gold_scorer(response_texts[:cfg.eval_samples])
            gold_mean = np.mean(gold_scores)
            kl_mean = stats.get("objective/kl", 0.0)
            if isinstance(kl_mean, torch.Tensor):
                kl_mean = kl_mean.item()

            log["step"].append(step)
            log["proxy_reward_mean"].append(proxy_mean)
            log["gold_reward_mean"].append(gold_mean)
            log["kl_mean"].append(kl_mean)

            print(f"  Step {step:4d}: proxy={proxy_mean:.4f}, "
                  f"gold={gold_mean:.4f}, kl={kl_mean:.4f}")

        # Online RM update
        if online_rm and step > 0 and step % cfg.rm_update_freq == 0:
            print(f"  [RM UPDATE at step {step}]")
            update_proxy_rm_online(
                proxy_rm, encoder_model, tokenizer, gold_scorer, prompts,
                encoder_model, cfg, device,
            )

    # Save logs
    log_path = os.path.join(output_dir, "log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved logs to {log_path}")

    return log


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 RLHF: Frozen vs Online RM"
    )
    parser.add_argument("--beta", type=float, required=True,
                        help="KL penalty coefficient")
    parser.add_argument("--online_rm", action="store_true",
                        help="Enable online reward model updates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="results/rlhf")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    cfg.total_steps = args.total_steps
    cfg.output_dir = args.output_dir
    cfg.use_wandb = args.use_wandb

    run_experiment(args.beta, args.online_rm, args.seed, cfg)


if __name__ == "__main__":
    main()
