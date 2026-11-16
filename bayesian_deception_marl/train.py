"""Main training executable: co-evolutionary self-play with logging and checkpoints.

Runs the SelfPlayTrainer for a number of episodes, periodically logging
metrics to console and CSV, and saving model checkpoints.
"""

import csv
import os
import sys
from typing import List

import torch

# Allow running both as a module (python -m bayesian_deception_marl.train)
# and as a standalone script (python train.py) from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EnvConfig, PPOConfig, DeceptionConfig
from algos.self_play_trainer import SelfPlayTrainer
from utils.visualization import plot_training_curves, plot_trajectory_heatmap


def main() -> None:
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    dec_cfg = DeceptionConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Using device: {device}")
    print(f"[train] Env:       grid={env_cfg.grid_size}, horizon={env_cfg.horizon}")
    print(f"[train] PPO:       lr={ppo_cfg.lr}, gamma={ppo_cfg.gamma}, clip={ppo_cfg.clip_coef}")
    print(f"[train] Deception: lambda_0={dec_cfg.lambda_0}, t_max={dec_cfg.t_max}, c_step={env_cfg.c_step}")

    trainer = SelfPlayTrainer(env_cfg, ppo_cfg, dec_cfg, device=device)

    num_episodes = 2000
    log_interval = 50
    save_interval = 200
    eval_episodes = 32

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    csv_path = os.path.join("logs", "train_metrics.csv")
    fieldnames = [
        "episode", "theta", "steps", "goal_reached", "intercepted",
        "total_attacker_reward", "entropy_at_midfield", "mean_entropy",
        "final_dist_to_goal", "att_policy_loss", "att_value_loss", "att_entropy",
        "def_policy_loss", "def_value_loss", "def_entropy",
        "belief_loss", "tom_loss",
    ]

    metrics_log: List[dict] = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(num_episodes):
            info = trainer.train_episode()
            metrics_log.append(info)

            row = {"episode": episode}
            row.update(info)
            writer.writerow(row)

            if episode % log_interval == 0 or episode == num_episodes - 1:
                msg = (
                    f"[train] Episode {episode:5d} | "
                    f"goal={info['goal_reached']:.0f} "
                    f"intercept={info['intercepted']:.0f} "
                    f"steps={info['steps']:.0f} "
                    f"ent_mid={info['entropy_at_midfield']:.3f} "
                    f"rew={info['total_attacker_reward']:.2f} "
                    f"belief_loss={info['belief_loss']:.4f} "
                    f"tom_loss={info['tom_loss']:.4f}"
                )
                print(msg)

            if episode % save_interval == 0 or episode == num_episodes - 1:
                ckpt_path = os.path.join("checkpoints", f"ckpt_{episode}.pt")
                trainer.save(ckpt_path)

    # Final checkpoint
    trainer.save(os.path.join("checkpoints", "final.pt"))

    # Save training curves
    plot_training_curves(metrics_log, os.path.join("figures", "training_curves.png"))

    # Final evaluation
    print("\n[train] Running final evaluation...")
    eval_metrics = trainer.evaluate(num_episodes=eval_episodes)
    print("[train] Evaluation results:")
    for k, v in eval_metrics.items():
        print(f"    {k}: {v:.4f}")

    # Save evaluation results
    eval_csv = os.path.join("logs", "eval_metrics.csv")
    with open(eval_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_metrics.keys()))
        writer.writeheader()
        writer.writerow(eval_metrics)

    print(f"\n[train] Done. Checkpoints in ./checkpoints, logs in ./logs, figures in ./figures")


if __name__ == "__main__":
    main()
