"""Benchmark evaluation: Naive PPO (lambda_0 = 0.0) vs Belief-Regularized PPO
(lambda_0 = 2.5).

Trains both configurations and reports empirical metrics:
  * Goal Success Rate
  * Interception Rate
  * Mean Entropy at Midfield
  * Steps to Goal
"""

import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EnvConfig, PPOConfig, DeceptionConfig
from algos.self_play_trainer import SelfPlayTrainer
from utils.visualization import (
    plot_trajectory_heatmap,
    plot_entropy_curve,
)
from utils.metrics import information_leakage_curve, entropy_at_midfield


def train_and_collect(
    dec_cfg: DeceptionConfig,
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    device: str,
    num_episodes: int = 800,
) -> Tuple[SelfPlayTrainer, List[dict]]:
    """Train a SelfPlayTrainer for num_episodes and return it with logs."""
    trainer = SelfPlayTrainer(env_cfg, ppo_cfg, dec_cfg, device=device)
    trajectories: List[dict] = []
    for episode in range(num_episodes):
        info = trainer.train_episode()
        trajectories.append(info)
        if episode % 200 == 0:
            print(f"    [lambda_0={dec_cfg.lambda_0}] episode {episode}/{num_episodes}")
    return trainer, trajectories


def evaluate_agent(trainer: SelfPlayTrainer, num_episodes: int = 64) -> Dict[str, float]:
    """Run num_episodes evaluation episodes and aggregate metrics."""
    return trainer.evaluate(num_episodes=num_episodes)


def main() -> None:
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[evaluate] Device: {device}")

    os.makedirs("figures", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    configs = {
        "naive_baseline (lambda_0=0.0)": DeceptionConfig(lambda_0=0.0),
        "regularized (lambda_0=2.5)": DeceptionConfig(lambda_0=2.5),
    }

    results: Dict[str, Dict[str, float]] = {}

    for name, dec_cfg in configs.items():
        print(f"\n[evaluate] Training {name} ...")
        trainer, _ = train_and_collect(
            dec_cfg, env_cfg, ppo_cfg, device, num_episodes=800
        )

        print(f"[evaluate] Evaluating {name} ...")
        metrics = evaluate_agent(trainer, num_episodes=64)
        results[name] = metrics

        # Collect trajectory data from a few episodes for visualization
        att_trajectories: List[np.ndarray] = []
        for _ in range(8):
            obs_dict, info = trainer.env.reset()
            att_obs = obs_dict["attacker"]
            def_obs = obs_dict["defender"]
            traj = [att_obs[:2].copy()]
            terminated = False
            truncated = False
            beliefs: List[np.ndarray] = []
            dists: List[float] = []
            for t in range(env_cfg.horizon):
                att_action, _, _, _ = SelfPlayTrainer._greedy_action(
                    trainer.attacker_agent, att_obs
                )
                def_action, _, _, _ = SelfPlayTrainer._greedy_action(
                    trainer.defender_agent, def_obs
                )
                # ToM belief
                att_pos_t = torch.as_tensor(
                    att_obs[:2], dtype=torch.float32, device=device
                ).unsqueeze(0)
                tom_logits, _ = trainer.attacker_tom.step(att_pos_t, None)
                tom_probs = torch.softmax(tom_logits, dim=-1).squeeze(0)
                beliefs.append(tom_probs.detach().cpu().numpy())
                dists.append(float(info.get("dist_to_goal", 0.0)))

                next_obs_dict, rewards, terminated, truncated, info = trainer.env.step(
                    att_action, def_action
                )
                att_obs = next_obs_dict["attacker"]
                def_obs = next_obs_dict["defender"]
                traj.append(att_obs[:2].copy())
                if terminated or truncated:
                    break
            att_trajectories.append(np.asarray(traj))

        # Visualization
        tag = name.split("(")[0].strip().replace(" ", "_").lower()
        plot_trajectory_heatmap(
            att_trajectories,
            os.path.join("figures", f"heatmap_{tag}.png"),
            grid_size=env_cfg.grid_size,
            title=f"Attacker Trajectory Heatmap — {name}",
        )
        # Entropy curve from the last collected episode beliefs/dists
        curve = information_leakage_curve(beliefs, dists)
        if curve:
            plot_entropy_curve(
                [c[1] for c in curve],
                [c[0] for c in curve],
                os.path.join("figures", f"leakage_curve_{tag}.png"),
                title=f"Information Leakage — {name}",
            )

    # ---- Comparison table ----
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS (64 evaluation episodes each)")
    print("=" * 70)
    header = f"{'Metric':<32} | {'Naive (l0=0.0)':>18} | {'Regularized (l0=2.5)':>20}"
    print(header)
    print("-" * len(header))
    metric_keys = [
        ("goal_success_rate", "Goal Success Rate"),
        ("interception_rate", "Interception Rate"),
        ("entropy_at_midfield", "Mean Entropy at Midfield"),
        ("steps", "Mean Steps to Goal"),
    ]
    for key, label in metric_keys:
        b = results["naive_baseline (lambda_0=0.0)"][key]
        r = results["regularized (lambda_0=2.5)"][key]
        print(f"{label:<32} | {b:>18.4f} | {r:>20.4f}")
    print("=" * 70)

    # Save results CSV
    csv_path = os.path.join("logs", "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Naive (lambda_0=0.0)", "Regularized (lambda_0=2.5)"])
        for key, label in metric_keys:
            b = results["naive_baseline (lambda_0=0.0)"][key]
            r = results["regularized (lambda_0=2.5)"][key]
            writer.writerow([label, f"{b:.4f}", f"{r:.4f}"])

    print(f"\n[evaluate] Results saved to {csv_path}")
    print("[evaluate] Figures saved to ./figures/")


if __name__ == "__main__":
    main()
