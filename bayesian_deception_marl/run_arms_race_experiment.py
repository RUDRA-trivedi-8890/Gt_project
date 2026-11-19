"""Three-Seed Before/After Benchmark Experiment for the Adaptive Arms Race.

Compares:
1. Baseline: Standard Self-Play PPO (Arms race disabled: mu_e=0, no penalty archive, no reveal delay).
2. Adaptive Arms Race: Phase 1 -> 2 -> 3 active (caught-route archive penalty, terminal reveal bonus,
   shaping cap, league training pool).

Across 3 distinct random seeds (42, 101, 2024).
Generates:
- figures/arms_race_phases.png (Multi-panel timeline of phases, rates, diversity, and mu_e)
- models/caught_routes.npz (Normalized caught-route frequency maps and phase state for live_demo.py)
- logs/arms_race_benchmark.csv (Full epoch-by-epoch telemetry)
"""

import os
import sys
import time
from typing import Dict, List
import numpy as np
import torch
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import EnvConfig, PPOConfig, DeceptionConfig, ArmsRaceConfig
from algos.self_play_trainer import SelfPlayTrainer
from utils.visualization import plot_arms_race_phases


def run_experiment(
    seeds: List[int] = [42, 101, 2024],
    num_epochs: int = 15,
    episodes_per_epoch: int = 16,
) -> Dict[str, dict]:
    """Run full before/after benchmark across 3 seeds."""
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    dec_cfg = DeceptionConfig()

    results_baseline = []
    results_arms_race = []
    best_arms_history = None

    print("================================================================================")
    print("STARTING 3-SEED BEFORE / AFTER ADAPTIVE ARMS-RACE BENCHMARK")
    print(f"Seeds: {seeds} | Epochs per run: {num_epochs} | Episodes per epoch: {episodes_per_epoch}")
    print("================================================================================\n")

    for seed in seeds:
        print(f"\n--- SEED {seed}: RUNNING BASELINE (Arms Race Disabled) ---")
        np.random.seed(seed)
        torch.manual_seed(seed)
        arms_cfg_base = ArmsRaceConfig(enabled=False)
        trainer_base = SelfPlayTrainer(env_cfg, ppo_cfg, dec_cfg, arms_cfg=arms_cfg_base, device="cpu", action_dim=9)

        base_history = []
        t0 = time.time()
        for ep in range(1, num_epochs + 1):
            s = trainer_base.train_epoch(num_episodes=episodes_per_epoch, epoch=ep)
            base_history.append(s)
            if ep % 5 == 0 or ep == num_epochs:
                print(f"  [Base Seed {seed}] Ep {ep:02d} | Intercept: {s['interception_rate']:.2f} | Goal: {s['goal_rate']:.2f} | RouteEnt: {s['route_diversity_entropy']:.2f} | t_rev: {s['mean_t_reveal']:.1f}")
        t_base = time.time() - t0
        print(f"  Baseline completed in {t_base:.1f}s.")

        final_base = base_history[-1]
        results_baseline.append({
            "seed": seed,
            "final_interception_rate": final_base["interception_rate"],
            "final_goal_rate": final_base["goal_rate"],
            "final_route_entropy": final_base["route_diversity_entropy"],
            "final_pairwise_dist": final_base["mean_pairwise_distance"],
            "final_t_reveal": final_base["mean_t_reveal"],
            "phase": final_base["phase"],
            "mu_e": final_base["mu_e"],
        })

        print(f"\n--- SEED {seed}: RUNNING ADAPTIVE ARMS RACE (Phases 1 -> 2 -> 3) ---")
        np.random.seed(seed)
        torch.manual_seed(seed)
        arms_cfg_active = ArmsRaceConfig(
            enabled=True,
            window_epochs=5,
            mu_max=1.0,
            kappa=5.0,
            shaping_cap_ratio=0.30,
            p2_interception_threshold=0.60,
            p2_consecutive_epochs=2,
            p3_ramp_epochs=5,
            midfield_row=5,
            use_league=True,
            league_checkpoint_freq=3,
        )
        trainer_arms = SelfPlayTrainer(env_cfg, ppo_cfg, dec_cfg, arms_cfg=arms_cfg_active, device="cpu", action_dim=9)

        arms_history = []
        t0 = time.time()
        for ep in range(1, num_epochs + 1):
            s = trainer_arms.train_epoch(num_episodes=episodes_per_epoch, epoch=ep)
            arms_history.append(s)
            if ep % 5 == 0 or ep == num_epochs:
                print(f"  [Arms Seed {seed}] Ep {ep:02d} | Phase: {int(s['phase'])} | mu_e: {s['mu_e']:.2f} | Intercept: {s['interception_rate']:.2f} | Goal: {s['goal_rate']:.2f} | RouteEnt: {s['route_diversity_entropy']:.2f} | t_rev: {s['mean_t_reveal']:.1f}")
        t_arms = time.time() - t0
        print(f"  Adaptive arms race completed in {t_arms:.1f}s.")

        final_arms = arms_history[-1]
        results_arms_race.append({
            "seed": seed,
            "final_interception_rate": final_arms["interception_rate"],
            "final_goal_rate": final_arms["goal_rate"],
            "final_route_entropy": final_arms["route_diversity_entropy"],
            "final_pairwise_dist": final_arms["mean_pairwise_distance"],
            "final_t_reveal": final_arms["mean_t_reveal"],
            "phase": final_arms["phase"],
            "mu_e": final_arms["mu_e"],
        })

        if best_arms_history is None or final_arms["route_diversity_entropy"] > best_arms_history[-1]["route_diversity_entropy"]:
            best_arms_history = arms_history
            # Emit telemetry for live_demo
            models_dir = os.path.join(CURRENT_DIR, "models")
            os.makedirs(models_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(models_dir, "caught_routes.npz"),
                C_theta=trainer_arms.caught_archive.C_theta,
                phase=trainer_arms.current_phase,
                mu_e=trainer_arms.current_mu_e,
            )

    # Save visualization plot
    fig_dir = os.path.join(CURRENT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_path = os.path.join(fig_dir, "arms_race_phases.png")
    plot_arms_race_phases(best_arms_history, plot_path)
    print(f"\n[Artifact] Phase timeline figure saved to: {plot_path}")

    # Build comparison dataframes
    df_base = pd.DataFrame(results_baseline)
    df_arms = pd.DataFrame(results_arms_race)

    logs_dir = os.path.join(CURRENT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    df_base.to_csv(os.path.join(logs_dir, "benchmark_baseline_3seeds.csv"), index=False)
    df_arms.to_csv(os.path.join(logs_dir, "benchmark_armsrace_3seeds.csv"), index=False)

    print("\n================================================================================")
    print("3-SEED SUMMARY COMPARISON TABLE")
    print("================================================================================")
    print("BASELINE (ARMS RACE DISABLED):")
    print(df_base.to_string(index=False))
    print("\nADAPTIVE ARMS RACE (PHASES 1 -> 2 -> 3 ENABLED):")
    print(df_arms.to_string(index=False))

    mean_base_ent = df_base["final_route_entropy"].mean()
    mean_arms_ent = df_arms["final_route_entropy"].mean()
    mean_base_dist = df_base["final_pairwise_dist"].mean()
    mean_arms_dist = df_arms["final_pairwise_dist"].mean()
    mean_base_rev = df_base["final_t_reveal"].mean()
    mean_arms_rev = df_arms["final_t_reveal"].mean()
    mean_base_goal = df_base["final_goal_rate"].mean()
    mean_arms_goal = df_arms["final_goal_rate"].mean()

    print("\nAGGREGATE DELTAS (ARMS RACE vs. BASELINE):")
    print(f"  Route Diversity Entropy: {mean_base_ent:.2f} -> {mean_arms_ent:.2f} (Delta: {mean_arms_ent - mean_base_ent:+.2f} bits)")
    print(f"  Pairwise Trajectory Dist: {mean_base_dist:.2f} -> {mean_arms_dist:.2f} (Delta: {mean_arms_dist - mean_base_dist:+.2f} cells)")
    print(f"  Mean Reveal Step t_reveal: {mean_base_rev:.2f} -> {mean_arms_rev:.2f} (Delta: {mean_arms_rev - mean_base_rev:+.2f} steps)")
    print(f"  Goal Success Rate:        {mean_base_goal:.2f} -> {mean_arms_goal:.2f} (Delta: {mean_arms_goal - mean_base_goal:+.2f})")
    print("================================================================================\n")

    return {"baseline": results_baseline, "arms_race": results_arms_race}


if __name__ == "__main__":
    run_experiment()
