import os
import sys
import time
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

from config import EnvConfig, PPOConfig, DeceptionConfig, ArmsRaceConfig
from algos.self_play_trainer import SelfPlayTrainer
from utils.visualization import plot_arms_race_phases

def run_test():
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    dec_cfg = DeceptionConfig()
    arms_cfg = ArmsRaceConfig(
        enabled=True,
        window_epochs=5,
        mu_max=1.0,
        kappa=5.0,
        shaping_cap_ratio=0.30,
        p2_interception_threshold=0.60,
        p2_consecutive_epochs=2,
        p3_ramp_epochs=5,
        midfield_row=5,
    )

    trainer = SelfPlayTrainer(env_cfg, ppo_cfg, dec_cfg, arms_cfg=arms_cfg, device="cpu", action_dim=9)
    print("Testing 10 epochs of 16 episodes...")
    t0 = time.time()
    history = []
    for ep in range(1, 11):
        summary = trainer.train_epoch(num_episodes=16, epoch=ep)
        history.append(summary)
        print(f"Epoch {ep:02d} | Phase {int(summary['phase'])} | mu_e {summary['mu_e']:.2f} | Intercept {summary['interception_rate']:.2f} | Goal {summary['goal_rate']:.2f} | RouteEnt {summary['route_diversity_entropy']:.2f}")
    
    elapsed = time.time() - t0
    print(f"10 epochs completed in {elapsed:.2f}s ({elapsed/10:.2f}s/epoch)")

if __name__ == "__main__":
    run_test()
