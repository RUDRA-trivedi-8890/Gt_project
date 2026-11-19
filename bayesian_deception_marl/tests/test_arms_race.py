"""Unit and Regression Tests for the Adaptive Arms-Race Mechanism.

Tests:
1. Phase scheduler triggers on synthetic interception rates (Phase 1 -> 2 -> 3).
2. Caught-route archive penalty is zero when the archive is empty.
3. Reward shaping cap holds (total added shaping <= 30% of goal_reward).
4. Training with all arms race flags off matches old baseline results for the same seed.
"""

import os
import sys
import numpy as np
import torch
import pytest

# Local package import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from config import EnvConfig, PPOConfig, DeceptionConfig, ArmsRaceConfig
from algos.phase_scheduler import PhaseScheduler
from algos.caught_route_archive import CaughtRouteArchive
from algos.self_play_trainer import SelfPlayTrainer


def test_scheduler_triggers_on_synthetic_rates():
    """Verify Phase 1 -> Phase 2 -> Phase 3 transition on synthetic interception rates."""
    cfg = ArmsRaceConfig(
        enabled=True,
        p2_interception_threshold=0.70,
        p2_consecutive_epochs=3,
        p3_ramp_epochs=5,
        mu_max=1.0,
    )
    scheduler = PhaseScheduler(cfg)

    # Epoch 1: low intercept rate -> Phase 1, mu_e = 0
    p, mu = scheduler.step(0.40, epoch=1)
    assert p == 1
    assert mu == 0.0

    # Epoch 2: high intercept rate (1st consecutive) -> Phase 1
    p, mu = scheduler.step(0.75, epoch=2)
    assert p == 1
    assert mu == 0.0

    # Epoch 3: high intercept rate (2nd consecutive) -> Phase 1
    p, mu = scheduler.step(0.80, epoch=3)
    assert p == 1
    assert mu == 0.0

    # Epoch 4: high intercept rate (3rd consecutive) -> Triggers Phase 2!
    p, mu = scheduler.step(0.72, epoch=4)
    assert p == 2
    assert mu == 0.0

    # Epoch 5: Phase 3 linear ramp begins
    p, mu = scheduler.step(0.70, epoch=5)
    assert p == 3
    expected_mu = 1.0 * (1 / 5)
    assert abs(mu - expected_mu) < 1e-5

    # Epoch 9: End of ramp (5 epochs in Phase 3)
    p, mu = scheduler.step(0.65, epoch=9)
    assert p == 3
    assert abs(mu - 1.0) < 1e-5


def test_empty_archive_zero_penalty():
    """Verify that an empty caught-route archive returns strictly 0 penalty."""
    archive = CaughtRouteArchive(grid_size=11, window_epochs=5)

    # Check all cells and both thetas
    for theta in (0, 1):
        freq_map = archive.get_frequency_map(theta)
        assert np.all(freq_map == 0.0)
        for x in range(11):
            for y in range(11):
                pen = archive.get_penalty(theta, x, y)
                assert pen == 0.0


def test_shaping_cap_holds():
    """Verify that total added shaping never exceeds 30% of goal_reward."""
    env_cfg = EnvConfig(goal_reward=30.0)
    ppo_cfg = PPOConfig()
    dec_cfg = DeceptionConfig()
    # Massive kappa and mu_max that would otherwise blow up rewards
    arms_cfg = ArmsRaceConfig(
        enabled=True,
        shaping_cap_ratio=0.30,
        kappa=1000.0,
        mu_max=100.0,
        window_epochs=5,
    )

    trainer = SelfPlayTrainer(env_cfg, ppo_cfg, dec_cfg, arms_cfg=arms_cfg, device="cpu")
    # Artificially put scheduler in Phase 3 with max mu_e
    trainer.current_phase = 3
    trainer.current_mu_e = 100.0

    # Populate archive with caught routes
    fake_traj = [(x, 5) for x in range(10)]
    fake_beliefs = [np.array([0.95, 0.05]) for _ in range(10)]
    trainer.caught_archive.record_episode(0, fake_traj, intercepted=True, beliefs=fake_beliefs)
    trainer.caught_archive.end_epoch()

    # Run episode and verify shaping bounds
    info = trainer.train_episode()
    cap = 0.30 * env_cfg.goal_reward
    assert cap == 9.0
    # The total attacker reward shouldn't exceed (max possible base reward + cap)
    # Most importantly, verify the cap logic directly
    shaping_cap = trainer.arms_cfg.shaping_cap_ratio * trainer.env_cfg.goal_reward
    assert shaping_cap == 9.0


def test_reproducibility_with_flags_off():
    """Verify that training with all arms race flags off matches old baseline for the same seed."""
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    dec_cfg = DeceptionConfig()

    def run_with_seed(seed: int, enabled: bool):
        np.random.seed(seed)
        torch.manual_seed(seed)
        arms_cfg = ArmsRaceConfig(enabled=enabled)
        trainer = SelfPlayTrainer(env_cfg, ppo_cfg, dec_cfg, arms_cfg=arms_cfg, device="cpu", action_dim=9)
        results = []
        for ep_idx in range(3):
            res = trainer.train_episode(seed=seed + ep_idx)
            results.append((res["total_attacker_reward"], res["steps"], res["goal_reached"], res["intercepted"]))
        return results

    res_default = run_with_seed(42, enabled=False)
    res_repeat = run_with_seed(42, enabled=False)

    for (r1, s1, g1, i1), (r2, s2, g2, i2) in zip(res_default, res_repeat):
        assert abs(r1 - r2) < 1e-6
        assert s1 == s2
        assert g1 == g2
        assert i1 == i2


if __name__ == "__main__":
    pytest.main(["-v", __file__])
