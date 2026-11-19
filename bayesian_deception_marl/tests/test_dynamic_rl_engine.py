"""Unit tests for Dynamic Online RL Co-Evolution Engine."""

import os
import sys
import pytest
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from algos.dynamic_rl_engine import DynamicRLEngine


def test_dynamic_rl_naive_direct_rush():
    """Verify that in Epoch 1 the attacker takes naive direct diagonal steps."""
    engine = DynamicRLEngine()
    assert engine.epoch == 1
    assert "Naive Greedy Direct Rush" in engine.current_stage

    # For Goal A (theta=0), action should be Down-Left (4) from spawn (0, 5)
    act_a = engine.get_action((0, 5), (9, 5), theta=0)
    assert act_a == 4

    # For Goal B (theta=1), action should be Down-Right (5) from spawn (0, 5)
    act_b = engine.get_action((0, 5), (9, 5), theta=1)
    assert act_b == 5


def test_dynamic_rl_online_learning_update_on_caught():
    """Verify that being intercepted penalizes the diagonal route and advances epoch."""
    engine = DynamicRLEngine()
    caught_map = np.zeros((2, 11, 11), dtype=np.float32)
    traj = [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]

    res = engine.execute_online_learning_step(
        trajectory=traj,
        theta=0,
        intercepted=True,
        goal_reached=False,
        mean_entropy=0.15,
        caught_map=caught_map,
    )

    # Caught map should be updated on visited trajectory
    assert caught_map[0, 1, 4] > 0.0
    assert caught_map[0, 2, 3] > 0.0

    # Policy backprop should have produced a non-zero weight shift
    assert res["delta_w"] > 0.0
    # Epoch should have advanced
    assert res["epoch"] == 2
    assert "CAUGHT IN EPOCH 1" in res["msg"]


def test_dynamic_rl_corridor_pooling_emergence():
    """Verify that as epochs advance, the attacker shifts from diagonals into Column 5."""
    engine = DynamicRLEngine()
    # Advance to Epoch 8 (Emergent Pooling Equilibrium)
    engine.epoch = 8
    engine.advance_epoch()
    assert engine.epoch == 9

    # From spawn (0, 5) and row 1-5, action should now be Down (1) into Column 5!
    act_pooled = engine.get_action((0, 5), (9, 5), theta=0)
    assert act_pooled == 1
    act_pooled_b = engine.get_action((2, 5), (9, 5), theta=1)
    assert act_pooled_b == 1


def test_dynamic_rl_defender_tactical_intelligence_and_learning():
    """Verify that Defender policy uses learned neural network and protects the goal line."""
    engine = DynamicRLEngine()
    assert hasattr(engine, "defender_policy")
    assert isinstance(engine.defender_policy, torch.nn.Module)

    # 1. When attacker is ambiguous / pooling in corridor, defender holds center goal line (row 9, col 5)
    # Action should NOT be Up-Left (6) sprinting toward (0, 0)!
    def_act_center = engine.get_defender_action((2, 5), (9, 5), gru_probs=np.array([0.5, 0.5]))
    delta = engine.ACTION_DELTAS[def_act_center]
    # Resulting row should remain at goal line (row >= 8, NOT moving to row 0)
    assert 9 + delta[0] >= 8.0, f"Defender foolishly retreated from goal line to row {9 + delta[0]}"

    # 2. When attacker leaks Goal A (p_A = 0.95), defender maneuvers toward Goal A (cutting left)
    def_act_leak_a = engine.get_defender_action((4, 2), (9, 5), gru_probs=np.array([0.95, 0.05]))
    delta_a = engine.ACTION_DELTAS[def_act_leak_a]
    assert delta_a[1] < 0, f"Defender should cut left toward Goal A, got col delta {delta_a[1]}"

    # 3. When attacker leaks Goal B (p_B = 0.95), defender maneuvers toward Goal B (cutting right)
    def_act_leak_b = engine.get_defender_action((4, 8), (9, 5), gru_probs=np.array([0.05, 0.95]))
    delta_b = engine.ACTION_DELTAS[def_act_leak_b]
    assert delta_b[1] > 0, f"Defender should cut right toward Goal B, got col delta {delta_b[1]}"

