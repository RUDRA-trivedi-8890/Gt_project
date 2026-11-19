"""Visualization utilities: heatmaps, ASCII renders, and entropy plots.

Generates Matplotlib figures showing:
  * 2D trajectory heatmaps (pooling vs. diverging behaviour)
  * Information leakage curves (entropy vs. normalised distance to goal)
  * Training curves across episodes
"""

import os
from typing import List, Sequence
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ascii_render(
    attacker_pos: Sequence[float],
    defender_pos: Sequence[float],
    theta: int,
    grid_size: int = 11,
    goal_a: Sequence[int] = (10, 0),
    goal_b: Sequence[int] = (10, 10),
) -> str:
    """ASCII-art render of the current grid state."""
    grid = np.full((grid_size, grid_size), ".", dtype=str)
    goal = goal_a if theta == 0 else goal_b
    grid[int(goal[0]), int(goal[1])] = "G"
    grid[int(attacker_pos[0]), int(attacker_pos[1])] = "A"
    grid[int(defender_pos[0]), int(defender_pos[1])] = "D"
    return "\n".join(" ".join(row) for row in grid)


def plot_trajectory_heatmap(
    attacker_trajectories: List[np.ndarray],
    save_path: str,
    grid_size: int = 11,
    title: str = "Attacker Trajectory Heatmap",
    goal_a: Sequence[int] = (10, 0),
    goal_b: Sequence[int] = (10, 10),
) -> None:
    """Save a 2D heatmap of Attacker state visitation.

    A concentrated heatmap along the median axis indicates pooling /
    deceptive behaviour; split heatmaps toward both corners indicate a
    separating (non-deceptive) equilibrium.
    """
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    for traj in attacker_trajectories:
        traj = np.asarray(traj, dtype=np.float32)
        for x, y in traj:
            heatmap[int(round(x)), int(round(y))] += 1.0

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap.T, origin="lower", cmap="hot", interpolation="nearest")
    plt.colorbar(label="State Visitation Count")
    plt.title(title)
    plt.xlabel("Y (column)")
    plt.ylabel("X (row)")
    # Mark goals
    plt.scatter(goal_a[1], goal_a[0], marker="*", s=200, color="cyan",
                label="Goal A", edgecolors="black")
    plt.scatter(goal_b[1], goal_b[0], marker="*", s=200, color="lime",
                label="Goal B", edgecolors="black")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    _save_fig(save_path)


def plot_entropy_curve(
    entropy_values: Sequence[float],
    distances: Sequence[float],
    save_path: str,
    title: str = "Information Leakage Curve",
) -> None:
    """Plot Shannon entropy vs. normalised distance to goal.

    High entropy at midfield with a drop near the goal signals a
    pooling-then-dash deceptive strategy.
    """
    plt.figure(figsize=(6, 4))
    xs = np.asarray(distances, dtype=np.float64)
    ys = np.asarray(entropy_values, dtype=np.float64)
    order = np.argsort(xs)
    plt.plot(xs[order], ys[order], marker="o", linestyle="-", color="tab:blue")
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="max entropy (1 bit)")
    plt.xlabel("Normalised Distance to Goal")
    plt.ylabel("Shannon Entropy (bits)")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    _save_fig(save_path)


def plot_training_curves(
    metrics_log: Sequence[dict],
    save_path: str,
    title: str = "Training Curves",
) -> None:
    """Plot key training metrics over episodes."""
    if not metrics_log:
        return

    episodes = np.arange(len(metrics_log))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(title)

    def _series(key):
        return [m.get(key, 0.0) for m in metrics_log]

    # Goal success / interception rate
    ax = axes[0, 0]
    ax.plot(episodes, _series("goal_reached"), label="Goal Reached", color="tab:green")
    ax.plot(episodes, _series("intercepted"), label="Intercepted", color="tab:red")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate (per episode)")
    ax.set_title("Outcome")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Entropy at midfield
    ax = axes[0, 1]
    ax.plot(episodes, _series("entropy_at_midfield"), color="tab:purple")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Entropy at Midfield")
    ax.grid(True, alpha=0.3)

    # Total attacker reward
    ax = axes[1, 0]
    ax.plot(episodes, _series("total_attacker_reward"), color="tab:orange")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Total Attacker Reward")
    ax.grid(True, alpha=0.3)

    # Belief / ToM loss
    ax = axes[1, 1]
    ax.plot(episodes, _series("belief_loss"), label="Defender Belief Loss", color="tab:brown")
    ax.plot(episodes, _series("tom_loss"), label="Attacker ToM Loss", color="tab:pink")
    ax.set_xlabel("Episode")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Belief Training Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(save_path)


def plot_arms_race_phases(
    metrics_history: List[dict],
    save_path: str,
    title: str = "Adaptive Arms-Race Evolution (Phases 1 -> 2 -> 3)",
) -> None:
    """Plot multi-panel timeline illustrating the 3-phase arms race.

    Panels:
      1. Interception Rate & Goal Rate vs. Epoch with Phase background shading.
      2. Route Diversity (Spatial Entropy & Mean Pairwise Trajectory Distance).
      3. Reveal Timing t_reveal vs. Epoch.
      4. Caught-Route Penalty Weight mu_e vs. Epoch.
    """
    if not metrics_history:
        return

    epochs = [m["epoch"] for m in metrics_history]
    phases = [m.get("phase", 1) for m in metrics_history]
    intercept_rates = [m.get("interception_rate", 0.0) for m in metrics_history]
    goal_rates = [m.get("goal_rate", 0.0) for m in metrics_history]
    diversity_ents = [m.get("route_diversity_entropy", 0.0) for m in metrics_history]
    pairwise_dists = [m.get("mean_pairwise_distance", 0.0) for m in metrics_history]
    t_reveals = [m.get("mean_t_reveal", 0.0) for m in metrics_history]
    mu_es = [m.get("mu_e", 0.0) for m in metrics_history]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Helper to add phase shading
    phase_colors = {1: ("#e6f7ff", "Phase 1: Baseline"), 
                    2: ("#fffbe6", "Phase 2: Dominance Detected"), 
                    3: ("#fff1f0", "Phase 3: Adaptive Ramp")}

    def shade_phases(ax):
        current_p = phases[0]
        start_ep = epochs[0]
        for ep, p in zip(epochs, phases):
            if p != current_p:
                col, label = phase_colors.get(current_p, ("#f5f5f5", f"Phase {current_p}"))
                ax.axvspan(start_ep, ep, color=col, alpha=0.6, zorder=0)
                current_p = p
                start_ep = ep
        col, label = phase_colors.get(current_p, ("#f5f5f5", f"Phase {current_p}"))
        ax.axvspan(start_ep, epochs[-1], color=col, alpha=0.6, zorder=0)

    # Panel 1: Rates
    ax = axes[0]
    shade_phases(ax)
    ax.plot(epochs, intercept_rates, "r-o", markersize=4, label="Interception Rate", linewidth=1.8)
    ax.plot(epochs, goal_rates, "g-s", markersize=4, label="Goal Success Rate", linewidth=1.8)
    ax.axhline(0.70, color="darkred", linestyle="--", alpha=0.6, label="Phase 2 Trigger (70%)")
    ax.set_ylabel("Rate", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Competitive Game Outcomes & Phase Transitions")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Route Diversity
    ax = axes[1]
    shade_phases(ax)
    ax.plot(epochs, diversity_ents, "b-^", markersize=4, label="Route Entropy (bits)", linewidth=1.8)
    ax2 = ax.twinx()
    ax2.plot(epochs, pairwise_dists, "m--d", markersize=4, label="Pairwise Distance (cells)", linewidth=1.5)
    ax.set_ylabel("Entropy (bits)", color="b", fontweight="bold")
    ax2.set_ylabel("Pairwise Dist", color="m", fontweight="bold")
    ax.set_title("Attacker Route Diversity & Trajectory Variance")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: t_reveal
    ax = axes[2]
    shade_phases(ax)
    ax.plot(epochs, t_reveals, color="#d46b08", marker="o", markersize=4, label="Mean Reveal Step (t_reveal)", linewidth=1.8)
    ax.set_ylabel("Timestep", fontweight="bold")
    ax.set_title("Information Revelation Delay (t_reveal: Defender belief in true theta > 0.9)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: mu_e
    ax = axes[3]
    shade_phases(ax)
    ax.plot(epochs, mu_es, "k-", linewidth=2.0, label="Penalty Weight mu_e")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("mu_e", fontweight="bold")
    ax.set_title("Caught-Route Penalty Schedule (Phase 3 Linear Ramp)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save_fig(save_path)


def _save_fig(save_path: str) -> None:
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
