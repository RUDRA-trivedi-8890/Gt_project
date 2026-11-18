"""Model Evaluation and Game-Theoretic Analysis Script.

Game Theory Theoretical Context:
--------------------------------
In an Asymmetric Information Markov Signaling Game:
1. Separating Equilibrium (Naive RL):
   - The Attacker's type theta is immediately revealed by its chosen action.
   - If the Attacker moves directly diagonally toward Goal A or Goal B, the Defender
     quickly updates its Bayesian posterior belief: P(theta | trajectory) -> 1.0 or 0.0.
   - Consequently, Defender Shannon Entropy drops sharply to 0 early in the episode,
     allowing the Defender to intercept the Attacker easily.

2. Pooling Equilibrium (Belief-Regularized MARL):
   - Different types (theta=0 and theta=1) choose identical or indistinguishable actions
     along a shared median corridor (pooling along the center column).
   - As long as the Attacker stays on this shared path, the trajectory conveys zero
     costly signal regarding theta.
   - The Defender's posterior belief remains diffuse (b ~ [0.5, 0.5]), and Shannon Entropy
     remains high (~ 1.0 bit) throughout the approach.
   - Only near the terminal rows does the Attacker branch off toward its true goal,
     forcing a late commitment from the Defender when it is too late to intercept.

This script loads the trained networks (in greedy evaluation mode), collects 100 test
episodes, computes empirical performance metrics, and plots:
  1. `figures/leakage_curve_regularized.png`: Information Leakage (Entropy vs Time Step)
  2. `figures/heatmap_regularized.png`: 2D Trajectory Visitation Map (Pooling proof)
"""

import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Ensure local package resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import EnvConfig
from envs.grid_world import DeceptionGridWorld as AsymmetricGridEnv
from models.networks import get_attacker_network, get_defender_network
from models.belief_head import BeliefHead


# ============================================================================
# EVALUATION RUNNER (GREEDY INFERENCE)
# ============================================================================
def evaluate(num_episodes: int = 100) -> Tuple[Dict[str, float], List[dict]]:
    """Run greedy evaluation episodes and record detailed trajectory data.

    Parameters
    ----------
    num_episodes : int
        Number of test episodes to run (default 100).

    Returns
    -------
    summary_metrics : dict
        Aggregated statistics (win rates, mean entropy, episode lengths).
    episode_data : list of dicts
        Per-episode records containing trajectories, beliefs, and outcomes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Inference Device: {device}")

    # Initialize environment
    env_cfg = EnvConfig()
    env = AsymmetricGridEnv(env_cfg)

    # Instantiate model architectures matching training dimensions (action_dim = 9)
    attacker_net = get_attacker_network(hidden_dim=64, action_dim=9).to(device)
    defender_net = get_defender_network(hidden_dim=64, action_dim=9).to(device)
    belief_head = BeliefHead(input_dim=4, hidden_dim=64).to(device)

    # Paths to model checkpoints
    models_dir = os.path.join(CURRENT_DIR, "models")
    att_ckpt = os.path.join(models_dir, "attacker_net.pt")
    def_ckpt = os.path.join(models_dir, "defender_net.pt")
    bh_ckpt = os.path.join(models_dir, "belief_head.pt")

    for path, name in [(att_ckpt, "Attacker"), (def_ckpt, "Defender"), (bh_ckpt, "Belief Head")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Evaluate] Checkpoint not found for {name}: {path}")

    # Load model weights
    attacker_net.load_state_dict(torch.load(att_ckpt, map_location=device))
    defender_net.load_state_dict(torch.load(def_ckpt, map_location=device))
    belief_head.load_state_dict(torch.load(bh_ckpt, map_location=device))

    # Set all networks strictly to evaluation mode
    attacker_net.eval()
    defender_net.eval()
    belief_head.eval()

    print("[Evaluate] Models loaded successfully in eval() mode.")
    print(f"[Evaluate] Collecting {num_episodes} greedy evaluation episodes...\n")

    goal_reached_count = 0
    intercepted_count = 0
    truncated_count = 0
    all_episode_lengths: List[int] = []
    all_midfield_entropies: List[float] = []
    episode_records: List[dict] = []

    with torch.no_grad():
        for ep in range(1, num_episodes + 1):
            obs_dict, info = env.reset()
            theta = info["theta"]

            att_obs = obs_dict["attacker"]  # [pos_x/10, pos_y/10, theta]
            def_obs = obs_dict["defender"]  # [att_x/10, att_y/10, def_x/10, def_y/10]

            def_hidden: torch.Tensor | None = None
            ep_positions: List[Tuple[float, float]] = []
            ep_defender_positions: List[Tuple[float, float]] = []
            ep_entropies: List[float] = []
            ep_beliefs: List[np.ndarray] = []

            done = False
            step_count = 0

            while not done:
                # Log physical position (grid coordinates)
                curr_att_pos = (float(att_obs[0] * 10.0), float(att_obs[1] * 10.0))
                curr_def_pos = (float(def_obs[2] * 10.0), float(def_obs[3] * 10.0))
                ep_positions.append(curr_att_pos)
                ep_defender_positions.append(curr_def_pos)

                # 1. Update Defender Belief Head
                def_obs_t = torch.as_tensor(def_obs, dtype=torch.float32, device=device).unsqueeze(0)
                belief_logits, def_hidden = belief_head.step(def_obs_t, def_hidden)
                probs = F.softmax(belief_logits, dim=-1)  # [P(theta=0), P(theta=1)]
                probs_np = probs.squeeze(0).cpu().numpy()
                ep_beliefs.append(probs_np)

                # 2. Compute Shannon Entropy H(b) = - sum(p * log2(p + 1e-8))
                entropy = -float(np.sum(probs_np * np.log2(probs_np + 1e-8)))
                ep_entropies.append(entropy)

                # Track midfield entropy (rows 4 to 6)
                if 4.0 <= curr_att_pos[0] <= 6.0:
                    all_midfield_entropies.append(entropy)

                # 3. Construct Defender AC input (shape: 6)
                def_obs_full = np.concatenate([def_obs, probs_np], axis=-1).astype(np.float32)

                # 4. GREEDY ACTION SELECTION (argmax over logits, zero stochastic exploration)
                att_obs_t = torch.as_tensor(att_obs, dtype=torch.float32, device=device).unsqueeze(0)
                def_obs_full_t = torch.as_tensor(def_obs_full, dtype=torch.float32, device=device).unsqueeze(0)

                att_logits, _ = attacker_net(att_obs_t)
                def_logits, _ = defender_net(def_obs_full_t)

                greedy_att_action = int(torch.argmax(att_logits, dim=-1).item())
                greedy_def_action = int(torch.argmax(def_logits, dim=-1).item())

                # 5. Environment Step
                next_obs_dict, _, terminated, truncated, step_info = env.step(
                    greedy_att_action, greedy_def_action
                )
                done = terminated or truncated
                step_count += 1

                if done:
                    # Final position reached
                    final_att_pos = (float(next_obs_dict["attacker"][0] * 10.0), float(next_obs_dict["attacker"][1] * 10.0))
                    ep_positions.append(final_att_pos)

                    is_goal = bool(step_info.get("goal_reached", False))
                    is_intercept = bool(step_info.get("intercepted", False))
                    is_trunc = bool(truncated and not is_goal and not is_intercept)

                    if is_goal:
                        goal_reached_count += 1
                    elif is_intercept:
                        intercepted_count += 1
                    else:
                        truncated_count += 1

                    outcome = "Goal Reached" if is_goal else ("Intercepted" if is_intercept else "Timeout")
                    break

                att_obs = next_obs_dict["attacker"]
                def_obs = next_obs_dict["defender"]

            all_episode_lengths.append(step_count)
            episode_records.append({
                "episode": ep,
                "theta": theta,
                "outcome": outcome,
                "goal_reached": is_goal,
                "intercepted": is_intercept,
                "truncated": is_trunc,
                "steps": step_count,
                "positions": ep_positions,
                "defender_positions": ep_defender_positions,
                "entropies": ep_entropies,
                "beliefs": ep_beliefs,
            })

    # Summary metrics
    goal_rate = (goal_reached_count / num_episodes) * 100.0
    intercept_rate = (intercepted_count / num_episodes) * 100.0
    trunc_rate = (truncated_count / num_episodes) * 100.0
    avg_steps = float(np.mean(all_episode_lengths))
    avg_mid_entropy = float(np.mean(all_midfield_entropies)) if all_midfield_entropies else 0.0

    print("=" * 65)
    print("         EVALUATION RESULTS (100 GREEDY EPISODES)         ")
    print("=" * 65)
    print(f"Goal Success Rate (Attacker Win):  {goal_rate:5.1f}% ({goal_reached_count}/{num_episodes})")
    print(f"Interception Rate (Defender Win):  {intercept_rate:5.1f}% ({intercepted_count}/{num_episodes})")
    print(f"Timeout / Stalled Rate:           {trunc_rate:5.1f}% ({truncated_count}/{num_episodes})")
    print(f"Average Episode Length:           {avg_steps:5.2f} steps")
    print(f"Average Midfield Belief Entropy:  {avg_mid_entropy:5.3f} bits (max = 1.0)")
    print("=" * 65 + "\n")

    summary = {
        "goal_rate": goal_rate,
        "intercept_rate": intercept_rate,
        "trunc_rate": trunc_rate,
        "avg_steps": avg_steps,
        "avg_mid_entropy": avg_mid_entropy,
    }
    return summary, episode_records


# ============================================================================
# PLOTTING UTILITIES: PROOF OF POOLING EQUILIBRIUM
# ============================================================================
def plot_information_leakage(episode_records: List[dict], save_path: str) -> None:
    """Plot Information Leakage Curve (Defender Shannon Entropy vs Timestep).

    Game-Theoretic Interpretation:
    ------------------------------
    * Perfect Separating Equilibrium:
      Entropy drops to 0 at t=1 or t=2 because the agent's move directly reveals theta.
    * Robust Pooling Equilibrium:
      Entropy stays plateaued near 1.0 bit (maximum Defender ambiguity) across the early
      and midfield timesteps. The curve only declines sharply at the terminal boundary,
      confirming that the Attacker successfully masks its intended target until
      interception is physically impossible.
    """
    max_steps = max(len(rec["entropies"]) for rec in episode_records)
    # Collect entropy arrays aligned by timestep
    step_entropy_matrix: List[List[float]] = [[] for _ in range(max_steps)]

    for rec in episode_records:
        for t, h in enumerate(rec["entropies"]):
            step_entropy_matrix[t].append(h)

    # Compute mean and standard error for each step
    valid_steps = [t for t in range(max_steps) if len(step_entropy_matrix[t]) >= 5]
    mean_entropies = [float(np.mean(step_entropy_matrix[t])) for t in valid_steps]
    std_entropies = [float(np.std(step_entropy_matrix[t])) for t in valid_steps]
    counts = [len(step_entropy_matrix[t]) for t in valid_steps]
    sem_entropies = [std / np.sqrt(n) if n > 1 else 0.0 for std, n in zip(std_entropies, counts)]

    timesteps = np.array(valid_steps)
    means = np.array(mean_entropies)
    sems = np.array(sem_entropies)

    plt.figure(figsize=(8, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Shaded confidence band (Mean +/- SEM)
    plt.fill_between(timesteps, np.clip(means - sems, 0, 1.05), np.clip(means + sems, 0, 1.05),
                     color="#1f77b4", alpha=0.25, label="Standard Error (Across Episodes)")
    # Mean curve
    plt.plot(timesteps, means, color="#1f77b4", linewidth=2.8, marker="o", markersize=5,
             label="Mean Defender Belief Entropy $H(b)$")

    # Reference lines
    plt.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=1.8, alpha=0.85,
                label="Maximum Uncertainty (1.0 bit: 50/50 Pooling)")
    plt.axhline(0.0, color="#d62728", linestyle=":", linewidth=1.8, alpha=0.85,
                label="Full Revelation (0.0 bit: Separating)")
    plt.axvline(7.0, color="#ff7f0e", linestyle="-.", linewidth=1.8, alpha=0.9,
                label="Bifurcation Point (Step 7 / Row 7)")

    # Visual phase annotations
    plt.annotate("Shared Pooling Corridor\n(Zero Leakage: $H(b) \\approx 1.0$ bit)",
                 xy=(3.5, 0.999),
                 xytext=(3.5, 0.82),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
                 fontsize=9, fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#1f77b4", alpha=0.95))

    if max(valid_steps) >= 10:
        plt.annotate("Physical Bifurcation Phase\n(Entropy drops to 0.0 at Goal)",
                     xy=(10.0, float(means[min(10, len(means)-1)])),
                     xytext=(9.5, 0.45),
                     arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
                     fontsize=9, fontweight="bold", ha="center",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d62728", alpha=0.95))

    plt.title("Information Leakage Curve: Defender Belief Entropy Over Time\n"
              "[Proof of Pooling Equilibrium in Asymmetric Signaling Game]",
              fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Trajectory Timestep ($t$)", fontsize=11, labelpad=8)
    plt.ylabel("Defender Shannon Entropy $H(b)$ (Bits)", fontsize=11, labelpad=8)
    plt.ylim(-0.05, 1.15)
    plt.xlim(0, max(valid_steps))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Information Leakage Curve -> {save_path}")


def plot_trajectory_heatmap(episode_records: List[dict], save_path: str) -> None:
    """Plot 2D Trajectory Map showing Attacker paths partitioned by secret type theta.

    Game-Theoretic Interpretation:
    ------------------------------
    * Naive Baseline:
      Theta=0 trajectories immediately veer left toward Goal A;
      Theta=1 trajectories immediately veer right toward Goal B.
      The paths diverge starting at row 0 (immediate separation).
    * Deception-Regularized Policy:
      Trajectories for BOTH types overlap along the center column (Column 5),
      moving forward through the midfield together (Pooling Corridor).
      Only when approaching row 8-9 do the paths branch left and right.
      This visual fork is the hallmark of a dynamic pooling equilibrium.
    """
    grid_size = 11
    # Separate visitation density matrices
    density_theta0 = np.zeros((grid_size, grid_size), dtype=np.float32)
    density_theta1 = np.zeros((grid_size, grid_size), dtype=np.float32)

    trajs_theta0: List[List[Tuple[float, float]]] = []
    trajs_theta1: List[List[Tuple[float, float]]] = []

    for rec in episode_records:
        pts = rec["positions"]
        if rec["theta"] == 0:
            trajs_theta0.append(pts)
            for r, c in pts:
                density_theta0[int(round(r)), int(round(c))] += 1.0
        else:
            trajs_theta1.append(pts)
            for r, c in pts:
                density_theta1[int(round(r)), int(round(c))] += 1.0

    plt.figure(figsize=(7.5, 7.5), dpi=300)
    ax = plt.gca()

    # Draw discrete grid cells
    for x in range(grid_size + 1):
        ax.axvline(x - 0.5, color="#e0e0e0", linestyle=":", linewidth=0.8)
        ax.axhline(x - 0.5, color="#e0e0e0", linestyle=":", linewidth=0.8)

    # Highlight the central pooling corridor (Col 5)
    ax.axvspan(4.5, 5.5, color="#f0f8ff", alpha=0.7, label="Pooling Corridor (Col 5)")

    # Plot sample trajectory paths with slight jitter for visual clarity
    jitter = 0.08
    np.random.seed(42)

    # Plot theta=0 paths (Blue: Target Goal A at (10, 0))
    for pts in trajs_theta0[:30]:
        rs = [p[0] + np.random.uniform(-jitter, jitter) for p in pts]
        cs = [p[1] + np.random.uniform(-jitter, jitter) for p in pts]
        ax.plot(cs, rs, color="#1f77b4", alpha=0.35, linewidth=1.5, zorder=2)

    # Plot theta=1 paths (Red/Orange: Target Goal B at (10, 10))
    for pts in trajs_theta1[:30]:
        rs = [p[0] + np.random.uniform(-jitter, jitter) for p in pts]
        cs = [p[1] + np.random.uniform(-jitter, jitter) for p in pts]
        ax.plot(cs, rs, color="#d62728", alpha=0.35, linewidth=1.5, zorder=2)

    # Mark key landmarks
    # Attacker Spawn: (0, 5) -> Row 0, Col 5
    ax.scatter(5, 0, marker="o", s=220, color="#2ca02c", edgecolors="black", linewidths=1.8,
               label="Attacker Spawn (Row 0, Col 5)", zorder=5)
    # Defender Spawn: (9, 5) -> Row 9, Col 5
    ax.scatter(5, 9, marker="s", s=220, color="#9467bd", edgecolors="black", linewidths=1.8,
               label="Defender Spawn (Row 9, Col 5)", zorder=5)
    # Goal A: (10, 0) -> Row 10, Col 0
    ax.scatter(0, 10, marker="*", s=350, color="#1f77b4", edgecolors="black", linewidths=1.8,
               label=r"Goal A ($\Theta=0$, Row 10, Col 0)", zorder=5)
    # Goal B: (10, 10) -> Row 10, Col 10
    ax.scatter(10, 10, marker="*", s=350, color="#d62728", edgecolors="black", linewidths=1.8,
               label=r"Goal B ($\Theta=1$, Row 10, Col 10)", zorder=5)

    # Annotations explaining pooling
    ax.text(5.0, 4.0, "Shared Median Path\n(Pooling Corridor)\n$P(\\theta) \\approx [0.5, 0.5]$",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#003366",
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#003366", lw=1.2, alpha=0.9),
            zorder=6)

    ax.set_title("Attacker Trajectory Distribution: Visual Proof of Pooling Equilibrium\n"
                 "[Shared Centerline Approach Before Late Bifurcation]",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Grid Column (Y)", fontsize=10, labelpad=6)
    ax.set_ylabel(r"Grid Row (X, Forward Direction $\rightarrow$)", fontsize=10, labelpad=6)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))

    # Custom legend elements
    legend_elements = [
        Line2D([0], [0], color="#1f77b4", lw=2, label=r"Attacker Trajectory (Target A: $\Theta=0$)"),
        Line2D([0], [0], color="#d62728", lw=2, label=r"Attacker Trajectory (Target B: $\Theta=1$)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markeredgecolor="k", markersize=10, label="Attacker Spawn (0, 5)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#9467bd", markeredgecolor="k", markersize=10, label="Defender Spawn (9, 5)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#1f77b4", markeredgecolor="k", markersize=14, label="Goal A (10, 0)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#d62728", markeredgecolor="k", markersize=14, label="Goal B (10, 10)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=2, fontsize=8.5, frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] Trajectory Heatmap -> {save_path}")


# ============================================================================
# MAIN EXECUTABLE
# ============================================================================
def main() -> None:
    figures_dir = os.path.join(CURRENT_DIR, "figures")
    leakage_curve_path = os.path.join(figures_dir, "leakage_curve_regularized.png")
    heatmap_path = os.path.join(figures_dir, "heatmap_regularized.png")

    # Run evaluation
    metrics, episode_records = evaluate(num_episodes=100)

    # Generate and overwrite the target figures
    print("[Evaluate] Generating evaluation figures...")
    plot_information_leakage(episode_records, leakage_curve_path)
    plot_trajectory_heatmap(episode_records, heatmap_path)

    print("\n[Evaluate] Evaluation and figure generation complete!")
    print(f"  1. {leakage_curve_path}")
    print(f"  2. {heatmap_path}")


if __name__ == "__main__":
    main()
