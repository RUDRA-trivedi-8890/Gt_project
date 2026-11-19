"""Metrics for the Bayesian Deception MARL project.

Exact computation of Shannon entropy, information leakage curves (entropy vs.
normalised distance to goal), and KL divergence between belief distributions.
"""

from typing import List, Sequence, Tuple
import numpy as np


def shannon_entropy(probs: np.ndarray, eps: float = 1e-8) -> float:
    """Shannon entropy H(b) = - sum b_i * log2(b_i + eps) in bits.

    Parameters
    ----------
    probs : np.ndarray
        Probability distribution (any shape).
    eps : float
        Numerical-stability clamp.

    Returns
    -------
    float
        Entropy in bits.
    """
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    probs = np.clip(probs, eps, 1.0)
    # Renormalise after clamping to keep a valid distribution
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log2(probs)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """KL divergence D_KL(P || Q) = sum p_i * log2(p_i / q_i) in bits."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def information_leakage_curve(
    beliefs: Sequence[np.ndarray],
    distances: Sequence[float],
) -> List[Tuple[float, float, float]]:
    """Build the information-leakage curve.

    For each timestep pairs (belief, normalised distance to goal) with
    (normalised_distance, entropy, confidence).

    Parameters
    ----------
    beliefs : sequence of np.ndarray
        Belief distributions b_t = [P(theta=0), P(theta=1)] at each step.
    distances : sequence of float
        Normalised distance to the active goal at each step (0 = at goal,
        1 = furthest).

    Returns
    -------
    list of (normalised_distance, entropy, confidence)
        Sorted by increasing normalised distance.
    """
    curve: List[Tuple[float, float, float]] = []
    for b, d in zip(beliefs, distances):
        b = np.asarray(b, dtype=np.float64)
        ent = shannon_entropy(b)
        conf = float(np.max(b))
        curve.append((float(d), ent, conf))
    curve.sort(key=lambda x: x[0])
    return curve


def entropy_at_midfield(
    beliefs: Sequence[np.ndarray],
    distances: Sequence[float],
    threshold: float = 0.5,
) -> float:
    """Mean entropy while the Attacker is at/near midfield."""
    ents = [
        shannon_entropy(b)
        for b, d in zip(beliefs, distances)
        if float(d) <= threshold
    ]
    return float(np.mean(ents)) if ents else 0.0


def belief_confidence(beliefs: Sequence[np.ndarray]) -> List[float]:
    """Max probability (confidence) of each belief distribution."""
    return [float(np.max(np.asarray(b, dtype=np.float64))) for b in beliefs]


def agreement_rate(
    beliefs: Sequence[np.ndarray], true_theta: int
) -> float:
    """Fraction of timesteps where the belief's argmax equals true_theta."""
    correct = 0
    for b in beliefs:
        b = np.asarray(b, dtype=np.float64)
        if int(np.argmax(b)) == true_theta:
            correct += 1
    return correct / len(beliefs) if beliefs else 0.0


def compute_t_reveal(
    beliefs: Sequence[np.ndarray],
    true_theta: int,
    threshold: float = 0.9,
    t_max: int = 25,
) -> int:
    """First timestep where the Defender's belief in the true theta exceeds threshold.

    If the threshold is never exceeded throughout the episode, returns t_max.
    """
    for t, b in enumerate(beliefs):
        b = np.asarray(b, dtype=np.float64)
        if float(b[true_theta]) >= threshold:
            return t
    return t_max


def route_diversity_entropy(
    trajectories: Sequence[Sequence[Tuple[float, float]]],
    grid_size: int = 11,
    eps: float = 1e-8,
) -> float:
    """Entropy of the 2D cell-visitation distribution across trajectories.

    Higher entropy indicates greater spatial exploration and diversity of routes.
    """
    if not trajectories:
        return 0.0
    counts = np.zeros((grid_size, grid_size), dtype=np.float64)
    for traj in trajectories:
        for pt in traj:
            x = int(np.clip(round(pt[0]), 0, grid_size - 1))
            y = int(np.clip(round(pt[1]), 0, grid_size - 1))
            counts[x, y] += 1.0
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    nz = probs[probs > 0]
    return float(-np.sum(nz * np.log2(nz + eps)))


def mean_pairwise_trajectory_distance(
    trajectories: Sequence[Sequence[Tuple[float, float]]],
    num_samples: int = 25,
) -> float:
    """Mean pairwise Euclidean distance between resampled trajectories.

    Resamples all trajectories to num_samples steps to enable uniform metric comparison.
    """
    if len(trajectories) < 2:
        return 0.0

    resampled = []
    for traj in trajectories:
        arr = np.asarray(traj, dtype=np.float64)
        if len(arr) == 0:
            continue
        if len(arr) == 1:
            resampled.append(np.repeat(arr, num_samples, axis=0))
            continue
        t_orig = np.linspace(0.0, 1.0, len(arr))
        t_new = np.linspace(0.0, 1.0, num_samples)
        interp_x = np.interp(t_new, t_orig, arr[:, 0])
        interp_y = np.interp(t_new, t_orig, arr[:, 1])
        resampled.append(np.column_stack([interp_x, interp_y]))

    if len(resampled) < 2:
        return 0.0

    dists = []
    n = len(resampled)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(resampled[i] - resampled[j], axis=-1).mean()
            dists.append(d)
    return float(np.mean(dists)) if dists else 0.0

