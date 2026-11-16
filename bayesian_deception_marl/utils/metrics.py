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
