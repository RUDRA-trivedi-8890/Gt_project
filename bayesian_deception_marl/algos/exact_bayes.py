"""Exact Bayesian Receiver Teacher and Posterior Diagnostic Filter.

Given the Attacker's stochastic policy pi_att(a | s, theta), computes the exact recursive
Bayesian posterior over private type theta:
    b_t(theta) proportional to b_{t-1}(theta) * pi_att(a_t | s_t, theta)

Starting from uniform prior b_0 = [0.5, 0.5].
Provides:
- KL(GRU belief || exact Bayes) diagnostic logging
- Auxiliary distillation loss L_distill
- Expected Calibration Error (ECE) metric calculation
"""

from typing import List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExactBayesFilter:
    """Exact sequential Bayesian receiver tracking the true mathematical posterior."""

    def __init__(self, attacker_policy: nn.Module, device: str = "cpu", eps: float = 1e-6) -> None:
        self.attacker_policy = attacker_policy
        self.device = torch.device(device)
        self.eps = eps
        self.belief = np.array([0.5, 0.5], dtype=np.float64)

    def reset(self, prior: Optional[Sequence[float]] = None) -> np.ndarray:
        """Reset belief to prior (default uniform 50/50)."""
        if prior is not None:
            self.belief = np.asarray(prior, dtype=np.float64) / sum(prior)
        else:
            self.belief = np.array([0.5, 0.5], dtype=np.float64)
        return self.belief.copy()

    def update(
        self,
        att_pos: Sequence[float],
        action: int,
        raw_att_obs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Execute one Bayes filter update on observing action a_t at state s_t.

        Parameters
        ----------
        att_pos : sequence of float
            Attacker coordinates [x, y] in range [0..10] or normalized [0..1].
        action : int
            Observed discrete action taken by the attacker.
        raw_att_obs : np.ndarray, optional
            Optional full attacker observation (if history input is enabled).

        Returns
        -------
        b_next : np.ndarray
            Updated posterior [P(theta=0), P(theta=1)].
        """
        x_norm = float(att_pos[0]) if att_pos[0] <= 1.0 else float(att_pos[0]) / 10.0
        y_norm = float(att_pos[1]) if att_pos[1] <= 1.0 else float(att_pos[1]) / 10.0

        likelihoods = np.zeros(2, dtype=np.float64)

        with torch.no_grad():
            for hyp_theta in (0, 1):
                if raw_att_obs is not None and len(raw_att_obs) > 3:
                    obs_hyp = raw_att_obs.copy()
                    obs_hyp[0] = x_norm
                    obs_hyp[1] = y_norm
                    obs_hyp[2] = float(hyp_theta)
                else:
                    obs_hyp = np.array([x_norm, y_norm, float(hyp_theta)], dtype=np.float32)

                obs_t = torch.as_tensor(obs_hyp, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits, _ = self.attacker_policy(obs_t)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                act_prob = float(probs[int(action)])
                likelihoods[hyp_theta] = max(self.eps, act_prob)

        # Bayesian Posterior Update: b_t(theta) proportional to b_{t-1}(theta) * L(theta)
        unnorm = self.belief * likelihoods
        total = unnorm.sum()
        if total > 0:
            self.belief = unnorm / total
        else:
            self.belief = np.array([0.5, 0.5], dtype=np.float64)

        return self.belief.copy()


def compute_kl_diagnostic(gru_probs: np.ndarray, exact_probs: np.ndarray, eps: float = 1e-8) -> float:
    """Compute KL divergence KL(GRU || Exact Bayes) in bits: sum p * log2(p / q)."""
    p = np.clip(np.asarray(gru_probs, dtype=np.float64), eps, 1.0)
    q = np.clip(np.asarray(exact_probs, dtype=np.float64), eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def compute_ece(
    predicted_probs: Sequence[float],
    true_labels: Sequence[int],
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) for binary classification.

    Parameters
    ----------
    predicted_probs : sequence of float
        Confidence / probability assigned to class 1 (or the winning class).
    true_labels : sequence of int
        Ground truth binary labels (0 or 1).
    num_bins : int
        Number of equal-width bins in [0, 1].

    Returns
    -------
    ece : float
        Scalar ECE metric in [0, 1].
    """
    preds = np.asarray(predicted_probs, dtype=np.float64)
    labels = np.asarray(true_labels, dtype=np.float64)
    n = len(preds)
    if n == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        mask = (preds >= bin_lower) & (preds < bin_upper if i < num_bins - 1 else preds <= bin_upper)
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(preds[mask])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)
