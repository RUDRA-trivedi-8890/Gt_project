"""Adaptive Arms-Race Phase Scheduler (Phases 1 -> 2 -> 3).

Controls the progression of the attacker's counter-adaptation against defender dominance:
- Phase 1: Baseline training. Caught-route penalty weight mu_e = 0.
- Phase 2: Defender dominance detected when interception rate > p2_interception_threshold
  for p2_consecutive_epochs consecutive epochs. mu_e remains 0.
- Phase 3: Arms race active. Linearly ramps mu_e from 0 to mu_max over p3_ramp_epochs epochs.
"""

from typing import Dict, List, Tuple
from config import ArmsRaceConfig


class PhaseScheduler:
    """Manages Phase 1 -> 2 -> 3 transitions and caught-route penalty weight mu_e."""

    def __init__(self, config: ArmsRaceConfig) -> None:
        self.cfg = config
        self.phase: int = 1
        self.consecutive_high_intercepts: int = 0
        self.phase2_epoch: int = -1
        self.phase3_epoch: int = -1
        self.current_epoch: int = 0
        self.current_mu_e: float = 0.0
        self.history: List[Dict[str, float]] = []

    def step(self, interception_rate: float, epoch: int) -> Tuple[int, float]:
        """Update phase state and return (current_phase, current_mu_e) for the epoch.

        Parameters
        ----------
        interception_rate : float
            Fraction of episodes in the current epoch that ended in interception.
        epoch : int
            Current epoch index (1-based or 0-based).

        Returns
        -------
        phase : int
            Current phase (1, 2, or 3).
        mu_e : float
            Current penalty weight.
        """
        self.current_epoch = epoch

        if not self.cfg.enabled:
            self.phase = 1
            self.current_mu_e = 0.0
            self._record(interception_rate)
            return self.phase, self.current_mu_e

        # Check Phase 1 -> Phase 2 trigger
        if self.phase == 1:
            if interception_rate >= self.cfg.p2_interception_threshold:
                self.consecutive_high_intercepts += 1
                if self.consecutive_high_intercepts >= self.cfg.p2_consecutive_epochs:
                    self.phase = 2
                    self.phase2_epoch = epoch
                    # Automatically transition to Phase 3 setup
                    self.phase3_epoch = epoch + 1
            else:
                self.consecutive_high_intercepts = 0
            self.current_mu_e = 0.0

        elif self.phase == 2:
            # Phase 2 was triggered in the previous epoch; start Phase 3 ramping
            self.phase = 3
            if self.phase3_epoch < 0:
                self.phase3_epoch = epoch
            progress = min(1.0, max(0.0, (epoch - self.phase3_epoch + 1) / max(1, self.cfg.p3_ramp_epochs)))
            self.current_mu_e = float(self.cfg.mu_max * progress)

        elif self.phase == 3:
            # In Phase 3: continue ramping up to mu_max
            progress = min(1.0, max(0.0, (epoch - self.phase3_epoch + 1) / max(1, self.cfg.p3_ramp_epochs)))
            self.current_mu_e = float(self.cfg.mu_max * progress)

        self._record(interception_rate)
        return self.phase, self.current_mu_e

    def _record(self, interception_rate: float) -> None:
        self.history.append({
            "epoch": float(self.current_epoch),
            "phase": float(self.phase),
            "mu_e": float(self.current_mu_e),
            "interception_rate": float(interception_rate),
        })

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.phase = 1
        self.consecutive_high_intercepts = 0
        self.phase2_epoch = -1
        self.phase3_epoch = -1
        self.current_epoch = 0
        self.current_mu_e = 0.0
        self.history.clear()
