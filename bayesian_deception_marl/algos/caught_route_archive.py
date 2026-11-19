"""Caught-Route Archive for the Adaptive Arms Race.

Maintains frequency maps C_theta(x, y) across a sliding window of the last W epochs
for episodes that were either intercepted or where the Defender's belief in the true
theta exceeded the reveal threshold (0.9) prior to midfield.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np


class CaughtRouteArchive:
    """Sliding-window archive of failed or leaked attacker trajectories."""

    def __init__(
        self,
        grid_size: int = 11,
        window_epochs: int = 5,
        midfield_row: int = 5,
        belief_threshold: float = 0.9,
    ) -> None:
        self.grid_size = grid_size
        self.window_epochs = window_epochs
        self.midfield_row = midfield_row
        self.belief_threshold = belief_threshold

        # Sliding window of epochs: deque of lists of (theta, trajectory)
        self.window: deque[List[Tuple[int, List[Tuple[int, int]]]]] = deque(maxlen=window_epochs)
        self.current_epoch_caught: List[Tuple[int, List[Tuple[int, int]]]] = []

        # Cached normalized frequency maps: shape (2, grid_size, grid_size)
        self.C_theta: np.ndarray = np.zeros((2, grid_size, grid_size), dtype=np.float32)

    def is_caught_or_leaked(
        self,
        intercepted: bool,
        theta: int,
        trajectory: List[Tuple[int, int]],
        beliefs: List[np.ndarray],
    ) -> bool:
        """Determine if an episode qualifies as caught or leaked before midfield.

        Criteria:
        1. Episode ended in interception.
        2. OR Defender belief in true theta > belief_threshold before midfield (row <= midfield_row).
        """
        if intercepted:
            return True

        for (x, y), b in zip(trajectory, beliefs):
            if x <= self.midfield_row:
                p_true = float(b[theta])
                if p_true >= self.belief_threshold:
                    return True
        return False

    def record_episode(
        self,
        theta: int,
        trajectory: List[Tuple[int, int]],
        intercepted: bool,
        beliefs: List[np.ndarray],
    ) -> bool:
        """Record an episode trajectory if it qualifies as caught or leaked."""
        qualifies = self.is_caught_or_leaked(intercepted, theta, trajectory, beliefs)
        if qualifies:
            self.current_epoch_caught.append((theta, list(trajectory)))
        return qualifies

    def end_epoch(self) -> None:
        """Finalize the current epoch, append to sliding window, and recompute C_theta."""
        self.window.append(list(self.current_epoch_caught))
        self.current_epoch_caught = []
        self._recompute_maps()

    def _recompute_maps(self) -> None:
        """Recompute normalized frequency map C_theta(x, y) in [0, 1] for theta in {0, 1}."""
        counts = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)

        for epoch_episodes in self.window:
            for theta, traj in epoch_episodes:
                for x, y in traj:
                    xi = int(np.clip(round(x), 0, self.grid_size - 1))
                    yi = int(np.clip(round(y), 0, self.grid_size - 1))
                    counts[theta, xi, yi] += 1.0

        for t in (0, 1):
            max_c = counts[t].max()
            if max_c > 0:
                self.C_theta[t] = counts[t] / max_c
            else:
                self.C_theta[t].fill(0.0)

    def get_penalty(self, theta: int, x: float, y: float) -> float:
        """Query C_theta(x, y) in [0, 1]. Returns 0.0 if archive is empty."""
        xi = int(np.clip(round(x), 0, self.grid_size - 1))
        yi = int(np.clip(round(y), 0, self.grid_size - 1))
        return float(self.C_theta[theta, xi, yi])

    def get_frequency_map(self, theta: int) -> np.ndarray:
        """Return a copy of the normalized frequency map for theta."""
        return self.C_theta[theta].copy()

    def total_caught_count(self) -> int:
        """Total number of caught episodes across the current sliding window."""
        return sum(len(episodes) for episodes in self.window)

    def clear(self) -> None:
        """Reset archive to empty state."""
        self.window.clear()
        self.current_epoch_caught.clear()
        self.C_theta.fill(0.0)
