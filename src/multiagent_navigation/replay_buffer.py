"""Experience replay buffer for off-policy TD3 training.

A FIFO ring over a deque of (state, action, reward, done, next_state) tuples —
note that `done` comes BEFORE `next_state`, and `sample_batch` returns column
arrays in that same order. Based on Patrick Emami's implementation.
"""

from __future__ import annotations

import random

import numpy as np

from collections import deque

# The stored transition: (s, a, r, t, s2) — done before next-state.
Experience = tuple[np.ndarray, np.ndarray, float, int, np.ndarray]

########################################
#            Replay buffer             #
########################################


class ReplayBuffer:
    """The right side of the deque holds the most recent experiences."""

    def __init__(self, buffer_size: int, random_seed: int = 123) -> None:
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer: deque[Experience] = deque()
        random.seed(random_seed)

    def add(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: float,
        t: int,
        s2: np.ndarray,
    ) -> None:
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self) -> int:
        return self.count

    def sample_batch(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniform sample of min(count, batch_size) stored transitions."""
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        # ── Column-wise batches; rewards / dones as (k, 1) columns ──
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self) -> None:
        self.buffer.clear()
        self.count = 0
