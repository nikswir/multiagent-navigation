"""Experience replay buffer for off-policy TD3 training.

A FIFO ring over a plain list of (state, action, reward, done, next_state)
tuples — note that `done` comes BEFORE `next_state`, and `sample_batch`
returns column arrays in that same order. A list gives O(1) indexing, so
uniform sampling stays cheap at full capacity (a deque would pay O(n) per
pick). Sampling uses a private `random.Random(seed)` — constructing a buffer
never touches module-global RNG state.
"""

from __future__ import annotations

import random

import numpy as np

# The stored transition: (s, a, r, t, s2) — done before next-state.
Experience = tuple[np.ndarray, np.ndarray, float, int, np.ndarray]

########################################
#            Replay buffer             #
########################################


class ReplayBuffer:
    """A fixed-capacity FIFO ring: the write cursor overwrites the oldest."""

    def __init__(self, buffer_size: int, random_seed: int = 123) -> None:
        self.buffer_size = buffer_size
        self.buffer: list[Experience] = []
        self.rng = random.Random(random_seed)
        self._cursor = 0

    def add(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: float,
        t: int,
        s2: np.ndarray,
    ) -> None:
        experience = (s, a, r, t, s2)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            # ── Ring is full: overwrite the oldest entry in place ──
            self.buffer[self._cursor] = experience
            self._cursor = (self._cursor + 1) % self.buffer_size

    def size(self) -> int:
        return len(self.buffer)

    def sample_batch(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniform sample of min(size, batch_size) stored transitions."""
        batch = self.rng.sample(self.buffer, min(len(self.buffer), batch_size))

        # ── Column-wise batches; rewards / dones as (k, 1) columns ──
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self) -> None:
        self.buffer.clear()
        self._cursor = 0
