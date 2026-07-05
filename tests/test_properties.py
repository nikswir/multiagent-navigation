"""Property-based tests for the TD3 navigation code (Hypothesis, stage 1).

Assert real invariants over generated inputs rather than hand-picked cases:
the pure reward function's terminal values and step formula, the actor's tanh
bound, the environment's 24-d reset contract over random layouts, and the
replay buffer's batch shapes. Layout randomness is env-owned: the generated
`seed` goes straight into `SimpleEnv(seed=...)`.
"""

from __future__ import annotations

import math

import torch
import pytest
import numpy as np

from hypothesis import given, settings
from hypothesis import strategies as st

from multiagent_navigation.agent import Actor
from multiagent_navigation.environment import SimpleEnv
from multiagent_navigation.replay_buffer import ReplayBuffer

# Finite, modestly-bounded floats: enough to exercise the maths without
# overflowing intermediate activations into non-finite territory.
FLOATS = st.floats(
    min_value=-1e3,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)

# Action components as the policy emits them (tanh-bounded).
ACTIONS = st.floats(
    min_value=-1.0,
    max_value=1.0,
    allow_nan=False,
    width=32,
)

# Lidar readings: non-negative, up to well past the proximity-penalty knee.
LASERS = st.floats(
    min_value=0.0,
    max_value=10.0,
    allow_nan=False,
    width=32,
)

########################################
#           Reward function            #
########################################


@settings(max_examples=50)
@given(a0=ACTIONS, a1=ACTIONS, min_laser=LASERS)
def test_get_reward_terminal_values(
    a0: float,
    a1: float,
    min_laser: float,
) -> None:
    """The terminal rewards are fixed: +100 at the goal, -100 on impact."""
    action = [a0, a1]
    assert SimpleEnv.get_reward(True, False, action, min_laser) == 100.0
    assert SimpleEnv.get_reward(False, True, action, min_laser) == -100.0

    # ── Reaching the goal outranks a simultaneous collision ──
    assert SimpleEnv.get_reward(True, True, action, min_laser) == 100.0


@settings(max_examples=50)
@given(a0=ACTIONS, a1=ACTIONS, min_laser=LASERS)
def test_get_reward_step_formula(
    a0: float,
    a1: float,
    min_laser: float,
) -> None:
    """The shaped step reward is finite and matches its formula exactly."""
    reward = SimpleEnv.get_reward(False, False, [a0, a1], min_laser)

    proximity = 1 - min_laser if min_laser < 1 else 0.0
    expected = a0 / 2 - abs(a1) / 2 - proximity - 0.05

    assert math.isfinite(reward)
    assert reward == pytest.approx(expected)


########################################
#               Networks               #
########################################


@settings(max_examples=40)
@given(state=st.lists(FLOATS, min_size=24, max_size=24))
def test_actor_output_is_within_tanh_bounds(state: list[float]) -> None:
    """The actor's tanh head keeps every action finite and in [-1, 1]."""
    torch.manual_seed(0)
    actor = Actor(24, 2, hidden1=16, hidden2=16)

    x = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        out = actor(x)

    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()
    assert bool((out.abs() <= 1.0).all())


########################################
#             Environment              #
########################################


@settings(max_examples=10, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**16),
    n_robots=st.integers(min_value=1, max_value=4),
)
def test_reset_returns_bounded_24d_states(seed: int, n_robots: int) -> None:
    """A reset yields one finite 24-d state per active robot."""
    env = SimpleEnv(
        robot_radius=0.25,
        max_steps=50,
        n_robots=n_robots,
        max_robots=4,
        seed=seed,
    )

    states = env.reset(n_robots=n_robots)

    assert len(states) == n_robots
    for state in states:
        assert len(state) == 24
        assert all(math.isfinite(v) for v in state)

        # ── Normalized goal distance >= 0; relative angle in [-1, 1] ──
        assert state[20] >= 0.0
        assert -1.0 <= state[21] <= 1.0


########################################
#            Replay buffer             #
########################################


@settings(max_examples=30)
@given(
    n_add=st.integers(min_value=1, max_value=30),
    batch_size=st.integers(min_value=1, max_value=40),
)
def test_replay_buffer_sample_shapes(n_add: int, batch_size: int) -> None:
    """Sampled columns keep (s, a, r, t, s2) shapes; r / t are (k, 1)."""
    buffer = ReplayBuffer(buffer_size=100, random_seed=0)
    for k in range(n_add):
        buffer.add(np.zeros(24), np.zeros(2), float(k), 0, np.zeros(24))

    s, a, r, t, s2 = buffer.sample_batch(batch_size)

    k = min(n_add, batch_size)
    assert s.shape == (k, 24)
    assert a.shape == (k, 2)
    assert r.shape == (k, 1)
    assert t.shape == (k, 1)
    assert s2.shape == (k, 24)
