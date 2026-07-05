"""Training-loop building blocks: schedules and evaluation math (stage 1).

Pins the ONE curriculum formula's breakpoints, the closed-form exploration
anneal, evaluate()'s outcome bookkeeping on a hand-scripted env (mutually
exclusive arrival/collision counting with arrival winning ties, timeout as
the remainder), and that the `random_near_obstacle` burst branch trains
without crashing in the multi-robot loop.
"""

from __future__ import annotations

import torch
import pytest
import numpy as np

from pathlib import Path

from multiagent_navigation import lib
from multiagent_navigation.agent import TD3
from multiagent_navigation.config_schema import Config

from multiagent_navigation.lib import (
    evaluate,
    curriculum_cap,
    exploration_noise,
)

CPU = torch.device("cpu")


def _tiny_network() -> TD3:
    torch.manual_seed(0)
    return TD3(24, 2, max_action=1.0, device=CPU, hidden1=8, hidden2=8)


########################################
#              Schedules               #
########################################


def test_curriculum_cap_ramp_breakpoints() -> None:
    """1 at t=0, +1 every ramp/max_robots steps, capped at max_robots."""
    assert curriculum_cap(8, 0, 800_000) == 1
    assert curriculum_cap(8, 99_999, 800_000) == 1
    assert curriculum_cap(8, 100_000, 800_000) == 2
    assert curriculum_cap(8, 699_999, 800_000) == 7
    assert curriculum_cap(8, 700_000, 800_000) == 8
    assert curriculum_cap(8, 10**9, 800_000) == 8


def test_exploration_noise_anneals_from_configured_initial() -> None:
    """Linear from the CONFIGURED initial to the floor, then flat."""
    assert exploration_noise(1.0, 0.1, 500_000, 0) == pytest.approx(1.0)
    assert exploration_noise(1.0, 0.1, 500_000, 250_000) == pytest.approx(
        0.55,
    )
    assert exploration_noise(1.0, 0.1, 500_000, 500_000) == pytest.approx(
        0.1,
    )
    assert exploration_noise(1.0, 0.1, 500_000, 10**7) == pytest.approx(0.1)

    # ── A non-default initial anneals over the SAME horizon ──
    assert exploration_noise(0.5, 0.1, 100, 50) == pytest.approx(0.3)
    assert exploration_noise(0.5, 0.1, 0, 0) == pytest.approx(0.1)


########################################
#         Evaluation counting          #
########################################


class _ScriptedEnv:
    """Replays hand-written (rewards, dones, infos) step scripts.

    Outcomes are fixed rather than simulated, so evaluate()'s bookkeeping
    is pinned exactly — including the alive-robot filter and the terminal
    outcome classification.
    """

    def __init__(self, script: list) -> None:
        self.script = script
        self.n_robots = len(script[0][0])
        self.episode_done = [False] * self.n_robots
        self._step = 0

    def reset(self, n_robots: int) -> list:
        self.episode_done = [False] * self.n_robots
        self._step = 0
        return [np.zeros(24) for _ in range(self.n_robots)]

    def step(self, action: np.ndarray) -> tuple:
        rewards, dones, infos = self.script[self._step]
        self._step += 1
        self.episode_done = list(dones)
        states = [np.zeros(24) for _ in range(self.n_robots)]
        return states, list(rewards), list(dones), list(infos)


def test_evaluate_counts_arrival_and_collision_exclusively() -> None:
    """A robot that arrives while touching another counts ONLY as arrived
    (the env pays it +100), so arrived + collision + timeout == 1."""
    script = [
        (
            [100.0, 1.0],
            [True, False],
            [
                {
                    "target_reached": True,
                    "collision": False,
                    "reason": "target_reached",
                },
                {"target_reached": False, "collision": False},
            ],
        ),
        (
            [0.0, 100.0],
            [True, True],
            [
                {
                    "target_reached": True,
                    "collision": False,
                    "reason": "target_reached",
                },
                {
                    "target_reached": True,
                    "collision": True,
                    "reason": "target_reached",
                },
            ],
        ),
    ]

    row = evaluate(
        _tiny_network(),
        _ScriptedEnv(script),  # type: ignore[arg-type]
        epoch=1,
        n_robots=2,
        eval_episodes=1,
    )

    # ── Both robots arrived; the simultaneous touch is NOT a collision ──
    assert row["Avg_arrived"] == pytest.approx(1.0)
    assert row["Avg_collision"] == pytest.approx(0.0)
    assert row["Avg_timeout"] == pytest.approx(0.0)

    # ── Rewards averaged per robot: (100 + 1 + 100) / 2; the frozen
    #    robot's step-2 zero is not double-counted ──
    assert row["Avg_reward"] == pytest.approx(100.5)
    assert row["Avg_N_robots"] == pytest.approx(2.0)


def test_evaluate_timeout_is_the_remainder() -> None:
    script = [
        (
            [1.0, -100.0],
            [False, True],
            [
                {"target_reached": False, "collision": False},
                {
                    "target_reached": False,
                    "collision": True,
                    "reason": "collision",
                },
            ],
        ),
        (
            [0.5, 0.0],
            [True, True],
            [
                {
                    "target_reached": False,
                    "collision": False,
                    "reason": "max_steps_reached",
                },
                {
                    "target_reached": False,
                    "collision": True,
                    "reason": "collision",
                },
            ],
        ),
    ]

    row = evaluate(
        _tiny_network(),
        _ScriptedEnv(script),  # type: ignore[arg-type]
        epoch=1,
        n_robots=2,
        eval_episodes=1,
    )

    assert row["Avg_arrived"] == pytest.approx(0.0)
    assert row["Avg_collision"] == pytest.approx(0.5)
    assert row["Avg_timeout"] == pytest.approx(0.5)
    assert row["Avg_reward"] == pytest.approx((1.0 - 100.0 + 0.5) / 2)


########################################
#         Random-burst branch          #
########################################


def test_random_near_obstacle_branch_trains_without_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The per-robot burst path runs end-to-end in the multi-robot loop."""
    # ── Force the burst to trigger on every step ──
    monkeypatch.setattr(lib, "RAND_ACTION_TRIGGER", -1.0)
    monkeypatch.setattr(lib, "RAND_ACTION_MIN_LASER", float("inf"))

    cfg = Config()
    cfg.train.seed = 0
    cfg.train.max_timesteps = 12
    cfg.train.eval_freq = 10**9
    cfg.train.n_robots = 2
    cfg.train.random_near_obstacle = True
    cfg.train.buffer_size = 50
    cfg.train.save_model = False
    cfg.model.hidden1 = 8
    cfg.model.hidden2 = 8
    cfg.model.batch_size = 4
    cfg.env.max_steps = 3
    cfg.env.max_robots = 2

    result = lib.train(cfg, device=CPU, results_dir=tmp_path / "results")

    assert result.agent is not None
