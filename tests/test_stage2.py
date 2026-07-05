"""Stage-2 (heavy) tests — run only with RUN_STAGE2=1.

The occupants of the two-stage policy's heavy tier: too slow for the
per-commit suite, still CPU-only. Currently one longer end-to-end training
smoke over the real curriculum loop.
"""

from __future__ import annotations

import math

import torch

from pathlib import Path

from tests.conftest import stage2

from multiagent_navigation import train, Config

########################################
#         Long training smoke          #
########################################


@stage2
def test_longer_training_run_produces_sane_evaluations(
    tmp_path: Path,
) -> None:
    """A few thousand timesteps of the real loop: finite rewards, rates
    that stay probabilities, and the curriculum ramp reaching its cap."""
    cfg = Config()
    cfg.train.seed = 1
    cfg.train.max_timesteps = 3000
    cfg.train.eval_freq = 1500
    cfg.train.eval_ep = 2
    cfg.train.n_robots = 2
    cfg.train.max_robots_timestamp = 2000
    cfg.train.buffer_size = 5000
    cfg.train.save_model = False
    cfg.model.hidden1 = 32
    cfg.model.hidden2 = 32
    cfg.model.batch_size = 32
    cfg.env.max_steps = 100
    cfg.env.max_robots = 2

    result = train(
        cfg,
        device=torch.device("cpu"),
        results_dir=tmp_path / "results",
    )

    assert len(result.evaluations) >= 1
    for row in result.evaluations:
        assert math.isfinite(row["Avg_reward"])
        assert 0.0 <= row["Avg_arrived"] <= 1.0
        assert 0.0 <= row["Avg_collision"] <= 1.0
        assert 0.0 <= row["Avg_timeout"] <= 1.0
        assert 1 <= row["Avg_N_robots"] <= cfg.train.n_robots
