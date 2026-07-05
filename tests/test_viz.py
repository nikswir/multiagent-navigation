"""Checkpoint resolution and best-epoch selection in viz (stage 1).

Pins the run-directory resolution order (explicit run_dir, CWD-relative
layout, newest Hydra output run, loud failure) and that `load_best_network`
picks the epoch with the highest Avg_reward and loads exactly its weights.
Headless: the Agg backend is pinned before viz imports pyplot.
"""

from __future__ import annotations

import os

# Pin the headless backend before viz imports pyplot below.
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import torch
import pytest

from pathlib import Path

from multiagent_navigation import viz
from multiagent_navigation.lib import build_agent
from multiagent_navigation.config_schema import Config

CPU = torch.device("cpu")


def _tiny_cfg() -> Config:
    cfg = Config()
    cfg.model.hidden1 = 8
    cfg.model.hidden2 = 12
    cfg.train.file_name = "tiny"
    return cfg


########################################
#          Run-dir resolution          #
########################################


def test_resolve_run_dirs_prefers_explicit_run_dir(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    cfg.animate.run_dir = str(tmp_path / "run")

    results_dir, models_dir = viz.resolve_run_dirs(cfg)

    assert results_dir == tmp_path / "run" / "results"
    assert models_dir == tmp_path / "run" / "pytorch_models"


def test_resolve_run_dirs_falls_back_to_newest_output_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _tiny_cfg()
    old = tmp_path / "outputs" / "2026-01-01" / "10-00-00" / "results"
    new = tmp_path / "outputs" / "2026-01-02" / "10-00-00" / "results"
    for results in (old, new):
        results.mkdir(parents=True)
        (results / "tiny").write_text("[]")

    results_dir, models_dir = viz.resolve_run_dirs(cfg)

    assert results_dir.resolve() == new
    assert models_dir.resolve() == new.parent / "pytorch_models"


def test_resolve_run_dirs_errors_without_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        viz.resolve_run_dirs(_tiny_cfg())


########################################
#          Best-epoch loading          #
########################################


def test_load_best_network_picks_argmax_avg_reward(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    cfg.animate.run_dir = str(tmp_path)

    # ── A three-epoch log whose best-by-reward is epoch 2 ──
    rows = [
        {
            "Epoch": 1,
            "Avg_reward": 1.0,
            "Avg_arrived": 0.1,
            "Avg_collision": 0.2,
        },
        {
            "Epoch": 2,
            "Avg_reward": 5.0,
            "Avg_arrived": 0.9,
            "Avg_collision": 0.0,
        },
        {
            "Epoch": 3,
            "Avg_reward": 2.0,
            "Avg_arrived": 0.5,
            "Avg_collision": 0.1,
        },
    ]
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "tiny").write_text(json.dumps(rows))

    saved = build_agent(cfg, CPU)
    (tmp_path / "pytorch_models").mkdir()
    saved.save("tiny_epoch-2", tmp_path / "pytorch_models")

    loaded = viz.load_best_network(cfg, CPU)

    pairs = zip(
        saved.actor.parameters(),
        loaded.actor.parameters(),
        strict=True,
    )
    for expected, actual in pairs:
        assert torch.equal(expected, actual)
