"""Smoke tests of the TD3 update on tiny CPU networks (stage 1).

Two update iterations with miniature hidden sizes must survive end-to-end
(sampling, twin-critic update, delayed actor update, Polyak soft update) and
hand back a usable bounded policy; the soft update itself is pinned exactly,
and a tiny `train()` run pins the config wiring and log/checkpoint contract.
"""

from __future__ import annotations

import json
import random

import torch
import numpy as np

from pathlib import Path

from multiagent_navigation import train, Config

from multiagent_navigation.agent import TD3
from multiagent_navigation.replay_buffer import ReplayBuffer

CPU = torch.device("cpu")
STATE_DIM = 24
ACTION_DIM = 2


def _tiny_agent(seed: int = 0) -> TD3:
    torch.manual_seed(seed)
    return TD3(
        STATE_DIM,
        ACTION_DIM,
        max_action=1.0,
        device=CPU,
        hidden1=16,
        hidden2=16,
    )


def _filled_buffer(n: int = 32) -> ReplayBuffer:
    """A buffer of right-shaped random transitions."""
    buffer = ReplayBuffer(buffer_size=100, random_seed=0)
    rng = np.random.default_rng(0)
    for _ in range(n):
        buffer.add(
            rng.normal(size=STATE_DIM),
            rng.uniform(-1, 1, size=ACTION_DIM),
            float(rng.normal()),
            int(rng.integers(0, 2)),
            rng.normal(size=STATE_DIM),
        )
    return buffer


########################################
#            Initialization            #
########################################


def test_targets_hard_copied_at_init() -> None:
    agent = _tiny_agent()

    pairs = list(
        zip(
            agent.actor.parameters(),
            agent.actor_target.parameters(),
            strict=True,
        ),
    ) + list(
        zip(
            agent.critic.parameters(),
            agent.critic_target.parameters(),
            strict=True,
        ),
    )
    for param, target_param in pairs:
        assert torch.equal(param, target_param)


########################################
#             Update smoke             #
########################################


def test_train_smoke_and_bounded_action() -> None:
    agent = _tiny_agent()
    buffer = _filled_buffer()

    # ── Two tiny iterations: batch 8, CPU, delayed actor update ──
    agent.train(buffer, iterations=2, batch_size=8, policy_freq=2)

    action = agent.get_action(np.zeros(STATE_DIM))
    assert action.shape == (ACTION_DIM,)
    assert np.isfinite(action).all()
    assert bool((np.abs(action) <= 1.0).all())
    assert agent.iter_count == 1


def test_soft_update_moves_targets_by_tau() -> None:
    agent = _tiny_agent(1)
    buffer = _filled_buffer()
    tau = 0.05

    targets_before = [
        p.detach().clone() for p in agent.critic_target.parameters()
    ]
    agent.train(buffer, iterations=1, batch_size=8, tau=tau, policy_freq=1)

    # ── target' == tau * online + (1 - tau) * target, exactly ──
    triples = zip(
        targets_before,
        agent.critic.parameters(),
        agent.critic_target.parameters(),
        strict=True,
    )
    for before, online, after in triples:
        expected = tau * online.detach() + (1 - tau) * before
        assert torch.allclose(after.detach(), expected, atol=1e-6)


########################################
#            Training entry            #
########################################


def test_train_wires_config_and_writes_artifacts(tmp_path: Path) -> None:
    """A tiny `train()` run returns evaluations and writes its artifacts."""
    random.seed(0)

    # ── Tiny but complete: 2-step episodes with eval every 2 timesteps, so
    #    6 timesteps are enough to train, evaluate and checkpoint ──
    cfg = Config()
    cfg.train.seed = 0
    cfg.train.max_timesteps = 6
    cfg.train.eval_freq = 2
    cfg.train.eval_ep = 1
    cfg.train.max_ep = 3
    cfg.train.n_robots = 1
    cfg.train.buffer_size = 50
    cfg.train.file_name = "tiny"
    cfg.model.hidden1 = 8
    cfg.model.hidden2 = 12
    cfg.model.batch_size = 4
    cfg.env.max_steps = 2
    cfg.env.max_robots = 2

    result = train(
        cfg,
        device=CPU,
        results_dir=tmp_path / "results",
        models_dir=tmp_path / "models",
    )

    # ── The actor's shape reflects the config (state -> h1 -> h2 -> action),
    #    so a swapped or dropped kwarg between train() and TD3 fails here ──
    state_dim = cfg.env.environment_dim + 4
    assert result.agent.actor.layer_1.in_features == state_dim
    assert result.agent.actor.layer_1.out_features == cfg.model.hidden1
    assert result.agent.actor.layer_2.out_features == cfg.model.hidden2
    assert result.agent.actor.layer_3.out_features == cfg.model.action_dim

    # ── At least one evaluation epoch, each row with the full log schema ──
    keys = {
        "Epoch",
        "Avg_N_robots",
        "Avg_reward",
        "Avg_arrived",
        "Avg_collision",
        "Avg_timeout",
    }
    assert len(result.evaluations) >= 1
    assert all(set(row) == keys for row in result.evaluations)

    # ── The eval log and the epoch checkpoint land where viz expects ──
    log = json.loads((tmp_path / "results" / "tiny").read_text())
    assert isinstance(log, list)
    assert len(log) >= 1
    assert set(log[0]) == keys
    assert (tmp_path / "models" / "tiny_epoch-1_actor.pth").exists()
    assert (tmp_path / "models" / "tiny_epoch-1_critic.pth").exists()
