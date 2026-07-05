"""TD3 update tests on tiny CPU networks (stage 1).

Pins the algorithm, not just the plumbing: every critic parameter receives
gradients (a `.data`-style detach would freeze layers silently), the Bellman
target is exactly r + (1 - done) * discount * min(Q1', Q2'), the delayed
actor update fires on a persistent every-`policy_freq` cadence, the Polyak
soft update matches its formula, and save/load round-trips the weights with
targets hard-synced. A tiny `train()` run pins the config wiring and the
log/checkpoint contract.
"""

from __future__ import annotations

import json

import torch
import pytest
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
    assert agent.total_it == 2


def test_get_actions_batches_match_single_queries() -> None:
    agent = _tiny_agent(6)
    states = np.random.default_rng(6).normal(size=(5, STATE_DIM))

    batched = agent.get_actions(states)

    assert batched.shape == (5, ACTION_DIM)
    for i in range(5):
        single = agent.get_action(states[i])
        assert np.allclose(batched[i], single, atol=1e-6)


########################################
#          Gradients & target          #
########################################


def test_every_critic_parameter_receives_gradients() -> None:
    """One critic update moves EVERY critic parameter — a `.data`-style
    detach freezing the mixing layers would fail here."""
    agent = _tiny_agent(2)
    buffer = _filled_buffer()
    before = [p.detach().clone() for p in agent.critic.parameters()]

    agent.train(buffer, iterations=1, batch_size=16, policy_freq=2)

    pairs = zip(before, agent.critic.parameters(), strict=True)
    for old, new in pairs:
        assert not torch.equal(old, new.detach())


def test_compute_target_q_matches_bellman() -> None:
    """target = r + (1 - done) * discount * min(Q1', Q2'), terminal-masked."""
    agent = _tiny_agent(3)
    torch.manual_seed(3)
    next_state = torch.randn(4, STATE_DIM)
    reward = torch.randn(4, 1)
    done = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    discount = 0.9

    # ── policy_noise=0 makes the smoothed target action deterministic ──
    target, min_q = agent.compute_target_q(
        next_state,
        reward,
        done,
        discount,
        policy_noise=0.0,
        noise_clip=0.5,
    )

    with torch.no_grad():
        next_action = agent.actor_target(next_state)
        q1, q2 = agent.critic_target(next_state, next_action)
        expected_min = torch.min(q1, q2)

    assert torch.allclose(min_q, expected_min, atol=1e-6)
    expected = reward + (1 - done) * discount * expected_min
    assert torch.allclose(target, expected, atol=1e-6)

    # ── The terminal rows bootstrap nothing: target == raw reward ──
    assert torch.allclose(target[1], reward[1], atol=1e-6)
    assert torch.allclose(target[3], reward[3], atol=1e-6)


def test_actor_updates_on_persistent_policy_freq_cadence() -> None:
    """The delayed update counts critic updates ACROSS train() calls."""
    agent = _tiny_agent(4)
    buffer = _filled_buffer()
    before = [p.detach().clone() for p in agent.actor.parameters()]

    # ── total_it = 1: off-cadence, the actor must not move ──
    agent.train(buffer, iterations=1, batch_size=8, policy_freq=2)
    pairs = zip(before, agent.actor.parameters(), strict=True)
    for old, new in pairs:
        assert torch.equal(old, new.detach())

    # ── total_it = 2: the delayed update fires on the next call ──
    agent.train(buffer, iterations=1, batch_size=8, policy_freq=2)
    pairs = zip(before, agent.actor.parameters(), strict=True)
    assert any(not torch.equal(old, new.detach()) for old, new in pairs)


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
#            Checkpointing             #
########################################


def test_save_load_round_trip_syncs_targets(tmp_path: Path) -> None:
    """load() restores the online nets and hard-copies them into targets."""
    agent = _tiny_agent(5)
    buffer = _filled_buffer()
    agent.train(buffer, iterations=3, batch_size=8, policy_freq=1)
    agent.save("ckpt", tmp_path)

    fresh = _tiny_agent(99)
    fresh.load("ckpt", tmp_path)

    # ── Online nets equal the saved ones, bit for bit ──
    for saved, loaded in zip(
        list(agent.actor.parameters()) + list(agent.critic.parameters()),
        list(fresh.actor.parameters()) + list(fresh.critic.parameters()),
        strict=True,
    ):
        assert torch.equal(saved, loaded)

    # ── Targets are hard-synced to the loaded weights, not random ──
    for online, target in zip(
        list(fresh.actor.parameters()) + list(fresh.critic.parameters()),
        list(fresh.actor_target.parameters())
        + list(fresh.critic_target.parameters()),
        strict=True,
    ):
        assert torch.equal(online, target)


def test_load_missing_checkpoint_raises(tmp_path: Path) -> None:
    agent = _tiny_agent(8)
    with pytest.raises(OSError, match="no_such"):
        agent.load("no_such", tmp_path)


########################################
#            Training entry            #
########################################


def _tiny_train_cfg() -> Config:
    """Tiny but complete: 2-step episodes with eval every 2 timesteps, so
    6 timesteps are enough to train, evaluate and checkpoint."""
    cfg = Config()
    cfg.train.seed = 0
    cfg.train.max_timesteps = 6
    cfg.train.eval_freq = 2
    cfg.train.eval_ep = 1
    cfg.train.n_robots = 1
    cfg.train.buffer_size = 50
    cfg.train.file_name = "tiny"
    cfg.model.hidden1 = 8
    cfg.model.hidden2 = 12
    cfg.model.batch_size = 4
    cfg.env.max_steps = 2
    cfg.env.max_robots = 2
    return cfg


def test_train_wires_config_and_writes_artifacts(tmp_path: Path) -> None:
    """A tiny `train()` run returns evaluations and writes its artifacts."""
    cfg = _tiny_train_cfg()

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

    # ── The eval log and the checkpoints land where viz / resume expect ──
    log = json.loads((tmp_path / "results" / "tiny").read_text())
    assert isinstance(log, list)
    assert len(log) >= 1
    assert set(log[0]) == keys
    assert (tmp_path / "models" / "tiny_epoch-1_actor.pth").exists()
    assert (tmp_path / "models" / "tiny_epoch-1_critic.pth").exists()
    assert (tmp_path / "models" / "tiny_actor.pth").exists()
    assert (tmp_path / "models" / "tiny_critic.pth").exists()


def test_train_load_model_resumes_or_fails_loudly(tmp_path: Path) -> None:
    """load_model=true resumes from the final checkpoint — and a missing
    checkpoint is an error, never a silent fresh start."""
    cfg = _tiny_train_cfg()
    train(
        cfg,
        device=CPU,
        results_dir=tmp_path / "results",
        models_dir=tmp_path / "models",
    )

    # ── Resume from the plain-name final checkpoint just written ──
    cfg.train.load_model = True
    result = train(
        cfg,
        device=CPU,
        results_dir=tmp_path / "results",
        models_dir=tmp_path / "models",
    )
    assert result.agent is not None

    # ── An empty models dir must raise, not train from scratch ──
    with pytest.raises(OSError, match="tiny"):
        train(
            cfg,
            device=CPU,
            results_dir=tmp_path / "results2",
            models_dir=tmp_path / "empty_models",
        )
