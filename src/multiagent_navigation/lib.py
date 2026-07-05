"""Library API — the TD3 training loop, evaluation and the result type.

Turns a composed `Config` into a trained shared policy. A curriculum ramps the
active robot count from 1 toward `cfg.train.n_robots` over
`max_robots_timestamp` timesteps; exploration noise decays at runtime from its
configured initial value toward `expl_min`; every `eval_freq` timesteps the
policy is evaluated, checkpointed to `{models_dir}/{file_name}_epoch-{epoch}`
and the evaluation log is dumped to `{results_dir}/{file_name}` (the same
contract `viz` reads). Independent of Hydra and the CLI — `run.py` composes a
config, picks a device and output dirs, and calls in here. `__init__`
re-exports this surface as the public API.
"""

from __future__ import annotations

import json
import torch
import numpy as np

from copy import copy
from pathlib import Path
from dataclasses import dataclass

from multiagent_navigation.agent import TD3
from multiagent_navigation.config_schema import Config
from multiagent_navigation.environment import SimpleEnv
from multiagent_navigation.replay_buffer import ReplayBuffer

########################################
#           Public contract            #
########################################

# ── `random_near_obstacle` knobs: trigger chance, laser gate, burst ──
RAND_ACTION_TRIGGER = 0.85
RAND_ACTION_MIN_LASER = 0.6
RAND_ACTION_STEPS = (8, 15)


@dataclass
class TrainResult:
    """What a training run produces — the public result type."""

    agent: TD3
    evaluations: list[dict[str, float]]


########################################
#              Evaluation              #
########################################


def evaluate(
    network: TD3,
    env: SimpleEnv,
    epoch: int,
    n_robots_rate: float,
    max_robots: int,
    max_ep: int,
    action_dim: int = 2,
    eval_episodes: int = 10,
) -> dict[str, float]:
    """Average reward / arrival / collision rates over fresh episodes."""
    avg_reward = 0.0
    avg_col = 0.0
    avg_arrived = 0.0
    avg_n_robots = 0.0

    for _ in range(eval_episodes):
        episode_timesteps = 0
        n_robots = min(max_robots, int(1 + max_robots * n_robots_rate))
        avg_n_robots += n_robots / eval_episodes
        state = env.reset(n_robots=n_robots)
        dones: list[bool] = []

        while sum(dones) < env.n_robots:
            # ── Shared policy per robot; throttle remapped to [0, 1] ──
            action = np.zeros((env.n_robots, action_dim))
            for i, robot_state in enumerate(state):
                action[i] = network.get_action(np.array(robot_state))
            action[:, 0] = (action[:, 0] + 1) / 2

            prev_dones = copy(dones)
            state, rewards, dones, infos = env.step(action)

            # ── Aggregate over robots that were alive this step ──
            for i in range(env.n_robots):
                if episode_timesteps == 0 or not prev_dones[i]:
                    dones[i] = (
                        True
                        if episode_timesteps + 1 == max_ep
                        else bool(dones[i])
                    )
                    avg_reward += rewards[i] / (eval_episodes * env.n_robots)
                    info = infos[i]
                    if info is not None and info["collision"]:
                        avg_col += 1 / (eval_episodes * env.n_robots)
                    if info is not None and info["target_reached"]:
                        avg_arrived += 1 / (eval_episodes * env.n_robots)
                else:
                    continue

            episode_timesteps += 1

    print("..............................................")
    print(
        f"Average Reward over {eval_episodes} Evaluation Episodes, "
        f"Epoch {epoch}: {avg_reward:f}, "
        f"Avg collisions: {avg_col:f} , "
        f"Avg arrived: {avg_arrived:f} , "
        f"Avg N_robots: {avg_n_robots:f}",
    )
    print("..............................................")

    return {
        "Epoch": epoch,
        "Avg_N_robots": avg_n_robots,
        "Avg_reward": avg_reward,
        "Avg_arrived": avg_arrived,
        "Avg_collision": avg_col,
        "Avg_timeout": 1 - avg_col - avg_arrived,
    }


########################################
#            Training entry            #
########################################


def train(
    cfg: Config,
    *,
    device: torch.device | None = None,
    results_dir: str | Path = "results",
    models_dir: str | Path = "pytorch_models",
) -> TrainResult:
    """Build env / agent / buffer from `cfg`, train, log and checkpoint."""

    # ── Resolve device and output dirs (created at run time only) ──
    dev = device if device is not None else torch.device("cpu")
    results_path = Path(results_dir)
    models_path = Path(models_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    if cfg.train.save_model:
        models_path.mkdir(parents=True, exist_ok=True)

    # ── Determinism: seed the RNGs the run touches ──
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    # ── Build the world (capacity), the agent and the replay buffer ──
    state_dim = cfg.env.environment_dim + 4
    env = SimpleEnv(
        world_width=cfg.env.world_width,
        world_height=cfg.env.world_height,
        environment_dim=cfg.env.environment_dim,
        robot_radius=cfg.env.robot_radius,
        max_steps=cfg.env.max_steps,
        n_robots=cfg.train.n_robots,
        max_robots=cfg.env.max_robots,
        time_delta=cfg.env.time_delta,
        goal_reached_dist=cfg.env.goal_reached_dist,
        lidar_max_range=cfg.env.lidar_max_range,
        obstacle_definitions=[list(o) for o in cfg.env.obstacle_definitions],
    )
    network = TD3(
        state_dim,
        cfg.model.action_dim,
        cfg.model.max_action,
        device=dev,
        actor_lr=cfg.model.actor_lr,
        critic_lr=cfg.model.critic_lr,
        hidden1=cfg.model.hidden1,
        hidden2=cfg.model.hidden2,
    )
    replay_buffer = ReplayBuffer(cfg.train.buffer_size, cfg.train.seed)

    if cfg.train.load_model:
        try:
            network.load(cfg.train.file_name, models_path)
        except (OSError, RuntimeError):
            print(
                "Could not load the stored model parameters, "
                "initializing training with random parameters",
            )

    # ── Loop state: exploration noise decays from its initial value ──
    evaluations: list[dict[str, float]] = []
    expl_noise = cfg.train.expl_noise
    state: list[np.ndarray] = []
    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_timesteps = 0
    dones: list[bool] = []
    epoch = 1
    count_rand_actions = 0
    random_action = np.zeros(cfg.model.action_dim)
    n_robots = 0

    while timestep < cfg.train.max_timesteps:
        # ── On episode end: train, maybe evaluate/checkpoint, reset ──
        if sum(dones) == n_robots:
            if timestep != 0:
                network.train(
                    replay_buffer,
                    episode_timesteps,
                    batch_size=cfg.model.batch_size,
                    discount=cfg.model.discount,
                    tau=cfg.model.tau,
                    policy_noise=cfg.model.policy_noise,
                    noise_clip=cfg.model.noise_clip,
                    policy_freq=cfg.model.policy_freq,
                )

            if timesteps_since_eval >= cfg.train.eval_freq:
                print("Validating")
                timesteps_since_eval %= cfg.train.eval_freq
                evaluations.append(
                    evaluate(
                        network,
                        env,
                        epoch=epoch,
                        n_robots_rate=(
                            timestep / cfg.train.max_robots_timestamp
                        ),
                        max_robots=cfg.train.n_robots,
                        max_ep=cfg.train.max_ep,
                        action_dim=cfg.model.action_dim,
                        eval_episodes=cfg.train.eval_ep,
                    ),
                )
                if cfg.train.save_model:
                    network.save(
                        f"{cfg.train.file_name}_epoch-{epoch}",
                        directory=models_path,
                    )
                with open(results_path / cfg.train.file_name, "w") as fh:
                    json.dump(evaluations, fh)
                epoch += 1

            # ── Curriculum: ramp active robots toward n_robots ──
            n_robots = np.random.randint(
                1,
                min(
                    cfg.train.n_robots + 1,
                    int(
                        1
                        + cfg.train.n_robots
                        * timestep
                        / cfg.train.max_robots_timestamp,
                    )
                    + 1,
                ),
            )
            state = env.reset(n_robots=n_robots)
            episode_timesteps = 0
            episode_num += 1

        # ── Exploration noise decay toward expl_min ──
        if expl_noise > cfg.train.expl_min:
            expl_noise = expl_noise - (
                (1 - cfg.train.expl_min) / cfg.train.expl_decay_steps
            )

        # ── Query the shared policy and add exploration noise ──
        action = np.zeros((env.n_robots, cfg.model.action_dim))
        for i, robot_state in enumerate(state):
            action[i] = network.get_action(np.array(robot_state))
        action = (
            action
            + np.random.normal(
                0,
                expl_noise,
                size=(env.n_robots, cfg.model.action_dim),
            )
        ).clip(-cfg.model.max_action, cfg.model.max_action)

        # ── Optional forced random bursts near obstacles ──
        if cfg.train.random_near_obstacle:
            if (
                np.random.uniform(0, 1) > RAND_ACTION_TRIGGER
                and min(state[4:-8]) < RAND_ACTION_MIN_LASER
                and count_rand_actions < 1
            ):
                count_rand_actions = np.random.randint(*RAND_ACTION_STEPS)
                random_action = np.random.uniform(-1, 1, 2)

            if count_rand_actions > 0:
                count_rand_actions -= 1
                action = random_action
                action[0] = -1

        # ── Step the env: throttle remapped to [0, 1] for the world ──
        action_in = action.copy()
        action_in[:, 0] = (action_in[:, 0] + 1) / 2
        prev_dones = copy(dones)
        next_state, rewards, dones, infos = env.step(action_in)

        # ── Store one transition per robot that was alive this step ──
        for i in range(env.n_robots):
            if episode_timesteps == 0 or not prev_dones[i]:
                done_bool = (
                    0
                    if episode_timesteps + 1 == cfg.train.max_ep
                    else int(dones[i])
                )
                dones[i] = (
                    True
                    if episode_timesteps + 1 == cfg.train.max_ep
                    else bool(dones[i])
                )
                replay_buffer.add(
                    state[i],
                    action[i],
                    rewards[i],
                    done_bool,
                    next_state[i],
                )
            else:
                continue

        state = next_state
        episode_timesteps += 1
        timestep += 1
        timesteps_since_eval += 1

    # ── Report the best epoch by average evaluation reward ──
    if evaluations:
        best = max(evaluations, key=lambda e: e["Avg_reward"])
        print(f"Training is finished. Best epoch is: {best}")

    return TrainResult(agent=network, evaluations=evaluations)
