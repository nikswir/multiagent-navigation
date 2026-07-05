"""Library API — the TD3 training loop, evaluation and the result type.

Turns a composed `Config` into a trained shared policy. A curriculum
(`curriculum_cap`) ramps the active robot count from 1 toward
`cfg.train.n_robots` over `max_robots_timestamp` timesteps; exploration noise
anneals linearly (`exploration_noise`) from its configured initial value to
`expl_min` over `expl_decay_steps`; every `eval_freq` timesteps the policy is
evaluated, checkpointed to `{models_dir}/{file_name}_epoch-{epoch}` and the
evaluation log is dumped to `{results_dir}/{file_name}` (the same contract
`viz` reads); the final weights are also saved under the plain `file_name`,
which is what `load_model` resumes from. The shared builders (`make_env`,
`build_agent`, `select_device`) are the single place an env / agent shell /
device is derived from a config. Independent of Hydra and the CLI — `run.py`
composes a config, picks a device and output dirs, and calls in here.
`__init__` re-exports this surface as the public API.
"""

from __future__ import annotations

import os
import json
import torch
import numpy as np

from typing import Any
from pathlib import Path
from dataclasses import dataclass

from multiagent_navigation.agent import TD3
from multiagent_navigation.config_schema import Config
from multiagent_navigation.replay_buffer import ReplayBuffer
from multiagent_navigation.environment import SimpleEnv, OUTCOME_LABELS

########################################
#           Public contract            #
########################################

# ── `random_near_obstacle` knobs: trigger chance, laser gate (metres,
#    body-relative), burst length range ──
RAND_ACTION_TRIGGER = 0.85
RAND_ACTION_MIN_LASER = 0.6
RAND_ACTION_STEPS = (8, 15)


@dataclass
class TrainResult:
    """What a training run produces — the public result type."""

    agent: TD3
    evaluations: list[dict[str, float]]


########################################
#          Device & builders           #
########################################


def select_device() -> torch.device:
    """The run device: MAN_DEVICE override, else cuda -> mps -> cpu.

    The one device-selection policy, shared by every entry point (`run`,
    `viz`, the report scripts) so MAN_DEVICE behaves the same everywhere.
    """
    forced = os.environ.get("MAN_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_env(
    cfg: Config,
    *,
    n_robots: int,
    max_steps: int | None = None,
    seed: int | None = None,
) -> SimpleEnv:
    """A `SimpleEnv` from the config's env group — the ONE construction site.

    `max_steps` overrides `cfg.env.max_steps` for callers that replay short
    episodes (animation, GIFs); everything else always comes from the config.
    """
    return SimpleEnv(
        world_width=cfg.env.world_width,
        world_height=cfg.env.world_height,
        environment_dim=cfg.env.environment_dim,
        robot_radius=cfg.env.robot_radius,
        max_steps=cfg.env.max_steps if max_steps is None else max_steps,
        n_robots=n_robots,
        max_robots=cfg.env.max_robots,
        time_delta=cfg.env.time_delta,
        goal_reached_dist=cfg.env.goal_reached_dist,
        lidar_max_range=cfg.env.lidar_max_range,
        obstacle_definitions=[list(o) for o in cfg.env.obstacle_definitions],
        seed=seed,
    )


def build_agent(cfg: Config, device: torch.device) -> TD3:
    """A TD3 shell shaped by the config (state dim is DERIVED: lidar + 4)."""
    state_dim = cfg.env.environment_dim + 4
    return TD3(
        state_dim,
        cfg.model.action_dim,
        cfg.model.max_action,
        device=device,
        actor_lr=cfg.model.actor_lr,
        critic_lr=cfg.model.critic_lr,
        hidden1=cfg.model.hidden1,
        hidden2=cfg.model.hidden2,
    )


def load_agent(
    cfg: Config,
    device: torch.device,
    directory: str | Path,
    filename: str | None = None,
) -> TD3:
    """Rebuild the network shell and load `{filename}` from `directory`,
    ready for deterministic evaluation (actor in eval mode)."""
    network = build_agent(cfg, device)
    network.load(filename or cfg.train.file_name, directory)
    network.actor.eval()
    return network


########################################
#              Schedules               #
########################################


def curriculum_cap(max_robots: int, timestep: int, ramp_steps: int) -> int:
    """The active-robot cap at `timestep`: 1 at t=0, `max_robots` from
    ramp completion on — the ONE curriculum formula, shared by training
    (which draws uniformly from [1, cap]) and evaluation (which uses the
    cap directly)."""
    return min(max_robots, int(1 + max_robots * timestep / ramp_steps))


def exploration_noise(
    initial: float,
    floor: float,
    decay_steps: int,
    timestep: int,
) -> float:
    """Linear anneal from `initial` at t=0 to `floor` at `decay_steps`."""
    if decay_steps <= 0:
        return floor
    frac = min(1.0, timestep / decay_steps)
    return initial - (initial - floor) * frac


########################################
#               Rollout                #
########################################


def greedy_rollout(
    network: TD3,
    env: SimpleEnv,
    n_robots: int,
) -> dict[str, Any]:
    """One deterministic episode: per-robot paths, outcomes and layout.

    The shared trajectory format consumed by the report figure scripts:
    per-robot xs/ys/thetas (poses while alive), goal and outcome label,
    plus the layout geometry the figures redraw.
    """
    state = env.reset(n_robots=n_robots)
    xs = [[env.robots[i].x] for i in range(env.n_robots)]
    ys = [[env.robots[i].y] for i in range(env.n_robots)]
    thetas = [[env.robots[i].theta] for i in range(env.n_robots)]

    # ── Batched shared policy until every robot settles ──
    while not all(env.episode_done):
        alive = [not done for done in env.episode_done]
        state, _rewards, _dones, _infos = env.step(
            network.get_actions(np.stack(state)),
        )
        for i in range(env.n_robots):
            if alive[i]:
                xs[i].append(env.robots[i].x)
                ys[i].append(env.robots[i].y)
                thetas[i].append(env.robots[i].theta)

    # ── Per-robot paths + the layout geometry the figures need ──
    robots = []
    for i in range(env.n_robots):
        info = env.info[i] or {}
        robots.append(
            {
                "xs": xs[i],
                "ys": ys[i],
                "thetas": thetas[i],
                "goal": [float(env.goals_x[i]), float(env.goals_y[i])],
                "outcome": OUTCOME_LABELS.get(
                    str(info.get("reason")),
                    "timeout",
                ),
            },
        )
    return {
        "robots": robots,
        "obstacles": [
            {"x": o.x, "y": o.y, "width": o.width, "height": o.height}
            for o in env.world.obstacles
        ],
        "width": env.world.width,
        "height": env.world.height,
        "goal_threshold": env.goal_reached_dist,
    }


########################################
#              Evaluation              #
########################################


def evaluate(
    network: TD3,
    env: SimpleEnv,
    epoch: int,
    n_robots: int,
    eval_episodes: int = 10,
) -> dict[str, float]:
    """Average reward / arrival / collision rates over fresh episodes.

    Terminal outcomes are mutually exclusive and follow the env's own
    priority: a robot that reaches its goal while touching another robot
    counts as an arrival (it was paid +100), never as a collision too —
    so arrived + collision + timeout always sums to 1.
    """
    avg_reward = 0.0
    avg_col = 0.0
    avg_arrived = 0.0

    for _ in range(eval_episodes):
        state = env.reset(n_robots=n_robots)

        while not all(env.episode_done):
            # ── Batched shared policy; env remaps throttles itself ──
            alive = [not done for done in env.episode_done]
            action = network.get_actions(np.stack(state))
            state, rewards, dones, infos = env.step(action)

            # ── Aggregate over robots that were alive this step ──
            for i in range(env.n_robots):
                if not alive[i]:
                    continue
                avg_reward += rewards[i] / (eval_episodes * env.n_robots)
                info = infos[i]
                if dones[i] and info is not None:
                    if info["target_reached"]:
                        avg_arrived += 1 / (eval_episodes * env.n_robots)
                    elif info["collision"]:
                        avg_col += 1 / (eval_episodes * env.n_robots)

    print("..............................................")
    print(
        f"Average Reward over {eval_episodes} Evaluation Episodes, "
        f"Epoch {epoch}: {avg_reward:f}, "
        f"Avg collisions: {avg_col:f} , "
        f"Avg arrived: {avg_arrived:f} , "
        f"Avg N_robots: {n_robots:f}",
    )
    print("..............................................")

    return {
        "Epoch": epoch,
        "Avg_N_robots": float(n_robots),
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

    # ── Determinism: torch / numpy globals here, env & buffer own theirs ──
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    # ── Build the world (capacity), the agent and the replay buffer ──
    env = make_env(cfg, n_robots=cfg.train.n_robots, seed=cfg.train.seed)
    network = build_agent(cfg, dev)
    replay_buffer = ReplayBuffer(cfg.train.buffer_size, cfg.train.seed)

    # ── Resume from `{file_name}` when asked; a broken checkpoint is an
    #    error, never a silent fresh start ──
    if cfg.train.load_model:
        network.load(cfg.train.file_name, models_path)

    # ── Loop state ──
    evaluations: list[dict[str, float]] = []
    timestep = 0
    timesteps_since_eval = 0
    episode_timesteps = 0
    epoch = 1
    count_rand_actions = np.zeros(cfg.env.max_robots, dtype=int)
    random_actions = np.zeros((cfg.env.max_robots, cfg.model.action_dim))

    def draw_n_robots() -> int:
        cap = curriculum_cap(
            cfg.train.n_robots,
            timestep,
            cfg.train.max_robots_timestamp,
        )
        return int(np.random.randint(1, cap + 1))

    state = env.reset(n_robots=draw_n_robots())

    while timestep < cfg.train.max_timesteps:
        # ── 1. Query the shared policy and add exploration noise ──
        expl_noise = exploration_noise(
            cfg.train.expl_noise,
            cfg.train.expl_min,
            cfg.train.expl_decay_steps,
            timestep,
        )
        action = network.get_actions(np.stack(state))
        action = (
            action
            + np.random.normal(
                0,
                expl_noise,
                size=(env.n_robots, cfg.model.action_dim),
            )
        ).clip(-cfg.model.max_action, cfg.model.max_action)

        # ── 2. Optional forced random bursts near obstacles, per robot ──
        if cfg.train.random_near_obstacle:
            for i in range(env.n_robots):
                min_laser = (
                    float(min(state[i][: cfg.env.environment_dim]))
                    * cfg.env.lidar_max_range
                )
                if (
                    np.random.uniform(0, 1) > RAND_ACTION_TRIGGER
                    and min_laser < RAND_ACTION_MIN_LASER
                    and count_rand_actions[i] < 1
                ):
                    count_rand_actions[i] = np.random.randint(
                        *RAND_ACTION_STEPS,
                    )
                    random_actions[i] = np.random.uniform(-1, 1, 2)
                    random_actions[i][0] = -1  # brake while steering
                if count_rand_actions[i] > 0:
                    count_rand_actions[i] -= 1
                    action[i] = random_actions[i]

        # ── 3. Step the env (it remaps throttles itself) ──
        alive = [not done for done in env.episode_done]
        next_state, rewards, dones, infos = env.step(action)

        # ── 4. Store one transition per robot alive this step; a timeout
        #    is truncation, not a terminal — its bootstrap stays on ──
        for i in range(env.n_robots):
            if not alive[i]:
                continue
            info = infos[i]
            timeout = (
                info is not None and info.get("reason") == "max_steps_reached"
            )
            done_bool = int(dones[i] and not timeout)
            replay_buffer.add(
                state[i],
                action[i],
                rewards[i],
                done_bool,
                next_state[i],
            )

        state = next_state
        episode_timesteps += 1
        timestep += 1
        timesteps_since_eval += 1

        # ── 5. On episode end: train, maybe evaluate/checkpoint, reset ──
        if all(dones):
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
                        n_robots=curriculum_cap(
                            cfg.train.n_robots,
                            timestep,
                            cfg.train.max_robots_timestamp,
                        ),
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
                    fh.write("\n")
                epoch += 1

            state = env.reset(n_robots=draw_n_robots())
            episode_timesteps = 0

    # ── Final weights under the plain name — the resume/report artifact ──
    if cfg.train.save_model:
        network.save(cfg.train.file_name, directory=models_path)

    # ── Report the best epoch by average evaluation reward ──
    if evaluations:
        best = max(evaluations, key=lambda e: e["Avg_reward"])
        print(f"Training is finished. Best epoch is: {best}")

    return TrainResult(agent=network, evaluations=evaluations)
