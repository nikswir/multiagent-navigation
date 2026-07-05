"""Train the shared TD3 policy and dump artifacts for the report.

Runs the curriculum training session that produced the curated
report/assets/TD3_simpleEnv.json log (10^6 timesteps, active robots ramping
1 -> 8), saves the final actor/critic checkpoint, dumps the evaluation log,
and rolls out one greedy multi-robot episode to capture a trajectory.
Curated outputs land in report/assets/.

    uv run python report/scripts/run_experiment.py
"""

from __future__ import annotations

import os
import json
import torch
import random
import numpy as np

from pathlib import Path

from multiagent_navigation.lib import train
from multiagent_navigation.agent import TD3
from multiagent_navigation.config_schema import Config
from multiagent_navigation.environment import SimpleEnv

SEED = 10
N_TRAJ_ROBOTS = 4
MAX_STEPS = 500

# The curated log's curriculum: the active-robot count ramps 1 -> CAP over
# CURRICULUM_STEPS timesteps (the config defaults ship a lighter 4 / 400k
# ramp; the report run overrode them to the values recovered from the log's
# Avg_N_robots column, which steps 1..8 exactly at epochs 20, 40, ..., 140).
CURRICULUM_CAP = 8
CURRICULUM_STEPS = 800_000

ROOT = Path(__file__).parents[2]
ASSETS = Path(__file__).parents[1] / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

OUTCOMES = {
    "target_reached": "arrived",
    "collision": "collision",
    "max_steps_reached": "timeout",
}

########################################
#                Device                #
########################################


def select_device() -> torch.device:
    # MAN_DEVICE overrides autodetection (e.g. MAN_DEVICE=cpu to dodge slow
    # MPS host-device synchronisation on the small per-step batches).
    forced = os.environ.get("MAN_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


########################################
#               Rollout                #
########################################


def greedy_trajectory(agent: TD3, cfg: Config) -> dict:
    """Roll out one deterministic (no-noise) multi-robot episode."""
    env = SimpleEnv(
        world_width=cfg.env.world_width,
        world_height=cfg.env.world_height,
        environment_dim=cfg.env.environment_dim,
        robot_radius=cfg.env.robot_radius,
        max_steps=MAX_STEPS,
        n_robots=N_TRAJ_ROBOTS,
        max_robots=cfg.env.max_robots,
        time_delta=cfg.env.time_delta,
        goal_reached_dist=cfg.env.goal_reached_dist,
        lidar_max_range=cfg.env.lidar_max_range,
        obstacle_definitions=[list(o) for o in cfg.env.obstacle_definitions],
    )
    state = env.reset(n_robots=N_TRAJ_ROBOTS)
    xs = [[env.robots[i].x] for i in range(env.n_robots)]
    ys = [[env.robots[i].y] for i in range(env.n_robots)]
    thetas = [[env.robots[i].theta] for i in range(env.n_robots)]

    # ── Shared policy per robot until every robot settles ──
    while sum(env.episode_done) < env.n_robots:
        alive = [not done for done in env.episode_done]
        action = np.zeros((env.n_robots, cfg.model.action_dim))
        for i, robot_state in enumerate(state):
            action[i] = agent.get_action(np.array(robot_state))
        action[:, 0] = (action[:, 0] + 1) / 2
        state, _rewards, _dones, _infos = env.step(action)
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
                "outcome": OUTCOMES.get(str(info.get("reason")), "timeout"),
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
#             Entry point              #
########################################


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = select_device()
    print(f"device = {device}")

    # ── The run config: package defaults + the report run's curriculum ──
    cfg = Config()
    cfg.train.n_robots = CURRICULUM_CAP
    cfg.train.max_robots_timestamp = CURRICULUM_STEPS

    # ── Train; per-epoch checkpoints land in the gitignored run dirs ──
    result = train(
        cfg,
        device=device,
        results_dir=ROOT / "results",
        models_dir=ROOT / "pytorch_models",
    )

    # ── Curated copies for the report pipeline (checkpoint + log) ──
    result.agent.save(cfg.train.file_name, ASSETS)
    log_path = ASSETS / f"{cfg.train.file_name}.json"
    log_path.write_text(json.dumps(result.evaluations))
    print(f"saved checkpoint + {len(result.evaluations)} eval epochs")

    traj = greedy_trajectory(result.agent, cfg)
    (ASSETS / "trajectory.json").write_text(json.dumps(traj))
    outcomes = [robot["outcome"] for robot in traj["robots"]]
    print(f"greedy rollout outcomes: {outcomes}")


if __name__ == "__main__":
    main()
