"""Evaluate the trained shared policy over fresh multi-robot episodes.

Loads the report/assets checkpoint and rolls out the deterministic policy on
freshly sampled episodes (8 robots each — the curated run's curriculum cap)
to estimate per-robot arrival / collision / timeout rates. Saves the fastest
all-arrived episode as trajectory_success.json and overwrites trajectory.json
with a collision episode, for the report's success/failure figure.

    uv run python report/scripts/eval_policy.py
"""

from __future__ import annotations

import os
import json
import torch
import random
import numpy as np

from pathlib import Path

from multiagent_navigation.agent import TD3
from multiagent_navigation.config_schema import Config
from multiagent_navigation.environment import SimpleEnv

SEED = 11
N_ROBOTS = 8
N_EPISODES = 100
MAX_STEPS = 500

ASSETS = Path(__file__).parents[1] / "assets"

OUTCOMES = {
    "target_reached": "arrived",
    "collision": "collision",
    "max_steps_reached": "timeout",
}

########################################
#              Checkpoint              #
########################################


def load_agent(cfg: Config, device: torch.device) -> TD3:
    """Rebuild the network shell and load the report checkpoint."""
    name = cfg.train.file_name
    if not (ASSETS / f"{name}_actor.pth").exists():
        raise SystemExit(
            f"missing checkpoint {ASSETS / name}_actor.pth — run "
            "`uv run python report/scripts/run_experiment.py` first",
        )

    state_dim = cfg.env.environment_dim + 4
    agent = TD3(
        state_dim,
        cfg.model.action_dim,
        cfg.model.max_action,
        device=device,
        hidden1=cfg.model.hidden1,
        hidden2=cfg.model.hidden2,
    )
    agent.load(name, ASSETS)
    agent.actor.eval()
    return agent


########################################
#               Rollout                #
########################################


def rollout(agent: TD3, env: SimpleEnv, cfg: Config) -> dict:
    """One deterministic episode; per-robot paths, outcomes and layout."""
    state = env.reset(n_robots=N_ROBOTS)
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

    device = torch.device(
        os.environ.get(
            "MAN_DEVICE",
            "mps" if torch.backends.mps.is_available() else "cpu",
        ),
    )
    cfg = Config()
    agent = load_agent(cfg, device)

    env = SimpleEnv(
        world_width=cfg.env.world_width,
        world_height=cfg.env.world_height,
        environment_dim=cfg.env.environment_dim,
        robot_radius=cfg.env.robot_radius,
        max_steps=MAX_STEPS,
        n_robots=N_ROBOTS,
        max_robots=cfg.env.max_robots,
        time_delta=cfg.env.time_delta,
        goal_reached_dist=cfg.env.goal_reached_dist,
        lidar_max_range=cfg.env.lidar_max_range,
        obstacle_definitions=[list(o) for o in cfg.env.obstacle_definitions],
    )

    # ── Roll out; keep the fastest all-arrived and one collision episode ──
    counts = {"arrived": 0, "collision": 0, "timeout": 0}
    best = None
    best_steps = float("inf")
    worst = None
    for _ in range(N_EPISODES):
        traj = rollout(agent, env, cfg)
        outcomes = [robot["outcome"] for robot in traj["robots"]]
        for outcome in outcomes:
            counts[outcome] += 1
        steps = max(len(robot["xs"]) for robot in traj["robots"])
        if all(o == "arrived" for o in outcomes) and steps < best_steps:
            best, best_steps = traj, steps
        if worst is None and "collision" in outcomes:
            worst = traj

    n = N_EPISODES * N_ROBOTS
    rate = counts["arrived"] / n
    print(f"per-robot arrival rate: {counts['arrived']}/{n} = {rate:.1%}")

    stats = {
        "n": n,
        "n_episodes": N_EPISODES,
        "n_robots": N_ROBOTS,
        "arrived": counts["arrived"],
        "collision": counts["collision"],
        "timeout": counts["timeout"],
        "rate": rate,
    }
    (ASSETS / "eval_stats.json").write_text(json.dumps(stats))

    if best is not None:
        (ASSETS / "trajectory_success.json").write_text(json.dumps(best))
        print(f"saved all-arrived episode ({best_steps} steps)")
    else:
        print("no all-arrived episode captured")

    if worst is not None:
        (ASSETS / "trajectory.json").write_text(json.dumps(worst))
        print("saved collision episode")


if __name__ == "__main__":
    main()
