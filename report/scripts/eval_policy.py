"""Evaluate the trained shared policy over fresh multi-robot episodes.

Loads the published checkpoint from model/ and rolls out the deterministic
policy on freshly sampled episodes (8 robots each — the curated run's
curriculum cap) to estimate per-robot arrival / collision / timeout rates.
Saves the fastest all-arrived episode as trajectory_success.json and
overwrites trajectory.json with a collision episode, for the report's
success/failure figure.

    uv run python report/scripts/eval_policy.py
"""

from __future__ import annotations

import json
import torch
import numpy as np

from pathlib import Path

from multiagent_navigation.agent import TD3
from multiagent_navigation.config_schema import Config

from multiagent_navigation.lib import (
    make_env,
    load_agent,
    select_device,
    greedy_rollout,
)

SEED = 11
N_ROBOTS = 8
N_EPISODES = 100

ASSETS = Path(__file__).parents[1] / "assets"
MODEL_DIR = Path(__file__).parents[2] / "model"

########################################
#              Checkpoint              #
########################################


def load_report_agent(cfg: Config, device: torch.device) -> TD3:
    """The published model/ checkpoint, with a friendly missing-file hint."""
    name = cfg.train.file_name
    if not (MODEL_DIR / f"{name}_actor.pth").exists():
        raise SystemExit(
            f"missing checkpoint {MODEL_DIR / name}_actor.pth — train with "
            "`uv run python report/scripts/run_experiment.py` and copy the "
            "chosen epoch's weights into model/",
        )
    return load_agent(cfg, device, MODEL_DIR)


########################################
#             Entry point              #
########################################


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = select_device()
    cfg = Config()
    agent = load_report_agent(cfg, device)
    env = make_env(cfg, n_robots=N_ROBOTS, seed=SEED)

    # ── Roll out; keep the fastest all-arrived and one collision episode ──
    counts = {"arrived": 0, "collision": 0, "timeout": 0}
    best = None
    best_steps = float("inf")
    worst = None
    for _ in range(N_EPISODES):
        traj = greedy_rollout(agent, env, n_robots=N_ROBOTS)
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
    (ASSETS / "eval_stats.json").write_text(json.dumps(stats) + "\n")

    if best is not None:
        (ASSETS / "trajectory_success.json").write_text(
            json.dumps(best) + "\n",
        )
        print(f"saved all-arrived episode ({best_steps} steps)")
    else:
        print("no all-arrived episode captured")

    if worst is not None:
        (ASSETS / "trajectory.json").write_text(json.dumps(worst) + "\n")
        print("saved collision episode")


if __name__ == "__main__":
    main()
