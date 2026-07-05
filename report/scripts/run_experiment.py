"""Train the shared TD3 policy and dump artifacts for the report.

Runs the curriculum training session behind the curated
report/assets/TD3_simpleEnv.json log (10^6 timesteps, active robots ramping
1 -> 8), dumps the evaluation log and curates TWO checkpoints: the final
weights (plain `{file_name}`) and the best-by-Avg_reward epoch
(`{file_name}_best`), so both rows of the README results story stay
reproducible. Also rolls out one greedy multi-robot episode to capture a
trajectory. Curated outputs land in report/assets/.

    uv run python report/scripts/run_experiment.py
"""

from __future__ import annotations

import json
import torch
import shutil
import numpy as np

from pathlib import Path

from multiagent_navigation.config_schema import Config

from multiagent_navigation.lib import (
    train,
    make_env,
    select_device,
    greedy_rollout,
)

SEED = 10
N_TRAJ_ROBOTS = 4

# The curated log's curriculum: the active-robot count ramps 1 -> CAP over
# CURRICULUM_STEPS timesteps (the config defaults ship a lighter 4 / 400k
# ramp; the report run overrides them here).
CURRICULUM_CAP = 8
CURRICULUM_STEPS = 800_000

ROOT = Path(__file__).parents[2]
ASSETS = Path(__file__).parents[1] / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

########################################
#             Entry point              #
########################################


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = select_device()
    print(f"device = {device}")

    # ── The run config: package defaults + the report run's curriculum ──
    cfg = Config()
    cfg.train.seed = SEED
    cfg.train.n_robots = CURRICULUM_CAP
    cfg.train.max_robots_timestamp = CURRICULUM_STEPS

    # ── Train; per-epoch checkpoints land in the gitignored run dirs ──
    result = train(
        cfg,
        device=device,
        results_dir=ROOT / "results",
        models_dir=ROOT / "pytorch_models",
    )

    # ── Curated copies for the report pipeline (checkpoints + log) ──
    result.agent.save(cfg.train.file_name, ASSETS)
    log_path = ASSETS / f"{cfg.train.file_name}.json"
    log_path.write_text(json.dumps(result.evaluations) + "\n")
    print(f"saved final checkpoint + {len(result.evaluations)} eval epochs")

    if result.evaluations:
        best = max(result.evaluations, key=lambda e: e["Avg_reward"])
        epoch = int(best["Epoch"])
        for part in ("actor", "critic"):
            shutil.copy2(
                ROOT
                / "pytorch_models"
                / f"{cfg.train.file_name}_epoch-{epoch}_{part}.pth",
                ASSETS / f"{cfg.train.file_name}_best_{part}.pth",
            )
        print(
            f"curated best epoch {epoch} (Avg_reward {best['Avg_reward']:.2f})",
        )

    # ── One greedy trajectory for the report's failure figure ──
    env = make_env(cfg, n_robots=N_TRAJ_ROBOTS, seed=SEED)
    traj = greedy_rollout(result.agent, env, n_robots=N_TRAJ_ROBOTS)
    (ASSETS / "trajectory.json").write_text(json.dumps(traj) + "\n")
    outcomes = [robot["outcome"] for robot in traj["robots"]]
    print(f"greedy rollout outcomes: {outcomes}")


if __name__ == "__main__":
    main()
