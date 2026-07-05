"""Hydra entry point for multiagent-navigation.

A run is composed from `configs/` (the env / model / train / animate groups)
and written into Hydra's per-run output directory, so repeated runs and
`--multirun` sweeps never collide. Things to try::

    python -m multiagent_navigation.run --cfg job          # print the config
    python -m multiagent_navigation.run train.n_robots=2   # override a field
    python -m multiagent_navigation.run --multirun train.seed=1,2,3
"""

from __future__ import annotations

import hydra

from typing import cast
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from multiagent_navigation import config_schema
from multiagent_navigation.config_schema import Config
from multiagent_navigation.lib import train, TrainResult, select_device

# Register the structured-config schema so Hydra type-checks the composed YAML.
config_schema.register()

########################################
#               Core run               #
########################################


def run(cfg: Config, out_dir: Path) -> TrainResult:
    """Train from the composed config, writing artifacts under `out_dir`."""
    # ── Pick the device once; the library stays device-agnostic ──
    device = select_device()
    return train(
        cfg,
        device=device,
        results_dir=out_dir / "results",
        models_dir=out_dir / "pytorch_models",
    )


########################################
#             Entry point              #
########################################


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    # ── Hydra gives each run (and each --multirun job) its own output dir ──
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    result = run(cast(Config, cfg), out_dir)
    print(f"trained {len(result.evaluations)} eval epochs -> {out_dir}")


if __name__ == "__main__":
    main()
