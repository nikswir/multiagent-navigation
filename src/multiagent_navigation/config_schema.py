"""Structured-config schema — the run's typed contract.

Hydra validates `configs/` against these dataclasses: each field has a type
and a literal default (overridden by the matching config-group option file).
Registering the root makes Hydra reject wrong types / unknown fields at
startup, instead of crashing deep inside the run. One dataclass per group.
"""

from __future__ import annotations

from dataclasses import field, dataclass
from hydra.core.config_store import ConfigStore

########################################
#            Group schemas             #
########################################


@dataclass
class EnvConfig:
    """The SimpleEnv world: geometry, lidar, timing and robot capacity.

    `max_robots` is the env's array *capacity*; the *active* robot count per
    episode comes from `train.n_robots` (training curriculum cap) or
    `animate.n_robots`. Rewards are fixed in `SimpleEnv.get_reward`
    (+100 goal / -100 collision / -0.05 per-step penalty).
    """

    world_width: float = 10.0
    world_height: float = 10.0
    environment_dim: int = 20
    robot_radius: float = 0.25
    max_steps: int = 500
    max_robots: int = 16
    time_delta: float = 0.1
    goal_reached_dist: float = 0.3
    lidar_max_range: float = 5.0
    # (x, y, width, height) axis-aligned rectangles.
    obstacle_definitions: list[list[float]] = field(
        default_factory=lambda: [
            [-3, 1, 1, 2],
            [1, -2, 2, 1],
            [-1, -3, 3, 0.5],
            [2, 2, 0.5, 3],
        ],
    )


@dataclass
class ModelConfig:
    """TD3 network sizes, learning rates and update hyper-parameters."""

    action_dim: int = 2
    max_action: float = 1.0
    actor_lr: float = 0.0001
    critic_lr: float = 0.0005
    hidden1: int = 800
    hidden2: int = 600
    discount: float = 0.9999
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    batch_size: int = 256


@dataclass
class TrainConfig:
    """Training schedule, exploration decay, curriculum and checkpoints.

    `expl_noise` is the INITIAL exploration noise — it anneals linearly to
    `expl_min` over `expl_decay_steps`. `n_robots` is the curriculum's
    active-robot cap (reached after `max_robots_timestamp` timesteps).
    Episode length is owned by the env (`env.max_steps`); there is no
    separate training-side cap.
    """

    seed: int = 10
    max_timesteps: int = 1_000_000
    eval_freq: int = 5000
    eval_ep: int = 100
    expl_noise: float = 1.0
    expl_min: float = 0.1
    expl_decay_steps: int = 500_000
    n_robots: int = 4
    max_robots_timestamp: int = 400_000
    buffer_size: int = 600_000
    random_near_obstacle: bool = False
    save_model: bool = True
    load_model: bool = False
    file_name: str = "TD3_simpleEnv"


@dataclass
class AnimateConfig:
    """Animation playback and the artifact paths shared with training.

    `run_dir` points at ONE training-run directory (the Hydra output dir or
    a repo-root layout) from which `results/` and `pytorch_models/` are
    derived; when empty, `viz` falls back to `results_dir`/`models_dir`
    relative to the CWD and then to the newest `outputs/` run.
    """

    n_robots: int = 10
    max_steps: int = 200
    interval: int = 50
    fig_width: float = 8.0
    fig_height: float = 8.0
    seed: int = 1
    run_dir: str = ""
    results_dir: str = "results"
    models_dir: str = "pytorch_models"
    report_csv: str = "report.csv"


########################################
#           Root & registry            #
########################################


@dataclass
class Config:
    """The composed run config — one field per group."""

    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    animate: AnimateConfig = field(default_factory=AnimateConfig)


def register() -> None:
    # ── Expose the schema as `config_schema` for config.yaml's defaults ──
    ConfigStore.instance().store(name="config_schema", node=Config)
