"""Multi-robot navigation with TD3 — a shared policy for lidar disc robots.

The public API: import from the package root, not from submodules.
"""

from __future__ import annotations

from multiagent_navigation.agent import TD3
from multiagent_navigation.config_schema import Config
from multiagent_navigation.environment import SimpleEnv
from multiagent_navigation.replay_buffer import ReplayBuffer
from multiagent_navigation.lib import train, evaluate, TrainResult

__version__ = "0.1.0"

__all__ = [
    "TD3",
    "train",
    "Config",
    "evaluate",
    "SimpleEnv",
    "TrainResult",
    "ReplayBuffer",
]
