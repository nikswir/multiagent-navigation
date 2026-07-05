"""Public-API stability — pin the exported surface and its signatures.

If a test here breaks, the public contract changed: update it deliberately
(and bump the version), don't just edit the assertion away.
"""

from __future__ import annotations

import inspect
import importlib
import dataclasses

PKG = importlib.import_module("multiagent_navigation")

########################################
#              Public API              #
########################################


def test_exports() -> None:
    """`__all__` is exactly the supported public surface.

    Deliberately extended (with the 0.2.0 bump) by the shared builders:
    `make_env`, `build_agent` and `select_device`.
    """
    assert set(PKG.__all__) == {
        "TD3",
        "train",
        "Config",
        "evaluate",
        "make_env",
        "SimpleEnv",
        "TrainResult",
        "build_agent",
        "ReplayBuffer",
        "select_device",
    }


def test_train_signature() -> None:
    """`train` keeps its (cfg, *, device, dirs) -> TrainResult contract."""
    sig = inspect.signature(PKG.train, eval_str=True)
    assert list(sig.parameters) == [
        "cfg",
        "device",
        "results_dir",
        "models_dir",
    ]
    assert sig.return_annotation is PKG.TrainResult


def test_trainresult_fields() -> None:
    """The TrainResult dataclass keeps its public fields."""
    fields = [f.name for f in dataclasses.fields(PKG.TrainResult)]
    assert fields == ["agent", "evaluations"]
