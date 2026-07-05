"""The two default sources can never drift (stage 1).

Defaults deliberately live in TWO places: the typed dataclasses in
config_schema.py (so `Config()` works without Hydra — scripts and tests
build it directly) and configs/*/default.yaml (so the shipped, tunable
values are visible as data). This test composes the real YAML tree with
Hydra and asserts it equals `Config()` field for field — a value changed in
one place but not the other fails the suite instead of silently winning by
merge order.
"""

from __future__ import annotations

from omegaconf import OmegaConf
from hydra import compose, initialize

from multiagent_navigation import config_schema
from multiagent_navigation.config_schema import Config

########################################
#           Schema <-> YAML            #
########################################


def test_yaml_defaults_match_schema_defaults() -> None:
    config_schema.register()

    # ── Compose exactly what `python -m multiagent_navigation.run` sees ──
    with initialize(version_base=None, config_path="../configs"):
        composed = compose(config_name="config")

    assert OmegaConf.to_object(composed) == Config()
