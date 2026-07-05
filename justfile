# Common dev tasks. Run `just` (no args) to list them.
#
# Thin wrappers over the real entrypoints (uv, pre-commit, pytest) — a
# convenience layer, not a source of truth. The pre-commit gate stays the
# single definition of what lint/format/types run (see AGENTS.md); these
# recipes only call it, never re-list the underlying tools.

# list available recipes
default:
    @just --list

# sync the dev environment from the lockfile
install:
    uv sync

# install the pre-commit git hook so the gate runs on every commit
hooks:
    uv run pre-commit install

# run the whole pre-commit gate (lint, format, types, style checks; auto-fixes what it can)
lint:
    uv run pre-commit run --all-files

# stage-1 tests: fast, CPU-only, deterministic (the default suite)
test *args:
    uv run pytest {{args}}

# stage-1 + stage-2 tests (heavy: downloads / GPU / long-running)
stage2 *args:
    RUN_STAGE2=1 uv run pytest {{args}}

# tests with branch coverage (informational, no threshold)
cov:
    uv run pytest --cov

# run with the default config; pass Hydra overrides, e.g. `just run model=default`
run *args:
    uv run python -m multiagent_navigation.run {{args}}
