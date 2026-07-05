# AGENTS.md — how this project is built

The engineering workflow: how the code is written, checked and run. Read it
first to be up to speed. Code-style rules live in the `code-style` rule
(`.claude/rules/code-style.md`), auto-loaded when a Python file is read.
Repo-specific commands (how to run the app) live in the README.

## Toolchain

| Concern                  | Tool                                       |
| ------------------------ | ------------------------------------------ |
| dependencies & packaging | uv (src layout)                            |
| lint + format            | Ruff (`ruff check`, `ruff format`)         |
| type checking            | mypy (strict over `src`/`tools`, relaxed over `report/scripts`) |
| tests                    | pytest (+ Hypothesis for property tests)   |
| config                   | Hydra structured configs                   |
| commit gate              | pre-commit                                 |

## Quick commands (`just`)

A `justfile` provides shortcuts for the common tasks — run `just` to list them:

| Recipe          | Runs                              |
| --------------- | --------------------------------- |
| `just install`  | `uv sync`                         |
| `just hooks`    | `uv run pre-commit install`       |
| `just lint`     | `uv run pre-commit run --all-files` |
| `just test`     | `uv run pytest`                   |
| `just stage2`   | `RUN_STAGE2=1 uv run pytest`      |
| `just cov`      | `uv run pytest --cov`             |
| `just run`      | `uv run python -m multiagent_navigation.run`    |

It is a thin convenience layer that **calls** these entrypoints — the
`pre-commit` gate stays the single source of truth, so `just` is never required
(the raw commands below and in the README work unchanged). `test`, `stage2` and
`run` forward extra args (e.g. `just run model=default`, `just test -k name`).

## Code style

The full rules live in `.claude/rules/code-style.md` — a path-scoped rule that
auto-loads when a Python file is read. In short: banners, length ladders,
four-block absolute imports, trailing commas, 80-column lines. The mechanical
rules are **enforced by pre-commit** through the `check-*` hooks below; the rest
is human/agent judgment. Each violation message cites the `code-style §N` rule
it breaks.

## pre-commit is the gate

Everything runs before each commit (`pre-commit install`), and CI runs the same
set — `.pre-commit-config.yaml` is the single source of truth, never duplicated.

Hooks, in order:

- **hygiene** — trailing-whitespace, end-of-file, typos
- **format** — `add-trailing-comma` → `ruff-format` (explode multi-line
  constructs one element per line)
- **lint** — `ruff check`
- **types** — `mypy`
- **style as code** — `check-imports`, `check-banners`, `check-device`: project
  rules encoded as executable checks (what ruff / mypy cannot express), each
  citing the `code-style` rule it enforces.

Principle: a rule worth keeping becomes a check, so neither a human nor an agent
can quietly break it. What the formatter / linter already enforces is never
re-checked (no duplication).

Run all: `pre-commit run --all-files`. Run one: `pre-commit run ruff-check`.

**`pre-commit` only sees git-tracked files** — a new, unstaged file is *silently
skipped* by `pre-commit run --all-files`, so a green run can hide a broken new
file. `git add` new files before trusting the gate, or audit uncommitted work by
running the tools directly over the filesystem (`ruff check src tests tools
report/scripts`,
`mypy`, the `check-*` scripts). On `git commit` the hooks run on the staged
files — that is when the gate is real.

## Linting & type checking

- **ruff** — `ruff check` (rules `E, F, UP, B, SIM, C4, PT`; isort `I` is off,
  imports are ordered by length — code-style §6) and `ruff format --check`.
- **mypy** — strict (`disallow_untyped_defs`, …) over `src` and `tools`, and
  also over `report/scripts` with untyped defs allowed (research glue). Every
  library function is fully annotated; the package ships a `py.typed` marker.

## Tests — two stages

- **Stage 1 (fast)** — CPU-only, deterministic (seeded, tiny tensors), no
  downloads. Runs on every commit and in CI. Includes property-based tests
  (Hypothesis) asserting invariants, and tests written to kill specific mutants.
- **Stage 2 (heavy)** — anything that downloads data/weights, needs a GPU, or
  runs long. A `conftest.py` fixture **skips these unless `RUN_STAGE2=1`**, so
  they are off by default and CI never triggers them.

Run stage 1 (the default): `pytest`. Run stage 2 as well: `RUN_STAGE2=1 pytest`.

## Coverage

`pytest --cov` with branch coverage, `source = <package>`. **Informational, not
a blocking gate** (no required threshold, no badge by default). `exclude_lines`
drops what can't be meaningfully exercised (`pragma: no cover`,
`if TYPE_CHECKING:`, `if __name__ == "__main__":`).

## CI — light then heavy

- **Stage 1 (every push / PR)** — `pre-commit run --all-files` (lint, format,
  types, style checks) + `pytest`, on cheap runners with no hardware. Docs-only
  pushes are skipped (`paths-ignore`).
- **Stage 2 (on demand)** — heavy tests on real hardware; not on every push.

CI invokes `pre-commit`, not the tools directly — one source of truth shared
with local commits.

## Commits

- Run pre-commit before committing; fix what the `check-*` hooks flag.
- `git add` new files before trusting the gate — pre-commit only sees tracked
  files, so untracked ones are silently skipped.
- Conventional prefixes (`feat:`, `fix:`, `docs:`, `style:`, `chore:`, `test:`),
  one logical change per commit.

For code review, use the built-in `/code-review` (or `/review`). For test
coverage + mutation quality of a change, delegate the read-only
`test-coverage-auditor` subagent with the change's intent in the prompt — it is
the one review pass with no built-in equivalent.
