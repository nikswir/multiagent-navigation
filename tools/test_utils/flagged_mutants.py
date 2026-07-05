"""Print the diff of every mutant worth a manual look, in one fast pass.

``mutmut results`` lists every mutant it did NOT kill. Of those we keep
``survived`` (a test ran the line but no assertion caught the change) plus
``timeout`` / ``suspicious`` (anomalies). None is a hole for certain: each is
a candidate to read and judge. ``no tests`` is dropped: pytest coverage is the
authority on physically-uncovered lines, and mutmut's ``no tests`` both
duplicates it and misfires on modules it cannot substitute. For each kept mutant
this reads
mutmut's ``mutants/`` copy and diffs the original function against the mutated
one — all in-process, so it is far faster than ``mutmut show`` per id (each
reloads the project and all its imports). Coupled to mutmut 3.x's on-disk
layout. Run from the repo root after ``mutmut run``::

    python tools/test_utils/flagged_mutants.py
"""

from __future__ import annotations

import ast
import difflib
import argparse
import subprocess

from pathlib import Path

MUTANTS = Path("mutants")

# Statuses worth a look. `no tests` is dropped on purpose: pytest coverage is
# the authority on physically-uncovered lines, and mutmut's `no tests` both
# duplicates it and misfires on modules it can't substitute.
WORTH_A_LOOK = {"survived", "timeout", "suspicious"}

########################################
#              Collecting              #
########################################


def flagged_mutants() -> list[tuple[str, str]]:
    # ── (id, status) for every mutant we keep: `mutmut results` already drops
    #    the killed ones, and we drop `no tests` too (see module docstring) ──
    proc = subprocess.run(
        ["mutmut", "results"],
        capture_output=True,
        text=True,
        check=False,
    )
    out: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if ": " in stripped:
            mutant_id, status = stripped.rsplit(": ", 1)
            mutant_id, status = mutant_id.strip(), status.strip()
            if "__mutmut_" in mutant_id and status in WORTH_A_LOOK:
                out.append((mutant_id, status))
    return out


def _func_source(text: str, tree: ast.Module, name: str) -> str:
    # ── Source of the function/method called `name`, anywhere in the tree ──
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == name
        ):
            return ast.get_source_segment(text, node) or ""
    return ""


def mutant_diff(mutant_id: str) -> str:
    # ── Diff mutmut's "orig" function against the mutated one ──
    module, _, func = mutant_id.rpartition(".")
    base = func.rpartition("__mutmut_")[0]
    if not base:
        return "(unrecognized id)"

    tail = str(Path(*module.split(".")).with_suffix(".py"))
    found = [p for p in MUTANTS.rglob(Path(tail).name) if str(p).endswith(tail)]
    if not found:
        return f"(source not found under {MUTANTS}/)"

    text = found[0].read_text()
    tree = ast.parse(text)
    orig = _func_source(text, tree, f"{base}__mutmut_orig")
    mutated = _func_source(text, tree, func)

    # ── Normalize mutmut's generated names so the `def` line matches and the
    #    diff shows only the real mutation, not the function rename ──
    orig = orig.replace(f"{base}__mutmut_orig", base, 1)
    mutated = mutated.replace(func, base, 1)

    diff = difflib.unified_diff(
        orig.splitlines(),
        mutated.splitlines(),
        lineterm="",
        n=1,
    )
    body = "\n".join(
        line for line in diff if not line.startswith(("---", "+++", "@@"))
    )
    return body or "(no diff)"


########################################
#             Entry point              #
########################################


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dump all flagged mutants.")
    p.parse_args(argv)

    flagged = flagged_mutants()
    if not flagged:
        print("# no flagged mutants")
        return 0

    counts: dict[str, int] = {}
    for _, status in flagged:
        counts[status] = counts.get(status, 0) + 1
    summary = " · ".join(f"{n} {s}" for s, n in sorted(counts.items()))
    print(f"# {len(flagged)} flagged mutants ({summary})\n")

    for mutant_id, status in sorted(flagged, key=lambda m: m[1]):
        print(f"## [{status}] {mutant_id}")
        print(mutant_diff(mutant_id))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
