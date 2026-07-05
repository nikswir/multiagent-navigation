"""Tests for the mutant-report helper (flagged_mutants)."""

from __future__ import annotations

import ast
import pytest
import flagged_mutants

from pathlib import Path

########################################
#            Mutant report             #
########################################


def test_func_source_extracts_named_function() -> None:
    src = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    tree = ast.parse(src)
    assert (
        flagged_mutants._func_source(src, tree, "b") == "def b():\n    return 2"
    )


def test_flagged_mutants_parses_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = (
        "    pkg.mod.x_a__mutmut_1: survived\n"
        "    pkg.mod.x_a__mutmut_2: no tests\n"  # dropped (pytest owns this)
        "    pkg.mod.x_a__mutmut_3: timeout\n"
        "    pkg.mod.x_a__mutmut_4: suspicious\n"
        "    1.2 mutants/s: done\n"  # has ': ' but is not a mutant id
        "    not a mutant line\n"
    )

    class _Proc:
        stdout = fake

    monkeypatch.setattr(
        flagged_mutants.subprocess,
        "run",
        lambda *a, **k: _Proc(),
    )
    assert flagged_mutants.flagged_mutants() == [
        ("pkg.mod.x_a__mutmut_1", "survived"),
        ("pkg.mod.x_a__mutmut_3", "timeout"),
        ("pkg.mod.x_a__mutmut_4", "suspicious"),
    ]


def test_mutant_diff_shows_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg = tmp_path / "mutants" / "src" / "m"
    pkg.mkdir(parents=True)
    (pkg / "f.py").write_text(
        "def x_g__mutmut_orig():\n    return 1\n\n\n"
        "def x_g__mutmut_1():\n    return 2\n",
    )
    monkeypatch.setattr(flagged_mutants, "MUTANTS", tmp_path / "mutants")

    diff = flagged_mutants.mutant_diff("src.m.f.x_g__mutmut_1")
    assert "-    return 1" in diff
    assert "+    return 2" in diff
