"""Tests for the import-hygiene check (check_imports, code-style §6)."""

from __future__ import annotations

import check_imports

from pathlib import Path

FIRST_PARTY = {"multiagent_navigation"}

########################################
#             Import check             #
########################################


def _check(tmp_path: Path, text: str) -> bool:
    path = tmp_path / "sample.py"
    path.write_text(text)
    return check_imports.check_file(path, FIRST_PARTY)


def test_relative_import_flagged(tmp_path: Path) -> None:
    assert _check(tmp_path, "from .foo import bar\n") is True


def test_clean_ladder_passes(tmp_path: Path) -> None:
    assert _check(tmp_path, "import ast\nimport argparse\n") is False


def test_line_ladder_break_flagged(tmp_path: Path) -> None:
    assert _check(tmp_path, "import argparse\nimport ast\n") is True


def test_block_order_flagged(tmp_path: Path) -> None:
    assert (
        _check(tmp_path, "from multiagent_navigation import x\nimport ast\n")
        is True
    )


def test_name_ladder_break_flagged(tmp_path: Path) -> None:
    assert _check(tmp_path, "from x import bbbb, a\n") is True


def test_first_party_roots_discovers_local_packages(
    tmp_path: Path,
) -> None:
    # Detected from disk, not hard-coded: a top-level `<pkg>/__init__.py`
    # package and a sibling module both count as first-party. No `src/` here,
    # so the missing-directory branch is exercised too.
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (tmp_path / "solo.py").write_text("")

    roots = check_imports.first_party_roots(tmp_path)
    assert "mypkg" in roots
    assert "solo" in roots
