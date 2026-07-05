"""Tests for the device-agnostic check (check_device)."""

from __future__ import annotations

import check_device

from pathlib import Path

########################################
#             Device check             #
########################################


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(text)
    return path


def test_flags_cuda_call(tmp_path: Path) -> None:
    path = _write(tmp_path, "x = t.cuda()\n")
    assert check_device.check_file(path) is True


def test_flags_to_cuda_string(tmp_path: Path) -> None:
    path = _write(tmp_path, 'x = t.to("cuda")\n')
    assert check_device.check_file(path) is True


def test_flags_device_kwarg(tmp_path: Path) -> None:
    path = _write(tmp_path, 'f(device="cuda")\n')
    assert check_device.check_file(path) is True


def test_device_ok_exempts(tmp_path: Path) -> None:
    path = _write(tmp_path, "x = t.cuda()  # device-ok\n")
    assert check_device.check_file(path) is False


def test_clean_file_passes(tmp_path: Path) -> None:
    path = _write(tmp_path, "x = t.to(dev)\n")
    assert check_device.check_file(path) is False
