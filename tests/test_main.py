"""The command-line entry point."""

from __future__ import annotations

import pytest


def test_main_prints_version_platform_and_solvers(capsys, monkeypatch) -> None:
    """The bare command reports version, platform, and usable solvers."""
    from ncpolopt.__main__ import main

    monkeypatch.setattr("sys.argv", ["ncpolopt"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("ncpolopt ")
    assert "Python " in out
    assert "solvers" in out


def test_main_version_flag(capsys, monkeypatch) -> None:
    """--version prints just the version (argparse's exit path)."""
    from ncpolopt.__main__ import main

    monkeypatch.setattr("sys.argv", ["ncpolopt", "--version"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip()


def test_main_reports_missing_solvers(capsys, monkeypatch) -> None:
    """Without any usable backend the CLI prints an install hint."""
    from ncpolopt import __main__

    monkeypatch.setattr(__main__, "available_solvers", lambda: [])
    monkeypatch.setattr("sys.argv", ["ncpolopt"])
    with pytest.raises(SystemExit):
        __main__.main()
    assert "No SDP solver found" in capsys.readouterr().out


def test_verbosity_levels() -> None:
    """The verbosity mapping: 0 warns, 1 informs, 2 and beyond debug."""
    import logging

    from ncpolopt._logging import verbosity_to_level

    assert verbosity_to_level(0) == logging.WARNING
    assert verbosity_to_level(1) == logging.INFO
    assert verbosity_to_level(2) == logging.DEBUG
    assert verbosity_to_level(99) == logging.DEBUG
    assert verbosity_to_level(-1) == logging.WARNING


def test_module_entry_point() -> None:
    """``python -m ncpolopt`` runs the CLI."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "ncpolopt"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.startswith("ncpolopt ")
