"""Smoke tests for the CLI entry point.

The unit tests never import main.py, so a syntax error or a broken
subcommand wiring there would slip through them. Running --help in a
subprocess catches both.
"""

import subprocess
import sys
from pathlib import Path

import pytest

MAIN = str(Path(__file__).resolve().parent.parent / "main.py")

SUBCOMMANDS = [
    "ingest",
    "ingest-fx",
    "metrics",
    "benchmark",
    "forecast",
    "drift",
    "var-test",
    "report",
]


def run_help(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, MAIN, *args, "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestCliWiring:
    def test_top_level_help(self):
        result = run_help()
        assert result.returncode == 0
        for command in SUBCOMMANDS:
            assert command in result.stdout

    @pytest.mark.parametrize("command", SUBCOMMANDS)
    def test_subcommand_help(self, command):
        result = run_help(command)
        assert result.returncode == 0, result.stderr
