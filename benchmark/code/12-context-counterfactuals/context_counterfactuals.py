"""Experiment 12 command-line entry point."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00-common"))
from evidence_run import cli

if __name__ == "__main__":
    cli(12)
