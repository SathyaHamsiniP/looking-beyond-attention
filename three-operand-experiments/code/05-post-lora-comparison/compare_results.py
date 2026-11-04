import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.comparison import compare

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare matched baseline and adapter patching runs without moving any source results."
    )
    p.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        required=True,
        help="Baseline first, then adapters",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if len(a.runs) < 2:
        p.error("Provide baseline and at least one adapter run")
    compare(a.runs, a.output)
