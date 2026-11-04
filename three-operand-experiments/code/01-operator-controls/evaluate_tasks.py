"""Evaluate V1/V2/V3 manifests; original operator controls remain separate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.evaluation import main

if __name__ == "__main__":
    main(1)
