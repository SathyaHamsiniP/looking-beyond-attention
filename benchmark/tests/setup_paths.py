"""Make the same shared imports available to unittest discovery."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code' / '00-common'))
import benchmark_paths

for folder in ("11-evidence-sufficiency", "12-context-counterfactuals"):
    sys.path.append(str(benchmark_paths.CODE / folder))
