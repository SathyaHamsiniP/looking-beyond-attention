"""Locations shared by the benchmark scripts; no experiment settings live here."""
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
BENCHMARK = CODE.parent
REPO = BENCHMARK.parent
FOLDERS = ('00-common', 'three-rules', '06-lora', '07-operator-controls',
           '08-activation-patching', '09-probing', 'tools')
for folder in FOLDERS:
    location = str(CODE / folder)
    if location not in sys.path:
        sys.path.append(location)


def source_path(name):
    """Resolve one shared source filename; duplicates and missing files fail."""
    matches = [CODE / folder / name for folder in FOLDERS
               if (CODE / folder / name).is_file()]
    if len(matches) != 1:
        raise ValueError(f'Expected one benchmark source file: {name}')
    return matches[0]
