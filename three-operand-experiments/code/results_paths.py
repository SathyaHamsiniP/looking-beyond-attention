"""Shared output location for three-operand experiments."""
from pathlib import Path

DEFAULT_RESULTS_ROOT = Path.home() / 'looking-beyond-attention-results'


def require_results_root(path):
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(
            f'Results root does not exist: {root}\n'
            'Pass an existing results directory with '
            '--experiment-dir. No results root was created.'
        )
    return root
