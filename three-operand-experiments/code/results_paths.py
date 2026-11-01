"""Shared output location for Colab experiments. Outputs stay on Drive."""
from pathlib import Path

DEFAULT_RESULTS_ROOT = Path(
    '/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results'
)


def require_results_root(path):
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(
            f'Results root does not exist: {root}\n'
            'Mount Google Drive and pass its actual existing results path with '
            '--experiment-dir. No results root was created.'
        )
    return root
