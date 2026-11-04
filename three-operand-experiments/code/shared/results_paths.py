from pathlib import Path

EXPERIMENTS = {
    1: "01-operator-controls",
    2: "02-probing",
    3: "03-activation-patching",
    4: "04-lora",
    5: "05-post-lora-comparison",
}
LEGACY_PATCHING = Path("activation_patching/qwen3_5/dataset-v1")


def run_path(root, experiment, task, run_id):
    root = Path(root).expanduser().resolve()
    for value in (task, run_id):
        if not value or "/" in value or value in (".", ".."):
            raise ValueError("Unsafe path component")
    if not root.is_dir():
        raise ValueError("Create the results root explicitly first")
    return root / EXPERIMENTS[experiment] / "qwen3_5" / "dataset-v1" / task / run_id
