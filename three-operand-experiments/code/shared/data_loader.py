import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from tasks.validate_sets import validate


def load_manifest(path, data_dir):
    manifest = json.loads(Path(path).read_text())
    validate(manifest, data_dir)
    return manifest


def option_labels(n):
    if not 2 <= n <= 26:
        raise ValueError("Expected 2..26 options")
    return list("XYZ") if n == 3 else list("DEFGHIJKLMNOPQRSTUVWXYZABC"[:n])


def rotated(item, rotation):
    order = item["option_order"]
    k = rotation % len(order)
    return {**item, "option_order": order[k:] + order[:k]}


def prompt(item, manifest, data_dir, symbolic=False):
    """Images occur in list order, followed by the correspondingly numbered labels."""
    paths = [str(Path(data_dir) / o["path"]) for o in item["observations"]]
    lines = ["A is a green cube. B is a blue sphere. C is a red cylinder."]
    if manifest["task_version"].startswith("v1"):
        lines += [
            "Every image satisfies one hidden rule. Which candidate rule explains the images?"
        ]
    else:
        lines += [
            "Each observation is labelled true or false by one hidden rule. Identify that rule."
        ]
    for i, o in enumerate(item["observations"], 1):
        if symbolic:
            lines.append(
                f"Observation {i}: A={o['bits'][0]}, B={o['bits'][1]}, C={o['bits'][2]}"
            )
        if not manifest["task_version"].startswith("v1"):
            lines.append(f"Image {i}: {str(o['label']).lower()}")
    mapping = dict(zip(option_labels(len(item["option_order"])), item["option_order"]))
    lines += [f"{label}: {manifest['rules'][rule]}" for label, rule in mapping.items()]
    lines += ["Reply with only the option label. ANSWER:"]
    return ([] if symbolic else paths), "\n".join(lines), mapping


def shortcut_control(item, manifest):
    """Remove all distinguishing observations. Mark ambiguity instead of assigning accuracy."""
    from tasks.rules import consistent, evaluate

    obs = [
        o
        for o in item["observations"]
        if len({evaluate(r, o["bits"]) for r in manifest["rules"]}) == 1
    ]
    candidates = consistent(obs, manifest["rules"])
    return {**item, "observations": obs}, candidates
