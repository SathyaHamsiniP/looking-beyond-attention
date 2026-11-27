"""Shared CPU preparation, resumable inference and paired reporting."""

import argparse
from collections import defaultdict, Counter
from importlib.metadata import version
from itertools import product
from pathlib import Path
import re
import sys

from benchmark_paths import REPO, BENCHMARK
import lora_7rules as seven
from evidence_design import BITS, RULES, allocation_audit, design
from evidence_metrics import baselines, paired_delta, response_status, summarize

RESULTS = BENCHMARK / "results/seven-rules/qwen3.5-4b"
WRAPPER = (
    REPO
    / "three-operand-experiments/code/01-operator-controls/operator_controls_qwen35.py"
)
MODES = ("visual", "symbolic")
MODELS = ("pretrained", "lora")


def outside(path):
    if path.resolve().is_relative_to(REPO):
        raise ValueError("Choose a persistent output directory outside the repository")
    return path


def inventory(data, historical):
    from PIL import Image
    import hashlib

    def pixels(path):
        with Image.open(path) as im:
            return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()

    _, fingerprint = seven.dataset_identity(historical)
    if (
        fingerprint
        != "f0710387bdd8133776977b0f129b2fe1d83e7cb8ceb534c0629b7bfdd6b459f9"
    ):
        raise ValueError(
            "Historical dataset differs from the seven-rule training source"
        )
    used = {
        pixels(p)
        for folder in (historical, REPO / "three-operand-experiments/data")
        for p in (folder / "images").glob("*.png")
    }
    meta = seven.read(data / "metadata.json")
    if meta["seed"] == 42 or len(meta["records"]) != meta["requested_images"]:
        raise ValueError("Use a complete, newly seeded evaluation render")
    pool = defaultdict(list)
    hashes = {}
    for row in meta["records"]:
        name = row["filename"]
        if Path(name).name != name or name in hashes:
            raise ValueError("Invalid or repeated filename")
        objects = {"_".join(obj) for obj in row["objects"]}
        bits = "".join(
            str(int(atom in objects))
            for atom in ("green_cube", "blue_sphere", "red_cylinder")
        )
        if list(map(int, bits)) != row["bits"]:
            raise ValueError(f"Object annotations disagree for {name}")
        path = data / "images" / name
        key = pixels(path)
        if key in used:
            raise ValueError(
                f"Repeated pixels in historical or fresh evaluation pool: {name}"
            )
        used.add(key)
        hashes[name] = seven.file_hash(path)
        pool[bits].append(name)
    if len({len(pool[b]) for b in BITS}) != 1:
        raise ValueError("Expected a balanced pool across all eight configurations")
    return {b: sorted(pool[b]) for b in BITS}, hashes, meta["seed"]


def prepare(number, args):
    pool, hashes, render_seed = inventory(args.data, args.historical)
    manifest = dict(
        experiment=number,
        seed=args.seed,
        repeats=args.repeats,
        render_seed=render_seed,
        rules=RULES,
        pool=pool,
        hashes=hashes,
        sets=design(number, pool, args.seed, args.repeats),
    )
    out = outside(args.out)
    out.mkdir(exist_ok=True, parents=True)
    target = out / "manifest.json"
    if target.exists() and seven.read(target) != manifest:
        raise ValueError("Existing manifest differs. Choose a new output directory")
    seven.atomic_json(target, manifest)
    audit = dict(
        render_seed=render_seed,
        design_seed=args.seed,
        repeats=args.repeats,
        manifest_sha256=seven.file_hash(target),
        fresh_images=len(hashes),
        sets=len(manifest["sets"]),
        groups=len({r["group"] for r in manifest["sets"]}),
        candidate_counts=dict(Counter(len(r["answers"]) for r in manifest["sets"])),
        allocation=allocation_audit(manifest["sets"], pool),
        conditions={},
    )
    for kind in sorted({r["condition"] for r in manifest["sets"]}):
        rows = [r for r in manifest["sets"] if r["condition"] == kind]
        audit["conditions"][kind] = dict(
            sets=len(rows),
            groups=len({r["group"] for r in rows}),
            unique_answer_counts=dict(
                Counter(r["answers"][0] for r in rows if len(r["answers"]) == 1)
            ),
            configuration_counts=dict(Counter(b for r in rows for b in r["bits"])),
        )
    seven.atomic_json(out / "design.json", audit)
    print(f"{len(manifest['sets'])} set presentations before rotations and input modes")


def prompt(row, rotation, mode, number):
    mapping = seven.comparison.mapping(7, rotation)
    options = [
        f"{n}. {next(r['expression'] for r in RULES if r['id'] == rid)}"
        for n, rid in mapping.items()
    ]
    lines = [
        "A means a green cube is present. B means a blue sphere is present.",
        "C means a red cylinder is present. Ignore other objects.",
        "Every scene is a positive example. OR is inclusive. XOR means exactly one.",
    ]
    if mode == "symbolic":
        lines.extend(
            ", ".join(f"{a}={v}" for a, v in zip("ABC", b)) for b in row["bits"]
        )
    lines.extend(options)
    lines.append(
        "Which rules are consistent with every scene? Select all that apply. "
        "Reply with distinct option numbers separated by commas, like 1,3,5. "
        "If only one fits, give its number. Do not add any other text."
        if number == 11
        else "Which rule is consistent with every scene? Reply with one option number from 1 to 7. "
        "Do not add any other text."
    )
    return "\n".join(lines)


def parse(answer, rotation, number):
    s = answer.strip()
    if not re.fullmatch(r"[1-7](?:\s*,\s*[1-7])*", s):
        return None
    labels = [n.strip() for n in s.split(",")]
    if len(set(labels)) != len(labels) or (number == 12 and len(labels) != 1):
        return None
    order = seven.comparison.mapping(7, rotation)
    return sorted(order[n] for n in labels)


def jobs(manifest, mode, rotations):
    for row, rotation in product(manifest["sets"], range(rotations)):
        yield f"{row['set_id']}-{rotation}", dict(row, rotation=rotation, mode=mode)


def checkpoint(folder, tasks, number, infer, limit=None):
    records = []
    pending = 0
    for key, row in tasks:
        path = folder / (key + ".json")
        if path.exists():
            saved = seven.read(path)
            if saved["key"] != key or not isinstance(saved["response"], str):
                raise ValueError(f"Invalid checkpoint: {path}")
        else:
            if infer is None:
                raise ValueError(f"Incomplete evaluation: missing {path}")
            if limit is not None and pending >= limit:
                continue
            response = infer(row)
            saved = dict(key=key, response=response)
            pending += 1
        prediction = parse(saved["response"], row["rotation"], number)
        status = response_status(prediction, row["answers"])
        if not path.exists():
            seven.atomic_json(
                path, dict(saved, prediction=prediction, response_status=status)
            )
        records.append(dict(row, prediction=prediction, response_status=status))
    return records


def load_chat(adapter):
    sys.path.insert(0, str(WRAPPER.parent))
    from operator_controls_qwen35 import Qwen35Adapter

    model, processor, _ = seven.comparison.shared.load_model()
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
    return Qwen35Adapter(model, processor)


def run(number, args, manifest):
    identity = None
    if args.model == "lora":
        if args.adapter is None:
            raise ValueError("Supply the selected epoch-3 --adapter")
        identity = seven.adapter_identity(args.adapter)
        if (
            identity
            != seven.read(RESULTS / "lora/training/complete.json")["adapter_identity"]
        ):
            raise ValueError("Wrong LoRA checkpoint")
    elif args.adapter:
        raise ValueError("Use --model lora with --adapter")
    if args.mode == "visual":
        for name, digest in manifest["hashes"].items():
            if seven.file_hash(args.data / "images" / name) != digest:
                raise ValueError(f"Image changed: {name}")
    sources = seven.code_identity()
    for path in [
        WRAPPER,
        Path(__file__),
        Path(__file__).with_name("evidence_design.py"),
        Path(__file__).with_name("evidence_metrics.py"),
    ]:
        sources[path.name] = seven.file_hash(path)
    cfg = dict(
        manifest=seven.digest(manifest),
        model=args.model,
        adapter=identity,
        revision=seven.MODEL["revision"],
        mode=args.mode,
        rotations=args.rotations,
        source=sources,
        software={
            k: version(k)
            for k in (
                "torch",
                "transformers",
                "peft",
                "bitsandbytes",
                "accelerate",
                "pillow",
            )
        },
    )
    folder = args.out / args.model / args.mode
    chat = None

    def infer(row):
        nonlocal chat
        if chat is None:
            chat = load_chat(args.adapter)
        files = (
            [args.data / "images" / f for f in row["image_files"]]
            if args.mode == "visual"
            else []
        )
        return chat.chat(
            files, prompt(row, row["rotation"], args.mode, number), max_new_tokens=32
        )

    tasks = list(jobs(manifest, args.mode, args.rotations))
    with seven.output(folder, cfg, resume=(folder / "config.json").exists()):
        rows = checkpoint(folder, tasks, number, infer, args.limit)
        print(f"{len(rows)}/{len(tasks)} saved")
        if len(rows) == len(tasks):
            seven.atomic_json(folder / "metrics.json", summarize(rows))


def report(number, out, manifest):
    from matplotlib import pyplot as plt

    by_condition, summary = {}, {}
    rotations = None
    protocol = None
    for model, mode in product(MODELS, MODES):
        folder = out / model / mode
        cfg = seven.read(folder / "config.json")
        if cfg["manifest"] != seven.digest(manifest) or (
            rotations and rotations != cfg["rotations"]
        ):
            raise ValueError("Report needs matching manifests and rotation counts")
        identity = {k: cfg[k] for k in ("source", "revision", "software", "rotations")}
        if (
            cfg["model"] != model
            or cfg["mode"] != mode
            or (protocol and identity != protocol)
        ):
            raise ValueError("Conditions used different software, sources or protocol")
        protocol = identity
        rotations = cfg["rotations"]
        rows = checkpoint(folder, jobs(manifest, mode, rotations), number, None)
        by_condition[model, mode] = rows
        summary[f"{model}/{mode}"] = summarize(rows)
    deltas = {
        f"lora-minus-pretrained/{m}": paired_delta(
            by_condition["pretrained", m], by_condition["lora", m]
        )
        for m in MODES
    }
    deltas.update(
        {
            f"symbolic-minus-visual/{m}": paired_delta(
                by_condition[m, "visual"], by_condition[m, "symbolic"]
            )
            for m in MODELS
        }
    )
    if number == 12:
        baseline_rows = defaultdict(list)
        for row in manifest["sets"]:
            for name, prediction in baselines(row).items():
                baseline_rows[name].append(dict(row, prediction=prediction, rotation=0))
        for name, rows in baseline_rows.items():
            summary[name] = summarize(rows)
    seven.atomic_json(
        out / "comparison.json",
        dict(
            conditions=summary,
            paired_differences=deltas,
            uncertainty=dict(
                method="95% percentile cluster bootstrap",
                draws=5000,
                seed=2026,
                unit="chain" if number == 11 else "matched pair",
            ),
        ),
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    if number == 11:
        for label, measurements in summary.items():
            sizes = sorted(
                int(k.split("-")[-1])
                for k in measurements["rates"]
                if k.startswith("exact/candidates-")
            )
            ax.plot(
                sizes,
                [measurements["rates"][f"exact/candidates-{n}"] for n in sizes],
                marker="o",
                label=label,
            )
        ax.set(
            xlabel="Number of logically consistent rules",
            ylabel="Exact set accuracy",
            ylim=(0, 1.05),
        )
        ax.legend(fontsize=8)
    else:
        ax.barh(
            list(summary),
            [s["rates"]["both_correct/context"] for s in summary.values()],
        )
        ax.set(xlim=(0, 1), xlabel="Both context answers correct")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)


def cli(number):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("prepare")
    gen.add_argument("--historical", type=Path, required=True)
    gen.add_argument("--seed", type=int, default=202611)
    gen.add_argument("--repeats", type=int, default=1)
    ev = sub.add_parser("run")
    ev.add_argument("--mode", choices=MODES, required=True)
    ev.add_argument("--model", choices=MODELS, required=True)
    ev.add_argument("--adapter", type=Path)
    ev.add_argument("--rotations", type=int, choices=(1, 7), default=7)
    ev.add_argument(
        "--limit", type=int, help="Maximum new predictions in this invocation"
    )
    rep = sub.add_parser("report")
    for cmd in (gen, ev, rep):
        cmd.add_argument("--out", type=Path, required=True)
    for cmd in (gen, ev):
        cmd.add_argument("--data", type=Path, required=True)
    args = p.parse_args()
    outside(args.out)
    if args.command == "prepare":
        return prepare(number, args)
    m = seven.read(args.out / "manifest.json")
    if (
        m["experiment"] != number
        or m["rules"] != RULES
        or m["sets"] != design(number, m["pool"], m["seed"], m["repeats"])
    ):
        raise ValueError("Manifest differs from the verified design")
    if args.command == "report":
        return report(number, args.out, m)
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    run(number, args, m)
