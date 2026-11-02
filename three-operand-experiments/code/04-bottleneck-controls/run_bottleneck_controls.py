#!/usr/bin/env python3
"""Experiment 04: locate bottlenecks in Qwen3.5's evidence-based rule selection.

Default: 60 model calls over the same three held-out families used in Experiment 02.
--smoke-test: six calls for checking a Colab setup.
--validate-only: no GPU. Results print to console AND are checkpointed to Drive.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "02-probing"))
sys.path.insert(0, str(CODE / "01-operator-controls"))
sys.path.insert(0, str(CODE))
import rule_evidence as evidence
from results_paths import DEFAULT_RESULTS_ROOT, require_results_root

ex = evidence.experiment
MODES = ("oracle", "original", "interleaved")
RULES = ex.RULES
LABELS = ex.LABELS


def example_lines(examples):
    return "\n".join(
        f"Image {i}: A={e['bits'][0]}, B={e['bits'][1]}, C={e['bits'][2]}, "
        f"{'positive' if e['positive'] else 'negative'}"
        for i, e in enumerate(examples, 1)
    )


def make_prompt(job, mode, task, candidate=None, readout=None):
    intro = ("A is a green cube; B is a blue sphere; C is a red cylinder. "
             "A concept is true only when both its color and shape match. "
             "Other objects are irrelevant.\n"
             "Positive means the hidden Boolean rule is true; negative means false.\n"
             "AND requires both inputs. OR requires at least one. Obey parentheses.\n")
    if mode == "oracle":
        intro += "Exact ground-truth scene descriptions:\n" + example_lines(job["examples"]) + "\n"
    elif mode == "original":
        intro += "The image labels, in the same order as the four images, are:\n"
        intro += "\n".join(
            f"Image {i}: {'positive' if e['positive'] else 'negative'}"
            for i, e in enumerate(job["examples"], 1)
        ) + "\n"
    else:
        intro += "Each image has its positive/negative label directly beside it.\n"
    if task == "choice":
        choices = "\n".join(
            f"{label}: {ex.EXPRESSIONS[job['mapping'][label]]}"
            for label in job["display"]
        )
        return (intro + "Find the ONE rule consistent with ALL four examples.\n"
                + choices + "\nReply with only X, Y, or Z.\nANSWER:")
    if task == "verify":
        return (intro + "Is the following candidate rule consistent with ALL four examples?\n"
                + ex.EXPRESSIONS[candidate] + "\nReply with exactly YES or NO.\nANSWER:")
    if task == "binding":
        return (intro + "For each numbered image, give its A/B/C bits and its "
                "provided positive/negative label. Exactly four lines; format:\n"
                "1: 010 negative\n2: 100 negative\n3: 111 positive\n4: 000 negative\n"
                "These are FORMAT EXAMPLES ONLY; read the actual images.\nANSWER:")
    if task == "readout":
        if readout == "rule_id":
            return (intro + "Which rule is consistent with ALL examples?\n"
                    + "\n".join(f"{r}: {ex.EXPRESSIONS[r]}" for r in RULES)
                    + "\nReply with exactly R1, R2, or R3.\nANSWER:")
        return (intro + "Which Boolean rule is consistent with ALL examples?\n"
                + "\n".join(ex.EXPRESSIONS[r] for r in RULES)
                + "\nReply with ONLY the complete Boolean expression, not a letter or number.\nANSWER:")
    raise ValueError(task)


def interleaved_inputs(vlm, examples, prompt):
    """Put labels directly beside their images instead of all images before text."""
    from PIL import Image
    items = [{"type": "text", "text": "Read these four labeled images in sequence.\n"}]
    for i, e in enumerate(examples, 1):
        items.append({"type": "text", "text": f"\nImage {i} ({'positive' if e['positive'] else 'negative'}):\n"})
        path = ex.IMAGE_DIR / e["filename"]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.width * img.height > ex.MAX_PIXELS:
                factor = (ex.MAX_PIXELS / (img.width * img.height)) ** 0.5
                img = img.resize((max(1, int(img.width * factor)),
                                  max(1, int(img.height * factor))),
                                 Image.Resampling.LANCZOS)
            items.append({"type": "image", "image": img})
    items.append({"type": "text", "text": "\n" + prompt})
    messages = [{"role": "user", "content": items}]
    return vlm.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True, return_tensors="pt",
    ).to(vlm.model.device)


def infer(vlm, job, mode, task, candidate=None, readout=None):
    import torch
    prompt = make_prompt(job, mode, task, candidate, readout)
    if mode == "interleaved":
        inputs = interleaved_inputs(vlm, job["examples"], prompt)
    else:
        paths = [] if mode == "oracle" else [
            str(ex.IMAGE_DIR / e["filename"]) for e in job["examples"]
        ]
        inputs = vlm._inputs(paths, prompt)
    with torch.inference_mode():
        if task == "choice":
            output = vlm.model(**inputs, logits_to_keep=1, use_cache=False)
            scores = output.logits[0, -1].float()
            values = torch.stack([scores[ex.TOKEN_IDS[l]] for l in LABELS])
            probs = torch.softmax(values, dim=0).cpu().tolist()
            del output, scores, values
            return {job["mapping"][l]: float(p) for l, p in zip(LABELS, probs)}
        max_tokens = 96 if task == "binding" else (40 if readout == "expression" else 12)
        output = vlm.model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=vlm.processor.tokenizer.eos_token_id,
        )
        length = inputs["input_ids"].shape[1]
        response = vlm.processor.decode(
            output[0, length:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        del output
        return response


def parse_yes_no(s):
    m = re.match(r"^\s*(YES|NO)\b", s, re.I)
    return m.group(1).upper() if m else None


def parse_rule(s, readout):
    text = s.strip().upper()
    if readout == "rule_id":
        m = re.match(r"^(R[123])\b", text)
        return m.group(1) if m else None
    normalized = re.sub(r"\s+", "", text)
    for rule, expression in ex.EXPRESSIONS.items():
        if normalized.startswith(re.sub(r"\s+", "", expression.upper())):
            return rule
    return None


def parse_binding(s):
    found = {}
    for line in s.splitlines():
        m = re.match(r"^\s*([1-4])\s*[:.)-]\s*([01])\s*,?\s*([01])\s*,?\s*([01])"
                     r"\s*[,;:-]?\s*(positive|negative)\b", line, re.I)
        if m:
            found[int(m.group(1))] = ("".join(m.group(i) for i in (2, 3, 4)),
                                     m.group(5).lower())
    return found


def build_jobs(families, smoke=False):
    selected = [f for f in families if f["split"] == "test"]
    if smoke:
        selected = [f for f in selected if f["rule"] == "R1"]
    jobs = []
    for family in selected:
        presentations = [j for j in evidence.presentations(family)
                         if j["condition"] == "both" and j["rotation"] == 0]
        for job in presentations:
            if smoke and job["layout"] != 0:
                continue
            for mode in MODES:
                jobs.append(dict(id=f"{job['job_id']}_{mode}_choice", task="choice",
                                 mode=mode, job=job))
        for mode in ("oracle", "interleaved"):
            for candidate in (("R1", "R2") if smoke else RULES):
                job = presentations[0]
                jobs.append(dict(id=f"{job['job_id']}_{mode}_verify_{candidate}",
                                 task="verify", mode=mode, job=job,
                                 candidate=candidate))
        job = presentations[0]
        jobs.append(dict(id=f"{job['job_id']}_interleaved_binding",
                         task="binding", mode="interleaved", job=job))
        if not smoke:
            for mode in ("oracle", "interleaved"):
                for readout in ("rule_id", "expression"):
                    jobs.append(dict(id=f"{job['job_id']}_{mode}_readout_{readout}",
                                     task="readout", mode=mode, job=job,
                                     readout=readout))
    # Full: 3 families x (9 choices + 6 verifications + 1 binding + 4 readouts) = 60.
    assert len(jobs) == (8 if smoke else 60), len(jobs)
    return jobs


def validate_design(families, jobs):
    evidence.validate(families)
    assert len({j["id"] for j in jobs}) == len(jobs)
    for item in jobs:
        job = item["job"]
        assert evidence.candidates(job["examples"]) == [job["rule"]]
        assert len(job["examples"]) == 4
        if item["task"] == "choice":
            assert set(job["mapping"].values()) == set(RULES)
    held = {f["rule"]: f for f in families if f["split"] == "test"}
    for a, b in (("R1", "R2"), ("R2", "R3")):
        x, y = held[a]["conditions"]["both"], held[b]["conditions"]["both"]
        assert sum(k["filename"] != v["filename"] for k, v in zip(x, y)) == 1
    return held


def result_for(item, response):
    job = item["job"]
    task, truth, mode = item["task"], job["rule"], item["mode"]
    row = {"id": item["id"], "task": task, "mode": mode, "family": job["family_id"],
           "truth": truth, "layout": job["layout"]}
    if task == "choice":
        probs = response
        best = max(probs.values())
        winners = [r for r in RULES if abs(probs[r] - best) <= 1e-8]
        row.update({f"p_{r}": probs[r] for r in RULES})
        row.update(prediction="|".join(winners), correct=float(truth in winners) / len(winners),
                   p_true=probs[truth],
                   chosen_label="|".join(l for l in LABELS if job["mapping"][l] in winners))
    elif task == "verify":
        expected = "YES" if item["candidate"] == truth else "NO"
        predicted = parse_yes_no(response)
        row.update(candidate=item["candidate"], expected=expected, prediction=predicted,
                   correct=int(predicted == expected), raw=response)
    elif task == "readout":
        predicted = parse_rule(response, item["readout"])
        row.update(readout=item["readout"], prediction=predicted,
                   correct=int(predicted == truth), raw=response)
    else:
        parsed = parse_binding(response)
        scene_correct = 0
        label_correct = 0
        for i, exm in enumerate(job["examples"], 1):
            got_bits, got_label = parsed.get(i, ("", ""))
            scene_correct += got_bits == "".join(map(str, exm["bits"]))
            label_correct += got_label == ("positive" if exm["positive"] else "negative")
        row.update(scene_correct=scene_correct, label_correct=label_correct,
                   parsed_lines=len(parsed), correct=int(scene_correct == 4 and label_correct == 4),
                   raw=response)
    return row


def write_outputs(rows, out):
    with (out / "results.json").open("w") as f:
        json.dump(rows, f, indent=2)
    fields = sorted({k for row in rows for k in row})
    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    print("\n" + "=" * 65 + "\nEXPERIMENT 04 — CONSOLE SUMMARY\n" + "=" * 65)
    for mode in MODES:
        data = [r for r in rows if r["task"] == "choice" and r["mode"] == mode]
        if data:
            accuracy = np.mean([r["correct"] for r in data])
            avgp = np.mean([r["p_true"] for r in data])
            counts = Counter(l for r in data for l in r["chosen_label"].split("|"))
            print(f"Rule choice [{mode:11s}] {accuracy:.1%} correct; "
                  f"mean correct-rule probability {avgp:.1%}; "
                  f"X/Y/Z choices: {counts['X']}/{counts['Y']}/{counts['Z']}")
    for mode in ("oracle", "interleaved"):
        data = [r for r in rows if r["task"] == "verify" and r["mode"] == mode]
        if data:
            by_family = defaultdict(list)
            for row in data:
                by_family[row["family"]].append(row["correct"])
            print(f"Candidate checks [{mode:11s}] {np.mean([r['correct'] for r in data]):.1%} "
                  f"individual; {sum(all(x) for x in by_family.values())}/{len(by_family)} "
                  "families fully correct")
    for row in (r for r in rows if r["task"] == "binding"):
        print(f"Binding [{row['truth']}] bits {row['scene_correct']}/4, "
              f"labels {row['label_correct']}/4, parsed {row['parsed_lines']}/4")
        if not row["correct"]:
            print("  Model output:", repr(row["raw"][:180]))
    for mode in ("oracle", "interleaved"):
        for form in ("rule_id", "expression"):
            subset = [r for r in rows if r["task"] == "readout"
                      and r["mode"] == mode and r["readout"] == form]
            if subset:
                print(f"Generated answer [{mode}, {form}] "
                      f"{sum(r['correct'] for r in subset)}/{len(subset)}")
    for mode in MODES:
        data = {(r["truth"], r["layout"]): r for r in rows
                if r["task"] == "choice" and r["mode"] == mode}
        for a, b in (("R1", "R2"), ("R2", "R3")):
            pairs = [(data[a, layout], data[b, layout]) for layout in range(3)
                     if (a, layout) in data and (b, layout) in data]
            if pairs:
                tv = [sum(abs(x[f"p_{r}"] - y[f"p_{r}"]) for r in RULES) / 2
                      for x, y in pairs]
                print(f"Counterfactual [{mode}] {a}->{b}: "
                      f"mean distribution change {np.mean(tv):.3f}")
    wrong = [r for r in rows if r.get("correct") == 0]
    for row in wrong[:5]:
        print(f"Example: {row['task']}/{row['mode']}/{row['truth']}: "
              f"predicted {row.get('prediction', row.get('raw', ''))!s}"[:180])
    print("\nInterpretation: only three held-out image-sharing families; "
          "many calls are repeated presentations. "
          "Restricted X/Y/Z probabilities are conditional on those tokens.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    root = Path(".") if args.validate_only else require_results_root(args.experiment_dir)
    out = root / "bottleneck-controls" / ("smoke" if args.smoke_test else "quick")
    ex.configure_data(args.data_dir, out)
    families = evidence.make_dataset(ex.records)
    jobs = build_jobs(families, smoke=args.smoke_test)
    validate_design(families, jobs)
    print(f"Validated {len(families)} families; {len(jobs)} planned inference calls.")
    if args.validate_only:
        print("CPU-only validation complete. No model loaded.")
        return
    out.mkdir(parents=True, exist_ok=True)
    config = {"experiment": 4, "model": ex.MODEL_ID,
              "jobs": [item["id"] for item in jobs],
              "data_hash": ex.DATA_HASH}
    path = out / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Saved run has a different configuration; use a fresh output folder.")
    path.write_text(json.dumps(config, indent=2))
    checkpoint = out / "results.jsonl"
    rows = [json.loads(x) for x in checkpoint.read_text().splitlines() if x.strip()] \
        if checkpoint.exists() else []
    done = {r["id"] for r in rows}
    if not done <= set(config["jobs"]) or len(done) != len(rows):
        raise ValueError("Checkpoint contains duplicate or unknown IDs.")
    if len(rows) < len(jobs):
        import torch
        torch.manual_seed(2026)
        ex.load_native_model(ex.MODEL_ID, "4bit")
        print("Model loaded. Console shows every result; files are also saved.")
        for i, item in enumerate(jobs, 1):
            if item["id"] in done:
                print(f"[{i:02d}/{len(jobs)}] resume: already completed")
                continue
            response = infer(ex.vlm, item["job"], item["mode"], item["task"],
                             item.get("candidate"), item.get("readout"))
            row = result_for(item, response)
            rows.append(row)
            tmp = out / "results.tmp.jsonl"
            tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
            tmp.replace(checkpoint)
            print(f"[{i:02d}/{len(jobs)}] {row['task']:7s} {row['mode']:11s} "
                  f"{row['truth']} -> {row.get('prediction', str(row.get('scene_correct', '-')) + '/4')} "
                  f"{'OK' if row['correct'] else 'CHECK'}", flush=True)
    write_outputs(rows, out)
    summarize(rows)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
