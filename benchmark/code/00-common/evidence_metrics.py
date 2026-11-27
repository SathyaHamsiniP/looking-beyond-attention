"""Score complete matched runs. Malformed responses count as errors."""

from collections import defaultdict
from itertools import combinations, groupby
from statistics import mean

from evidence_design import RULES, solve


def summarize(records):
    values = defaultdict(list)

    def record(name, value):
        values[name].append(float(value))

    by_set = defaultdict(list)
    by_group = defaultdict(list)
    for row in records:
        by_set[row["set_id"]].append(row)
        by_group[row["group"], row["rotation"]].append(row)
        gold, pred = set(row["answers"]), row["prediction"]
        hit = pred is not None and set(pred) == gold
        record("exact", hit)
        record("invalid_format", pred is None)
        record(f"exact/candidates-{len(gold)}", hit)
        record("exact/" + row["condition"], hit)
        overlap = len(gold.intersection(pred or []))
        record("precision_macro_set", overlap / len(pred) if pred else 0)
        record("recall_macro_set", overlap / len(gold))
        if row["condition"] != "progressive":
            record(f"exact/member-{row['member']}", hit)
            record("exact/" + row["answers"][0], hit)

    for rule in RULES:
        rid = rule["id"]
        positives = [r for r in records if rid in r["answers"]]
        selected = [r for r in records if rid in (r["prediction"] or [])]
        for row in positives:
            record("recall/" + rid, rid in (row["prediction"] or []))
        for row in selected:
            record("precision/" + rid, rid in row["answers"])
    for rows in by_set.values():
        for left, right in combinations(rows, 2):
            record(
                "order_agreement",
                left["prediction"] is not None
                and left["prediction"] == right["prediction"],
            )
    for rows in by_group.values():
        rows.sort(key=lambda r: r["member"])
        if rows[0]["condition"] == "progressive":
            absent = set()
            for a, b in zip(rows, rows[1:]):
                removed = set(a["answers"]) - set(b["answers"])
                p, q = a["prediction"], b["prediction"]
                if removed:
                    record(
                        "correct_elimination",
                        p is not None
                        and q is not None
                        and removed <= set(p)
                        and removed.isdisjoint(q),
                    )
                if p is not None:
                    absent.update(
                        r["id"]
                        for r in RULES
                        if r["id"] not in a["answers"] and r["id"] not in p
                    )
                if absent and q is not None:
                    record("invalid_rule_reintroduced", bool(absent.intersection(q)))
            continue
        a, b = rows
        p, q = a["prediction"], b["prediction"]
        kind = a["condition"]
        both = p == a["answers"] and q == b["answers"]
        record("both_correct/" + kind, both)
        if kind == "control":
            if p is not None and q is not None:
                record("incorrect_switch/control", p != q)
        else:
            record("correct_switch/" + kind, both)
            if p is not None and q is not None:
                record("any_switch/" + kind, p != q)
    return dict(
        sets=len(by_set),
        groups=len({r["group"] for r in records}),
        presentations=len(records),
        rates={k: mean(v) for k, v in values.items()},
        denominators={k: len(v) for k, v in values.items()},
    )


def baselines(row):
    # Ground-truth annotations give these shortcuts perfect perception.
    index = row["focal"] if row["focal"] is not None else 0
    candidates = solve([row["bits"][index]])
    marginals = [mean(int(b[i]) for b in row["bits"]) for i in range(3)]

    def distance(rule):
        return sum(
            (mean(int(b[i]) for b in rule["true_bits"]) - m) ** 2
            for i, m in enumerate(marginals)
        )

    return {
        "single_image": [candidates[0]],
        "marginal_frequency": [min(RULES, key=distance)["id"]],
    }


def paired_delta(left, right):
    """Matched exact accuracy change, equally weighted by chain or pair."""

    def groups(rows):
        return {
            key: mean(r["answers"] == r["prediction"] for r in group)
            for key, group in groupby(
                sorted(rows, key=lambda r: r["group"]), key=lambda r: r["group"]
            )
        }

    a, b = groups(left), groups(right)
    if a.keys() != b.keys():
        raise ValueError("Unmatched groups")
    diffs = [b[k] - a[k] for k in a]
    return dict(
        groups=len(diffs),
        mean=mean(diffs),
        improved=sum(d > 0 for d in diffs),
        tied=diffs.count(0),
        declined=sum(d < 0 for d in diffs),
    )
