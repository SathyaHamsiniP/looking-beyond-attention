"""Positive-only evidence designs. Bits and filenames stay on the evaluator side."""

from collections import Counter, defaultdict
from itertools import combinations, permutations
import random

from benchmark_paths import source_path
from build_sets import compatible_rules
from compare_3_vs_7 import validate_rules

RULES = validate_rules(source_path("rules_7.json"))["rules"]
BITS = tuple(f"{i:03b}" for i in range(8))


def solve(bits):
    if not set(bits) <= set(BITS):
        raise ValueError("Unknown A/B/C configuration")
    return compatible_rules(bits, RULES)


def unique_sets():
    results = defaultdict(list)
    for size in range(1, len(BITS) + 1):
        for bits in combinations(BITS, size):
            answer = solve(bits)
            if len(answer) == 1:
                results[answer[0]].append(bits)
    return results


def example(group, member, condition, bits, files, focal=None):
    answers = solve(bits)
    if not answers or len(bits) != len(files):
        raise ValueError("Invalid design")
    return dict(
        set_id=f"{group}-{member}",
        group=group,
        member=member,
        condition=condition,
        bits=list(bits),
        image_files=list(files),
        answers=answers,
        focal=focal,
    )


def chains(allocate, rng, repeats):
    rows = []
    sizes = Counter()
    for rep in range(repeats):
        for rule, options in unique_sets().items():
            # Sample equivalent endpoints, then favor underrepresented ambiguity levels.
            paths = list(permutations(rng.choice(options)))
            rng.shuffle(paths)

            def score(path):
                counts = Counter(len(solve(path[:i])) for i in range(1, len(path) + 1))
                return sum(1 / (1 + sizes[k]) for k in counts)

            path = max(paths, key=score)
            files = [allocate(b) for b in path]
            for i in range(1, len(path) + 1):
                row = example(f"{rep}-{rule}", i, "progressive", path[:i], files[:i])
                sizes[len(row["answers"])] += 1
                rows.append(row)
    return rows


def pairs(allocate, rng, repeats):
    unique = unique_sets()
    rows = []

    def add(group, kind, left, right, focal=None):
        # Keep all shared configurations in their original positions and pixels.
        left = list(left)
        rng.shuffle(left)
        additions = sorted(set(right) - set(left))
        rng.shuffle(additions)
        right = [b if b in right else additions.pop() for b in left]
        files = {b: allocate(b) for b in sorted(set(left + right))}
        first = [files[b] for b in left]
        second = [files[b] for b in right]
        if kind == "control":
            i = rng.randrange(len(right))
            second[i] = allocate(right[i], exclude=second[i])
        focus = left.index(focal) if focal else None
        rows.extend(
            [
                example(group, 0, kind, left, first, focus),
                example(group, 1, kind, right, second, focus),
            ]
        )

    for rep in range(repeats):
        for rule, options in unique.items():
            bits = rng.choice(options)
            add(f"{rep}-control-{rule}", "control", bits, bits)
        for a, b in combinations(sorted(unique), 2):
            for kind in ("replacement", "context"):
                matches = []
                for x in unique[a]:
                    for y in unique[b]:
                        changes = len(set(x) - set(y))
                        shared = sorted(set(x) & set(y))
                        if (
                            len(x) == len(y)
                            and shared
                            and (
                                changes == 1 if kind == "replacement" else changes >= 2
                            )
                        ):
                            matches.append((x, y, shared))
                if matches:
                    left, right, shared = rng.choice(matches)
                    if rng.randrange(2):
                        left, right = right, left
                    focal = rng.choice(shared) if kind == "context" else None
                    add(f"{rep}-{kind}-{a}-{b}", kind, left, right, focal)
    return rows


def design(number, pool, seed, repeats):
    if repeats < 1 or any(len(pool[b]) < 2 for b in BITS):
        raise ValueError(
            "Need positive repeats and two fresh variants per configuration"
        )
    rng = random.Random(seed)
    # Shuffle once to make tie-breaking deterministic without tying logical
    # sampling to pool size. Least-used files minimize repeat allocations.
    image_rng = random.Random(f"{seed}:images:{number}")
    variants = {bits: sorted(files) for bits, files in pool.items()}
    for files in variants.values():
        image_rng.shuffle(files)
    usage = Counter()

    def allocate(bits, exclude=None):
        file = min((f for f in variants[bits] if f != exclude), key=usage.__getitem__)
        usage[file] += 1
        return file

    rows = (chains if number == 11 else pairs)(allocate, rng, repeats)
    grouped = defaultdict(list)
    for row in rows:
        inferred = [next(b for b in BITS if f in pool[b]) for f in row["image_files"]]
        if inferred != row["bits"] or solve(inferred) != row["answers"]:
            raise ValueError("Evidence and manifest answer disagree")
        grouped[row["group"]].append(row)
    for group in grouped.values():
        for a, b in zip(group, group[1:]):
            if number == 11:
                valid = a["image_files"] == b["image_files"][:-1] and set(
                    b["answers"]
                ) <= set(a["answers"])
            else:
                changes = sum(x != y for x, y in zip(a["bits"], b["bits"], strict=True))
                pixels = sum(
                    x != y
                    for x, y in zip(a["image_files"], b["image_files"], strict=True)
                )
                kind = a["condition"]
                if kind == "control":
                    valid = changes == 0 and pixels == 1
                elif kind == "replacement":
                    valid = changes == pixels == 1
                else:
                    i = a["focal"]
                    valid = (
                        changes == pixels
                        and changes >= 2
                        and a["image_files"][i] == b["image_files"][i]
                    )
                valid = (
                    valid
                    and len(a["answers"]) == len(b["answers"]) == 1
                    and ((a["answers"] == b["answers"]) == (kind == "control"))
                )
            if not valid:
                raise ValueError(f"Invalid matched evidence in {a['group']}")
    return rows


def allocation_audit(rows, pool):
    """Count each file once per chain or pair, ignoring deliberate within-group reuse."""
    groups = defaultdict(set)
    for row in rows:
        groups[row["group"]].update(row["image_files"])
    usage = Counter(file for files in groups.values() for file in files)
    by_configuration = {}
    for bits, files in pool.items():
        demands = [usage[f] for f in files]
        by_configuration[bits] = dict(
            available=len(files),
            required_without_reuse=sum(demands),
            distinct_used=sum(n > 0 for n in demands),
            repeated_allocations=sum(max(n - 1, 0) for n in demands),
        )
    slots = sum(usage.values())
    required = max(c["required_without_reuse"] for c in by_configuration.values())
    return dict(
        group_image_slots=slots,
        distinct_files=len(usage),
        repeated_allocations=slots - len(usage),
        files_in_multiple_groups=sum(n > 1 for n in usage.values()),
        groups_with_shared_files=sum(
            any(usage[f] > 1 for f in files) for files in groups.values()
        ),
        balanced_pool_for_disjoint_groups=len(BITS) * required,
        by_configuration=by_configuration,
    )
