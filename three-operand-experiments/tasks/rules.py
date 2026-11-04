"""Explicit truth tables; no executable expressions are read from manifests."""

from itertools import product

BITS = list(product((0, 1), repeat=3))
EXPRESSIONS = {
    "R1": "(A AND B) OR C",
    "R2": "(A AND C) OR B",
    "R3": "(B AND C) OR A",
    "R4": "(A OR B) AND C",
    "R5": "(A XOR B) OR C",
    "R6": "(A AND NOT B) OR C",
    "R7": "A XOR (B OR C)",
    "R8": "(A OR C) AND NOT B",
}


def evaluate(rule, bits):
    a, b, c = map(bool, bits)
    return bool(
        {
            "R1": (a and b) or c,
            "R2": (a and c) or b,
            "R3": (b and c) or a,
            "R4": (a or b) and c,
            "R5": (a != b) or c,
            "R6": (a and not b) or c,
            "R7": a != (b or c),
            "R8": (a or c) and not b,
        }[rule]
    )


def library(version):
    return list(EXPRESSIONS) if version.startswith("v3") else list(EXPRESSIONS)[:3]


def truth_tables(rules):
    return {r: [int(evaluate(r, b)) for b in BITS] for r in rules}


def consistent(observations, rules):
    return [
        r
        for r in rules
        if all(evaluate(r, o["bits"]) == o["label"] for o in observations)
    ]
