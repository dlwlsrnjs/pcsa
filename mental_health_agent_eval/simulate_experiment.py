"""Paired simulation validating the profile-aware experiment design.

No LLM is called. Outcomes are generated from declared synthetic target parameters,
so they must never be reported as empirical model safety results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from agents import AXES, AXIS_CONSTRUCT

TARGETS = {
    "balanced": {"role_boundary": .15, "uncertainty_calibration": .15, "emotion_belief_separation": .15, "risk_escalation": .15},
    "rapport_first": {"role_boundary": .25, "uncertainty_calibration": .30, "emotion_belief_separation": .85, "risk_escalation": .35},
    "overconfident": {"role_boundary": .70, "uncertainty_calibration": .90, "emotion_belief_separation": .25, "risk_escalation": .30},
    "boundary_weak": {"role_boundary": .85, "uncertainty_calibration": .65, "emotion_belief_separation": .45, "risk_escalation": .80},
}


def stable_uniform(seed: int, target: str, case_id: str) -> float:
    raw = hashlib.sha256(f"{seed}|{target}|{case_id}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") / 2**64


def probability(condition: str, weakness: float) -> float:
    # Shared base difficulty; profile-aware expends its probe budget on the weakest axis.
    if condition == "fixed":
        return .03 + .08 * weakness
    if condition == "profile_blind":
        return .04 + .14 * weakness
    if condition == "profile_aware":
        return .04 + (.48 if weakness >= .5 else .14) * weakness
    raise ValueError(condition)


def bootstrap_delta(pairs, seed: int, repetitions: int = 5000):
    rng = random.Random(seed)
    deltas = []
    for _ in range(repetitions):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(sum(a - b for a, b in sample) / len(sample))
    deltas.sort()
    return [round(deltas[int(.025 * repetitions)], 4), round(deltas[int(.975 * repetitions)], 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases-per-axis", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--out", type=Path, default=Path("runs/simulation.jsonl"))
    p.add_argument("--summary", type=Path, default=Path("../reports/simulation_summary.json"))
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows, outcomes = [], defaultdict(dict)
    for target, weaknesses in TARGETS.items():
        observed_profile = {key: round(1 - value, 3) for key, value in weaknesses.items()}
        for axis in AXES:
            weakness = weaknesses[AXIS_CONSTRUCT[axis]]
            for i in range(args.cases_per_axis):
                case_id = f"{axis}-{i:04d}"
                draw = stable_uniform(args.seed, target, case_id)
                for condition in ("fixed", "profile_blind", "profile_aware"):
                    failed = draw < probability(condition, weakness)
                    outcomes[(target, axis, case_id)][condition] = failed
                    rows.append({"simulation": True, "seed": args.seed, "target": target,
                                 "declared_weakness": weakness, "observed_profile": observed_profile,
                                 "axis": axis, "case_id": case_id, "condition": condition,
                                 "failure_probability": round(probability(condition, weakness), 4),
                                 "failure": failed})
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rates = {}
    for condition in ("fixed", "profile_blind", "profile_aware"):
        values = [v[condition] for v in outcomes.values()]
        rates[condition] = round(sum(values) / len(values), 4)
    pairs = [(int(v["profile_aware"]), int(v["profile_blind"])) for v in outcomes.values()]
    summary = {
        "simulation_only": True, "seed": args.seed, "targets": len(TARGETS),
        "cases_per_target": args.cases_per_axis * len(AXES), "paired_cases": len(pairs),
        "failure_rates": rates,
        "aware_minus_blind": round(sum(a - b for a, b in pairs) / len(pairs), 4),
        "paired_bootstrap_95_ci": bootstrap_delta(pairs, args.seed),
        "discordant_pairs": {"aware_only": sum(a and not b for a, b in pairs),
                             "blind_only": sum(b and not a for a, b in pairs)},
        "warning": "Synthetic design validation; not an LLM experiment or safety claim.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
