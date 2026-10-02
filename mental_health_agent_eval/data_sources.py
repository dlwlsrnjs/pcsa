"""Dataset loaders with a clean separation of concerns, mirroring PCSA.

PCSA draws *persona construction* material (client characteristics, cognitive
distortions, dialogue style) from counseling corpora (Cactus, CBT-Bench/CBT-DP,
Cheeseburger Therapy), and draws *attack goals* from a separate mental-health
safety-probe set that CARES/the judge evaluates responses against. Our
profile-aware agent adds a third, agent-specific source: a benign *calibration
probe* set used only to measure the target's behavioral profile.

Sources handled here:
- ``personas``          : corpus-derived personas (``data/processed/personas.jsonl``,
                          built by ``build_personas_from_corpus.py`` when the raw
                          private corpora are present) OR the committed approved
                          fallback (``data/personas_fallback.jsonl``).
- ``attack_goals``      : committed synthetic, approved, intent-level goals
                          (``data/attack_goals.jsonl``) — separate from personas.
- ``calibration_probes``: benign profiling probes (``data/calibration_probes.jsonl``).

Persona and goal are *attacker-side conditions*: they shape client-utterance
generation and are never sent to the target as separate fields. The target only
ever receives the generated client turns and the accumulating dialogue history.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_personas(processed: Path | None = None, fallback: Path | None = None) -> tuple[list[dict], str]:
    """Return (personas, source_tag). Prefer corpus-derived personas, else fallback."""
    processed = processed or (DATA_DIR / "processed" / "personas.jsonl")
    fallback = fallback or (DATA_DIR / "personas_fallback.jsonl")
    if processed.exists():
        rows = load_jsonl(processed)
        if rows:
            return rows, "corpus_derived"
    return load_jsonl(fallback), "synthetic_fallback"


def load_attack_goals(path: Path | None = None) -> list[dict]:
    return load_jsonl(path or (DATA_DIR / "attack_goals.jsonl"))


def load_susceptibility_probes(path: Path | None = None) -> list[dict]:
    """Fixed, approved probe battery (axis x strategy x register) for the pre-test."""
    path = path or (DATA_DIR / "susceptibility_probes.jsonl")
    return load_jsonl(path) if path.exists() else []


def load_jargon(path: Path | None = None) -> list[str]:
    """Psychiatric/medical term seeds for the Phase I complexity perturbation."""
    path = path or (DATA_DIR / "psych_jargon.jsonl")
    if not path.exists():
        return []
    return [r["term"] for r in load_jsonl(path) if r.get("term")]


def load_calibration_probes(path: Path | None = None) -> list[dict]:
    path = path or (DATA_DIR / "calibration_probes.jsonl")
    if path.exists():
        return load_jsonl(path)
    from agents import CALIBRATION_PROBES  # stdlib default if the file is absent
    return list(CALIBRATION_PROBES)


def build_cases(axis: str, personas: list[dict], goals: list[dict], k: int, seed: int) -> list[dict]:
    """Deterministically pair goals (axis-specific) with personas to form k cases."""
    axis_goals = [g for g in goals if g.get("axis") == axis]
    if not axis_goals:
        raise ValueError(f"no attack goals for axis {axis}")
    rng = random.Random(f"{seed}|{axis}")
    cases = []
    for i in range(k):
        goal = axis_goals[i % len(axis_goals)]
        persona = personas[rng.randrange(len(personas))]
        cases.append({"case_id": f"{axis}-{i:03d}", "axis": axis, "persona": persona, "goal": goal})
    return cases
