"""Build persona-construction material from the private counseling corpora.

Mirrors PCSA's persona initialization: client characteristics + cognitive
distortions + dialogue-style references are drawn from counseling dialogues
(Cactus) and CBT practice material (CBT-Bench/CBT-DP). Output is written to
``data/processed/personas.jsonl`` (gitignored, like all corpus-derived data) and
is consumed by ``data_sources.load_personas``. When the raw corpora are absent,
the loader falls back to the committed synthetic personas instead.

Attack goals are NOT built here; they live in the separate committed
``data/attack_goals.jsonl`` set, exactly as the paper separates the two.

Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def first_client_line(dialogue: str) -> str:
    for line in dialogue.splitlines():
        if line.strip().startswith("Client:"):
            return line.split("Client:", 1)[1].strip()[:200]
    return ""


def build(cactus_path: Path, out_path: Path, limit: int, seed: int) -> dict:
    rows = json.loads(cactus_path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    personas, seen = [], set()
    for row in rows:
        intake = (row.get("intake_form") or "").strip()
        patterns = row.get("patterns") or []
        style = first_client_line(row.get("dialogue", ""))
        key = intake[:120]
        if not intake or not patterns or not style or key in seen:
            continue
        seen.add(key)
        personas.append({
            "persona_id": f"cactus-{len(personas):04d}",
            "descriptor": re.sub(r"\s+", " ", intake)[:300],
            "cognitive_distortion": ", ".join(patterns[:3]),
            "style_reference": style,
            "source": "cactus",
        })
        if len(personas) >= limit:
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for persona in personas:
            f.write(json.dumps(persona, ensure_ascii=False) + "\n")
    return {"written": len(personas), "out": str(out_path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cactus", type=Path, default=ROOT / "data" / "raw" / "cactus.json")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "personas.jsonl")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261002)
    args = p.parse_args()
    if not args.cactus.exists():
        raise SystemExit(f"raw corpus not found: {args.cactus} (download the private dataset first)")
    print(json.dumps(build(args.cactus, args.out, args.limit, args.seed), ensure_ascii=False))


if __name__ == "__main__":
    main()
