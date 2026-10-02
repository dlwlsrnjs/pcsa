"""Reproducible structural audit for the PCSA source datasets (stdlib only)."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


def summarize_cactus(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    patterns, techniques, attitudes = map(collections.Counter, ({}, {}, {}))
    hashes, duplicate_count, turn_counts, dialogue_chars = set(), 0, [], []
    email_re = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
    phone_re = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
    pii_like = collections.Counter()
    required = {"thought", "patterns", "intake_form", "cbt_technique", "cbt_plan", "attitude", "dialogue"}
    missing = collections.Counter()
    for row in rows:
        for key in required - row.keys():
            missing[key] += 1
        patterns.update(row.get("patterns") or [])
        techniques.update([row.get("cbt_technique", "<missing>")])
        attitudes.update([row.get("attitude", "<missing>")])
        dialogue = row.get("dialogue", "")
        digest = hashlib.sha256((row.get("thought", "") + "\0" + dialogue).encode()).hexdigest()
        duplicate_count += digest in hashes
        hashes.add(digest)
        turn_counts.append(len(re.findall(r"(?m)^(?:Counselor|Client):", dialogue)))
        dialogue_chars.append(len(dialogue))
        text = row.get("intake_form", "") + "\n" + dialogue
        pii_like["email"] += bool(email_re.search(text))
        pii_like["phone"] += bool(phone_re.search(text))
    return {
        "rows": len(rows), "fields": sorted(set().union(*(r.keys() for r in rows))),
        "missing_required": dict(missing), "exact_content_duplicates": duplicate_count,
        "attitudes": dict(attitudes), "unique_patterns": len(patterns),
        "top_patterns": patterns.most_common(20), "unique_cbt_techniques": len(techniques),
        "top_cbt_techniques": techniques.most_common(20),
        "turns": stats(turn_counts), "dialogue_characters": stats(dialogue_chars),
        "pii_like_regex_hits": dict(pii_like),
        "privacy_note": "Synthetic personas still contain realistic names/demographics; treat as sensitive research data.",
    }


def summarize_cbt_dp(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    statements = {r.get("Input.client_statement", "") for r in rows}
    exercises = collections.Counter(r.get("Input.exe", "<missing>") for r in rows)
    levels = collections.Counter(r.get("Input.level", "<missing>") for r in rows)
    orders = collections.Counter(r.get("Input.order", "<missing>") for r in rows)
    ids = [r.get("Input.id") for r in rows]
    return {
        "comparison_rows": len(rows), "unique_client_statements": len(statements),
        "unique_input_ids": len(set(ids)), "duplicate_input_ids": len(ids) - len(set(ids)),
        "exercises": dict(exercises), "levels": dict(levels), "presentation_order": dict(orders),
        "fields": sorted(set().union(*(r.keys() for r in rows))),
        "note": "Pairwise-comparison rows repeat the 156 underlying exercises across systems/orderings.",
    }


def stats(values):
    values = sorted(values)
    if not values:
        return {}
    n = len(values)
    return {"min": values[0], "median": values[n // 2], "mean": round(sum(values) / n, 2), "max": values[-1]}


def write_candidates(cbt_path: Path, output: Path):
    rows = json.loads(cbt_path.read_text(encoding="utf-8"))
    seen, candidates = set(), []
    high_risk = re.compile(r"\b(?:suicid|self[- ]?harm|kill myself|overdose|starv|purge)\b", re.I)
    for row in rows:
        statement = row.get("Input.client_statement", "").strip()
        if not statement or statement in seen:
            continue
        seen.add(statement)
        candidates.append({
            "candidate_id": f"cbt-dp-{len(candidates):03d}", "source": "CBT-Bench/CBT-DP",
            "client_statement": statement, "exercise": row.get("Input.exe"),
            "level": row.get("Input.level"), "automated_risk_flag": bool(high_risk.search(statement)),
            "review_status": "needs_human_review",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"candidate_rows": len(candidates), "risk_flagged": sum(r["automated_risk_flag"] for r in candidates)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cactus", type=Path, default=Path("data/raw/cactus.json"))
    p.add_argument("--cbt-dp", type=Path, default=Path("data/raw/cbt-bench-dp-pairwise-comparison.json"))
    p.add_argument("--out", type=Path, default=Path("reports/dataset_audit.json"))
    p.add_argument("--candidates", type=Path, default=Path("data/processed/cbt_dp_candidates.jsonl"))
    args = p.parse_args()
    report = {
        "cactus": summarize_cactus(args.cactus), "cbt_dp": summarize_cbt_dp(args.cbt_dp),
        "candidate_export": write_candidates(args.cbt_dp, args.candidates),
        "cheeseburger_therapy": {
            "downloaded": False, "reported_sessions": 116,
            "reason": "No public dataset artifact found; source paper obtained it by request with licensing/consent and IRB approval.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
