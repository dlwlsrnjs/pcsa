import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agents import AXES, profile_target, respond, select_approved_message


def parse_command(raw, parser):
    command = json.loads(raw) if raw else None
    if command is not None and (not isinstance(command, list) or not command or
                                not all(isinstance(x, str) for x in command)):
        parser.error("adapter command must be a nonempty JSON string array")
    return command


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/demo-v2.jsonl")
    p.add_argument("--turns", type=int, default=2)
    p.add_argument("--condition", choices=("fixed", "profile_blind", "profile_aware"), default="profile_aware")
    p.add_argument("--target-command", help="JSON array of trusted executable argv")
    p.add_argument("--analyzer-command", help="JSON array for the calibration scoring adapter")
    p.add_argument("--target-model", default="mock-safe-v1")
    args = p.parse_args()
    if not 1 <= args.turns <= 8:
        p.error("turns must be 1..8")
    target = parse_command(args.target_command, p)
    analyzer = parse_command(args.analyzer_command, p)

    if args.condition == "fixed":
        profile, calibration = {}, []
    else:
        profile, calibration = profile_target(target, analyzer)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        p.error("output exists; choose a new output path")

    run_id = datetime.now(timezone.utc).isoformat()
    with output.open("x", encoding="utf-8") as f:
        for i, axis in enumerate(AXES):
            history, decisions = [], []
            for turn in range(args.turns):
                persona, message, variant = select_approved_message(axis, turn, args.condition, profile)
                decisions.append({"turn": turn, "axis": axis, "persona": persona, "variant": variant})
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": respond(history, target)})
            record = {
                "schema_version": "2.0", "run_id": run_id,
                "case": {"id": f"{args.condition}-fixture-{i}", "axis": axis, "synthetic": True},
                "condition": args.condition, "history": history, "decisions": decisions,
                "behavior_profile": profile, "calibration": calibration,
                "target_model": args.target_model, "target_mode": "adapter" if target else "mock",
                "analyzer_mode": "adapter" if analyzer else "mock",
                "labels": {a: None for a in AXES}, "evidence": {},
                "reviewer_id": None, "status": "unreviewed",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Created {output}; calibration is descriptive and no safety outcomes were assigned.")


if __name__ == "__main__":
    main()
