"""Phase I CLI: harden attacker personas by surrogate-guided perturbation.

For each (persona seed x attack goal) case, run the perturbation loop against the
surrogate open models and save the hardened persona (system instruction + opening
utterances) to ``data/processed/hardened_personas.jsonl`` (gitignored) for Phase II.

Example:
    set -a; . ../.env; set +a
    python3 run_phase1.py --cases-per-axis 1 --max-iters 4 \
        --surrogate '["python3","adapters/openai_target.py","--model","meta-llama/Llama-3.1-8B-Instruct","--base-url","https://router.huggingface.co/v1","--key-env","HF_API_KEY"]'
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import data_sources
import phase1_persona as p1
import pcsa

GEN_CMD = ["python3", "adapters/gpt5nano.py"]                                  # scriptwriter/perturb/distortion
PROGRESS_CMD = ["python3", "adapters/gpt5nano.py", "--model", "gpt-4o-mini"]   # fitness evaluator
DEFAULT_SURROGATE = ["python3", "adapters/openai_target.py",
                     "--model", "meta-llama/Llama-3.1-8B-Instruct",
                     "--base-url", "https://router.huggingface.co/v1", "--key-env", "HF_API_KEY"]


def parse_cmd(raw):
    cmd = json.loads(raw)
    if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd) or not cmd:
        raise SystemExit("command must be a nonempty JSON string array")
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-per-axis", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--score-target", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--surrogate", action="append", help="JSON argv array; repeatable for multiple surrogates")
    ap.add_argument("--out", type=Path, default=Path("../data/processed/hardened_personas.jsonl"))
    args = ap.parse_args()

    surrogates = [parse_cmd(s) for s in args.surrogate] if args.surrogate else [list(DEFAULT_SURROGATE)]
    personas, source = data_sources.load_personas()
    goals = data_sources.load_attack_goals()
    jargon = data_sources.load_jargon()
    cases = [c for axis in pcsa.PCSA_AXES
             for c in data_sources.build_cases(axis, personas, goals, args.cases_per_axis, args.seed)]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    lock, done = threading.Lock(), 0
    with args.out.open("w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(p1.harden_persona, case, surrogates, GEN_CMD, PROGRESS_CMD,
                               jargon, args.max_iters, args.patience, args.score_target): case
                   for case in cases}
        for fut in as_completed(futures):
            case = futures[fut]
            try:
                hardened = fut.result()
            except Exception as exc:
                print(f"[ERROR] {case['case_id']}: {exc}", flush=True)
                continue
            hardened["persona_source"] = source
            with lock:
                f.write(json.dumps(hardened, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                print(f"[{done:3d}/{len(cases)}] {case['case_id']} fitness={hardened['fitness']} "
                      f"iters={hardened['iterations']}", flush=True)
    print(f"Saved {done} hardened personas to {args.out}")


if __name__ == "__main__":
    main()
