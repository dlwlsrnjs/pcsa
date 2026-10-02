# Dataset acquisition and smoke experiment report

Date: 2026-10-02 (Asia/Seoul)

## Acquisition status

| Dataset | Status | Local artifact | Notes |
|---|---|---|---|
| Cactus | Downloaded | `data/raw/cactus.json` | Official Hugging Face artifact; SHA-256 matches upstream LFS OID |
| CBT-Bench / CBT-DP | Downloaded | `data/raw/cbt-bench-dp-pairwise-comparison.json` | Official Hugging Face artifact |
| Cheeseburger Therapy | Not publicly downloadable | None | Human-session data was provided to the CHI paper authors under licensing/consent and IRB; no general download/reuse artifact found |

## Structural results

- Cactus: 31,577 rows, 21–35 turns per dialogue (mean 30.53), 0 exact thought+dialogue duplicates, 11 cognitive-pattern labels, 34 raw technique spellings, and no email/phone regex matches. Synthetic intake forms contain realistic names and demographics and remain sensitive research data.
- Cactus attitudes: negative 9,469; neutral 10,882; positive 11,226.
- CBT-DP: 468 pairwise-comparison rows, 156 unique client statements, 10 exercise types. The rows repeat underlying exercises across reference/model comparisons and randomized presentation order.
- The CBT-DP export produced 156 candidate statements. All remain `needs_human_review`; the simple high-risk keyword screen flagged 0 and is not a clinical or safety approval.

Machine-readable results are in `reports/dataset_audit.json`.

## Smoke experiment

The `fixed`, `profile_blind`, and `profile_aware` conditions each completed four two-turn synthetic cases against `mock-safe-v1`. All 12 cases intentionally remain unreviewed, so failure rate and axis rates are `null`. Three unit tests passed.

This validates execution and missing-label handling only. It is not a model safety result: no real target/analyzer adapter, approved adversarial test set, or independent human/LLM labels were available.

## Next experimental gate

Before a defensible target-model experiment, a qualified reviewer must approve or exclude candidate cases, define inappropriate target intents separately from normal CBT requests, and confirm data-use terms. Then run the same case IDs across conditions with fixed model versions/decoding/seeds and blinded independent review.

## Paired design simulation

A parameterized simulation verified the paired experimental logic before spending model calls. It used four declared synthetic target profiles, 1,000 cases per target, and a shared deterministic random draw for all three conditions (4,000 paired cases; seed 20261002).

- fixed: 0.0622 simulated failure-discovery rate
- profile-blind: 0.1005
- profile-aware: 0.2040
- aware minus blind: +0.1035; paired bootstrap 95% interval [0.0943, 0.1133]
- discordant pairs: aware-only 414, blind-only 0

These values follow from the declared simulation function and only show that the runner, pairing, and bootstrap recover the intended treatment effect. They are not observed LLM safety rates. Raw rows are in `mental_health_agent_eval/runs/simulation-seed-20261002.jsonl`; the compact result is `reports/simulation_summary.json`.

No real-model experiment was run because this environment exposes no model API credentials, local inference server, or GPU. The remaining external dependency is a target/analyzer adapter plus reviewed test cases.
