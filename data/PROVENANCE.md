# Dataset provenance

- Code repository: https://github.com/dlwlsrnjs/pcsa
- Private dataset repository: https://huggingface.co/datasets/jin-kwon/pcsa-data

## Cactus

- Official repository: https://github.com/coding-groot/cactus
- Official dataset: https://huggingface.co/datasets/LangAGI-Lab/cactus
- Local raw file: `raw/cactus.json`
- SHA-256: `be3421495f9dd76dd47d5fd4abd9fdabed9c97fcb7f7bba34ecfc78fe07d3d18`
- Dataset card/repository reports GPL licensing. The corpus is synthetic, but contains realistic persona names and demographics.

## CBT-Bench / CBT-DP

- Official repository: https://github.com/mianzhang/CBT-Bench
- Official dataset: https://huggingface.co/datasets/CBT-LLM/CBT-Bench
- Local raw file: `raw/cbt-bench-dp-pairwise-comparison.json`
- SHA-256: `bda3cf0c862b98b13045c8134b0e60b91ebf5d3bbee5b799044f4791c91579ec`
- No explicit license file was present in the cloned GitHub repository at retrieval time. Treat use and redistribution as research-only pending author clarification.

## Cheeseburger Therapy

Not downloaded. The CHI 2024 source paper reports 116 human peer-support sessions. It says the researchers received de-identified data from the platform with licensing, consent, and IRB approval after requesting access. No public dataset download artifact or general reuse license was found, so this workspace does not scrape or reconstruct those conversations.

`processed/cbt_dp_candidates.jsonl` contains unmodified client-statement candidates with automated risk flags. Every row remains `needs_human_review`; it is not an approved red-team set.

## Committed synthetic datasets (separate from the private corpora)

Following PCSA, persona-construction material and attack goals are kept separate. These three files are synthetic, approved, and committed to the public code repository; they contain no corpus-derived or personal data.

- `attack_goals.jsonl`: the separate mental-health safety-probe goal set (intent-level, non-actionable). The judge/CARES evaluates target responses against these goals. One axis per row across the four PCSA dimensions.
- `personas_fallback.jsonl`: approved synthetic adult personas (descriptor + cognitive distortion + style reference) used when the private corpus-derived personas are absent. `build_personas_from_corpus.py` regenerates corpus-derived personas into the gitignored `processed/personas.jsonl` when the raw corpora are available.
- `calibration_probes.jsonl`: benign profiling probes used only to measure the target's behavioral profile (our agent-specific addition). No attack content.

Persona, style, and goal are attacker-side conditions only; the target model receives just the generated client turns and the accumulating dialogue history.
