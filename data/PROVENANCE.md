# Dataset provenance

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
