---
pretty_name: PCSA Research Data (Private)
language:
- en
license: other
task_categories:
- text-generation
---

# PCSA Research Data

Private research storage for the profile-aware counseling safety evaluation project.

## Related repository

- Code, experiment design, and reports: https://github.com/dlwlsrnjs/pcsa
- This private dataset: https://huggingface.co/datasets/jin-kwon/pcsa-data
- Research design: https://github.com/dlwlsrnjs/pcsa/blob/main/mental_health_agent_eval/research_design.md

## Contents

- `raw/cactus.json`: official synthetic Cactus corpus, SHA-256 `be3421495f9dd76dd47d5fd4abd9fdabed9c97fcb7f7bba34ecfc78fe07d3d18`.
- `raw/cbt-bench-dp-pairwise-comparison.json`: official CBT-Bench CBT-DP pairwise data, SHA-256 `bda3cf0c862b98b13045c8134b0e60b91ebf5d3bbee5b799044f4791c91579ec`.
- `processed/cbt_dp_candidates.jsonl`: 156 extracted client-statement candidates. Every row requires human review.
- `PROVENANCE.md`: source, licensing, and access notes.

## Restrictions

Cactus is synthetic but includes realistic persona names and demographics. CBT-Bench did not expose an explicit license file in the retrieved GitHub repository, so redistribution and non-research use require clarification from the authors. This private repository must not be made public without a separate license and privacy review.

Cheeseburger Therapy human-session transcripts are not included. The source study obtained them through direct licensing, consent, and IRB-approved access; no public reusable artifact was found.
