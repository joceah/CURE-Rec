
## 2026-10-04 — Phase 0B processed data audit and freeze

- Added reusable auditor `scripts/data/audit_freeze_processed.py`. It hashes source and processed artifacts, reconstructs each user timeline from raw review JSONL, checks saved train histories and leave-two-out targets, validates ID mappings, target coverage, k-core degrees, sequence lengths, and metadata missingness.
- Beauty 2018–2023 passed: 446,314 users, 148,338 items, 3,203,489 train interactions, 446,314 validation and 446,314 test samples. Sports passed: 169,680 users, 81,571 items, 1,081,335 train interactions, 169,680 validation and 169,680 test samples. All temporal histories and targets matched raw timestamp ordering exactly; all held-out targets had item and metadata coverage.
- Froze manifests and ordered validation/test sample IDs under `data/manifests/`; all six frozen-file checksums verified with `data/manifests/freeze.sha256`. Timestamp ties were resolved by stable source JSONL order, matching the preprocessing implementation.

## 2026-10-13 — CARE baseline implementation

- Added independent `baselines/care/` implementation adapted from CARE upstream commit `9af734f2b48a9643c5246ac68c1a9f16770bc2a8`: query-anchored reasoning, progressive SID-stage history masks, diversity loss, LoRA training, and constrained SID beam evaluation.
- Reused frozen Beauty/Sports 2018–2023 processed splits, three-token SID artifacts, and manifest sample IDs. Validation uses train history; test uses train plus valid. Metrics follow existing SID-level HR/NDCG; duplicate SIDs are reported with all matching item IDs.
- Added local snapshot resolution for fully offline model loading, independent checkpoint/result paths, and tests for stage mask, train/inference logits, leave-two-out history, adapter checkpoint round-trip, and constrained decoding.
- Verified four focused tests, real Beauty Qwen2.5-1.5B forward stage parity (max logit difference 0.0), one real LoRA backward/optimizer step, and Beauty/Sports data loading. Full-domain training and evaluation were not started.
