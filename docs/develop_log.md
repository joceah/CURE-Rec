
## 2026-10-04 — Phase 0B processed data audit and freeze

- Added reusable auditor `scripts/data/audit_freeze_processed.py`. It hashes source and processed artifacts, reconstructs each user timeline from raw review JSONL, checks saved train histories and leave-two-out targets, validates ID mappings, target coverage, k-core degrees, sequence lengths, and metadata missingness.
- Beauty 2018–2023 passed: 446,314 users, 148,338 items, 3,203,489 train interactions, 446,314 validation and 446,314 test samples. Sports passed: 169,680 users, 81,571 items, 1,081,335 train interactions, 169,680 validation and 169,680 test samples. All temporal histories and targets matched raw timestamp ordering exactly; all held-out targets had item and metadata coverage.
- Froze manifests and ordered validation/test sample IDs under `data/manifests/`; all six frozen-file checksums verified with `data/manifests/freeze.sha256`. Timestamp ties were resolved by stable source JSONL order, matching the preprocessing implementation.
