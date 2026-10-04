# CURE-Rec

**CURE-Rec** explores **Causally Utilized Latent Reasoning for Generative Recommendation**.

The project starts from a clean generative-recommendation baseline migrated from [MIR](https://github.com/joceah/MIR): Amazon sequential-data processing, text-to-item embeddings, RQ-VAE semantic IDs, Qwen SFT, constrained SID generation, evaluation utilities, and reusable analysis/test code.

## Research question

> Does the recommender actually use what it reasons?

The current direction studies whether latent reasoning states are not only predictive or semantically aligned, but **causally utilized and personally relevant** to the final recommendation decision.

See `docs/research_plan.md` for the full research plan.

## Migrated baseline

Included from MIR:
- Amazon preprocessing
- text-to-embedding pipeline
- RQ-VAE / Semantic ID pipeline
- Qwen SFT baseline
- prefix-constrained beam search and generic evaluation
- reusable Hydra/DeepSpeed configs
- generic diagnostics and tests

Intentionally excluded:
- historical CoT / CSA / RL experiments
- old MIR research notes and markdown plans
- backups, papers, checkpoints, generated data, and evaluation artifacts

## Migration cleanup

The migrated evaluator removes the legacy CoT/thought-query inference dependency and fixes the leave-two-out evaluation history:

- validation: `history = train`, `target = valid`
- test: `history = train + valid`, `target = test`

The data-processing split remains leave-two-out; CURE-Rec should keep this protocol explicit in all experiment reports.

## Current status

Phase 0 is in progress. Local preprocessing is complete for:

- Amazon Beauty 2018–2023
- Amazon Sports 2018–2023

Beauty is the development domain; Sports is the confirmation domain after the baseline recipe is frozen.

## Next steps

1. Audit and freeze the processed Beauty / Sports datasets with manifests and hashes.
2. Build text embeddings and freeze RQ-VAE / SID artifacts, starting with Beauty.
3. Train and freeze the Beauty Qwen SFT baseline, then reproduce the frozen recipe on Sports.
4. Run reproducible baseline evaluation and freeze all artifact identities.
5. Enter Phase 1 with the minimal K=1 latent pathway and intervention tests.
