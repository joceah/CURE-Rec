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

## Next steps

1. Freeze a reproducible SFT baseline.
2. Fix validation/test history construction.
3. Add a minimal K=1 latent pathway.
4. Implement zero / random / shuffle / matched-counterfactual interventions.
5. Measure causal utilization before adding more complex reasoning modules.
