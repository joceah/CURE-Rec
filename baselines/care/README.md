# CARE for CURE-Rec

This is a project-protocol adaptation of [CARE](https://github.com/Linxyhaha/CARE),
based on upstream commit `9af734f2b48a9643c5246ac68c1a9f16770bc2a8`.
It retains query-anchored reasoning, stage-specific progressive history
visibility, query diversity regularization, and constrained SID decoding.
The implementation uses the public Qwen2 attention-mask mapping in the
project's Transformers environment rather than copying the upstream Qwen2
internals. It trains on the frozen Beauty/Sports 2018–2023 leave-two-out splits
and their existing three-code SIDs. Upstream's example uses four-code SIDs,
Qwen2.5-0.5B, and different dataset dates; these results are therefore not
the paper's original numerical reproduction.

## Run

Use the project Python environment and start from the repository root:

```bash
python -m baselines.care.train data.domain=beauty
python -m baselines.care.evaluate data.domain=beauty eval.split=valid
python -m baselines.care.evaluate data.domain=beauty eval.split=test
```

Set `data.domain=sports` after fixing the Beauty recipe. Hydra overrides in
`config.yaml` control the model, query count, sample size, and output paths.
`eval.sample_size=null` evaluates every frozen sample. Checkpoints are saved
under `data/checkpoints/<domain>/care/`; metrics and per-sample predictions go
under `data/eval_results/<domain>/`.

The prompt is a concatenation of history SIDs followed by one answer token.
This layout is needed for progressive visibility: stage *k* queries see only
history codes through level *k* (and the answer token), while code generation
can use the previous reasoning stages. Validation uses train history; test uses
train plus the validation item. The frozen sample IDs are checked row by row.

The current evaluator recomputes the Qwen forward pass at each SID stage, so
full-domain evaluation is costly. Use the configured fixed 5,000-sample run
for the first comparison, then `eval.sample_size=null` for final results.

The frozen three-token index can assign the same SID to multiple items.
Metrics therefore follow the existing SFT evaluator and count SID matches;
each prediction records all item IDs sharing that SID. These are SID-level
ranking metrics, not unambiguous item-level ranking metrics.
