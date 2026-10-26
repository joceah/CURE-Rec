
## 2026-10-04 — Phase 0B processed data audit and freeze

- Added reusable auditor `scripts/data/audit_freeze_processed.py`. It hashes source and processed artifacts, reconstructs each user timeline from raw review JSONL, checks saved train histories and leave-two-out targets, validates ID mappings, target coverage, k-core degrees, sequence lengths, and metadata missingness.
- Beauty 2018–2023 passed: 446,314 users, 148,338 items, 3,203,489 train interactions, 446,314 validation and 446,314 test samples. Sports passed: 169,680 users, 81,571 items, 1,081,335 train interactions, 169,680 validation and 169,680 test samples. All temporal histories and targets matched raw timestamp ordering exactly; all held-out targets had item and metadata coverage.
- Froze manifests and ordered validation/test sample IDs under `data/manifests/`; all six frozen-file checksums verified with `data/manifests/freeze.sha256`. Timestamp ties were resolved by stable source JSONL order, matching the preprocessing implementation.

## 2026-10-13 — CARE baseline implementation

- Added independent `baselines/care/` implementation adapted from CARE upstream commit `9af734f2b48a9643c5246ac68c1a9f16770bc2a8`: query-anchored reasoning, progressive SID-stage history masks, diversity loss, LoRA training, and constrained SID beam evaluation.
- Reused frozen Beauty/Sports 2018–2023 processed splits, three-token SID artifacts, and manifest sample IDs. Validation uses train history; test uses train plus valid. Metrics follow existing SID-level HR/NDCG; duplicate SIDs are reported with all matching item IDs.
- Added local snapshot resolution for fully offline model loading, independent checkpoint/result paths, and tests for stage mask, train/inference logits, leave-two-out history, adapter checkpoint round-trip, and constrained decoding.
- Verified four focused tests, real Beauty Qwen2.5-1.5B forward stage parity (max logit difference 0.0), one real LoRA backward/optimizer step, and Beauty/Sports data loading. Full-domain training and evaluation were not started.

## 2026-10-16 — Beauty SFT 5k test evaluation

- Beauty Qwen SFT baseline completed a 5,000-sample test evaluation at `data/eval_results/beauty/sft_5k.json` using the merged checkpoint, 50 beams, max history length 20, 3-code SID, seed 42, and `train + valid → test` history.
- SID-level metrics: HR@1 0.0278, HR@5 0.0386, HR@10 0.0448, HR@20 0.0516, HR@50 0.0626; NDCG@5 0.03366, NDCG@10 0.03559, NDCG@20 0.03726, NDCG@50 0.03944.
- Updated `docs/research_plan.md`. Remaining Phase 0 closure work: checkpoint/artifact and sample identity in results, deterministic repeat, diagnostics, and Sports confirmation.

## 2026-10-19 — CURE v1 Phase 1 implementation

- Added isolated `src/models/cure/` dataset, one-query history reasoner, frozen SFT backbone wrapper, gated residual fusion, Latent-CE loss and trainer; added Hydra train/eval entry points and CURE evaluator using the existing SID prefix tree and leave-two-out protocol.
- Explicit history SID positions and item masks are passed from the dataset; item reconstruction disables latent fusion. Only reasoner, latent projection and gate are trainable. `normal` and `zero` modes share the same forward/generation fusion path. Phase 2 losses remain guarded until counterfactual training is implemented.
- Seven targeted tests pass, covering Zero-Z equality, target isolation, gradient ownership, checkpoint round trip, train/generation parity including beam expansion, dataset masks, and bf16/float32 mixing. Real Beauty merged SFT checkpoint Zero-Z maximum logit error was 0; one real backward step and constrained two-beam decoding succeeded.
- Full Beauty Latent-CE training and HR/NDCG comparison remain pending. The training set contains about 2.76 million recommendation samples; the default three-epoch run was not launched during this implementation pass.

### 2026-10-19 — CURE distributed training parity

- Added SFT-compatible `distributed.mixed_precision` and DeepSpeed ZeRO-2 configuration to `configs/cure.yaml`, plus optional `wandb_project`, run name, and step logging interval.
- Updated the CURE trainer to create `DeepSpeedPlugin`, aggregate train/validation loss across ranks, log only on the main rank, and call collective `accelerator.save_state` before saving the compact CURE weights on rank 0.
- Removed hard-coded float32 inputs in reasoner and fusion so DeepSpeed bf16 casting works. A two-GPU Accelerate/DeepSpeed smoke test completed forward, backward, optimizer steps, collective training-state save, and compact CURE save. Seven CURE regression tests, compile check, configuration interpolation, and diff check passed.

### 2026-10-21 — CURE evaluation beam memory fix

- Diagnosed evaluation OOM: CURE projected all hidden states of the beam-expanded prompt to float32 vocabulary logits, unlike Qwen's last-token-only generation path. `k_values` and stored predictions were CPU-side and not responsible for GPU memory growth.
- Added `logits_to_keep` to CURE forward and set it to one in `prepare_inputs_for_generation`; training with labels continues to project the full sequence. Added a regression test for last-token equivalence and beam generation behavior.
- Eight CURE tests passed. The trained Beauty checkpoint completed evaluation of eight test samples at the default `batch_size=8` and `num_beams=50` on one 24 GB GPU without OOM. The eight-sample metrics are only a smoke result.

### 2026-10-21 — 如何定位并修复 CURE 评估的 GPU OOM

**现象与结论。** 使用 `batch_size=8`、`num_beams=50` 评估 CURE 时显存不足，而同样参数下 SFT 可以运行。原因是 CURE 包装模型在 beam 展开后，把整段 prompt 的每个 hidden state 都投影成 float32 词表 logits；生成下一 token 实际只需要最后一个位置。`k_values` 只用于生成完成后的 HR/NDCG 计算，预测列表也位于 CPU，因此把结果提前写盘不能解决这次 GPU OOM。

**排查过程。**

1. 先检查 GPU 空闲显存、评估参数和实际调用路径：`nvidia-smi`，然后查 `configs/cure.yaml`、`src/inference/cure_evaluator.py`、`src/models/cure/model.py`，并与 SFT 的 `src/inference/beam_search.py` 比较。两者默认都是每卡 batch 8、beam 50，所以重点检查模型生成实现，而不是先假定 K 值或样本总数占用 GPU。
2. 沿数据流区分 GPU 与 CPU 对象。`generate_cure_topk()` 每批调用 `model.generate()`；生成的 SID 随后转成 Python tuple，累计的预测和最后的 `compute_metrics()` 在 CPU。`k_values` 只传给 `compute_metrics()`，不会让模型按每个 K 重跑。
3. 检查 Qwen 的接口：`inspect.signature(Qwen2ForCausalLM.forward)` 显示当前 Transformers 版本支持 `logits_to_keep`。SFT 的原生生成路径利用该参数只输出最后位置的 logits；当时的 `CUREForCausalLM.forward()` 却总是对完整 `hidden` 执行 `lm_head(...).float()`。
4. 用形状估算峰值。beam 展开后有效 batch 为 `8 × 50 = 400`。若 prompt 长约 80–100 token、词表约 15 万，完整 float32 logits 约为 `400 × (80–100) × 150000 × 4` 字节，即约 19–24 GB；投影前后的临时张量和模型权重还会增加峰值。该估算能解释为什么 24 GB GPU 会 OOM，而仅减少评估总样本数未必有用：峰值由**单批**大小决定。
5. 修复时在 CURE `forward()` 增加 `logits_to_keep`，先运行 Qwen backbone，再在 vocab projection 前只保留所需的最后 hidden state；`prepare_inputs_for_generation()` 固定传入 `logits_to_keep=1`。带 `labels` 的训练 forward 保持完整序列 logits，避免改变训练 loss。latent 编码与 gated fusion 的逻辑不变。
6. 验证分两层：单元测试确认最后位置的 logits 与完整 forward 的对应位置一致，并确认 beam 生成每步只对长度为 1 的 hidden state 做 fusion；全套 CURE 测试 8 项通过。随后用已训练 Beauty checkpoint 在**单张 24 GB GPU** 上以原默认 `batch_size=8`、`num_beams=50` 评估 8 条 test 样本，完整走通且无 OOM。8 条样本只验证内存和功能，不能用于评价推荐性能；正式 5000 条评估仍需单独运行。

**以后遇到类似 OOM 的最短排查路径。** 先记录报错发生在加载、prefill、解码还是指标阶段；用 `nvidia-smi` 看运行前显存；保持单批 `batch_size` 和 `num_beams` 不变，只把 `sample_size` 缩到一批复现峰值；列出最大的张量形状并按 `元素数 × dtype 字节数` 估算；对比可运行的 baseline 是否有 `logits_to_keep`、cache 或 logits dtype 差异。若 OOM 来自 GPU 中间张量，优先消除不必要的计算或调整每批负载；只有 CPU 结果累计或写盘时才考虑流式写入。

### 2026-10-22 — CURE 本地日志拆分与补全

- 原 `configs/cure.yaml` 的训练和评估共用 `logs/cure_beauty.log` 且 `mode: w`。后续评估启动会覆盖训练记录；训练 epoch 使用 `accelerator.print()`、评估指标使用 `print()`，这些内容仅到终端，不进入文件。检查时旧日志只剩评估加载样本的四行，先前训练文件内容无法从该文件恢复。
- 现在默认分别写 `logs/cure_${data.domain}_train.log` 和 `logs/cure_${data.domain}_eval.log`，并用追加模式保留重复运行记录。`logging.output_path` 仍可作为命令行覆盖路径。
- 训练记录 step loss/学习率、epoch train/valid loss 与 checkpoint；评估记录开始参数、每 50 批进度、结果文件和 HR/NDCG。只有全局主进程写共享文件。
- 配置解析、Python 语法、8 个 CURE 测试和 diff 检查通过。使用已训练 Beauty checkpoint 的两条样本评估验证了默认 `logs/cure_beauty_eval.log` 确实包含进度、输出路径和指标。两条样本仅用于日志 smoke test，不代表正式推荐效果。
