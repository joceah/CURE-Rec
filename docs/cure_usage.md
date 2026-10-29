# Safe Personalized Residual CURE 使用说明

本实现对应 `research_plan.md` 和 `cure_implementation_plan.md`。入口默认使用 Beauty validation，冻结完整 SFT，只训练单层 decision-conditioned reasoner、scalar gate 和 candidate residual。没有 item reconstruction auxiliary task，不复用旧 latent-residual implementation 或其 checkpoint。

## 公共配置命名

训练参数沿用 `configs/sft.yaml`，评估参数沿用 `configs/eval.yaml`。主要迁移如下；旧 CLI 参数名不再作为别名保留，已有 checkpoint 权重及历史 metadata 仍可加载。

| 旧 CURE 名称 | 统一后的名称 | 含义 |
|---|---|---|
| `train.batch_size` | `train.per_device_batch_size` | 每卡训练 microbatch；和 SFT 相同 |
| `train.cache_batch_size` | `train.cache_per_device_batch_size` | 每卡离线 difficulty scoring batch，独立于训练 batch |
| `train.log_steps` | `train.wandb_log_steps` | 每多少次 optimizer update 记录指标，和 SFT 相同 |
| `model.sft_checkpoint` | `model.llm.sft_model_path` | 冻结 SFT 模型目录；复用 SFT 的字段名 |
| `model.tokenizer_path` | `model.llm.tokenizer_path` | SFT tokenizer 目录；复用 SFT 的字段名 |
| `eval.checkpoint` | `eval.model_path` | 待评估 CURE branch `.pt`；对应 eval.yaml 的待评估模型路径 |
| `eval.output_path` | `eval.output_dir` + `eval.output_name` | 输出到 `<output_dir>/<output_name>.json`，和原 evaluator 一样 |
| 隐式使用 `train.seed` | `eval.random_seed` | 评估抽样和 donor matching 的随机种子，默认继承 train.seed |

`data.valid_path` / `data.test_path` 也与 SFT 对齐，通过 `eval.split` 选择。`eval.batch_size`、`eval.num_beams`、`eval.sample_size`、`eval.k_values` 原本已同名，保留。`eval.max_history_len` / `eval.max_seq_len` 默认继承训练设置，独立评估时与 checkpoint 记录的评估预处理核对；历史 checkpoint 没有 eval 预处理字段时回退到其 train 字段。

训练 seed 与评估 seed 可独立设置：改变 `train.seed` 会通过默认插值同步改变评估 seed；如果希望固定评估样本池，用 `eval.random_seed=42` 显式覆盖。`model.llm.local_files_only=true` 与 SFT 相同，只使用已有本地模型文件/缓存。CURE 的 `reasoner`、`residual`、`method`、`matching`、`train.init_checkpoint` 没有同义 SFT 参数，保留名称，并在 YAML 逐项注明单位和作用。

例如，设置每卡训练 batch 和评估输出位置：

```bash
python scripts/train/run_cure.py train.per_device_batch_size=8 train.wandb_log_steps=10
python scripts/eval/run_cure_eval.py \
  eval.model_path=data/checkpoints/beauty/cure_v1/residual_ce_seed42/best.pt \
  eval.output_dir=data/eval_results/beauty eval.output_name=cure_residual_ce_valid \
  eval.random_seed=42
```

## 验证

在安装项目依赖的 Python 环境中执行：

```bash
PYTHONPATH=src pytest -q tests/test_cure.py tests/test_prefix_tree.py tests/test_metrics.py
CURE_REAL_SFT=1 PYTHONPATH=src pytest -q tests/test_cure.py -k real_sft
```

第二条使用 `configs/cure.yaml` 中的本地 Beauty SFT，在两条 validation 样本上检查关闭 correction 后的精确 Top-10 恢复、target 隔离、冻结梯度和 checkpoint round-trip。常规测试还覆盖三个 stage 的合法 children、实际 generation logits processor、KL 方向、四种训练变体、singleton batch 与缓存身份校验。训练与 generation 的不同 batch/cache GEMM 在 BF16 下允许数值舍入误差；关闭 correction 后与原生 SFT beam 的输出必须精确一致。

仓库原有的 `tests/test_evaluator.py` 会读取 test split，因此不包含在开发阶段的上述测试命令中。

## 训练顺序

每个命令创建独立的新 run，分别使用相同冻结 SFT 初始化。应在检查前一变体的 validation 结果后再启动下一变体。

```bash
python scripts/train/run_cure.py method.variant=residual_ce
python scripts/train/run_cure.py method.variant=cure_g
python scripts/train/run_cure.py method.variant=cure_gp
python scripts/train/run_cure.py method.variant=cure_full
```

配置中的 loss weights、margins、bucket boundaries 和容量是实现默认值，尚未通过方法实验冻结。正式方法选择需要 validation HR/NDCG 信号以及多个 validation 子集、至少两个 seed。

输出位于 `data/checkpoints/beauty/cure_v1/<variant>_seed<seed>/`：

- `config.yaml`：解析后的完整配置。
- `valid_step_*.json`：同一批样本上的 SFT / correct / matched 结果与样本、donor 身份。
- `best.pt`、`last.pt`：只有新增 branch 权重、结构配置与 provenance，不复制 frozen SFT。
- `best_validation.json`：最佳 checkpoint 的完整 validation 结果。
- `training_summary.json`：实际 optimizer update 数、停止原因、最佳 checkpoint 所在 step 和早停状态。

选择顺序为 NDCG@10、HR@10、target-score gain、较小的 FD@10。默认拒绝覆盖已有 checkpoint 或 validation 结果的 run。`train.init_checkpoint=<path>` 仅在新 run 中载入 branch 权重，重置 optimizer/scheduler；它不是中断训练恢复。

需要在原输出目录重新训练时，在原命令末尾加 `train.overwrite_output_dir=true`，或在 YAML 中设置：

```yaml
train:
  overwrite_output_dir: true
```

显式开启后，主进程会清理该 run 的 `best.pt`、`last.pt`、旧 `valid_step_*.json` / `valid_epoch_*.json`、`best_validation.json`、`training_summary.json`，重写配置；训练 CLI 同时将本次日志改为覆盖写入。目录中的其他文件、W&B 历史 run 与外部 difficulty 缓存保留。默认仍为 false。该选项重新开始训练，不恢复 optimizer/scheduler；若显式指定同目录的 `train.init_checkpoint`，会先读取其权重再清理旧结果。

只检查 pipeline 时可使用较小配置，例如：

```bash
python scripts/train/run_cure.py method.variant=residual_ce \
  train.sample_size=32 train.max_steps=1 train.epochs=1 \
  eval.sample_size=8 eval.num_beams=10 'eval.k_values=[1,5,10]' \
  train.output_dir=outputs/cure_v1_smoke
```

这类 smoke 不能作为推荐性能结论。

## Step 验证与 early stopping

CURE 和普通 SFT 共用以下配置，步数全部指 **optimizer update**，gradient accumulation 内的 microbatch 不计入步数：

```yaml
train:
  eval_steps: 500
  early_stop:
    enabled: true
    patience_steps: 2500
    min_delta: 0.0001
```

默认每 500 次更新验证一次；epoch 结束或达到训练上限时，如果当前 step 尚未验证，也会补一次。`eval_steps: null` 只在 epoch 结束验证。早停在验证时判断：距离上次超过 `min_delta` 的显著提升达到 2500 次更新，就停止。例如 step 500 首次验证，之后一直没有显著提升，会在 step 3000 的验证后停止。若验证间隔不能整除 patience，停止发生在达到 patience 后的下一次验证。`epochs` 和可选的 `train.max_steps` 仍是训练上限，`early_stop.enabled=false` 可关闭早停而保留 step 验证。

CURE 默认 `early_stop.monitor=NDCG@10`、`mode=max`，只监控 correct branch；也可选择 `eval.k_values` 中已有的 HR/NDCG 指标。普通 SFT 默认 `monitor=loss`、`mode=min`，使用全局 target-token 加权的 validation loss。两者的 `eval.sample_size=1000` 均固定抽样一次，`eval.random_seed` 控制抽样；设为 `null` 使用全量 validation。不会读取 test 来决定早停。

`min_delta` 是绝对差值，只控制是否重置 patience，不阻止保存更好的 checkpoint。CURE 继续按上面的四项顺序选择 `best.pt`；SFT 按最低 validation loss 保存 `best_ckpt/`，结束时所有 rank 恢复最佳 checkpoint，再导出 `merged/`，因此导出的不是最后一步权重。ZeRO-3 的 SFT 合并在 CPU 上重建权重，需要容纳完整模型的主存。所有 rank 使用主进程广播的早停决定；W&B 记录 `early_stop/*`，两个训练入口均写 `training_summary.json`。

例如缩短等待窗口：

```bash
python scripts/train/run_cure.py train.eval_steps=200 train.early_stop.patience_steps=1000
python scripts/train/run_sft.py train.eval_steps=200 train.early_stop.patience_steps=1000
```

合成模型回归（不读取研究数据）：

```bash
PYTHONPATH=src pytest -q tests/test_early_stopping.py tests/test_sft_training.py tests/test_sft_distributed.py
CUDA_VISIBLE_DEVICES=0,1 CURE_ZERO_TEST=1 PYTHONPATH=src pytest -q tests/test_sft_distributed.py tests/test_cure_zero.py
```

## 训练缓慢时的开发流程

`configs/cure_dev.yaml` 是可直接运行的开发 preset：ZeRO-3 保留参数分片，parameter/optimizer CPU offload 均关闭；固定 seed=42 的 32768 条 train 和 200 条 validation，beam=10，只报告 HR/NDCG@1/5/10；每卡 batch=128、gradient accumulation=1，双卡有效 batch=256。运行 3 个 epoch，默认每个 epoch 验证一次，双卡共 384 次 optimizer update。固定样本曝光量可以让不同 batch 的比较更清楚。原有 cure.yaml 的全量设置不受这个 preset 影响。

先运行这一组：

```bash
conda activate cure_rec
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --use_deepspeed --num_processes 2 --mixed_precision bf16 \
  scripts/train/run_cure.py --config-name cure_dev
```

查看控制台或 `logs/cure_dev_*.log` 的 `train/seconds_per_update`（排除 validation 耗时）、`train/baseline_rec`、`train/rec`、`train/mean_target_score_gain`。前两项 loss 来自相同 batch，可直接比较；`baseline_rec - rec` 应约等于 target-score gain（允许浮点误差）。刚开始的 CUDA 初始化会影响首段计时，比较后续稳定区间。不同 batch 的速度应换算为 samples/second：双卡、accumulation=1 时，约为 `2 × per_device_batch_size / seconds_per_update`。

需要比较 batch 和 learning rate 时，依次运行这四组。它们使用同一 train/validation 子集、相同初始化 seed、相同 3 个 epoch；对照阶段关闭 early stop，让四组具有相同训练样本曝光量。每组写到独立目录，默认拒绝覆盖已有结果的目录：

```bash
for batch in 64 128; do
  for lr in 0.0001 0.0003; do
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
      --use_deepspeed --num_processes 2 --mixed_precision bf16 \
      scripts/train/run_cure.py --config-name cure_dev \
      train.per_device_batch_size="$batch" train.learning_rate="$lr" \
      train.early_stop.enabled=false \
      train.output_dir="data/checkpoints/beauty/cure_dev_sweep/b${batch}_lr${lr}_seed42"
  done
done
```

四组共用 32768 条训练样本，每组最多曝光 98304 条次；每卡 batch=64/128 分别是双卡有效 batch=128/256，3 个 epoch 分别为 768/384 次更新。不要用较大 batch 的 step 数直接判断它更快。

比较每个 run 的 `best_validation.json`：先看 `metrics.correct.NDCG@10`，再看 HR@10，以及同文件中 `metrics.baseline`；训练 gain 只能检查是否在学习，不能替代推荐指标。`training_summary.json` 记录实际步数、best checkpoint step、停止原因。200 条 validation 仅适合筛选：有希望的设置还要在固定 1000 条 validation、beam=50 上重新确认。四组都没有正增益时，先检查 residual 的学习机制，不直接扩大到完整数据。

选定设置后，恢复全量训练和正式开发验证配置，并重新启用 early stop。例如每卡 batch=128、lr=0.0001 被选中时：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --use_deepspeed --num_processes 2 --mixed_precision bf16 \
  scripts/train/run_cure.py \
  distributed.use_deepspeed=true distributed.mixed_precision=bf16 \
  distributed.deepspeed.zero_optimization.stage=3 \
  distributed.deepspeed.zero_optimization.offload_param.device=none \
  distributed.deepspeed.zero_optimization.offload_optimizer.device=none \
  train.per_device_batch_size=128 train.gradient_accumulation_steps=1 \
  train.learning_rate=0.0001 train.sample_size=null \
  train.early_stop.enabled=true \
  train.output_dir=data/checkpoints/beauty/cure_v1/residual_ce_fast_seed42
```

这里重新从 frozen SFT 初始化 branch；`train.init_checkpoint` 仅加载权重，不恢复 optimizer/scheduler。已有进程不会自动获得新代码，必须用新的输出目录重新启动。若关闭 offload 后显存不足，先减小每卡 microbatch；只有显存预算确实需要时再启用 CPU offload。

2026-10-27 的本机诊断：真实 Beauty frozen SFT、microbatch=128、同一 batch，单卡 residual 分支前向＋反向的中位耗时由逐样本参考的约 0.498 秒降为批量版约 0.00353 秒；frozen prepare 仍约 0.459 秒，因此不能把分支加速比当成整体加速比。另一次真实双卡 ZeRO-3 无 CPU offload 的四步 CLI smoke，每卡 batch=128、accumulation=1，在首步初始化后约 0.59–0.61 秒/update，每卡 peak CUDA allocated 约 2.79 GiB（统计涵盖训练及 4 条 validation、beam=10，不含 driver/NCCL 分配）。这只验证吞吐和 pipeline，不是正式方法效果或默认验证规模的显存结论。

候选评分已同时批量化 sample/stage 与 beam 内 residual 调用；合法 children 在 CPU 上组装后整体传输到 GPU，训练不构造 `[B,3,K,H]` 候选 embedding 副本。只对合法 candidates 做归一化，保留全词表 frozen LM-head 计算、target 隔离、关闭 correction 的结构性恢复和旧 branch checkpoint 参数名。safe KL 使用同一 padding mask，避免无效位置 `-inf - -inf` 引起 NaN。

## Accelerate 多卡与 W&B

支持 Accelerate DDP 与 DeepSpeed ZeRO，也保留普通单卡 / CPU 路径。每个 rank 自动使用自己的 GPU，无需手动设置 `model.device=cuda:0`。DDP 每卡保留完整 frozen SFT，只同步可训练 branch 梯度；需要减少 frozen SFT 权重显存时使用下述 ZeRO-3 preset。

双卡训练并启用 W&B：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu --num_processes 2 --mixed_precision no \
  scripts/train/run_cure.py method.variant=residual_ce \
  train.wandb_enabled=true train.wandb_project=genrec \
  train.wandb_run_name=cure_beauty_residual_ce_seed42
```

W&B 还可配置 `train.wandb_entity`、`train.wandb_group`、`train.wandb_tags`、`train.wandb_mode=offline`。默认 `wandb_enabled=false`，使用 online 模式时沿用本机已有 W&B 登录或 `WANDB_API_KEY` 配置。每次多卡训练只创建一个 run；offline 模式也只创建一个本地 run。记录全量配置、实际 world size / effective batch size、全 rank 平均 loss 分项、gate / correction norm、学习率、三分支 HR/NDCG 与 RG/PG/FD 等指标。日志频率由 `train.wandb_log_steps` 控制。

`train.per_device_batch_size` 是**每张卡的 microbatch size**。例如每卡 microbatch=8、gradient accumulation=4 时，双卡有效 batch 为 `8 × 2 × 4 = 64`；如果想保持原单卡有效 batch 32，设置 `train.gradient_accumulation_steps=2`。GP/Full 要求每卡 batch 至少为 2，donor 在本卡 batch 内匹配。多卡训练只保留完整 global batch，每个 epoch 舍弃不足一组的尾样本，避免重复采样、自匹配或不同 rank 更新步数不一致；下一 epoch 使用新的确定性 shuffle。单卡仍把 singleton 尾 batch 合并到前一 batch。

Frozen SFT 保持 `model.dtype=bfloat16`；默认 `distributed.mixed_precision='no'` 让小型 residual branch 用 FP32，主要计算仍由 BF16 SFT 完成。可额外设置 `distributed.mixed_precision=bf16` 启用训练 autocast，该路径已经通过真实双 GPU smoke。`--mixed_precision no` 是 launcher 的设置，训练 autocast 以本项目 `distributed.mixed_precision` 配置为准。

独立双卡评估：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu --num_processes 2 --mixed_precision no \
  scripts/eval/run_cure_eval.py method.variant=residual_ce
```

训练中的 validation、独立评估、首次 difficulty 缓存均按样本分片并行。有效结果中每个 sample index 恰好出现一次，不丢弃尾样本。ZeRO-3 额外补齐仅供通信对齐的计算，补齐样本不进入 matching 或指标；DDP 和 ZeRO-1/2 不需要此补齐。各卡先汇总全局 latent / difficulty，再在完整评估池中统一 matching，随后并行生成三分支结果。结果按 sample index 排序，所有 rank 获得相同指标供 checkpoint selection 使用；只有 global rank 0 写 cache、JSON、checkpoint 和 W&B。共享输出及缓存路径需对所有 rank 可见。

双进程回归测试使用合成数据，不读取研究 test split：

```bash
PYTHONPATH=src pytest -q tests/test_cure_distributed.py
CUDA_VISIBLE_DEVICES=0,1 CURE_DDP_DEVICE=cuda CURE_DDP_PRECISION=bf16 \
  PYTHONPATH=src pytest -q tests/test_cure_distributed.py
```

测试检查每次更新后的实际梯度和参数同步、backbone 冻结、跨卡训练样本无重叠、partial accumulation、仅主进程保存、全局 donor / 评估指标与单卡一致、缓存一致，以及 W&B offline 只生成一个 run 且包含 loss 和评估指标。

## DeepSpeed / ZeRO 与显存

`configs/cure.yaml` 已列出 DeepSpeed 配置；`distributed.use_deepspeed=false` 保留默认 DDP。另提供两个可直接选择的 preset：

| 配置 | 实际作用 |
|---|---|
| `cure_zero2` | 分片 trainable branch 的 optimizer states / gradients；frozen SFT 权重仍复制在各卡上 |
| `cure_zero3` | 分片包括 frozen SFT 在内的参数，默认额外使用 CPU parameter offload |

CURE 的主要权重已冻结，没有相应的 optimizer state，因此仅开启 ZeRO-2 对 frozen backbone 显存帮助有限。该分工与 [DeepSpeed ZeRO 文档](https://deepspeed.readthedocs.io/en/stable/zero3.html) 一致。

省显存的双卡启动方式：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --use_deepspeed --num_processes 2 --mixed_precision bf16 \
  scripts/train/run_cure.py --config-name cure_zero3 \
  method.variant=residual_ce train.wandb_enabled=true
```

独立评估也使用相同 preset：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --use_deepspeed --num_processes 2 --mixed_precision bf16 \
  scripts/eval/run_cure_eval.py --config-name cure_zero3 \
  method.variant=residual_ce
```

可配置 `distributed.deepspeed.zero_optimization` 中的 stage、communication bucket、parameter persistence、live/reuse budget、CPU optimizer / parameter offload 及 pin memory。若显存充足、希望减少 CPU/PCIe 传输，可设置 `distributed.deepspeed.zero_optimization.offload_param.device=none`，保留 GPU 参数分片。若还要卸载小型 branch 的 optimizer，设置 `distributed.deepspeed.zero_optimization.offload_optimizer.device=cpu`。CPU parameter offload 会增加主存占用和传输成本。

学习率、batch、gradient accumulation 和 clipping 继续由 `train.*` 控制，DeepSpeed 对应字段通过插值保持一致；冲突会在初始化前报错。optimizer / scheduler 不在 DeepSpeed JSON 中重复定义。默认 `zero_force_ds_cpu_optimizer=false`，CPU optimizer offload 使用 PyTorch AdamW；设为 true 时使用 DeepSpeed 选择的 CPU optimizer，需要相应的运行/编译环境。

ZeRO-3 默认 `distributed.zero3_init=true`，使用 Hugging Face 的分片 checkpoint 加载，避免先把完整 SFT 放上每张 GPU。模型只缓存一份很小的 frozen SID basis，评分和生成过程中不为 residual 聚合整个 LM head；MHA 的 functional output projection 显式注册为外部参数。保存 `best.pt/last.pt` 时所有 rank 只 gather 新增 branch，由 rank 0 写入；不会为保存而聚合整个 frozen SFT。checkpoint 保留 branch dtype，支持重新载入后继续做同精度评估。

ZeRO-3 的参数 all-gather 需要各 rank 保持相同 forward 调用次数；因此 cache 和 eval 的最后一组工作会补齐，并明确排除虚拟样本。生成保持 `synced_gpus`，禁止按 rank 独立提前结束。ZeRO-1/2 独立评估没有 optimizer states / gradients 可分片，使用复制模型直接推理；ZeRO-3 独立评估维持参数分片及 offload。

结果 metadata 保存 engine、ZeRO stage、精度、各 rank 的 allocated / peak CUDA bytes。训练/评估峰值从 `prepare` 完成后重置统计，包含 difficulty cache、训练和 validation，不含 CUDA driver / NCCL 的额外分配，也不代表其他 batch/beam 设置的显存上限。difficulty 缓存绑定评分后端配置，切换后端时需使用兼容缓存或新的路径。

真实 Beauty SFT 的一次显存 smoke 对照如下：两张 RTX 3090，microbatch=2、gradient accumulation=2、beam=10，8 条 train / 3 条 validation，一次优化更新。统计从 prepare 后重置，涵盖 difficulty cache、训练和 validation：

| 后端 | 每卡峰值 CUDA allocated memory |
|---|---:|
| DDP（BF16 backbone + BF16 autocast） | 3.41–3.42 GiB |
| ZeRO-3 + CPU parameter offload | 0.916 GiB |

这只验证该小规模配置的显存变化，不代表 batch=8、beam=50 的显存上限，也不用于判断推荐效果。CPU offload 在该 smoke 中显著增加传输开销，应按显存与吞吐需求选择。

双 GPU ZeRO 集成测试（合成数据，不读取研究 test）：

```bash
CUDA_VISIBLE_DEVICES=0,1 CURE_ZERO_TEST=1 PYTHONPATH=src \
  pytest -q tests/test_cure_zero.py
```

覆盖 ZeRO-2、ZeRO-3、parameter/optimizer CPU offload、分片初始化、实际 engine update 次数与 partial accumulation、冻结权重、各 rank 参数一致、完整 branch 保存，以及奇数样本评估。CUDA 测试需要已安装项目 DeepSpeed 依赖。

## Matching 与缓存

`cure_gp` / `cure_full` 在优化前离线计算选定 train 样本的 frozen target score，缓存位置由 `matching.cache_path` 指定。默认全量 train 的首次缓存需要一次 frozen SFT scoring pass；Residual-CE 和 CURE-G 不需要此缓存。

缓存绑定 SFT 权重、tokenizer、SID artifact、dataset manifest、train sequence、实际 history 截断长度、样本顺序和 bucket 配置。改变样本子集或预处理后，应使用新的缓存路径；不兼容的现有缓存会明确报错。

训练在每卡 batch 内按 `all three → difficulty+history → difficulty → random non-self` 选择 donor，不要求形成 permutation。单卡最后一个 singleton 会合并到前一 batch；多卡使用完整 global batch。评估在完整选定样本池中匹配，保存 donor sample IDs 与各 relaxation level 的数量。target popularity 只统计 train interaction；target score/popularity 仅用于训练和机制评估的匹配，不进入正常 correct-latent inference。

## 配对评估与解码协议

```bash
python scripts/eval/run_cure_eval.py method.variant=residual_ce
```

默认载入相应 run 的 `best.pt`，评估 validation。JSON 包含 HR/NDCG、RG/PG/FD、target-score gain、personalized score gap、safe KL、gate、gated correction 的 L2 norm 和 target-rank shift。rank shift 为 `baseline rank - correct rank`，未出现在 beam 中的 target 记为 `beam_size + 1`，因此该指标是截断 rank。

本实现协议名为 `sid-candidate-logsoftmax-3stage-v1`：只对三个 SID token 评分，不训练 EOS；每个 prefix 仅在 `SidPrefixTree` 的合法 children 上归一化。generation 复用 SFT 原生 `generate()`、beam search 和 KV cache，forward hook 只读取当前 prefix hidden，logits processor 只添加候选 residual。

原有 SFT `generate_topk_sids()` 默认保留历史行为。使用 `candidate_normalized=True` 可取得与 CURE 对齐的 SFT reference。历史 SFT 指标来自未重新归一化的旧 beam 协议，不能直接当作 CURE 新协议的配对 reference；CURE 评估总会在相同样本与协议下重新计算 frozen SFT。此次实现未重新运行 Beauty test。

checkpoint 记录 frozen SFT 文件哈希、tokenizer、SID index hash、manifest、train sequence identity、git SHA/dirty、method 配置和训练 seed。评估会核对 checkpoint 身份与 history/sequence 截断配置。数据加载还核对当前 train/held-out 文件及 sample IDs 是否匹配 frozen manifest。

test 与 Sports 入口要求显式提供 `eval.design_frozen=true eval.recipe_id=<frozen_recipe>`，并且 test history 固定为 `train + valid`。这些标记记录设计冻结，不替代研究计划中的 Go/No-Go 验证。
