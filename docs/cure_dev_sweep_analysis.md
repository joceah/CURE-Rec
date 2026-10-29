# Beauty CURE development sweep 结果分析

分析日期：2026-10-27。来源：`data/checkpoints/beauty/cure_dev_sweep` 四组 `config.yaml`、三次 `valid_step_*.json`、`best_validation.json`、`training_summary.json` 和对应训练日志。只分析已有 train/validation 结果，没有新运行训练、test 或 Sports。

## 结论

lr=0.0003 比 0.0001 更有希望，但当前结果不足以确认稳定推荐增益。200 条 validation 的 HR@10 从 baseline 3 次命中提高到 4 次；correct 和 matched 的 HR/NDCG 在全部 12 次验证中均相同，尚未显示正确匹配 latent Z 的额外价值。lr=0.0003 的两组 best checkpoint 值得进一步复评；不据此直接扩大为全量长训练。

## 对照有效性

- 四组 frozen SFT、tokenizer、SID、train sequence 与 manifest 身份一致；200 条 validation sample IDs、beam=10、seed=42、decode protocol 完全相同。
- 配置差异仅为每卡 batch、learning rate 及相应输出/日志路径；均为 residual_ce、固定 32768 条 train、3 epoch、双卡、accumulation=1、ZeRO-3、无 CPU offload、early stop 关闭。训练样本曝光量相同为 98304 条次。
- donor assignment 四组一致，全部非自身；donor 不要求构成 permutation。本次有 133 个不同 donor IDs。
- 四组都正常完成 3 epoch，best 都在 epoch 3；batch=64/128 分别为 768/384 次更新。

## 最佳 checkpoint

baseline：HR@10=0.015（3/200），NDCG@10=0.008730707。

| 每卡 batch | LR | 更新数 | correct HR@10 | correct NDCG@10 | matched NDCG@10 | target-score gain | personalized score gap |
|---|---|---|---|---|---|---|---|
| 128 | 0.0001 | 384 | 0.015 | 0.009166667 | 0.009166667 | 0.000990 | 0.00001458 |
| 128 | 0.0003 | 384 | 0.020 | 0.011215300 | 0.011215300 | 0.002119 | -0.00000463 |
| 64 | 0.0001 | 768 | 0.015 | 0.009166667 | 0.009166667 | 0.001601 | 0.00002018 |
| 64 | 0.0003 | 768 | 0.020 | 0.011368528 | 0.011368528 | 0.002788 | -0.00001050 |

按当前数值 b64/lr=0.0003 最好，其 NDCG@10 绝对增益为 0.002637821；b128/lr=0.0003 的绝对增益为 0.002484593。两者相差仅 0.000153228，HR@10 均为 4/200，不足以稳定区分 batch。不能把较大的相对百分比提升解释成充分证据；只净多命中 1 条，且只有一个 seed、一个小验证池。

## 学习与 latent 机制

- 分支确实在学习：lr=0.0003 两组 epoch 3 日志平均 train target-score gain 分别约为 0.01015（b64）和 0.00795（b128）；对应 validation gain 约 0.00279、0.00212。训练、验证增益均为正，不能再把原因简单归结为梯度断开或完全学不动。日志是间隔采样，epoch 平均值不是全量 train metric。
- 额外 latent 的个性化收益尚未出现：b64/lr=0.0003 的 correct-vs-matched 平均分差约 -0.00001050；b128/lr=0.0003 约 -0.00000463；所有验证的 HR/NDCG personalized lift 都为零。
- 当前聚合关系更接近 correct ≈ matched > baseline，而研究计划要求 correct > baseline ≈ matched。更可能主要学到了不依赖正确 Z 的 residual 修正，但这仍是待消融确认的解释。baseline decision state h 本身已包含用户历史，不能因此宣称模型完全不使用用户信息。
- correct/matched 汇总指标相同不意味着每条样本的预测完全相同；JSON 没有保存逐样本预测/target rank，现有聚合结果不能支持这个断言。
- 最好两组 gate 已达 0.99 左右，不能继续用“gate 一直关闭”解释。高 gate 本身也不等于不安全：当前 matched 指标提高而非降低，FD 为负。需要的是正确 Z 优于错误 Z 的证据。
- 本轮只有 residual_ce，没有启用 personal/safe 目标；不能据此认定完整 CURE formulation 失败。

## 速度与当前耗时瓶颈

以下按最近一次完整运行统计，避免把 b128/lr=0.0001 追加日志中的前一次运行混入。validation 时间按日志时间差减去未打印的更新耗时估算，非独立 profiler 数据；从 trainer 初始化完成到最后 validation 结束，不含模型加载。

| 每卡 batch | 稳定秒/update | 双卡 samples/s | 3 epoch 训练估计 | 每次 validation 估计 | 验证耗时占比 |
|---|---|---|---|---|---|
| 64 | 0.389–0.390 | 328–329 | 约 5.0 分钟 | 约 6.0–6.2 分钟 | 约 78% |
| 128 | 0.606–0.608 | 421–422 | 约 3.9 分钟 | 约 6.0–6.1 分钟 | 约 82% |

每组整体约 22–23 分钟，约 18 分钟是三次验证。批量化和关闭 offload 后，训练吞吐已改善；目前主要耗时是逐样本做 baseline/correct/matched 三次 beam generation，虽然单次 generation 内的 beam residual 已经批量化。应优先优化跨样本 validation generation 或减少无必要的重复验证，而不是继续仅调整训练 batch。

## 下一步

1. 保留 lr=0.0003 的 b64、b128 两个 best.pt，直接复评，无需先重训。b128 吞吐约高 28%，可作为效率优先候选，最终由更大 validation 决定。
2. 优先解决 validation 逐样本生成开销，再扩大到固定 1000 条 validation。可先保持 beam=10 验证扩大样本池后的信号，再按 beam=50 的开发口径确认；每个口径都重新配对 frozen SFT，不能跨口径直接比较。
3. 补充 correct/shuffled/zero/mean latent 的敏感性检查，分别看同一样本上的 target-score gap 与推荐指标，区分 h 路径修正和额外 Z 的贡献。
4. 若 residual 增益在更大验证池成立，按实施计划顺序进入 CURE-G、CURE-GP，再检查 safe 项。personalized lift 不出现时，检查 latent conditioning/训练信号，不单纯扩大训练时间或 latent depth。
5. recipe 有希望后再增加 seed 和第二个固定 validation 子集；继续不使用 test 做方法选择。

## 各 epoch 指标

| Run | epoch | correct NDCG@10 | correct HR@10 | target-score gain | personalized score gap | mean gate |
|---|---|---|---|---|---|---|
| b128_lr0.0001_seed42 | 1 | 0.009077324 | 0.015 | 0.000011 | 0.00000121 | 0.0382 |
| b128_lr0.0001_seed42 | 2 | 0.009077324 | 0.015 | 0.000697 | 0.00000663 | 0.7458 |
| b128_lr0.0001_seed42 | 3 | 0.009166667 | 0.015 | 0.000990 | 0.00001458 | 0.8918 |
| b128_lr0.0003_seed42 | 1 | 0.009077324 | 0.015 | 0.000013 | 0.00000018 | 0.0191 |
| b128_lr0.0003_seed42 | 2 | 0.009281036 | 0.015 | 0.002040 | -0.00004769 | 0.9091 |
| b128_lr0.0003_seed42 | 3 | 0.011215300 | 0.020 | 0.002119 | -0.00000463 | 0.9899 |
| b64_lr0.0001_seed42 | 1 | 0.009077324 | 0.015 | 0.000011 | 0.00000057 | 0.0325 |
| b64_lr0.0001_seed42 | 2 | 0.009166667 | 0.015 | 0.001336 | 0.00002238 | 0.9374 |
| b64_lr0.0001_seed42 | 3 | 0.009166667 | 0.015 | 0.001601 | 0.00002018 | 0.9771 |
| b64_lr0.0003_seed42 | 1 | 0.009077324 | 0.015 | 0.000009 | 0.00000008 | 0.0105 |
| b64_lr0.0003_seed42 | 2 | 0.009281036 | 0.015 | 0.002022 | -0.00009712 | 0.7807 |
| b64_lr0.0003_seed42 | 3 | 0.011368528 | 0.020 | 0.002788 | -0.00001050 | 0.9922 |
