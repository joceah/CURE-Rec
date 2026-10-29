## 沟通
- 面向用户的叙述默认使用简体中文；代码、命令和技术标识保持英文。
- 先给影响与结论，再给行动、待决策和必要证据；没有对应内容就省略。
- 使用简洁、连贯的段落；只有确实适合并列比较或按步骤执行时才使用列表。
- 使用具体、简单的词，避免无意义术语、套话、重复总结和未经请求的对比。
- 技术细节只保留对理解结论、判断风险或复现结果有帮助的部分。

## 执行方式
- 用户表示要开始新工作或修复现有问题时，持续推进，直到用户的目标完成，在目标方向上自主推进。
- 向用户提问之前，先完成上下文里已经授权、并且能把下一步变成可审查结果的工作。
- 用户建议不适合目标时直接说，不要迎合。
- 不要因为假想风险主动增加无关流程。

## 测试与验证
- 不要为可逆、影响小、只是复述实现的改动写测试。
- 跑与本次改动相称的测试，并完成必要检查；通过后除非出现新失败或新疑点，不重复扩大测试。
- 收尾删除本次产生、之后用不上的临时文件。

## 工具与并行
- 搜索文件或文本优先使用 `rg`、`rg --files`；独立读取和查询尽量批量执行。
- 只有存在真正独立的工作流，且委派能提升效率或质量时才使用子 Agent。
- 共享状态、连续决策和简单任务由当前 Agent 直接完成。

## 目前开发阶段（可实时更新）
- **Phase 0 已闭环**：Beauty / Sports 2018–2023 的 processed data、audit/freeze、text embedding、RQ-VAE、3-code SID artifact 与基础 Qwen SFT baseline 已建立；Beauty SFT 已完成正式评估。
- leave-two-out test evaluation 固定为 `train + valid → test`；Beauty 是 development domain，Sports 是冻结 recipe 后的 confirmation domain。
- 当前处于 **Phase 1 — Safe Personalized Residual CURE**：基础代码、关键 invariant 与真实 SFT smoke 已完成；尚未开展正式 Beauty validation 方法实验。
- 研究路线保持 **method-first**：首先让 CURE 在 HR / NDCG 上产生真实增益，再用针对自身的机制证据解释增益。
- 当前核心关系固定为：`Correct personalized Z > frozen SFT baseline ≈ matched wrong Z`，对应 positive lift、personalized lift、safe fallback。
- 不把“增加 latent depth”“residual learning”“marginal likelihood gain”“KL preservation”单独宣称为创新；这些已有相关工作。CURE 的候选贡献是把三者组织成针对 Semantic-ID generative recommendation 的非对称 safe personalized residual 训练原则。
- 外部方法主要作为后期性能 baseline；当前不展开跨架构 faithfulness audit。
- 新实现计划见 `docs/cure_implementation_plan.md`。

## 下一步计划（可实时更新）
- 已从冻结 Beauty SFT checkpoint 完成新的 CURE 基础实现，未复用旧 latent-residual implementation；入口及验证命令见 `docs/cure_usage.md`。
- baseline/correct/counterfactual 已共享合法 candidate scoring，原生 SFT beam/KV cache 在 correction disabled 时通过真实 BF16 Top-10 精确恢复测试。CURE 使用 candidate-normalized decoding；历史 SFT 指标不能直接作为新协议 reference。
- latent reasoner 使用 baseline decision state 作为 query，再查询 history evidence；latent 不能读取 target。
- CURE 只学习对 frozen SFT decision 的候选空间 residual correction；correction disabled 时必须结构性恢复 SFT。
- CURE 已支持 Accelerate DDP / DeepSpeed ZeRO-2、ZeRO-3 及 CPU parameter/optimizer offload，具备并行缓存/评估与可选 W&B。ZeRO-3 使用通信补齐但排除虚拟样本计数，只 gather 新增 branch 保存；真实 Beauty SFT train/eval CLI 和双 GPU 测试已通过。省显存 preset 为 `configs/cure_zero3.yaml`，启动方式及配置命名迁移表见 `docs/cure_usage.md`；公共训练参数沿用 SFT，公共评估参数沿用 eval.yaml。
- CURE 与普通 SFT 已支持按 optimizer update 进行 validation / early stopping：默认 eval_steps=500、patience_steps=2500、min_delta=0.0001；CURE 监控 correct NDCG@10，SFT 监控全局 token 加权 validation loss。SFT 结束恢复 best_ckpt 后导出 merged；DDP、ZeRO-2/3 的同步停止与最佳权重导出已有合成模型回归覆盖。
- 候选评分已按 sample/stage 批量计算，beam 内 residual 也已批量化；保留合法候选归一化、target 隔离和关闭修正恢复。开发 preset `configs/cure_dev.yaml` 保留 ZeRO-3、关闭 CPU offload，固定 train/validation 子集；具体对照步骤见 `docs/cure_usage.md`。日志包含同 batch 的 baseline_rec 与排除验证的 seconds_per_update。已完成数值/梯度对照、双卡 ZeRO 回归和真实 SFT CLI smoke，尚不代表推荐效果提升。
- 固定 train=32768 / validation=200 的 Residual-CE 四组 batch/LR 开发 sweep 已完成：lr=0.0003 的 HR@10 从 baseline 3/200 到 4/200，correct/matched HR/NDCG 全部一致，尚无稳定或额外 latent 个性化增益结论；分析见 `docs/cure_dev_sweep_analysis.md`。训练吞吐约 329/422 samples/s（每卡 batch=64/128），目前约 80% 时间在 validation。下一步扩大 validation 并检查 Z 敏感性，再按 CURE-G → CURE-GP → CURE-Full 顺序验证。
- matched donor 第一版只控制 history length、target popularity 和 baseline difficulty；不得与当前样本相同。
- 方法开发只看 validation；设计冻结后再重新运行 Beauty test，并随后在 Sports 上使用同一 recipe confirmation。
- CARE 保留在 `baselines/care/`，未来只作为 SID-based reasoning 性能 baseline。
- 完成某一开发阶段后把记录写入 `docs/develop_log.md`；除非用户明确要求审阅，正常情况下**禁止读取**该文档，避免浪费 token 阅读历史记录。
