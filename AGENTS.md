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
- 当前进入新的 **Phase 1 — Safe Personalized Residual CURE formulation & implementation**。
- 研究路线保持 **method-first**：首先让 CURE 在 HR / NDCG 上产生真实增益，再用针对自身的机制证据解释增益。
- 当前核心关系固定为：`Correct personalized Z > frozen SFT baseline ≈ matched wrong Z`，对应 positive lift、personalized lift、safe fallback。
- 不把“增加 latent depth”“residual learning”“marginal likelihood gain”“KL preservation”单独宣称为创新；这些已有相关工作。CURE 的候选贡献是把三者组织成针对 Semantic-ID generative recommendation 的非对称 safe personalized residual 训练原则。
- 外部方法主要作为后期性能 baseline；当前不展开跨架构 faithfulness audit。
- 新实现计划见 `docs/cure_implementation_plan.md`。

## 下一步计划（可实时更新）
- 从冻结 Beauty SFT checkpoint 出发实现新的 CURE，不复用旧 latent-residual implementation。
- 首先实现与 prefix-constrained SID decoding 完全一致的 baseline/correct/counterfactual candidate scoring。
- latent reasoner 使用 baseline decision state 作为 query，再查询 history evidence；latent 不能读取 target。
- CURE 只学习对 frozen SFT decision 的候选空间 residual correction；correction disabled 时必须结构性恢复 SFT。
- 核心训练项依次为 `L_rec`、`L_gain`、`L_personal`、`L_safe`；full objective 只有在各分项实现和验证正确后才启用。
- matched donor 第一版只控制 history length、target popularity 和 baseline difficulty；不得与当前样本相同。
- 方法开发只看 validation；设计冻结后再重新运行 Beauty test，并随后在 Sports 上使用同一 recipe confirmation。
- CARE 保留在 `baselines/care/`，未来只作为 SID-based reasoning 性能 baseline。
- 完成某一开发阶段后把记录写入 `docs/develop_log.md`；除非用户明确要求审阅，正常情况下**禁止读取**该文档，避免浪费 token 阅读历史记录。
