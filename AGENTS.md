## 沟通
- 面向用户的叙述默认使用简体中文；代码、命令和技术标识保持英文。
- 先给影响与结论，再给行动、待决策和必要证据；没有对应内容就省略。
- 使用简洁、连贯的段落；只有确实适合并列比较或按步骤执行时才使用列表。
- 使用具体、简单的词，避免无意义术语、套话、重复总结和未经请求的对比。
- 技术细节只保留对理解结论、判断风险或复现结果有帮助的部分。
## 执行方式
- 用户表示要开始新工作或修复现有问题时，持续推进，直到用户的目标完成，在目标方向上自主推进。
- 向用户提问之前，先完成上下文里已经授权、并且能把下一步变成可审查结果的工作。用户批准的应该是具体、可检查的结果。
- 用户建议不适合目标时直接说，不要迎合。
- 不要因为假想风险，主动加警告、免责声明、审批流程或安全/合规清单。
## 测试与验证
- 不要为可逆、影响小、只是复述实现的改动写测试。
- 跑与本次改动相称的测试，并完成必要检查。这些通过之后，只有出现新改动、新失败或尚未解决的疑点时，才扩大或重复测试；否则继续把任务做完。
- 收尾删掉本次产生、之后用不上的临时文件。
## 工具与并行
- 搜索文件或文本优先使用 `rg`、`rg --files`；独立的读取和查询尽量批量执行。
- 只有存在真正独立的工作流，且委派能节省时间或提升质量时才使用子 Agent。
- 共享状态、连续决策和简单任务由当前 Agent 直接完成；委派任务必须有明确输入、输出和完成判据，最终结论由主 Agent 汇总并验证。
## 目前开发阶段（可实时更新）
- **Phase 0 已闭环**：基础 SFT baseline 已完成训练与评估，当前正式进入 **Phase 1 — Minimal Latent-CE / CURE v1 implementation**。
- Amazon **Beauty 2018–2023** 与 **Sports 2018–2023** 已完成本地数据处理、数据审计 / freeze、text embedding、RQ-VAE 与 3-code Semantic ID artifacts。
- leave-two-out test evaluation 按 `train + valid → test` 执行；Beauty 作为 development domain，Sports 作为冻结 recipe 后的 confirmation domain。
- 研究路线明确采用 **method-first**：优先提升 CURE 自身推荐性能，再用 CURE-specific intervention 证明 latent reasoning 对性能提升有实际贡献。
- 当前 CURE v1 实现计划见 `docs/cure_v1_implementation_plan.md`。
- Phase 1 第一轮固定使用 **K=1 latent reasoner + gated residual fusion**；优先从冻结的 merged SFT checkpoint 初始化，不使用 forward hook，不使用 utilization / anchor / multi-intent。
- 不展开跨架构 latent-reasoning faithfulness audit；CARE 等外部方法主要作为后期标准推荐性能 baseline。
## 下一步计划（可实时更新）
- 新建 `src/models/cure/`，先实现 `dataset.py`、`reasoner.py`、`model.py`、`losses.py`、`trainer.py`。
- CURE v1 的 latent 只读取 history contextual states，禁止读取 target SID；dataset 显式返回 history positions，避免通过 magic offset 推断。
- 第一轮 Latent-CE 冻结 Qwen backbone / LM head / SID embeddings，只训练 reasoner、latent projection 和 fusion gate。
- 训练与 generation 必须共用同一 fusion 路径；禁止使用 MIR 风格 forward hook。
- 优先完成六个关键验证：Zero-Z≈SFT、无 target leakage、backbone 无梯度、reasoner/fusion 有梯度、checkpoint round-trip、train/generation fusion parity。
- 上述测试通过后，在 Beauty 跑 Latent-CE，并以 HR / NDCG 对比冻结 SFT baseline。
- 只有 Latent-CE 至少达到或基本持平 SFT 后，才进入 Phase 2 的 CURE-U / CURE-UA；否则先改 representation / fusion，不通过 utilization loss 强迫依赖低质量 latent。
- Phase 3 只针对 CURE 自身做 normal / zero / shuffle / matched intervention，用于证明 latent pathway 对增益有可验证贡献。
- CARE 已保留在 `baselines/care/`，未来在 Phase 5 作为 SID-based reasoning 性能 baseline；当前不做 CARE-specific faithfulness audit。
- 只有 CURE 主结果稳定后，再考虑是否加入 S²GR、IBA、SCoTER 等额外性能 baseline。
- 注意，为了防止该文档无效堆积过度膨胀，当完成某一阶段的开发任务时，请把开发记录存在![开发进度文档](docs/develop_log.md)里，开发进度文档除了我告知你审阅的情况下，正常情况下**禁止**阅读该文档，避免浪费大量token阅读过去开发记录
