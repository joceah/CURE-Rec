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
- 当前处于 **Phase 0 — Baseline Sanitation & Reproducibility**，其中 Phase 0A–0C 已完成，当前重点是 Phase 0D–0E 的 SFT baseline freeze 与统一 evaluation。
- Amazon **Beauty 2018–2023** 与 **Sports 2018–2023** 已完成本地数据处理、数据审计 / freeze、text embedding、RQ-VAE 与 3-code Semantic ID artifacts。
- leave-two-out test evaluation 按 `train + valid → test` 执行；Beauty 作为 development domain，Sports 作为冻结 recipe 后的 confirmation domain。
- **CARE** 已作为第一个外部 SID-based reasoning baseline 适配到项目协议，代码位于 `baselines/care/`；当前应先完成 Beauty numerical sanity，再决定 intervention audit 的扩展。
- CARE 属于 method-level external baseline，不与 SFT / Latent-CE / CURE 的 controlled ablation 混为一类。其 stage query state 是 SID prediction 的结构性必经表示，不能机械把 zero-query-state 解释为与 CURE Zero-Z 相同的 Necessity Gap。
## 下一步计划（可实时更新）
- 完成并冻结 Beauty Qwen SFT baseline，随后以冻结 recipe 跑 Sports confirmation baseline。
- 完成统一 baseline evaluation 与 artifact identity 记录，闭环 Phase 0。
- 对 CARE 先跑 Beauty train → valid → test numerical sanity，并确认 frozen sample IDs、SID collision reporting 与 checkpoint reload 行为。
- Phase 1 实现 K=1 Latent-CE，并加入 normal / zero / random / shuffle intervention hooks。
- Phase 2 以 CARE 作为第一外部 SID reasoning baseline；CARE 使用 architecture-aware audit（如 progressive-off、matched reasoning-state swap、history sensitivity），不要强行与 CURE 的 Zero-Z NG 等价。
- 只有 CURE / CARE 主结果稳定后，再考虑 S²GR、IBA、SCoTER 等第二批外部方法。
- 注意，为了防止该文档无效堆积过度膨胀，当完成某一阶段的开发任务时，请把开发记录存在![开发进度文档](docs/develop_log.md)里，开发进度文档除了我告知你审阅的情况下，正常情况下**禁止**阅读该文档，避免浪费大量token阅读过去开发记录
