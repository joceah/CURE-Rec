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
- 当前处于 **Phase 0 — Baseline Sanitation & Reproducibility**。
- Amazon **Beauty 2018–2023** 与 **Sports 2018–2023** 已完成本地数据处理，processed artifacts 位于本地 `data/processed/`。
- leave-two-out test evaluation 已按 `train + valid → test` 修正。
- 当前子阶段是 **Phase 0B — Processed Data Audit & Freeze**：先审计并冻结数据，再进入 text embedding、RQ-VAE / SID、Qwen SFT 和 baseline evaluation。
- Beauty 作为 development domain；Sports 作为冻结 recipe 后的 confirmation domain。
## 下一步计划（可实时更新）
- 审计 Beauty / Sports processed data：规模、序列长度、时间顺序、target coverage、metadata 缺失和 ID 映射一致性。
- 为两个 domain 生成 dataset manifest 与 artifact hash，固定 validation / test sample IDs。
- 先在 Beauty 上生成 text embeddings 并训练 / 冻结 RQ-VAE 与 SID artifacts，再用同一配置复制到 Sports。
- 在 Beauty 上训练并冻结 Qwen SFT baseline，随后以冻结 recipe 跑 Sports confirmation baseline。
- 完成统一 baseline evaluation 后再进入 Phase 1 的 K=1 latent pathway。
- 注意，为了防止该文档无效堆积过度膨胀，当完成某一阶段的开发任务时，请把开发记录存在![开发进度文档](docs/develop_log.md)里，开发进度文档除了我告知你审阅的情况下，正常情况下**禁止**阅读该文档，避免浪费大量token阅读过去开发记录
