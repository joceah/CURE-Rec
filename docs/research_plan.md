# CURE-Rec Research Plan

> 更新时间：2026-10-13  
> Working title: **CURE-Rec: Causal Utilization-Regularized rEasoning for Generative Recommendation**  
> 主线定位：**method-first generative recommendation**。核心目标是先做出更强的推荐方法，再证明 CURE 中的 latent reasoning 确实对性能提升产生了可验证的贡献。

---

# 0. 项目定位与研究边界

CURE-Rec 当前不试图回答一个过于宽泛的问题：

> “现有所有隐式推理推荐模型到底有没有真正利用 reasoning？”

这个问题本身很有研究价值，但如果认真展开，需要针对大量不同架构分别设计 intervention，容易演变成一个大型 evaluation / survey-style project。当前项目不以此为目标。

CURE-Rec 的论文优先级明确为：

1. **Recommendation utility first**：CURE 必须首先在 HR / NDCG 等主要推荐指标上表现有竞争力，最好稳定优于基础 SFT / TIGER-style baseline。
2. **Method contribution second**：提出一个轻量、清晰、可复现的 latent reasoning framework，而不是堆叠大量 reasoning 模块。
3. **Mechanistic evidence as supporting story**：只针对 CURE 自身证明 latent reasoning 不是装饰性表示，而是对性能提升有实际贡献。
4. **External methods are performance baselines**：CARE、S²GR、IBA、SCoTER 等外部方法主要用于最终性能比较；除非后期论文方向发生变化，不要求为每种架构设计专属 faithfulness audit。

因此论文理想叙事是：

> **CURE improves generative recommendation by learning a useful latent reasoning pathway, and controlled interventions show that the learned latent state is genuinely responsible for part of the performance gain.**

这和“建立一个通用 latent-reasoning faithfulness benchmark”是两个不同的项目。当前只做前者。

---

# 1. 从 MIR 到 CURE-Rec

最初的研究动机是：

> 显式自然语言 CoT 成本高，能否把额外推理压缩成低成本的 latent computation？

MIR 与早期 implicit-CoT 路线暴露了几个关键问题：

- teacher CoT 可能只是 post-hoc explanation，而不是 predictive reasoning；
- MSE / cosine 对齐只能说明 representation 相似，不说明它对推荐有用；
- 多个 latent intent 可以彼此不同，但不一定功能不同；
- generator 可以绕过 latent pathway，继续主要依赖 history → target 的直接路径；
- 训练和推理路径一旦不一致，latent module 的实验结论就不可信。

因此 CURE 不再从复杂的 multi-intent / CoT distillation 开始，而先解决一个更基本的问题：

> **能否设计一个简单的 latent reasoning pathway，使它既提升推荐性能，又能通过干预证明它确实贡献了这部分提升？**

---

# 2. 当前核心假设

## H1 — Useful Latent Reasoning

一个好的 latent state 不应该只“看起来合理”，而应该带来额外的 recommendation utility：

$$
P(y \mid H, Z)
>
P_0(y \mid H)
$$

其中 $P_0$ 是没有 latent reasoning 的 SFT baseline。

## H2 — Personalized Contribution

如果 latent reasoning 真正在建模当前用户，它应该比其他用户的 latent 更有帮助：

$$
\log P(y_i \mid H_i, Z_i)
>
\log P(y_i \mid H_i, \tilde Z_i)
$$

其中 $\tilde Z_i$ 是来自 matched donor 的 counterfactual latent。

## H3 — Utilization Can Be Explicitly Trained

普通 CE 训练不保证 generator 会主动使用 latent pathway，因此可以通过 utilization regularization 让正确 latent 提供额外增益，同时避免模型通过“破坏错误 latent 分支”作弊。

---

# 3. CURE Framework

## 3.1 最小结构

第一版保持简单：

~~~text
History H
   ↓
Qwen / TIGER-style SID Generator
   ├────────────── direct recommendation path
   │
   ↓
Latent Reasoner R
   ↓
Z (K = 1)
   ↓
Gated Fusion
   ↓
SID Prediction
~~~

Phase 1–3 只使用 $K=1$。

暂时不加入：

- explicit CoT；
- teacher CoT distillation；
- K=4 multi-intent；
- GRPO；
- posterior-prior；
- complex dynamic routing；
- semantic hierarchy；
- 额外 diversity loss。

原则是：

> **先证明一个 latent 有用，再讨论多个 latent。**

---

## 3.2 Recommendation Loss

$$
\mathcal{L}_{rec}
=
-\log P(y \mid H,Z)
$$

$y$ 为目标 item 的 SID 序列。

---

## 3.3 Utilization Loss

对当前样本构造 matched counterfactual latent $\tilde Z$：

$$
\mathcal{L}_{util}
=
\max
\left(
0,
\;m-
\left[
\log P(y\mid H,Z)
-
\log P(y\mid H,\tilde Z)
\right]
\right)
$$

它要求：

> 当前用户自己的 latent 应比一个“相似但错误”的 latent 更有助于预测正确目标。

matched donor 优先按以下因素控制：

- history length；
- target popularity；
- baseline difficulty；
- 可选 coarse category。

这样比完全随机 shuffle 更不容易被解释为 OOD perturbation。

---

## 3.4 Baseline Anchor

单独优化 $\mathcal{L}_{util}$ 可能产生一个坏解：模型故意让 counterfactual branch 变差，从而轻易拉大 margin。

因此引入冻结的 SFT baseline：

$$
P_0(\cdot\mid H)
$$

并约束错误 latent 下的输出不要偏离 baseline 太多：

$$
\mathcal{L}_{anchor}
=
D_{KL}
\left(
P(\cdot\mid H,\tilde Z)
\;\|\;
P_0(\cdot\mid H)
\right)
$$

理想行为：

~~~text
Wrong / unhelpful latent
→ prediction falls back toward SFT baseline

Correct personalized latent
→ prediction improves beyond SFT baseline
~~~

---

## 3.5 CURE Objective

$$
\mathcal{L}_{CURE}
=
\mathcal{L}_{rec}
+
\lambda_u\mathcal{L}_{util}
+
\lambda_a\mathcal{L}_{anchor}
$$

核心 ablation：

| Variant | Rec | Util | Anchor |
|---|---:|---:|---:|
| SFT | ✓ | – | – |
| Latent-CE | ✓ | – | – |
| CURE-U | ✓ | ✓ | – |
| CURE-UA | ✓ | ✓ | ✓ |

---

# 4. CURE 自身的机制验证

这里的 intervention 只用于回答：

> **CURE 自己的 latent pathway 是否真的贡献了它的性能增益？**

它不是要形成一个跨所有 reasoning architectures 的统一 benchmark。

## 4.1 Normal

$$
P(y\mid H,Z)
$$

## 4.2 Zero Latent

$$
Z\rightarrow0
$$

测试 CURE latent pathway 的必要性。

## 4.3 Global Shuffle

$$
Z_i\rightarrow Z_j
$$

测试模型是否只依赖一个 generic latent bias。

## 4.4 Matched Counterfactual

$$
Z_i\rightarrow Z_{matched(j)}
$$

测试 CURE latent 是否具有当前用户特有的信息价值。

## 4.5 Supporting Metrics

传统推荐指标始终是主指标：

- HR@K；
- NDCG@K。

机制指标作为辅助证据：

### Necessity Gap

$$
NG@K
=
HR@K_{normal}
-
HR@K_{zero}
$$

### Personalization Gap

$$
PG@K
=
HR@K_{normal}
-
HR@K_{matched}
$$

也可以记录：

- target log-prob change；
- target rank change；
- Top-K Jaccard；
- candidate distribution divergence。

这些指标用于解释 **CURE 的内部机制**，不是论文的独立 benchmark contribution。

成功标准不是“NG / PG 越大越好”，而是：

~~~text
Recommendation accuracy      ↑
CURE > SFT / Latent-CE        ideally
NG / PG                       > 0 and stable
Counterfactual branch         not catastrophically worse than SFT
~~~

如果出现：

~~~text
NG ↑
PG ↑
HR ↓
~~~

说明模型只是被强迫依赖一个低质量 latent，不算成功。

---

# 5. 当前项目状态

当前处于 **Phase 0 — Baseline Sanitation & Reproducibility** 的收尾阶段。

已完成：

- Amazon Beauty 2018–2023 数据处理；
- Amazon Sports 2018–2023 数据处理；
- leave-two-out protocol 修正；
- processed-data audit；
- frozen validation / test sample IDs；
- dataset manifests；
- text embeddings；
- RQ-VAE；
- 3-code Semantic ID artifacts。

当前优先级：

1. 完成并冻结 Beauty Qwen SFT baseline；
2. 完成统一 validation / test evaluation；
3. 用冻结 recipe 跑 Sports confirmation baseline；
4. 完成 Phase 0 artifact identity 与 reproducibility closure；
5. 进入 Phase 1 的 K=1 latent pathway。

Beauty 是 development domain，Sports 是 confirmation domain。

---

# 6. 研究路线：Phase 0 → Phase 5

## Phase 0 — Clean Baseline & Reproducibility

### 已完成

- 0A Data Processing；
- 0B Processed Data Audit & Freeze；
- 0C Text Embedding / RQ-VAE / SID Artifact。

### 当前任务：0D–0E

#### 0D — Qwen SFT Baseline

Beauty：

~~~text
pipeline debugging
→ hyperparameter sanity
→ freeze SFT recipe
~~~

Sports：

~~~text
reuse frozen recipe
→ confirmation run
~~~

固定：

- backbone；
- LoRA / full-FT strategy；
- learning rate；
- effective batch size；
- max history length；
- SID tokenizer；
- random seed；
- checkpoint selection rule。

#### 0E — Evaluation & Freeze

至少报告：

- HR@1 / 5 / 10 / 20 / 50；
- NDCG@5 / 10 / 20 / 50；
- prediction coverage；
- Top-1 concentration / popularity bias；
- SID collision statistics；
- basic history-sensitivity diagnostic。

每个结果记录：

- git SHA；
- dataset manifest；
- RQ-VAE / item-index identity；
- tokenizer；
- model checkpoint；
- seed；
- decoding config；
- sample IDs。

### Phase 0 Go / No-Go

只有当 Beauty 完整 pipeline 可复现、Sports 能复用冻结 recipe，并且不再存在 train / inference / evaluation protocol mismatch，才进入 Phase 1。

---

## Phase 1 — Minimal Latent-CE

### 目标

建立最简单的 $K=1$ latent pathway，并回答：

> 单纯加入一个 latent reasoner，在普通 CE 下是否能提高推荐性能？

实现：

~~~text
src/models/cure/
    model.py
    reasoner.py
    losses.py
~~~

要求：

- $Z$ 只依赖 history；
- gated residual fusion；
- 不使用 utilization loss；
- 不使用 multi-intent；
- 不使用 teacher signal。

必须验证：

1. gate = 0 时退化到 SFT-like behavior；
2. train / generation 使用同一 latent 路径；
3. save / reload 输出一致；
4. normal / zero / shuffle 能产生可测差异。

核心比较：

| Model | 目的 |
|---|---|
| SFT | 无 latent baseline |
| Latent-CE | 判断 latent architecture 本身是否有用 |

### Go / No-Go

如果 Latent-CE 连推荐性能都无法达到 SFT，先解决 representation / fusion，不进入复杂 regularization。

---

## Phase 2 — CURE Training

### 目标

在 Latent-CE 基础上加入 utilization regularization，使正确 latent 对最终预测产生额外增益。

顺序：

1. Latent-CE；
2. CURE-U：$\mathcal{L}_{rec}+\lambda_u\mathcal{L}_{util}$；
3. CURE-UA：再加入 $\lambda_a\mathcal{L}_{anchor}$。

主要看：

- HR / NDCG 是否提高；
- 相比 Latent-CE 的增益是否稳定；
- anchor 是否避免 counterfactual branch 被人为破坏。

### 核心成功标准

> **CURE 至少在 Beauty 上稳定优于或不弱于 Latent-CE，并有明确的机制证据说明增益来自 latent utilization，而不是额外参数本身。**

---

## Phase 3 — Mechanistic Validation of CURE

### 目标

只对 CURE 自身做有针对性的 intervention，支持论文故事：

> CURE 的 latent reasoning 确实促成了推荐性能提升。

实验：

- Normal；
- Zero-Z；
- Global Shuffle-Z；
- Matched Counterfactual-Z。

报告：

- HR / NDCG；
- NG；
- PG；
- target-rank shift；
- candidate-set shift。

可按 baseline difficulty 分成 easy / medium / hard，检查 latent gain 是否主要出现在困难样本。

这一阶段不要求对 CARE / S²GR / IBA 等外部架构做同等 intervention。

---

## Phase 4 — Method Refinement

只有在 Phase 2–3 的最小 CURE 已经成立后才做。

优先考虑的扩展顺序：

1. difficulty-aware utilization weighting；
2. 更稳健的 matched counterfactual sampling；
3. fusion / reasoner capacity ablation；
4. K > 1 multi-intent；
5. functional diversity。

不要同时加入多个扩展。

如果进入 K > 1，不再只看 cosine diversity，而更关注 functional specialization：

$$
P(y\mid H,Z)
\neq
P(y\mid H,Z_{\setminus k})
$$

---

## Phase 5 — Paper-Scale Validation

这一阶段才扩展数据集和外部 baselines。

### 数据集

最低必做：

- Amazon Beauty 2018–2023；
- Amazon Sports 2018–2023。

第三个数据集只在核心方法稳定后再决定。

### 外部 Baselines

外部方法主要做 **standard recommendation performance comparison**，不要求统一的 latent-faithfulness intervention。

优先级：

1. SFT / TIGER-style baseline；
2. **CARE**：已实现，未来作为 SID-based reasoning 性能 baseline；
3. 视时间与代码可用性再考虑 S²GR / IBA / SCoTER。

对 CARE 的当前要求仅为：

- 在同一 frozen dataset / SID / sample protocol 下训练与评估；
- 报告标准 HR / NDCG；
- 清楚说明 project adaptation 与原论文设置的差异。

不把 CARE 的 progressive-off、query-state intervention 等专属分析作为当前 CURE 论文的必做实验。

### Scaling / Robustness

主结果稳定后可增加：

- 不同 backbone size；
- 不同 seed；
- 不同 max history length；
- LoRA vs full fine-tuning（如果算力允许）。

---

# 7. 最小论文实验矩阵

| Group | Experiments |
|---|---|
| Base | SFT |
| Architecture | Latent-CE |
| Method | CURE-U / CURE-UA |
| CURE intervention | Normal / Zero / Shuffle / Matched |
| Dataset | Beauty → Sports |
| Performance baseline | CARE（后期） |
| Optional baseline | S²GR / IBA / SCoTER |
| Analysis | difficulty / latent gain / target-rank shift |

这已经足够形成一篇 method-first paper，不需要把项目扩张成大型隐式推理评测工程。

---

# 8. 论文主问题

论文可以围绕三个问题展开，但都只服务于 CURE：

### RQ1 — Does CURE improve generative recommendation?

用 HR / NDCG 回答。

### RQ2 — Does utilization regularization outperform a naive latent pathway?

比较 Latent-CE、CURE-U、CURE-UA。

### RQ3 — Is CURE's latent reasoning actually responsible for part of the gain?

用 Zero / Shuffle / Matched intervention 回答。

这三个问题已经构成完整方法论文故事，不需要额外证明“所有 latent reasoning 方法是否 faithful”。

---

# 9. Working Contributions

如果主实验成立，最终贡献优先写成：

1. **Method**  
   提出 CURE-Rec，一个面向 Semantic-ID generative recommendation 的轻量 latent reasoning framework。

2. **Training Objective**  
   提出 personalized utilization regularization 与 baseline anchoring，使 latent reasoning 提供真正的增量推荐信息，而不是仅形成辅助表示。

3. **Recommendation Performance**  
   在 Beauty / Sports 等数据集上证明 CURE 相比 SFT、naive latent architecture 和代表性 generative-reasoning baselines 具有竞争力。

4. **Mechanistic Evidence**  
   通过 CURE-specific controlled interventions 证明其 latent pathway 对性能提升具有可测贡献。

注意第 4 点是 **supporting evidence for our method**，不是“提出一个适用于所有隐式推理模型的新 benchmark”。

---

# 10. 主要风险与预案

## 风险 A：Latent-CE 本身就足够强

如果简单 latent architecture 已明显提升性能，CURE 必须证明 utilization objective 还能带来额外、稳定的增益，否则方法贡献不够强。

## 风险 B：NG / PG 提高但推荐性能下降

说明模型只是被强迫依赖 latent。

优先改 representation / fusion，不继续增大 $\lambda_u$。

## 风险 C：CURE 只在 Beauty 有效

先冻结 Beauty recipe，再在 Sports 做 confirmation。不要在 Sports 上重新大规模调参掩盖泛化问题。

## 风险 D：外部 baseline 适配成本过高

CARE 已作为第一个 SID-compatible baseline。其他方法只在主方法成立后按工程成本选择，不追求 baseline 数量。

## 风险 E：论文被理解成 evaluation paper

全文始终保持优先级：

> **performance → method → mechanism evidence**

而不是：

> evaluation metric → audit many models → method as appendix。

---

# 11. 近期里程碑

## Milestone A — Clean SFT Baseline

完成 Phase 0D–0E。

成功标志：

> Beauty SFT fully reproducible; Sports reproduces the frozen recipe.

## Milestone B — Latent-CE Works

完成 Phase 1。

成功标志：

> K=1 latent architecture 至少不弱于 SFT，并且 train / inference 路径完全一致。

## Milestone C — CURE Improves Utility

完成 Phase 2。

成功标志：

> CURE-U / CURE-UA 在主要推荐指标上稳定优于 Latent-CE。

## Milestone D — CURE Reasoning Is Used

完成 Phase 3。

成功标志：

> Zero / Shuffle / Matched interventions 能解释 CURE 的部分性能增益，而不是只有 representation-level 差异。

## Milestone E — Paper-Scale Evidence

完成 Phase 5。

成功标志：

> Beauty + Sports 结果成立，并完成 CARE 等代表性性能 baseline 与必要 ablation。

---

# 12. 当前暂不做

Phase 0–3 明确不做：

- 大规模 latent-reasoning model audit；
- 为每个外部架构设计专属 faithfulness intervention；
- explicit CoT generation；
- teacher CoT distillation；
- GRPO；
- K=4 multi-intent；
- complex dynamic routing；
- posterior-prior；
- baseline zoo。

这些都只能在核心 CURE 方法成立后再决定是否值得扩展。

---

# 13. 相关工作方向

## 13.1 SID-based generative reasoning

- **CARE — Bringing Reasoning to Generative Recommendation Through the Lens of Cascaded Ranking**  
  Query-Anchored Reasoning + Progressive History Encoding。当前项目已经完成 protocol adaptation，后续主要作为性能 baseline。  
  Paper: https://arxiv.org/abs/2602.03692  
  Code: https://github.com/Linxyhaha/CARE

- **S²GR** — stepwise semantic-guided latent reasoning；
- **IBA / Where Reasoning Matters** — SID position-wise reasoning budget；
- **SCoTER** — structured reasoning transfer。

## 13.2 Latent reasoning recommendation

- LatentR³；
- LaRec；
- HiLaR；
- ReaRec；
- LARES；
- RecRec。

这些工作主要用于 motivation / related work。是否进入实验由架构兼容性和代码成本决定。

## 13.3 Semantic ID / tokenizer / protocol

- ReSID；
- UniGRec；
- PIT；
- AsymRec；
- GenPAS。

---

# 14. 一句话项目原则

> **先把 CURE 做成一个更强的推荐方法，再用最小但可信的 intervention 证明：它的 latent reasoning 确实参与了这部分提升。**

---

## Appendix — Markdown 数学公式约定

行内公式使用：

$P(y\mid H,Z)$

块级公式使用：

$$
\mathcal{L}
=
\mathcal{L}_{rec}
+
\lambda_u\mathcal{L}_{util}
+
\lambda_a\mathcal{L}_{anchor}
$$

统一使用 `$...$` / `$$...$$`，避免不同 Markdown renderer 对 `\[...\]` 支持不一致。
