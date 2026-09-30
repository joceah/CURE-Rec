# CURE-Rec Research Plan

> 更新时间：2026-09-30  
> Working title: **CURE-Rec: Causal Utilization-Regularized rEasoning for Generative Recommendation**  
> 核心问题：生成式推荐模型能否学习不仅“与目标相关”，而且真正被最终推荐决策所利用的 latent reasoning？

---

## 0. 文档定位

这份文档不再只是 research idea 汇总，而是 CURE-Rec 的执行路线图。

项目分成两条互相支撑但工作量不同的主线：

1. **Method track**：提出一个新的 CURE framework，让 latent reasoning 不仅存在，而且被 generator 真正读取和利用。
2. **Research / evaluation track**：用 intervention-based protocol 判断 latent reasoning 是否必要、是否个性化、是否只是冗余表示。

原则上，论文不能只剩下“我们发明了几个新指标”，也不能只剩下“我们又加了一个 latent module”。更完整的故事应该是：

> 先发现 latent reasoning 可能存在 utilization gap，再提出 CURE framework 主动缩小这个 gap，并用 intervention protocol 验证模型确实发生了机制变化。

为了控制工作量，项目采用 **progressive validation**：

- 前期只在自己的 TIGER/Qwen backbone 上验证现象；
- 只有当现象和方法都成立后，再扩展外部 latent-reasoning baselines；
- 不在 Phase 0–2 就大规模复现所有近期论文。

---

# 1. 从 MIR 到 CURE-Rec：真正保留什么

最初的目标不是 multi-intent 本身，而是：

> **显式自然语言 CoT 太昂贵，能否把额外推理压缩成低成本的内部 latent computation？**

早期路线大致是：

~~~text
Large Recommendation LLM
        ↓
Structured CoT
        ↓
Small Transformer
        ↓
Thought Tokens
        ↓
Generative Recommender
~~~

后来出现的问题包括：

- teacher CoT 更像 post-hoc explanation，而不是 predictive reasoning；
- MSE / cosine 对齐只保证表示相似，不保证对推荐有用；
- 人工定义的 “用户画像 / 商品特点 / matching” 并不一定形成真正的 reasoning chain；
- thought token 可以被训练出来，但 generator 未必真的读取它；
- MIR 的多个 latent intent 可能只是“多个不同向量”，不一定是多个功能不同的 intent；
- 旧 MIR 实验还暴露了 train / inference 路径不一致的问题。

因此 CURE-Rec 不再把 “K 个 latent intent” 作为起点，而把问题退回到最基本的一层：

> **一个 latent reasoning state，能否成为最终推荐决策真正使用的信息通路？**

---

# 2. 研究假设

CURE-Rec 主要验证三个假设。

### H1 — Latent Utilization Gap

一个模型可以学到可预测、可解释甚至与 teacher 对齐的 latent state，但最终 generator 仍然主要依赖直接的 history-to-item 路径。

也就是说：

$$
\text{good latent representation}
\not\Rightarrow
\text{latent is used by prediction}
$$

### H2 — Personalized Utilization

真正有用的 latent reasoning 不应该只提供一个通用 bias。

对用户 $i$，正确 latent $Z_i$ 应比来自另一个相似用户的 counterfactual latent $\tilde Z_i$ 更有助于预测当前目标：

$$
\log P(y_i \mid H_i, Z_i)
>
\log P(y_i \mid H_i, \tilde Z_i)
$$

### H3 — Utilization Can Be Trained

如果我们显式优化“正确 latent 应带来额外预测增益，而错误 latent 应回退到 baseline”，就有可能让 latent reasoning 从 auxiliary representation 变成真正的 decision pathway。

---

# 3. CURE Framework

## 3.1 总体结构

第一版 CURE 故意保持简单：

~~~text
                   ┌────────────────────────────┐
                   │     Base Generator G       │
                   │   Qwen + SID Generation    │
                   └─────────────┬──────────────┘
                                 │
History H ──────── Backbone Hidden States
                                 │
                                 ├─────────────── direct path
                                 │
                                 ↓
                         Latent Reasoner R
                                 │
                                 ↓
                              Z (K=1)
                                 │
                                 ↓
                           Gated Fusion
                                 │
                                 ↓
                         SID Prediction
~~~

第一阶段只做 $K=1$。

原因很简单：

> 如果一个 latent 都无法被稳定利用，那么研究 K=4 multi-intent 没有可靠基础。

后续只有在 K=1 utilization 成立后，才进入 multi-intent / functional diversity。

---

## 3.2 Base Recommendation Loss

正常推荐目标：

$$
\mathcal{L}_{rec}
=
-\log P(y \mid H, Z)
$$

这里 $y$ 是目标商品对应的 SID 序列。

---

## 3.3 Counterfactual Latent

对样本 $i$，构造另一个 latent：

$$
\tilde Z_i = Z_j,\quad j \neq i
$$

但 $j$ 不应该完全随机。

优先使用 matched counterfactual：

- 相似 history length；
- 相似 target popularity；
- 相似 baseline difficulty；
- 可选：相同 coarse category；
- 但来自不同用户 / 不同 history。

这样避免 reviewer 质疑：

> 随机 latent 只是 OOD noise，性能下降并不能证明 personalization。

---

## 3.4 Utilization Loss

希望正确 latent 比 matched counterfactual latent 更有帮助：

$$
\mathcal{L}_{util}
=
\max
\left(
0,\;
m -
\left[
\log P(y \mid H,Z)
-
\log P(y \mid H,\tilde Z)
\right]
\right)
$$

它训练的是：

> correct latent should provide useful personalized information.

而不是：

> latent should look like some teacher embedding.

---

## 3.5 Baseline Anchor

单独使用 utilization margin 有一个漏洞：

模型可以通过“故意破坏 counterfactual branch”获得很大的 margin。

因此先准备普通 SFT baseline：

$$
P_0(\cdot \mid H)
$$

对于 counterfactual latent，希望模型回退到 baseline，而不是崩坏：

$$
\mathcal{L}_{anchor}
=
D_{KL}
\left(
P(\cdot \mid H,\tilde Z)
\;\|\;
P_0(\cdot \mid H)
\right)
$$

直觉是：

~~~text
No useful latent
      ↓
baseline-like prediction

Correct personalized latent
      ↓
additional useful information
      ↓
better prediction
~~~

---

## 3.6 CURE Objective

第一版只保留三个 loss：

$$
\mathcal{L}_{CURE}
=
\mathcal{L}_{rec}
+
\lambda_u \mathcal{L}_{util}
+
\lambda_a \mathcal{L}_{anchor}
$$

暂时不要加入：

- CoT distillation；
- diversity loss；
- K=4；
- GRPO；
- posterior-prior；
- complex SID routing；
- 多层人工 semantic reasoning。

如果最小机制都没有建立，复杂模块只会让问题更难定位。

---

# 4. Intervention-Based Evaluation

CURE 的 evaluation 不替代 HR / NDCG，而是在它们之外回答：

> 模型是否真的依赖当前 latent？

对于同一个训练好的模型、同一个 history $H$，只改变 latent。

---

## 4.1 Normal

$$
P(y \mid H,Z)
$$

正常推理。

## 4.2 Zero Latent

$$
Z \rightarrow 0
$$

测试 latent pathway 是否必要。

## 4.3 Random Latent

$$
Z \rightarrow \epsilon
$$

测试模型是否对 latent input 敏感。

## 4.4 Global Shuffle

$$
Z_i \rightarrow Z_j
$$

测试 latent 是否只是通用 bias。

## 4.5 Matched Counterfactual Shuffle

$$
Z_i \rightarrow Z_{\text{matched}(j)}
$$

测试 latent 是否携带当前用户专属信息。

---

# 5. 核心诊断指标

传统推荐指标继续报告：

- HR@K
- NDCG@K

除此之外增加机制指标。

### Necessity Gap

$$
NG@K
=
HR@K_{normal}
-
HR@K_{zero}
$$

### Shuffle Gap

$$
SG@K
=
HR@K_{normal}
-
HR@K_{shuffle}
$$

### Personalization Gap

$$
PG@K
=
HR@K_{normal}
-
HR@K_{matched}
$$

### Distribution Sensitivity

$$
DS
=
D_{KL}
\left(
P(\cdot \mid H,Z)
\;\|\;
P(\cdot \mid H,\tilde Z)
\right)
$$

还可以记录：

- Top-K Jaccard；
- rank correlation；
- target-rank change；
- category / semantic candidate shift。

重要提醒：

> 这些指标应描述为 **interventional utilization metrics**。  
> “causal” 是研究动机和 intervention 视角，但除非后续给出严格 SCM / identification assumptions，不应把结果表述成强因果效应估计。

这样可以减少 reviewer 对 causal terminology 的攻击。

---

# 6. 清晰研究路线：Phase 0 → Phase 5

下面是建议真正执行的项目顺序。

---

## Phase 0 — Baseline Sanitation & Reproducibility

### 目标

建立一个可信、冻结、后续所有实验都不会再改变的 TIGER/Qwen baseline。

### 必须完成

1. 修复 leave-two-out evaluation：
   - validation：history = train，target = valid；
   - test：history = train + valid，target = test。
2. 确认 RQ-VAE / SID tokenizer / Qwen SFT / evaluator 从头到尾可跑。
3. 固定：
   - dataset version；
   - preprocessing；
   - item mapping；
   - RQ-VAE checkpoint；
   - item.index.json；
   - tokenizer；
   - SFT starting model；
   - random seed；
   - beam size；
   - sample IDs。
4. result JSON 记录：
   - git commit SHA；
   - model/config identifier；
   - seed；
   - data split；
   - evaluation settings。
5. 明确 SID collision rate，并决定：
   - 至少同时报告 SID-level metric；
   - 后续是否增加 collision-aware item-level evaluation。

### 数据集策略

不要一开始铺太开。

Phase 0 先只用：

- Amazon Beauty

等 pipeline 固定后，再复制到：

- Amazon Sports；
- Amazon Toys。

### 交付物

- 一个冻结 SFT baseline；
- 一个固定 RQ-VAE / item index；
- baseline evaluation JSON；
- reproducibility README；
- basic smoke tests。

### Go / No-Go

只有当同一 checkpoint 重复评测结果稳定，并且完整 pipeline 可复现时，才进入 Phase 1。

---

## Phase 1 — Minimal Latent Pathway

### 目标

先回答一个最基本的问题：

> 普通 CE 训练下，一个 K=1 latent pathway 会不会自然被 generator 使用？

### 实现

新增：

~~~text
src/models/cure/
    model.py
    reasoner.py
    losses.py
~~~

第一版：

- K = 1；
- 一个 lightweight Transformer / pooling reasoner；
- latent $Z$ 只依赖 history；
- 用 gated residual fusion 接入 generator；
- 不使用 CoT；
- 不使用 utilization loss；
- 不使用 multi-intent。

### 必须做的 unit tests

1. gate = 0 时，logits 应接近 SFT baseline；
2. normal Z 与 zero Z logits 应可区分；
3. normal Z 与 shuffle Z logits 应可区分；
4. save → reload 后输出一致；
5. generation 时 latent pathway 必须真的被调用。

### 实验

只比较：

| Model | 作用 |
|---|---|
| SFT | 无 latent baseline |
| Latent-CE | K=1，只用 recommendation CE |

然后对 Latent-CE 跑：

- normal；
- zero；
- random；
- shuffle。

### 关键问题

可能出现三种结果。

**A. Latent-CE 提升 accuracy，zero/shuffle 明显下降**

说明自然 utilization 已经存在。后续重点转向 personalization / matched counterfactual。

**B. Latent-CE 提升 accuracy，但 zero/shuffle 几乎不变**

这是最理想的 CURE motivation：存在 latent bypass / weak utilization。

**C. Latent-CE 连 accuracy 都不提升**

先不要做 CURE loss。需要检查 latent representation / fusion 是否本身无效。

### 交付物

- 最小 K=1 latent model；
- intervention hooks；
- 第一张 utilization diagnostic table。

### Go / No-Go

至少要证明：

- latent pathway 工程上确实参与 forward；
- intervention 能稳定测量变化；
- Latent-CE 不出现明显训练/推理错位。

---

## Phase 2 — CURE Diagnostic Protocol

### 目标

把“latent 是否被使用”从单次 ablation 变成一个稳定 protocol。

### 新增 intervention

- zero；
- random；
- global shuffle；
- matched shuffle。

### Matched Counterfactual v1

先用简单可控条件：

- history length bin；
- target popularity bin；
- baseline confidence bin。

如果 metadata 可靠，再增加：

- category matching。

### 实验范围

这一阶段仍然不要复现十个 baseline。

优先：

1. SFT；
2. Latent-CE；
3. 如果工程成本可接受，再选 **1 个最兼容的公开 latent-reasoning baseline**。

目的不是立即建立 benchmark，而是确认 protocol 能区分不同 utilization behavior。

### 期望发现

例如：

~~~text
Model A
Normal ≈ Zero ≈ Shuffle
→ latent bypass

Model B
Normal > Zero
Normal ≈ Matched Shuffle
→ latent 有信息，但 personalization 弱

Model C
Normal > Matched Shuffle > Zero
→ latent 同时具有必要性和个性化信息
~~~

### 交付物

- intervention evaluator；
- NG / SG / PG；
- matched-counterfactual sampler；
- utilization behavior taxonomy。

### Go / No-Go

如果所有模型 normal / zero / shuffle 都高度一致：

- 说明这个问题很强，继续 Phase 3。

如果所有模型天然都有很强 utilization：

- 不再强调 “latent bypass”；
- 转向 “personalized faithfulness / functional specificity”。

---

## Phase 3 — CURE Training Framework

### 目标

从“测量 utilization”进入真正的方法贡献：

> 能否通过训练主动让 latent 成为有用的 decision pathway？

### 模型

保持 Phase 1 的 K=1 architecture 不变。

只加入：

$$
\mathcal{L}_{rec}
+
\lambda_u \mathcal{L}_{util}
$$

然后再加入：

$$
+\lambda_a \mathcal{L}_{anchor}
$$

### 核心 ablation

| Variant | Rec | Util | Anchor |
|---|---:|---:|---:|
| SFT | ✓ | – | – |
| Latent-CE | ✓ | – | – |
| CURE-U | ✓ | ✓ | – |
| CURE-UA | ✓ | ✓ | ✓ |

### 成功标准

不能只看 PG / NG 增大。

真正理想的是：

~~~text
Recommendation accuracy      ↑ or at least not degraded
Necessity / Personalization  ↑
Counterfactual branch        remains baseline-like
~~~

### 失败模式

如果：

~~~text
PG ↑
NG ↑
HR ↓
~~~

说明模型只是被强迫依赖 latent，而 latent 信息质量不足。

此时优先改 latent representation，不要继续加更强 utilization penalty。

### 交付物

- CURE loss；
- training code；
- lambda ablation；
- mechanism + accuracy joint analysis。

### Go / No-Go

在 Beauty 上至少满足：

- accuracy 不明显低于 Latent-CE；
- PG / NG 中至少一个稳定改善；
- matched counterfactual 不通过“故意破坏输出”获得 margin。

否则暂停后续复杂化。

---

## Phase 4 — Selective Utilization

### 目标

回答：

> latent reasoning 应该在哪些样本上真正有用？

不是做 adaptive inference depth，而是研究 **selective usefulness**。

### Difficulty Weight

使用冻结 baseline：

$$
w_i = 1 - P_0(y_i \mid H_i)
$$

或：

- entropy；
- target rank；
- top-1 / top-2 margin。

然后：

$$
\mathcal{L}_{util}^{weighted}
=
w_i \cdot \mathcal{L}_{util}
$$

### 分桶分析

按 baseline difficulty 分为：

- easy；
- medium；
- hard。

分别报告：

- HR/NDCG gain；
- NG；
- PG；
- gate magnitude；
- distribution shift。

### 关键问题

我们不应该要求简单样本也产生人工 latent dependency。

真正想看到的是：

> hard samples 获得更明显的 utilization gain，而 easy samples 接近 baseline。

### 交付物

- difficulty-aware CURE；
- easy/medium/hard analysis；
- utilization-vs-difficulty figure。

---

## Phase 5 — Generalization, Baselines & Paper-Scale Validation

这一阶段才开始扩展工作量。

### 5.1 多数据集

至少：

- Amazon Beauty；
- Amazon Sports；
- Amazon Toys。

如果时间和算力允许，再考虑一个非 Amazon 数据集。

### 5.2 外部 Baselines

不要追求数量。

建议按“范式代表性”选择 2–3 个：

- 一个基础 generative recommender；
- 一个 latent reasoning recommender；
- 一个较新的 multi-step / adaptive latent reasoning 方法。

优先选择：

- 有公开代码；
- 数据接口能适配；
- latent state 可以被 intervention；
- 不需要极高训练成本。

### 5.3 两种论文规模

#### Method-first paper

如果 CURE training 明显提高：

- accuracy；
- NG / PG；
- hard-case performance；

那么主线是：

> **CURE is a new framework for learning causally utilized latent reasoning.**

Evaluation protocol 是 supporting contribution。

#### Evaluation-first paper

如果我们发现大量 latent-reasoning models 都存在显著 utilization gap，但 CURE 方法提升有限：

主线可以变成：

> **Current latent-reasoning recommenders may not use what they reason.**

这时需要更多 baselines，工作量更大，但 research contribution 更偏 benchmark / methodology。

### 推荐优先级

**优先争取 Method-first。**

因为这样不需要一开始就复现大量外部模型，也更适合当前时间和工程条件。

---

# 7. Multi-Intent 什么时候回来？

只有 Phase 3–4 成功后再重新考虑 K > 1。

此时不再使用简单的 cosine diversity：

$$
\cos(z_i,z_j) < m
$$

而研究 **functional diversity**。

例如：

$$
P(y \mid H,Z)
\neq
P(y \mid H,Z_{\setminus k})
$$

并检查 mask 不同 latent 后是否影响不同 candidate subsets。

理想现象：

~~~text
mask z1
→ 一组候选明显下降

mask z2
→ 另一组候选明显下降
~~~

只有这种 functional specialization 出现后，才重新使用 “multi-intent” 这个 claim。

---

# 8. 最小论文实验矩阵

第一篇 CURE 论文最小可以控制在：

| Group | Experiments |
|---|---|
| Baseline | SFT |
| Architecture | Latent-CE |
| Method | CURE-U / CURE-UA |
| Intervention | normal / zero / random / shuffle / matched |
| Difficulty | easy / medium / hard |
| Dataset | Beauty → Sports → Toys |
| External model | 先 1 个，最终视结果扩到 2–3 个 |

这样不会一开始把项目做成一个庞大的 benchmark 工程。

---

# 9. 当前最重要的工程原则

### 9.1 一次只改变一个变量

不要再同时修改：

- latent representation；
- fusion；
- loss；
- tokenizer；
- decoding；
- evaluation protocol。

### 9.2 每个新模块必须有 intervention test

任何 latent 模块合入主实验前必须回答：

- zero 它会怎样？
- shuffle 它会怎样？
- reload 后还会怎样？
- generation 时它真的执行了吗？

### 9.3 新实验必须记录 artifact identity

每个结果至少记录：

- git SHA；
- dataset；
- RQ-VAE / item index identity；
- tokenizer；
- model checkpoint；
- seed；
- decoding config。

### 9.4 机制提升不能代替推荐效果

CURE 的目标不是制造依赖。

最终必须同时关心：

> utility + utilization

而不是只把 NG / PG 做得很大。

---

# 10. 主要风险与预案

### 风险 A：naive latent 本来就有很强 utilization

那就把研究重点从 necessity 转向：

- personalization；
- matched counterfactual；
- functional specificity。

### 风险 B：CURE 让 utilization 提升但 accuracy 下降

说明 latent quality 不足。

优先改善 reasoner / fusion，而不是增大 $\lambda_u$。

### 风险 C：matched counterfactual 设计不严谨

做多个 matching protocol：

- popularity；
- history length；
- baseline confidence；
- category；
- nearest-history embedding。

### 风险 D：外部 baseline 工作量过大

论文早期只要求一个兼容 baseline。

只有 Method-first 主结果成立后才扩大 baseline 数量。

### 风险 E：“causal” 被 reviewer 质疑

正文中明确区分：

- intervention-based causal motivation；
- empirical utilization test；
- formal causal effect estimation。

除非后续建立完整 SCM 与 identification assumptions，否则不要声称估计了严格的 causal effect。

---

# 11. 近期里程碑

### Milestone A — Clean Baseline

完成 Phase 0。

成功标志：

> Beauty SFT pipeline fully reproducible.

### Milestone B — First Mechanism Result

完成 Phase 1–2。

成功标志：

> 能明确画出 Normal / Zero / Shuffle / Matched 的差异。

### Milestone C — CURE Works

完成 Phase 3。

成功标志：

> CURE 相比 Latent-CE 同时改善 utilization，并保持或提高 recommendation accuracy。

### Milestone D — Selective Story

完成 Phase 4。

成功标志：

> hard samples 获得更明显 latent benefit。

### Milestone E — Paper-Scale Evidence

完成 Phase 5。

成功标志：

> 至少 3 个数据集 + 代表性外部 baseline + 完整 ablation。

---

# 12. 投稿时最理想的故事

论文开头可以压成三个问题：

### Q1. Does latent reasoning really influence recommendation?

用 intervention protocol 回答。

### Q2. Is the influence personalized?

用 matched counterfactual 回答。

### Q3. Can we explicitly learn useful latent dependence?

用 CURE framework 回答。

于是论文不再只是：

> 我们提出几个新指标。

也不只是：

> 我们加一个 latent module。

而是：

> **We identify a latent-utilization problem, introduce an interventional protocol to measure it, and propose CURE to explicitly learn personalized, useful latent dependence.**

---

# 13. Working Contributions

如果 Phase 3–5 顺利，最终 contribution 可以写成：

1. **Problem**  
   提出 latent utilization 这一问题：latent state 可被学习，但未必真正参与推荐决策。

2. **Framework**  
   提出 **CURE-Rec**，通过 personalized counterfactual utilization regularization 和 baseline anchoring，使 latent reasoning 成为真正有用的 decision pathway。

3. **Evaluation**  
   提出 intervention-based protocol，区分 latent necessity、generic sensitivity 与 personalized utilization。

4. **Analysis**  
   研究 utilization 与 sample difficulty、recommendation accuracy、functional specialization 之间的关系。

---

# 14. 当前暂不做的事情

为了防止项目再次失控，Phase 0–3 明确不做：

- explicit CoT generation；
- teacher CoT distillation；
- GRPO；
- K=4 multi-intent；
- complex dynamic routing；
- posterior-prior；
- semantic hierarchy；
- 大规模 baseline zoo。

这些都只能在核心机制建立以后作为扩展，而不能成为新的起点。

---

# 15. 最终研究问题

> **Can generative recommenders learn latent reasoning states that are not only predictive, but actually utilized and personally relevant to the final recommendation decision?**

中文：

> **生成式推荐模型能否学习不仅与目标相关，而且真正被最终推荐决策利用、并具有个性化信息价值的 latent reasoning？**

---

# 16. 一句话项目原则

> **先证明模型真的在用 reasoning，再讨论 reasoning 应该有多少、分成几种、或者有多复杂。**

---

# 17. 相关工作方向

正式写论文前需要再次核对最终发表版本和最新工作。

- ReaRec — inference-time / latent reasoning for recommendation
- LARES — recurrent latent reasoning
- LatentR³ — latent reasoning + reinforcement learning
- SCoTER — structured reasoning transfer
- S²GR — semantic supervision for generative reasoning
- LaRec — latent reasoning + multi-path reasoning
- RecRec — latent interests + recursive reasoning
- IBA / Where Reasoning Matters — reasoning budget allocation
- HiLaR — hierarchical latent reasoning / marginal contribution
- ReSID / UniGRec / PIT / AsymRec — Semantic ID / tokenizer research
- GenPAS — generative recommendation training sample construction
- General latent-reasoning faithfulness work — intervention / activation patching

---

## Appendix A — Markdown 数学公式约定

为了同时兼容 GitHub Markdown 和现代 VSCode Markdown Preview，本项目统一使用：

行内公式：

$P(y \mid H,Z)$

块级公式：

$$
\mathcal{L}
=
\mathcal{L}_{rec}
+
\lambda_u\mathcal{L}_{util}
+
\lambda_a\mathcal{L}_{anchor}
$$

不再使用旧文档中的 LaTeX display delimiters：

~~~text
\[
...
\]
~~~

如果 VSCode 仍不显示数学公式，请确认使用的是 Markdown Preview，而不是纯文本编辑视图，并检查 VSCode 的 Markdown math rendering 设置或相关扩展是否禁用了数学渲染。
