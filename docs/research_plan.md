# 生成式推荐中的 Causally Faithful Latent Reasoning：研究空位与论文方案

> 更新时间：2026-09-28  
> 项目背景：MIR / Generative Recommendation / Semantic ID / Latent Reasoning  
> 目标：从现有 MIR 与早期 implicit-CoT 探索中提炼出一个更干净、可验证、可投稿的研究问题。

---

## 1. 从原始动机重新定义问题

最初的研究动机其实一直很清楚：

> **显式自然语言 Chain-of-Thought（CoT）虽然可能提升推荐模型的推理能力与可解释性，但会引入额外的 token、存储、延迟与在线推理成本。能否把昂贵的显式推理压缩成一种低成本、内部完成的 latent reasoning？**

最早的尝试是：

```text
Large Recommendation LLM
        ↓
Structured CoT
(User Profile / Product Features / Matching / ...)
        ↓
Small Transformer
        ↓
Thought Tokens
        ↓
Generative Recommender
```

并通过 MSE 等方式，让小模型产生的 thought tokens 对齐 teacher LLM 的结构化 CoT 表示。

这个思路后来暴露出几个根本问题：

1. **MSE 对齐太粗糙**：一个 latent token 与 teacher embedding 接近，并不意味着它真的包含对推荐有用的信息。
2. **规范化 CoT 本身未必是真正的“链式推理”**：例如“用户画像”和“商品特征”往往是并列因素，而不是严格的 step-by-step reasoning。
3. **解释任务和预测任务不一致**：teacher 往往看到 `history + target` 后解释“为什么用户会购买该商品”，这更接近 post-hoc explanation，而真实推荐任务是：

```text
history
   ↓
uncertain future preference
   ↓
target
```

4. **有 latent representation 不等于模型真的使用了 latent representation**：模型可能一边把 latent 学得很好，一边仍然主要依赖直接的 `history → target` 路径。

后续 MIR 的多意图设计，本质上仍然围绕同一个问题：

```text
History
  ↓
K Latent Intents
  ↓
Dynamic Routing
  ↓
SID Generation
```

但 MIR 又引入了新的困难：

- 一个 next-item target 很难监督 K 个真实不同的 intent；
- `diversity loss` 只能保证向量不同，不能保证功能不同；
- SID 的不同 residual quantization 层未必对应不同用户意图；
- latent module 可能训练存在，但推理阶段拔掉后 backbone 仍然能工作；
- 如果 zero/shuffle latent 不影响结果，就不能证明模型真的依赖 latent。

因此，现在真正值得保留的，不是 MIR 当前的具体结构，而是最初的问题意识：

> **如何让生成式推荐获得额外的内部推理能力，同时避免显式 CoT 的高成本，并且能够证明这些 latent reasoning state 真正参与了最终决策？**

---

# 2. 2025–2026 生成式推荐研究热点

近期研究已经明显从“是否需要 latent reasoning”进入到更细粒度的问题：

```text
2024:
加入 reasoning 是否有帮助？

2025:
latent reasoning 能否替代 explicit CoT？

2026:
reasoning state 应该是什么？
应该监督什么？
应该 reasoning 几步？
哪些 token / 哪些样本值得多想？
这些 latent state 到底有没有被真正使用？
```

当前主要研究方向包括：

| 方向 | 主要问题 | 代表性趋势 |
|---|---|---|
| Latent Reasoning / Inference-time Compute | 不输出自然语言，直接在 latent space 中增加计算 | ReaRec、LARES、LatentR³、LaRec、RecRec |
| Adaptive Reasoning Depth | 不同样本动态决定 reasoning 深度 | LASAR、ManCAR |
| Position-wise Reasoning Budget | 不同 SID 位置使用不同计算预算 | IBA / Where Reasoning Matters |
| Multi-path / Multi-interest Reasoning | 一个用户对应多个 latent path / interests | LaRec、RecRec、PLR |
| Latent Semantic Supervision | 给中间 reasoning state 添加语义或过程监督 | S²GR、LaRec、IntuRec |
| Credit Assignment | 判断哪一步 reasoning 真正提高 target likelihood | HiLaR、SAPO、Retrieval-Grounded Credit Assignment |
| Semantic ID Tokenizer | 重新设计推荐专用 SID / tokenizer | ReSID、UniGRec、PIT、AsymRec |
| Efficient Generative Serving | 降低 SID 解码和 Transformer 推理成本 | RPG、SID-MLP、GenRec |
| Training/Evaluation Protocol | 样本构造、SID collision、历史窗口等对结果影响巨大 | GenPAS、collision-aware evaluation |

---

# 3. 哪些方向现在已经不适合作为核心创新

## 3.1 固定 K 个 latent intent + diversity loss

形式：

```text
History
  ↓
K Latent Intents
  ↓
Routing
  ↓
Recommendation
```

问题：

- Multi-interest / multi-path latent reasoning 已经成为明确方向；
- K 个不同向量并不等于 K 个不同 intent；
- pairwise cosine diversity 只保证 representation 不同，不保证 function 不同；
- 一个 next-item target 对多个 intent 的 identifiability 很弱。

因此 MIR 原版不适合作为论文核心。

## 3.2 显式 CoT → latent token 蒸馏

如果只是 MSE / cosine / InfoNCE 对齐，目前已经不够新。近期工作已经推进到 step-level alignment、trajectory alignment、process supervision、contrastive alignment、latent RL 和 multi-path latent reasoning。

因此，“换一个 loss 蒸馏 CoT”已经很难形成有力 contribution。

## 3.3 动态决定 reasoning depth

这一方向已经出现 per-sample adaptive stopping、policy head 控制 reasoning depth、adjustable inference-time reasoning depth。因此单纯讲“简单用户少想，困难用户多想”已经不够。

## 3.4 按 SID token 分配 reasoning compute

已有工作直接研究不同 SID position 的 information gain，并把更多 compute 分给高信息量位置。

## 3.5 每一步 reasoning 带来多少 target likelihood gain

近期工作已经开始做 reasoning-step credit assignment：

\[
\Delta_k =
\log P(y|H,z_{1:k})
-
\log P(y|H,z_{1:k-1})
\]

因此，只报告某一步 latent 是否提高 target probability，也不再足够新。

---

# 4. 当前真正值得做的研究空位

## Causally Faithful Latent Reasoning for Generative Recommendation

核心问题：

> **Latent reasoning state 是否真正因果地参与了推荐决策？**

进一步：

> **能否训练出“被模型真正利用”的 latent reasoning，而不仅仅是语义上对齐、表征上漂亮、或与 target 相关？**

最核心的 research question：

> **Does the recommender actually use what it reasons?**

---

# 5. 为什么这是一个独立于现有工作的研究问题

考虑：

```text
History H
   ↓
Reasoner
   ↓
Latent Z
   ↓
Generator
   ↓
Target y
```

即使实验观察到：

```text
with latent: HR@10 = 4.2
baseline:    HR@10 = 3.8
```

也不能证明 `Z → prediction`。

真实训练后的模型完全可能是：

```text
          ┌────────────────┐
H ───────→│    Backbone     │────→ y
│         └────────────────┘
│
└→ Reasoner → Z ── weak / unused
```

同时满足：

- Z 与 teacher CoT 很相似；
- Z 能预测 target；
- Z 的 auxiliary loss 很低；
- 整体模型比 baseline 强；
- 但 decoder 并没有真正依赖 Z。

这就是：

> **representation learning ≠ causal utilization**

---

# 6. 与现有 credit assignment 工作的区别

现有工作通常研究：

```text
加入 z_k 后
target probability 是否提高？
```

例如：

\[
\Delta_k =
\log P(y|H,z_{1:k})
-
\log P(y|H,z_{1:k-1})
\]

这是 incremental predictive gain。

但这里提出的问题是 intervention：

\[
P(y|H,Z)
\]

与：

\[
P(y|H,\operatorname{do}(Z=\tilde Z))
\]

之间是否产生稳定、符合个性化逻辑的变化。

换句话说：

> 保持用户历史 H 不变，只替换 latent reasoning state Z，推荐结果是否系统性改变？

这是 **causal utilization / interventional faithfulness**，不是普通 likelihood contribution。

---

# 7. 核心实验：Latent Intervention Battery

对于每个样本 \(i\)，正常状态：

\[
P(y_i|H_i,Z_i)
\]

然后做一系列 intervention。

## 7.1 Zero Latent

```text
Z_i → 0
```

测试 latent 被完全移除后，性能是否下降。

如果：

```text
normal ≈ zero
```

说明模型很可能绕过了 latent pathway。

## 7.2 Random Latent

```text
Z_i → ε
```

测试模型是否对 latent input 敏感。

## 7.3 Global Shuffle

```text
Z_i → Z_j
```

其中 \(j \neq i\)。

如果性能几乎不变，latent 可能只是一个通用 bias，而不是个性化 reasoning。

## 7.4 Matched Counterfactual Shuffle

普通 shuffle 容易被质疑为 OOD noise，因此选择另一个满足相似条件的用户：

```text
same target category
similar item popularity
similar history length
similar baseline difficulty
different user / different history
```

构造：

```text
H_i + Z_j
```

其中 \(Z_j\) 看起来是一个合理的 latent representation，只是不是当前用户自己的。

如果：

```text
P(y_i | H_i, Z_i)
≈
P(y_i | H_i, Z_j)
```

说明 latent representation 并没有承载真正用户专属的推理信息。

## 7.5 Latent Mask / Component Intervention

如果未来扩展到：

\[
Z=\{z_1,\ldots,z_K\}
\]

可以逐个 mask：

```text
Z \ z_k
```

观察每个 latent component 对推荐候选集合的影响。

---

# 8. 新的评价指标

传统 HR@K、NDCG@K 保留，同时增加机制指标。

## 8.1 Necessity Gap

\[
NG@K =
HR@K_{\text{normal}}
-
HR@K_{\text{zero}}
\]

意义：latent 是否是模型完成推荐所需要的信息。

## 8.2 Personalization Gap

\[
PG@K =
HR@K_{\text{normal}}
-
HR@K_{\text{matched-shuffle}}
\]

意义：latent 是否包含当前用户特有的信息。

## 8.3 Shuffle Gap

\[
SG@K =
HR@K_{\text{normal}}
-
HR@K_{\text{shuffle}}
\]

## 8.4 Distribution Sensitivity

不仅比较 target rank，还比较完整输出分布：

\[
D_{KL}
\left(
P(\cdot|H,Z)
\|
P(\cdot|H,\tilde Z)
\right)
\]

## 8.5 Candidate Shift

对 Top-K recommendation set 做：

- Jaccard similarity；
- rank correlation；
- category shift；
- semantic displacement。

目标不是简单证明“变了”，而是判断替换 latent 是否产生稳定、个性化、语义合理的 candidate shift。

---

# 9. Proposed Method：Causal Utilization Regularization

第一版模型应该刻意保持简单：

```text
            ┌──────────────→ Generator
            │
History ────┤
            │
            ↓
      Small Reasoner
            ↓
            Z
            │
            └──────────────→ Gated Fusion
```

初始阶段建议：

```text
K = 1
```

原因：如果单个 latent vector 都无法被模型可靠利用，那么直接研究 K=4 multi-intent 没有意义。

---

# 10. 基础推荐目标

\[
\mathcal L_{rec}
=
-\log P(y|H,Z)
\]

---

# 11. Utilization Loss

希望：

```text
correct latent
>
matched counterfactual latent
```

定义：

\[
\mathcal L_{util}
=
\max
\left(
0,
m
-
[
\log P(y|H,Z)
-
\log P(y|H,\tilde Z)
]
\right)
\]

其中：

\[
\tilde Z = Z_{\text{matched user}}
\]

含义：当前用户自己的 latent 必须比“另一个看起来同样合理的用户 latent”更有助于预测当前 target。

---

# 12. 为什么单独使用 Utilization Loss 会有漏洞

模型可能作弊：不是把 correct latent 做得更好，而是故意把 wrong latent 做得极差。

例如：

```text
correct Z → normal
wrong Z   → intentionally corrupted prediction
```

于是 utilization margin 很大，但这并不代表 latent 真正提供了有用信息。

---

# 13. Baseline Anchor

先训练 / 冻结一个普通 SFT baseline：

\[
P_0(y|H)
\]

对于 counterfactual latent \(\tilde Z\)，要求模型尽量回到 baseline，而不是崩坏：

\[
\mathcal L_{anchor}
=
D_{KL}
\left(
P(\cdot|H,\tilde Z)
\|
P_0(\cdot|H)
\right)
\]

训练目标变成：

```text
No useful latent
      ↓
baseline-like prediction

Correct personalized latent
      ↓
better prediction
```

而不是：

```text
correct latent → normal
wrong latent   → destroyed
```

---

# 14. Difficulty-aware Utilization

并非所有样本都需要 latent reasoning。

例如：

```text
history:
牙膏 → 牙膏 → 牙膏 → 牙膏

target:
牙膏
```

baseline 已经极度确定。如果强迫 latent 对这种样本必须产生巨大贡献，模型会制造 artificial dependency。

因此可用 baseline difficulty：

\[
w_i = 1-P_0(y_i|H_i)
\]

或使用：

- baseline entropy；
- target rank；
- margin；
- confidence。

得到：

\[
\mathcal L_{util}
=
w_i
\cdot
\max
\left(
0,
m-
[
\log P(y|H,Z)
-
\log P(y|H,\tilde Z)
]
\right)
\]

这样：

```text
easy sample:
latent 可以没有明显贡献

hard sample:
latent 应该提供额外 personalized information
```

注意：这不应包装成 “adaptive compute”。真正的 contribution 是：

> **selective causal usefulness**

---

# 15. 最终训练目标

第一版控制在三个 loss：

\[
\boxed{
\mathcal L
=
\mathcal L_{rec}
+
\lambda_u\mathcal L_{util}
+
\lambda_a\mathcal L_{anchor}
}
\]

第一版不要加入：

- diversity loss；
- CoT MSE；
- GRPO；
- posterior/prior；
- K=4；
- complex routing；
- multi-stage semantic hierarchy。

先证明：

> **一个 latent representation 能否稳定地成为模型决策真正使用的信息通路。**

---

# 16. 最小实验矩阵

| Model | Rec Loss | Latent | Utilization Loss | Anchor | Zero | Shuffle | Matched Shuffle |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT | ✓ | – | – | – | – | – | – |
| Latent | ✓ | ✓ | – | – | ✓ | ✓ | ✓ |
| + Util | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ |
| + Util + Anchor | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

理想现象不是单纯：

```text
HR@10:
2.84 → 3.20
```

而是：

```text
SFT                     2.84

Latent                  2.90
Latent-zero             2.89
→ latent bypass

Util                    3.10
Util-zero               2.84
Util-shuffle            2.85
Util-matched-shuffle    2.86
→ latent causally utilized
```

这样的结果在机制上比单纯 accuracy gain 更有说服力。

---

# 17. 如果 K=1 成功，再扩展 Multi-Intent

之后才考虑：

\[
Z=\{z_1,\ldots,z_K\}
\]

此时不再使用简单的：

\[
\cos(z_i,z_j)<m
\]

来定义 diversity。

因为：

> representation 不同 ≠ function 不同。

真正值得研究的是 **Functional Diversity**。

例如逐个移除：

\[
P(y|H,Z)
\neq
P(y|H,Z_{\setminus k})
\]

以及不同 \(z_k\) 的 intervention 是否影响不同 recommendation candidate subsets。

如果：

```text
mask z1
→ category A candidates disappear

mask z2
→ category B candidates disappear
```

才能更有力地声称不同 latent components 对应不同 functional intent。

---

# 18. 可以保留的 MIR 部分

MIR 不必完全废弃。

可以保留：

```text
History
  ↓
Latent / Intent Encoder
  ↓
Z
  ↓
Gated Fusion
  ↓
Generator
```

但暂时放弃以下 headline：

- multi-intent；
- dynamic SID routing；
- K=4；
- diversity loss。

MIR 可以退化成一个最简单的 latent-reasoning testbed。

---

# 19. 当前 MIR / 实验系统必须先修的问题

## 19.1 MIR inference path

过去实验中存在：

- MIR module 训练后，没有被当前 evaluator 正确加载；
- evaluator 只加载普通 `AutoModelForCausalLM`；
- zero / shuffle / gate-zero 结果完全相同；
- 当前 repo 中缺少可审计的 MIR implementation。

因此新版本需要：

```text
MIR / Latent model
作为完整 nn.Module 保存
      ↓
完整 reload
      ↓
generate()
必须实际调用 latent pathway
```

并加入 unit test：

```text
normal Z logits
≠
zero Z logits

normal Z logits
≠
shuffle Z logits
```

## 19.2 Test History Bug

当前数据逻辑：

```text
train = all except last two
valid = second last
test  = last
```

但 test inference history 使用的仍然只是 `train history`，因此漏掉 validation item。

标准 sequential evaluation 应改成：

```text
Validation:
history = train
target  = valid

Test:
history = train + valid
target  = test
```

这个 bug 特别容易伤害依赖“近期意图”的模型。

## 19.3 Baseline Version Drift

过去项目中已经出现过：

- SFT v1 / SFT v2 混用；
- RQ-VAE codebook 改变；
- checkpoint 起点不一致；
- CoT 实验曾错误加载 base Qwen 而非 SFT checkpoint。

因此新的实验必须固定：

- dataset；
- user split；
- item index；
- RQ-VAE；
- SID tokenizer；
- base checkpoint；
- decoding strategy；
- evaluation users；
- random seed；
- beam size；
- git SHA；
- artifact hash。

## 19.4 SID Collision

当前 RQ-VAE SID 存在 collision。

如果 item A / item B 拥有相同 SID，evaluation 仅按 SID 判断 hit，那么模型不能区分真正 item identity。

这会产生：

```text
SID-level accuracy
≠
item-level accuracy
```

长期建议：

- 加 unique suffix token；
- 或使用 atomic collision-resolving token；
- 或至少同时报告 item-level collision-aware metric。

---

# 20. 为什么 Future Posterior → History Prior 不适合作为唯一 headline

曾考虑：

\[
q(Z|H,\text{Future})
\rightarrow
p(Z|H)
\]

训练时 posterior 看未来行为：

```text
t+1
t+2
t+3
```

然后蒸馏到 history-only prior。

这个想法仍然有价值，尤其适合解决一个 next-item 无法监督 K 个 intent 的问题。

但类似 future-aware posterior、history prior、future behavior teacher distillation 已经存在相关工作。因此更适合作为：

> 后续组件 / multi-intent extension

而不是核心 novelty。

---

# 21. 论文最适合的主线

## Proposed Title 1

**Does Latent Reasoning Really Matter? Causally Faithful Reasoning for Generative Recommendation**

## Proposed Title 2

**Learning Causally Utilized Latent Reasoning for Generative Recommendation**

## Proposed Title 3

**When Latent Reasoning Matters: Interventional Learning for Generative Recommendation**

---

# 22. Introduction 的核心逻辑

1. Generative recommendation 越来越多地引入 latent reasoning；
2. 现有工作关注 reasoning depth、semantic alignment、multi-path reasoning、adaptive computation、fine-grained reward；
3. 但一个被忽略的问题是：

> **well-aligned latent representation 并不代表 latent state causally drives prediction。**

4. 模型可能同时学到：

```text
History → Item
```

和：

```text
History → Latent
```

但两条路径之间只有弱联系。

5. 因此提出：

> **Does the recommender actually use what it reasons?**

6. 用 interventional evaluation 测量 latent necessity / personalization；
7. 用 utilization regularization + baseline anchor 训练真正被模型读取的 latent states。

---

# 23. 建议的三条论文贡献

### Contribution 1 — Problem

首次系统研究 generative recommendation 中的：

> **latent reasoning 的 causal utilization**

区别于传统的 accuracy、representation alignment、auxiliary target prediction 和 reasoning-step likelihood gain。

### Contribution 2 — Evaluation

提出 intervention-based evaluation protocol：

- zero latent；
- random latent；
- shuffle latent；
- matched-counterfactual latent；
- component mask。

并定义：

- Necessity Gap；
- Personalization Gap；
- Distribution Sensitivity；
- Candidate Shift。

### Contribution 3 — Method

提出 lightweight **Causal Utilization Regularization**，使：

```text
correct personalized latent
→ improvement over baseline

wrong / counterfactual latent
→ baseline-like behavior
```

避免：

```text
wrong latent
→ intentionally broken model
```

---

# 24. Potential Reviewer Objections

## Q1. 这不就是 robustness test？

不是。普通 robustness 关心 noise 后模型是否仍然稳定；这里关心 latent pathway 是否真的携带决策必要的信息。
如果 latent 在 zero / matched-shuffle 后完全不影响结果，说明该 pathway 很可能被 bypass。

## Q2. 这不就是 ablation？

普通 ablation：

```text
remove module
retrain
```

研究的是 architecture 是否有帮助。

这里：

```text
same trained model
same H
intervene Z
```

研究的是给定同一个模型，其 prediction 是否依赖当前 latent state。

## Q3. 这不就是 HiLaR 的 marginal gain？

不是。HiLaR 风格是：

\[
P(y|H,z_{1:k})
-
P(y|H,z_{1:k-1})
\]

这里强调：

\[
P(y|H,Z)
\quad vs. \quad
P(y|H,\operatorname{do}(Z=\tilde Z))
\]

尤其是 matched personalized counterfactual intervention。

## Q4. 为什么 wrong latent 一定应该差？

不要求所有 wrong latent 都大幅变差。

利用 difficulty weighting、baseline anchor、matched counterfactual，只要求：

> 对真正需要额外 personalized information 的样本，正确 latent 应提供比不匹配 latent 更稳定的增益。

---

# 25. 风险

## 风险 1：所有 latent 模型本来就能通过 intervention

如果大多数 baseline：

```text
Normal >> Zero / Shuffle
```

那么“latent bypass”问题可能不普遍。

解决：重点转向 personalization faithfulness / matched counterfactual sensitivity。

## 风险 2：Utilization Loss 提高 dependency 但降低 accuracy

可能出现：

```text
NG ↑
PG ↑
HR ↓
```

说明模型学会依赖 latent，但 latent 本身质量不足。

这会产生一个新的研究问题：

> utilization 与 information quality 之间的 trade-off。

## 风险 3：matched counterfactual 定义不够严谨

需要设计多个 matching protocol：

- same category；
- same popularity bin；
- same history length；
- same baseline confidence；
- embedding-nearest history。

并做 robustness analysis。

## 风险 4：只有一个数据集有效

至少应该准备多个公开序列推荐数据集。

理想情况：

- Amazon Beauty；
- Amazon Sports / Toys；
- MovieLens / Yelp / Steam 中适合生成式推荐的至少一个。

---

# 26. 推荐实验顺序

## Phase 0 — 修复实验平台

先完成：

- test history 修复；
- MIR/latent inference path；
- checkpoint save/load；
- zero/shuffle unit test；
- baseline artifact freezing。

## Phase 1 — K=1 Latent Diagnostic

只训练：

```text
SFT
Latent
```

先回答：

> 普通 latent module 是否真的被使用？

## Phase 2 — Intervention Benchmark

增加：

- zero；
- random；
- shuffle；
- matched shuffle。

分析：

- HR；
- NDCG；
- Necessity Gap；
- Personalization Gap；
- candidate-set shift。

## Phase 3 — Utilization Training

加入 \(L_{util}\)，然后加入 \(L_{anchor}\)。

验证：

```text
accuracy ↑
causal utilization ↑
```

是否能同时成立。

## Phase 4 — Difficulty-aware Utilization

按 baseline uncertainty 分桶：

```text
easy
medium
hard
```

看 latent utilization 是否主要集中在 hard cases。

## Phase 5 — Multi-intent Extension

只有 Phase 1–4 成功后再扩展：

```text
K = 2 / 4
```

研究 functional diversity，而不是 representation diversity。

---

# 27. 当前最重要的研究原则

这次项目应该坚持一个原则：

> **一次只改变一个核心问题。**

过去的问题是同时改变：

```text
Reasoning representation
Reasoning supervision
Reasoning injection
SID quality
training procedure
evaluation protocol
```

六个变量同时变化以后，即使结果提升或下降，也无法知道原因。

新的研究路线应该依次回答：

```text
1. latent 是否真的被使用？
2. 能否主动提高 utilization？
3. utilization 是否提高 accuracy？
4. 哪些样本真正需要 latent？
5. 多个 latent 是否具有不同 functional role？
```

---

# 28. 对 MIR 的最终定位

当前最合理的处理方式不是继续“修 MIR”，也不是完全推倒重来。

而是：

> **把 MIR 从一个 multi-intent framework，降级为研究 latent utilization 的实验平台。**

保留：

- latent encoder；
- gated fusion；
- generative recommendation backbone。

暂时移除：

- dynamic SID routing；
- K=4；
- diversity loss；
- multi-intent claim。

如果 K=1 的 causal utilization 都无法建立，那么原 MIR 中更复杂的 multi-intent 解释没有坚实基础。

---

# 29. 最终研究问题

可以把整个项目最终浓缩成一句：

> **Can generative recommenders learn latent reasoning states that are not only predictive, but causally necessary and personally relevant to the final recommendation decision?**

中文：

> **生成式推荐模型能否学习不仅“与目标相关”，而且真正对最终推荐决策具有因果贡献和个性化作用的 latent reasoning？**

---

# 30. 一句话总结

不要再追问：

> “怎么设计一个更复杂的 latent reasoning 模块？”

而应该问：

> **“模型究竟有没有在使用它声称学会的 reasoning？”**

如果这个问题能够被系统地测量、证明并优化，它本身就有机会形成一篇比继续堆 MIR 模块更干净、更容易解释、也更有研究价值的论文。

---

# 参考方向与近期相关工作

> 正式写论文前应再次核查最新版本、最终出版信息与具体技术细节。

- **ReaRec** — latent / inference-time reasoning for recommendation  
  https://arxiv.org/abs/2503.22675

- **LARES** — recurrent latent reasoning  
  https://arxiv.org/abs/2505.16865

- **LatentR³** — latent reasoning + reinforcement learning  
  https://arxiv.org/abs/2505.19092

- **SCoTER** — structured reasoning transfer / structure preservation  
  https://arxiv.org/abs/2511.19514

- **S²GR** — semantic supervision for generative reasoning  
  https://arxiv.org/abs/2601.18664

- **LaRec** — latent reasoning, teacher alignment, multi-path reasoning  
  https://arxiv.org/abs/2607.24617

- **RecRec** — latent interests + recursive reasoning  
  https://arxiv.org/abs/2607.12945

- **Where Reasoning Matters / IBA** — SID-position reasoning budget  
  https://arxiv.org/abs/2607.12425

- **HiLaR** — hierarchical latent reasoning / marginal contribution  
  https://arxiv.org/abs/2607.27760

- **Semantic ID scaling analysis**  
  https://arxiv.org/abs/2509.25522

- **UniGRec** — joint / differentiable semantic identifier learning  
  https://arxiv.org/abs/2601.17438

- **ReSID** — recommendation-native semantic ID  
  https://arxiv.org/abs/2602.02338

- **AsymRec** — asymmetric continuous input / discrete output formulation  
  https://arxiv.org/abs/2605.14512

- **SID-MLP** — efficient SID generation  
  https://arxiv.org/abs/2605.12617

- **GenRec** — efficient industrial generative recommendation  
  https://arxiv.org/abs/2604.14878

- **RPG** — efficient / parallel semantic-ID generation  
  https://arxiv.org/abs/2506.05781

- **Recent latent reasoning faithfulness work** — causal intervention / activation patching for hidden reasoning states  
  https://arxiv.org/abs/2607.06648

---

## 下一步最值得做的事情

1. 修复当前 MIR repo 的 evaluation 闭环；
2. 构建一个最小 K=1 latent model；
3. 实现 zero / shuffle / matched-shuffle intervention；
4. 先跑出一张“latent 到底有没有被使用”的诊断图；
5. 再决定是否值得实现 Causal Utilization Regularization；
6. 如果机制成立，再写 Method / Experiments / Ablation；
7. 最后才扩展 multi-intent。

---

> **项目新的核心关键词：**
>
> `Generative Recommendation`  
> `Latent Reasoning`  
> `Causal Utilization`  
> `Interventional Evaluation`  
> `Faithfulness`  
> `Matched Counterfactual Latent`  
> `Selective Utilization`  
> `Functional Diversity`