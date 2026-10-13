# CURE-Rec Research Plan

> 更新时间：2026-10-13  
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

### 当前项目状态（2026-10-13）

当前仍处于 **Phase 0 — Baseline Sanitation & Reproducibility**，但基础数据与 SID artifact 已经从“准备阶段”推进到可冻结状态。

已完成：

- CURE-Rec 仓库从 MIR 迁移出干净的 generative recommendation baseline；
- leave-two-out test history 已修正为 `train + valid → test`；
- Amazon **Beauty** 与 **Sports** 两个 domain 的 **2018–2023** 数据处理已经完成；
- Beauty / Sports processed data 已完成 audit、sample-ID freeze 与 dataset manifest；
- Beauty / Sports text embedding、RQ-VAE 与 3-code Semantic ID artifact 已完成；
- CARE 已以独立 external baseline 形式适配到当前冻结协议，代码位于 `baselines/care/`。

当前重点：

- 完成并冻结 Beauty Qwen SFT baseline；
- 用冻结 recipe 在 Sports 上做 confirmation baseline；
- 完成统一 baseline evaluation；
- 对 CARE 先做 Beauty numerical sanity，再决定何时加入 intervention audit；
- 在 Phase 0 闭环后进入 Phase 1 的最小 K=1 latent pathway。

实验推进采用两层结构：

- **Beauty = development domain**：优先用于调通 pipeline、实现 CURE 和做快速 ablation；
- **Sports = confirmation domain**：在关键设计冻结后复现实验，避免所有结论只依赖单一 domain。

第三个数据集不作为 Phase 0–4 的硬性要求。只有当 CURE 主方法成立并进入 paper-scale validation 后，再决定是否加入 Toys 或其他非 Amazon 数据集。

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

建立一个可信、冻结、后续所有 CURE 实验都基于同一协议的 TIGER/Qwen baseline。

Phase 0 不再视为一个单一任务，而拆成五个连续子阶段。当前 **0A–0C 已完成，0D–0E 待闭环**。

---

### Phase 0A — Data Processing ✅ 已完成

本地已经完成：

- Amazon Beauty 2018–2023；
- Amazon Sports 2018–2023。

processed artifacts 位于本地 `data/processed/`，大体包括：

- user sequence；
- validation target；
- test target；
- item metadata；
- raw ID ↔ remapped ID 映射。

当前采用 leave-two-out sequential protocol：

~~~text
train = 除最后两次交互外的历史
valid = 倒数第二次交互
test  = 最后一次交互
~~~

对应评测必须保持：

~~~text
Validation:
history = train
target  = valid

Test:
history = train + valid
target  = test
~~~

数据本身不要求提交 Git；但它们的生成配置、统计信息和 artifact identity 必须可追踪。

---

### Phase 0B — Processed Data Audit & Freeze ✅ 已完成

在继续训练 RQ-VAE 之前，先证明输入数据本身可信。

#### 必查统计

Beauty 与 Sports 分别记录：

- user 数；
- item 数；
- interaction 数；
- valid / test 样本数；
- sequence length mean / median / P90 / P95 / max；
- item frequency distribution；
- user frequency distribution；
- metadata 缺失率；
- 过滤后仍无有效 metadata 的 item 数；
- train / valid / test target 是否全部存在于 item universe。

#### 必查一致性

至少验证：

1. 每个用户的交互严格按 timestamp 排序；
2. valid 时间不早于 train 最后一次交互；
3. test 时间不早于 valid；
4. test target 没有被错误加入训练 target；
5. remapped user/item ID 连续且映射稳定；
6. `user_sequences` 只包含 train 部分；
7. evaluator 在 test 时会把 valid item 补回 history；
8. 同一 processed dataset 重跑统计结果完全一致。

#### Artifact manifest

为每个 domain 生成一个轻量 manifest，建议记录：

~~~text
domain
date_range
k_core
n_users
n_items
n_interactions
n_valid
n_test
source/config version
git_sha
sha256(user_sequences)
sha256(valid)
sha256(test)
sha256(item_meta)
sha256(id mappings)
~~~

manifest 可以提交 Git，但原始 / processed 大文件继续留在本地。

#### 交付物

- Beauty data audit summary；
- Sports data audit summary；
- 两个 dataset manifest；
- 固定的 validation / test sample IDs。

#### Go / No-Go

只有当两个 domain 都通过时间顺序、映射、样本数量和 target coverage 检查后，才进入 0C。

---

### Phase 0C — Text Embedding & Semantic ID Artifacts ✅ 已完成

对每个 domain 独立生成：

~~~text
item_meta
   ↓
Text Encoder
   ↓
item_embeddings
   ↓
RQ-VAE
   ↓
item.index.json
   ↓
SID tokenizer
~~~

先在 Beauty 上跑通，再用冻结配置复制到 Sports。

必须记录：

- embedding model；
- text fields；
- embedding dimension；
- RQ-VAE architecture；
- codebook size / number of layers；
- training seed；
- codebook usage；
- SID collision rate；
- max collision group size；
- checkpoint hash；
- `item.index.json` hash。

RQ-VAE 的目标不是追求某个单独 reconstruction loss，而是得到稳定、覆盖良好、collision 可接受的推荐输出空间。

#### 交付物

每个 domain：

- item embeddings；
- frozen RQ-VAE checkpoint；
- `item.index.json`；
- SID tokenizer；
- RQ-VAE diagnostics。

#### Go / No-Go

Beauty 的 SID artifact 先冻结；Sports 使用同一套模型超参跑通后，再进入 0D。

---

### Phase 0D — Qwen SFT Baseline

先训练 Beauty SFT，用它作为所有 Phase 1–4 的主 development baseline。

固定：

- base model；
- LoRA / full fine-tuning strategy；
- learning rate；
- effective batch size；
- max history length；
- SID tokenizer；
- random seed；
- checkpoint selection rule。

Sports 不用于前期反复调参。

推荐流程：

~~~text
Beauty:
pipeline debugging
→ hyperparameter sanity
→ freeze SFT recipe

Sports:
reuse frozen recipe
→ confirmation run
~~~

这样可以减少在两个 domain 上同时调参带来的实验自由度。

#### 交付物

- Beauty frozen SFT baseline；
- Sports confirmation SFT baseline；
- training logs；
- checkpoint manifests。

---

### Phase 0E — Baseline Evaluation & Freeze

对冻结 checkpoint 运行统一 evaluator。

至少报告：

- HR@1 / 5 / 10 / 20 / 50；
- NDCG@5 / 10 / 20 / 50；
- prediction coverage；
- Top-1 concentration / popularity bias；
- basic history-sensitivity diagnostic；
- SID collision statistics。

每个 result JSON 至少写入：

- git SHA；
- domain；
- dataset manifest ID；
- RQ-VAE / item-index identity；
- tokenizer identity；
- model checkpoint；
- seed；
- decoding strategy；
- beam size；
- max history length；
- sample IDs / sample manifest。

同一 checkpoint 至少重复 evaluation，确认结果在确定性设置下完全一致；若存在 sampling，则明确记录随机性。

### Phase 0 最终交付物

- 2 个冻结 dataset manifests；
- 2 套 RQ-VAE / SID artifacts；
- Beauty SFT baseline；
- Sports confirmation baseline；
- baseline evaluation JSON；
- reproducibility / artifact manifest；
- smoke tests。

### Phase 0 总体 Go / No-Go

只有当：

1. Beauty 从 processed data → SID → SFT → evaluation 完整可复现；
2. Sports 能使用冻结 recipe 重现同一 pipeline；
3. 所有关键 artifact 都有明确 identity；
4. 不再存在 train / inference / evaluation protocol 不一致；

才正式进入 Phase 1。

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
3. **CARE**：作为第一个外部 SID-based reasoning baseline，目前项目内已经完成 protocol adaptation；
4. 只有 CARE 与 CURE 的核心现象跑通后，再考虑 S²GR / IBA / SCoTER 等第二批方法。

CARE 的价值在于它与当前 pipeline 高度兼容：同样以 Semantic ID 为输出、使用 constrained SID decoding，并显式在每个 SID stage 插入 query-anchored reasoning。它比 ReaRec / LARES 这类 ID-based sequential recommender 更适合作为当前项目的外部 reasoning baseline。

### CARE 的角色与干预边界

CARE 不能机械套用 CURE 的全部 intervention 定义。

CARE 的 stage query hidden state 本身就是当前 SID token 的直接预测表示：最后一个 query hidden state 经 `lm_head` 后产生该 stage logits。因此：

> **直接将最终 query hidden state 置零，再计算 “CARE-NG” 没有太强研究意义。**

这种干预几乎等价于直接清空 pre-logit representation，只能证明 architecture 按设计使用该状态，而不能证明 reasoning 本身具有额外信息价值。

CARE 更适合做以下 architecture-aware audit：

1. **Normal CARE**：标准 query-anchored reasoning + progressive history encoding；
2. **Progressive-off**：关闭 progressive visibility，衡量 progressive history encoding 的增益；
3. **Query/reasoning ablation**：作为 CARE 原方法内部 ablation 报告，但不要和 CURE 的 same-model Zero-Z Necessity Gap 直接等同；
4. **Matched reasoning-state swap**：在同一 SID stage 用 matched donor 的 contextualized query state 替换当前用户状态，重点测 personalization / state specificity；
5. **History-sensitivity test**：保持全局 query embeddings 不变，shuffle / replace history，检查 contextualized reasoning state 是否真的随用户历史变化。

如果实现 matched reasoning-state swap，可以定义 CARE-specific personalization effect：

$$
PG^{CARE}_{state}@K
=
HR@K_{normal}
-
HR@K_{matched\ state\ swap}
$$

但该指标应明确写成 **state personalization / specificity**，而不是与 CURE 的 Necessity Gap 做完全同义比较。

### CARE 当前实现状态

项目内的实现位于：

~~~text
baselines/care/
    model.py
    data.py
    train.py
    evaluate.py
    config.yaml
    README.md
~~~

当前 adaptation：

- 基于官方 CARE commit `9af734f2b48a9643c5246ac68c1a9f16770bc2a8`；
- 保留 query-anchored reasoning；
- 保留 stage-specific progressive history visibility；
- 保留 query diversity regularization；
- 使用项目统一的 exact SID-trie constrained decoding；
- 使用 Beauty / Sports 2018–2023 frozen leave-two-out protocol；
- 使用当前项目的 3-code SID，而官方示例主要使用 4-code SID；
- 使用 Qwen2.5-1.5B 与项目 LoRA recipe，而不是把官方数值设置原样复制过来。

因此论文中应表述为：

> **CARE adapted to the CURE-Rec experimental protocol**

而不是声称复现了 CARE 论文原表中的绝对数值。

CARE 进入正式实验前至少需要：

- Beauty train → valid → test smoke run；
- 与 SFT 使用相同 frozen sample IDs；
- 记录当前 project adaptation 与 upstream setting 的差异；
- 确认 query diversity loss、progressive mask 与 checkpoint reload 行为和官方定义一致；
- 最终实验同时报告 SID collision / ambiguous SID 情况。

### 期望发现

对于 CURE / Latent-CE，可以继续使用：

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

CARE 则重点回答另一个互补问题：

> 当 reasoning 被结构性地放在 SID token 的必经预测路径上时，这些 contextualized reasoning states 是否仍然包含当前用户特有的信息，而不是仅作为固定 query / stage bias？

这使 CARE 成为一个很有价值的对照：

~~~text
CURE:
optional latent pathway
→ 研究模型是否主动利用 latent

CARE:
structurally enforced reasoning pathway
→ 研究 reasoning state 是否真正 history-dependent / personalized
~~~

### 交付物

- intervention evaluator；
- NG / SG / PG；
- matched-counterfactual sampler；
- utilization behavior taxonomy；
- CARE accuracy baseline；
- CARE architecture-aware reasoning audit。

### Go / No-Go

如果 Latent-CE normal / zero / shuffle 高度一致：

- 说明 latent bypass 问题很强，继续 Phase 3。

如果 Latent-CE 天然已有很强 utilization：

- 不再强调 “latent bypass”；
- 转向 “personalized faithfulness / functional specificity”。

如果 CARE 的 normal accuracy 有竞争力，但 contextualized reasoning state 对 matched swap / history perturbation 不敏感：

- 这是非常有价值的诊断结果，说明“结构上经过 reasoning token”仍不自动等于 personalized reasoning。

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

当前已经准备好的正式数据集：

- Amazon Beauty 2018–2023；
- Amazon Sports 2018–2023。

这两个 domain 是论文最低必做集合：

- Beauty 用于 development / ablation；
- Sports 用于跨 domain confirmation。

第三个数据集只在主方法已经成立后再加入。优先候选可以是 Amazon Toys 或一个非 Amazon sequential recommendation dataset，但不要为了“凑三个数据集”延迟核心方法验证。

### 5.2 外部 Baselines

不要追求数量，也不要把“latent reasoning 相关”与“能直接纳入 SID-based generative recommendation protocol”混为一谈。

当前优先级：

1. **CARE — 第一优先级，已完成项目适配**  
   SID-based generative recommendation；query-anchored reasoning；progressive history encoding；与 TIGER/SID pipeline 高度兼容。
2. **S²GR — 第二优先级候选**  
   SID generation 前插入 latent thinking state，并用 codebook semantics 做监督；概念兼容度高，但复现成本高于 CARE。
3. **IBA / Where Reasoning Matters — 第二优先级候选**  
   研究不同 SID position 的 reasoning budget；与 CURE 的“reasoning 是否真的被使用”形成互补问题。
4. **SCoTER — 条件允许时加入**  
   structured reasoning transfer，论文直接在 TIGER 类 generative recommendation 上验证，但 teacher / transfer pipeline 更重。

以下工作优先作为 related work，而不是当前实验 baseline：

- ReaRec；
- LARES；
- RecRec。

原因是它们主要建立在 ID-based / text-based sequential recommendation，而不是当前 Semantic-ID autoregressive generation protocol 上。除非论文后续转成跨推荐范式的大规模 utilization audit，否则不值得为它们引入大量适配变量。

LatentR³ / LaRec / HiLaR 处于中间位置：research question 很近，但输出范式与 TIGER SID generation 不完全一致。只有在主实验已经稳定后才考虑适配。

### 5.3 Controlled Comparison vs Method-level Comparison

需要区分两类实验。

**Controlled CURE ablation**：

~~~text
SFT
Latent-CE
CURE-U
CURE-UA
~~~

这些模型必须尽量共享：

- backbone；
- tokenizer / SID；
- prompt；
- tuning strategy；
- sample IDs；
- decoding；
- evaluation protocol。

这样才能把变化归因到 CURE。

**External method comparison**：

CARE 等论文方法应尽量保留其核心 architecture / objective，只适配到统一数据、SID artifact 与 evaluation protocol。

因此 CARE 与 SFT 的差异不应该被解释成“纯 reasoning module 的 controlled gain”；它是 method-level comparison。

### 5.4 两种论文规模

#### Method-first paper

如果 CURE training 明显提高：

- accuracy；
- NG / PG；
- hard-case performance；

那么主线是：

> **CURE is a new framework for learning causally utilized latent reasoning.**

Evaluation protocol 是 supporting contribution。

#### Evaluation-first paper

如果我们发现多个 latent-reasoning models 都存在显著 utilization / personalization gap，但 CURE 方法提升有限：

主线可以变成：

> **Current latent-reasoning recommenders may not use what they reason.**

这时再扩大外部 baseline 数量，工作量更大，但 research contribution 更偏 benchmark / methodology。

### 推荐优先级

**优先争取 Method-first。**

CARE 当前已经足以作为第一个外部 SID reasoning reference；没有必要在 CURE 核心结果出现以前继续扩张 baseline zoo。

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
| External SID reasoning baseline | CARE（已实现，method-level comparison） |
| Intervention | normal / zero / random / shuffle / matched |
| CARE-specific audit | progressive-off / matched state swap / history sensitivity |
| Difficulty | easy / medium / hard |
| Dataset | Beauty（development）→ Sports（confirmation）→ optional third dataset |
| Additional external model | 只有主结果稳定后再决定是否加入 S²GR / IBA / SCoTER |

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

对于 CARE 这类结构性 reasoning baseline，还必须先判断 intervention 是否与 CURE 具有相同语义；不能为了统一表格而强行把 architecture ablation 叫作 Necessity Gap。

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

CARE 已作为第一批高度兼容 baseline。论文早期不要求继续扩张 baseline 数量。

只有 Method-first 主结果成立后才扩大到 S²GR / IBA / SCoTER 等方法。

### 风险 E：“causal” 被 reviewer 质疑

正文中明确区分：

- intervention-based causal motivation；
- empirical utilization test；
- formal causal effect estimation。

除非后续建立完整 SCM 与 identification assumptions，否则不要声称估计了严格的 causal effect。

### 风险 F：不同 architecture 的 intervention 不可直接比较

CURE 是 optional latent path；CARE 的 query state 是 SID prediction 的结构性必经节点。

因此：

- CURE 的 Zero-Z 可以解释为 same-model latent necessity；
- CARE 的 zero final query hidden 更接近直接破坏输出表示；
- 两者不应直接放在同一 NG 数值列里宣称谁“更 faithful”。

应优先比较共同的 recommendation utility，并对每种 architecture 使用语义合理的 intervention。

---

# 11. 近期里程碑

### Milestone A — Clean Baseline

完成 Phase 0。

当前状态：

- Beauty / Sports 2018–2023 data processing：**done**；
- processed-data audit / freeze：**done**；
- text embedding / RQ-VAE / SID artifact：**done**；
- CARE external baseline implementation：**done, pending numerical validation**；
- SFT baseline freeze / unified evaluation：**current priority**。

最终成功标志：

> Beauty pipeline fully reproducible, and Sports reproduces the frozen recipe without protocol changes.

### Milestone B — First Mechanism Result

完成 Phase 1–2。

成功标志：

> 能明确画出 CURE / Latent-CE 的 Normal / Zero / Shuffle / Matched 差异，并得到第一组 CARE architecture-aware audit 结果。

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

> Beauty + Sports 跨 domain 结果成立，CARE 作为兼容 SID reasoning baseline 完成比较；若结果需要再增加 S²GR / IBA / SCoTER 或第三数据集。

---

# 12. 投稿时最理想的故事

论文开头可以压成三个问题：

### Q1. Does latent reasoning really influence recommendation?

用 intervention protocol 回答。

### Q2. Is the influence personalized?

用 matched counterfactual 回答。

### Q3. Can we explicitly learn useful latent dependence?

用 CURE framework 回答。

CARE 提供一个有价值的结构性对照：

> 即使一个方法把 reasoning query 放进 SID 生成的必经路径，也仍然需要区分“结构上必须经过该 state”和“该 state 是否包含用户特异、可解释为 reasoning 的信息”。

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
   提出 intervention-based protocol，区分 latent necessity、generic sensitivity 与 personalized utilization，并强调 intervention 应与 architecture 的信息路径语义一致。

4. **Analysis**  
   研究 utilization 与 sample difficulty、recommendation accuracy、functional specialization 之间的关系，并以 CARE 作为 SID-based structurally enforced reasoning 的代表性对照。

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

## 17.1 SID-based / TIGER-compatible reasoning

- **CARE — Bringing Reasoning to Generative Recommendation Through the Lens of Cascaded Ranking**  
  Query-Anchored Reasoning + Progressive History Encoding；直接作用于 Semantic-ID autoregressive generation，是当前 CURE-Rec 最兼容的外部 reasoning baseline。  
  Paper: https://arxiv.org/abs/2602.03692  
  Code: https://github.com/Linxyhaha/CARE

- **S²GR — Stepwise Semantic-Guided Reasoning in Latent Space for Generative Recommendation**  
  在每个 SID generation step 前插入 latent thinking state，并使用 codebook semantics 进行监督。

- **IBA / Where Reasoning Matters**  
  研究 SID position-wise information gain，并把不同 reasoning budget 分配给不同 SID stage。

- **SCoTER**  
  Structured reasoning transfer；与 TIGER 类 generative recommendation 兼容，但 teacher / transfer pipeline 更重。

## 17.2 Latent reasoning：研究问题接近，但生成范式不同

- **LatentR³** — latent reasoning + reinforcement learning；
- **LaRec** — latent reasoning + multi-path reasoning；
- **HiLaR** — hierarchical latent reasoning / marginal contribution。

这些工作与 CURE 的 research question 很近，但不是当前第一优先级的 SID baseline。

## 17.3 Sequential-recommendation latent reasoning

- **ReaRec** — inference-time / latent reasoning for ID/text-based sequential recommendation；
- **LARES** — recurrent latent reasoning；
- **RecRec** — latent interests + recursive reasoning。

它们适合作为 related work；当前不为了 baseline 数量强行适配到 TIGER/SID pipeline。

## 17.4 Semantic ID / tokenizer 与实验协议

- **ReSID / UniGRec / PIT / AsymRec** — Semantic ID / tokenizer research；
- **GenPAS** — generative recommendation training sample construction。

## 17.5 General latent-reasoning faithfulness

- General latent-reasoning faithfulness work — counterfactual intervention / activation patching / hidden-state faithfulness。

这类工作是 CURE intervention design 的方法学参考，但不是 recommendation baseline。

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
