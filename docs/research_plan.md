# CURE-Rec Research Plan

> 更新时间：2026-10-26  
> Working title: **CURE-Rec: Counterfactual Utility-Regularized rEasoning for Generative Recommendation**  
> 主线：**method-first generative recommendation**。CURE 的目标不是增加更多 latent depth，而是学习一个对强 SFT baseline 有真实增量价值、具有个性化、且在错误 latent 下安全回退的 latent decision state。

---

# 0. 项目定位

CURE-Rec 不试图建立一个覆盖所有 latent-reasoning recommender 的统一 faithfulness benchmark。论文首先必须回答：

> **Can a latent decision state improve a strong Semantic-ID generative recommender by contributing personalized information beyond the baseline, rather than merely adding capacity or duplicating history representations?**

核心行为目标写成一个非对称关系：

$$
\boxed{
\text{Correct personalized } Z
\;>\;
\text{Frozen SFT baseline}
\;\approx\;
\text{Matched wrong } Z
}
$$

它包含三个条件：

1. **Positive lift**：正确 latent 必须优于 baseline；
2. **Personalized lift**：自己的 latent 必须优于 matched donor latent；
3. **Safe fallback**：错误 latent 不能通过破坏预测制造 margin，而应回退到 baseline-like behavior。

推荐性能始终是第一目标；机制指标只服务于解释 CURE 自身。

---

# 1. Novelty Boundary

正式写论文时不能把以下单点当作原创：

- 增加 latent computation depth；
- frozen recommender 上的 residual correction；
- latent 相对 no-latent 的 target-likelihood gain；
- KL preservation / distillation；
- 用 intervention 检查 reasoning 是否影响输出。

近期工作已经覆盖这些组成部分。例如 HiLaR 使用 latent state 的 marginal target-likelihood gain；PRL 显式学习 base recommender 的 residual；通用 reasoning 中也已有通过有/无 rationale distribution contrast 提高 reasoning dependence 的工作。

当前候选贡献不是某一个组件，而是下面这个 **asymmetric safe personalized residual formulation**：

> 在 SID generative recommendation 中，训练 latent state 只通过“为正确用户增加价值”获得增益；matched 错误 latent 必须失去这部分个性化增益，但不能让模型比 frozen baseline 更坏。

相关来源（正式投稿前再次核对最新版）：

- HiLaR: https://arxiv.org/abs/2607.27760
- Probabilistic Residual Learning: https://arxiv.org/abs/2607.20863
- Rationale Contrastive Decoding / context-aware CoT decoding: https://arxiv.org/abs/2610.29518

---

# 2. Frozen Baseline

基础模型是已经训练完成的 Qwen/TIGER-style SID generator：

$$
P_0(y\mid H)
$$

其中：

- $H$：用户交互历史；
- $y=(y_1,y_2,y_3)$：目标商品的三层 Semantic ID；
- baseline 使用与正式评估完全一致的 prefix-constrained SID decoding。

CURE 第一阶段冻结 baseline。这样性能变化更容易归因到新增 residual pathway，而不是再次微调整个 Qwen。

Beauty SFT 的当前 5k test reference：HR@10 = 0.0448，NDCG@10 = 0.03559。后续方法开发不继续使用 test 调参，只在 validation 上选择结构与超参。

---

# 3. CURE Architecture: Baseline-Aware Complementary State

## 3.1 Baseline decision state

冻结 SFT 对 history prompt 产生 baseline decision representation：

$$
B_i=h_i^0
$$

$B_i$ 只依赖 history/prompt，不包含 target。

同时保留 history item contextual states：

$$
E_i=[e_{i1},\ldots,e_{in}]
$$

## 3.2 Baseline-conditioned latent reasoner

不使用对所有用户相同的固定 learned query。让当前 baseline decision 生成 query：

$$
q_i=W_q B_i
$$

然后重新查询 history evidence：

$$
Z_i=R_\phi(q_i,E_i)
$$

语义不是“再总结一次用户历史”，而是：

> baseline 已经形成一个初步决策；latent reasoner 根据这个决策状态回看历史，寻找可用于修正当前预测的个性化 evidence。

第一版只使用一个 latent state，不把额外 depth 作为贡献点。

---

# 4. Residual Decision Correction

CURE 不替代 SFT，也不重新学习完整 vocabulary distribution。对第 $t$ 个 SID stage，baseline 给出：

$$
\ell^0_{i,t}(v)
$$

当前 prefix 对应的合法 SID children 集合记为：

$$
\mathcal C_{i,t}
$$

CURE 只生成 candidate-space correction：

$$
r_{i,t}(v\mid H_i,Z_i)
$$

最终 logits：

$$
\ell_{i,t}(v)
=
\ell^0_{i,t}(v)
+
\alpha_{i,t}r_{i,t}(v),
\quad v\in\mathcal C_{i,t}
$$

其中 $\alpha_{i,t}\in[0,1]$ 是 correction gate。

推荐实现使用冻结 LM-head/SID token embedding 作为 candidate basis：

$$
u_{i,t}=\operatorname{MLP}([h^0_{i,t};Z_i])
$$

$$
r_{i,t}(v)=\frac{(W_r u_{i,t})^\top e_v}{\sqrt d}
$$

其中 $e_v$ 是合法 SID token $v$ 的冻结输出 embedding。

关键结构性质：

$$
r=0\;\text{or}\;\alpha=0
\Rightarrow
P_\theta=P_0
$$

因此 baseline fallback 是模型结构中的显式参考点，而不是另一个重新训练的 decoder。

---

# 5. Target Score

训练和分析都只对三个 SID token 计算，不计算 EOS，并且 softmax 与 prefix-constrained decoding 使用同一合法 candidate set。

定义 correct-latent target score：

$$
S_i(Z)
=
\frac{1}{3}
\sum_{t=1}^{3}
\log P_\theta
(y_{i,t}\mid H_i,y_{i,<t},Z)
$$

baseline score：

$$
S_i^0
=
\frac{1}{3}
\sum_{t=1}^{3}
\log P_0
(y_{i,t}\mid H_i,y_{i,<t})
$$

所有核心 loss 都围绕同一个 score 定义。

---

# 6. Matched Counterfactual Latent

为样本 $i$ 选择另一个 donor $j\neq i$：

$$
\tilde Z_i=Z_j
$$

第一版 matching 只控制：

- history length bin；
- target popularity bin；
- frozen SFT baseline difficulty / target score bin。

这样 donor latent 不至于只是明显 OOD noise，同时不会依赖 deployment 时不可获得的信息。matching 只用于训练/分析，正常 inference 不需要 donor。

---

# 7. Core Objective

## 7.1 Recommendation Loss

$$
\mathcal L_{rec}=-S_i(Z_i)
$$

保证最终推荐本身正确。

## 7.2 Positive Gain Constraint

定义 latent 相对 frozen baseline 的增量：

$$
G_i=S_i(Z_i)-S_i^0
$$

要求正确 latent 至少提供 margin $m_g$：

$$
\mathcal L_{gain}
=
\max(0,m_g-G_i)
$$

它不是声称首次使用 likelihood gain，而是把“必须优于 frozen baseline”作为 CURE 三角关系的第一条约束。

## 7.3 Personalized Lift Constraint

定义：

$$
A_i=S_i(Z_i)-S_i(\tilde Z_i)
$$

要求：

$$
\mathcal L_{personal}
=
\max(0,m_p-A_i)
$$

它阻止模型只学习一个所有用户通用的 correction bias。

## 7.4 Safe Fallback Constraint

错误 latent 下，模型应接近 frozen baseline，而不是人为崩坏：

$$
\mathcal L_{safe}
=
\frac{1}{3}
\sum_{t=1}^{3}
D_{KL}
\left(
P_0^t
\;\|\;
P_\theta^t(\tilde Z_i)
\right)
$$

KL 只在当前合法 SID candidate set $\mathcal C_{i,t}$ 上计算。方向采用 $KL(P_0\|P_\theta)$，把 frozen baseline 视为需要 preservation 的 reference teacher。

## 7.5 Full CURE

$$
\boxed{
\mathcal L_{CURE}
=
\mathcal L_{rec}
+
\lambda_g\mathcal L_{gain}
+
\lambda_p\mathcal L_{personal}
+
\lambda_s\mathcal L_{safe}
}
$$

目标行为不是简单的 `correct Z > wrong Z`，而是：

$$
\boxed{
S_i(Z_i)>S_i^0\approx S_i(\tilde Z_i)
}
$$

这会迫使 margin 主要来自 **correct latent 带来的正向增益**，而不是 wrong latent 的灾难性退化。

---

# 8. Method Ablations

最小 ablation：

| Variant | Residual architecture | Gain | Personal | Safe |
|---|---:|---:|---:|---:|
| SFT | – | – | – | – |
| Residual-CE | ✓ | – | – | – |
| CURE-G | ✓ | ✓ | – | – |
| CURE-GP | ✓ | ✓ | ✓ | – |
| CURE-Full | ✓ | ✓ | ✓ | ✓ |

其中 Residual-CE 只用于证明“增加一个 correction branch”本身是否足够；它不是主方法。

---

# 9. Evaluation

主指标仍然是：

- HR@1/5/10/20/50；
- NDCG@5/10/20/50。

CURE 自身增加三个支持性指标：

### Residual Gain

$$
RG@K
=HR@K_{normal}-HR@K_{SFT}
$$

### Personalized Gap

$$
PG@K
=HR@K_{normal}-HR@K_{matched}
$$

### Fallback Degradation

$$
FD@K
=HR@K_{SFT}-HR@K_{matched}
$$

理想关系：

$$
\boxed{RG>0,\quad PG>0,\quad FD\approx0}
$$

同时报告 target-score gain、target-rank shift 和 correction magnitude，辅助解释机制。

---

# 10. Experimental Roadmap

## Phase 0 — Baseline & Artifacts ✅

已完成 Beauty/Sports 数据处理、audit/freeze、text embedding、RQ-VAE、3-code SID 与基础 SFT pipeline。Beauty SFT 已完成正式评估。

## Phase 1 — Residual Scoring Infrastructure ← current

目标：实现与实际 prefix-constrained decoding 完全一致的 baseline/correct/counterfactual scoring，并保证 correction disabled 时精确恢复 SFT。

先不做复杂 reasoner，不上多步 latent computation。

## Phase 2 — CURE Training

按以下顺序逐项启用：

~~~text
Residual-CE
→ CURE-G
→ CURE-GP
→ CURE-Full
~~~

所有方法选择只在 Beauty validation 上完成。

## Phase 3 — Mechanism Validation

只对最终 CURE 做：

- SFT / no correction；
- normal correct latent；
- matched counterfactual latent；
- optional global shuffle。

验证 `Correct > Baseline ≈ Matched` 是否成立。

## Phase 4 — Method Refinement

只有主方法已经产生正向性能信号后才考虑：

- difficulty-aware weighting；
-更稳健 matching；
- reasoner capacity；
- multiple latent states；
- joint LoRA tuning。

不把增加 depth 当默认改进方向。

## Phase 5 — Paper-Scale Validation

- Beauty：development；
- Sports：冻结 recipe confirmation；
- CARE：SID reasoning performance baseline；
- 视时间与代码可用性决定是否加入 S²GR / IBA / SCoTER；
- 主结果稳定后再做 seeds / backbone scaling / optional third dataset。

---

# 11. Go / No-Go

进入正式 paper-scale 实验前，至少满足：

1. CURE 在 Beauty validation 的主要 HR/NDCG 指标稳定优于 SFT；
2. CURE-Full 优于或至少不弱于 Residual-CE；
3. target-score 层面满足 $S(Z)>S^0$；
4. matched branch 接近 baseline，而不是明显崩坏；
5. correction-disabled inference 与原 SFT decoding 完全一致；
6. design freeze 后才重新评估 Beauty test，并在 Sports 使用相同 recipe。

如果只有 PG 变大但 HR/NDCG 不升，说明方法只制造了依赖，不算成功。

---

# 12. Working Contributions

如果实验成立，贡献优先表述为：

1. **Safe personalized residual formulation**：把 latent reasoning 定义为对 frozen generative recommender 的个性化增量修正，而非额外深度本身；
2. **Asymmetric counterfactual objective**：要求 `Correct Z > Baseline ≈ Matched Z`，联合 positive lift、personalized specificity 与 safe fallback；
3. **SID-aligned optimization**：训练目标与 prefix-constrained Semantic-ID candidate space 对齐；
4. **Method-first evidence**：在 Beauty/Sports 上首先证明推荐性能提升，再用 residual gain / personalized gap / fallback degradation 解释增益来源。

正式投稿前再次完成系统 related-work 查重；当前不声称单独的 residual learning、marginal gain 或 KL anchor 是原创。
