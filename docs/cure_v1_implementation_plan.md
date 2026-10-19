# CURE v1 Implementation Plan

> 更新时间：2026-10-19  
> 当前阶段：**Phase 1 — Minimal Latent-CE / CURE model skeleton**  
> 前提：基础 SFT baseline 已完成训练与评估，当前正式进入 CURE 方法实现。

---

## 1. 目标

第一版 CURE 不直接上完整 `L_util + L_anchor`，而先实现一个可扩展的统一模型骨架，并先以 **Latent-CE** 模式训练。

核心目标：

> 在冻结 SFT baseline 的基础上，引入一个只依赖用户历史的 `K=1` latent reasoning pathway，并验证该 pathway 本身是否能带来额外推荐收益。

第一阶段只优化：

$$
\mathcal{L}=\mathcal{L}_{rec}
$$

其中：

- 不使用 utilization loss；
- 不使用 anchor loss；
- 不使用 teacher signal；
- 不使用 multi-intent；
- 不使用 forward hook；
- 不改变现有 SID tokenizer / RQ-VAE / evaluation protocol。

只有 Latent-CE 至少达到或超过 SFT 后，才进入 CURE-U / CURE-UA。

---

# 2. 设计原则

## 2.1 训练与推理必须走同一 latent 路径

旧 MIR 最大的问题之一是 latent injection 在训练和推理阶段不完全一致。

CURE v1 禁止使用 forward hook 往 Qwen 中间层临时插入 latent。

应实现正式的 `CUREForCausalLM` model class，使以下路径共享同一个 fusion：

~~~text
training forward
validation forward
autoregressive generation
Zero-Z
Shuffle-Z
Matched-Z
~~~

模型行为必须由同一套显式代码控制。

---

## 2.2 第一版只使用 K=1

暂不做：

- K=4 multi-intent；
- diversity loss；
- SID routing；
- hierarchical intent；
- GRPO；
- CoT distillation。

原则：

> 如果一个 latent 都无法稳定提高推荐效果，那么多个 latent 暂时没有研究基础。

---

## 2.3 SFT baseline 作为固定起点

第一轮 Latent-CE 推荐直接使用已经训练完成的 merged SFT checkpoint 初始化，并冻结：

- Qwen backbone；
- LM head；
- SID token embeddings。

只训练：

- Latent Reasoner；
- latent projection；
- fusion gate。

这样如果 Latent-CE 提升，可以更清楚地归因到新增 latent pathway，而不是再次 fine-tune Qwen。

如果 frozen-backbone Latent-CE 无法达到 SFT，再考虑第二阶段 joint LoRA tuning。

---

# 3. 模型总体结构

~~~text
                      Frozen SFT Qwen
                            │
History SID ────────────────┤
                            ↓
                 contextual hidden states
                            │
                            ↓
                    Latent Reasoner
                            │
                            ↓
                         z  (K=1)
                            │
                            ↓
                  gated residual fusion
                            │
Qwen hidden h_t ────────────┤
                            ↓
                         h'_t
                            │
                            ↓
                         LM Head
                            │
                            ↓
                       next SID token
~~~

核心思想：

- Qwen 继续承担原 SFT 的直接 history → target 路径；
- latent reasoner 从 contextualized history 表示中提取一个额外 state `z`；
- `z` 通过 gated residual fusion 为 SID prediction 提供增量信息。

---

# 4. History Representation

当前每个商品由三个 SID token 表示：

$$
SID_i=(a_i,b_i,c_i)
$$

Qwen forward 后得到对应 contextual hidden states：

$$
h_i^a,\;h_i^b,\;h_i^c
$$

第一版把一个商品压成 item-level representation：

$$
e_i=
\frac{h_i^a+h_i^b+h_i^c}{3}
$$

用户历史得到：

$$
E=[e_1,e_2,\ldots,e_n]
$$

注意：

- latent reasoner 只能读取 history token hidden states；
- 不能读取 target SID hidden state；
- 不能通过 teacher-forcing target 泄漏信息。

因此 dataset 必须显式返回 history token / item position，而不能在 model 内通过 magic offset 猜位置。

---

# 5. Latent Reasoner v1

第一版 reasoner 保持轻量：

~~~text
contextual item states
        ↓
Linear: hidden_size → reasoner_dim
        ↓
+ recency positional embedding
        ↓
1 learned latent query
        ↓
1-layer Multi-Head Cross Attention
        ↓
FFN + LayerNorm
        ↓
z_r
        ↓
Linear: reasoner_dim → hidden_size
        ↓
z
~~~

推荐默认配置：

~~~yaml
reasoner:
  num_latents: 1
  hidden_dim: 256
  num_heads: 4
  num_layers: 1
  dropout: 0.1
~~~

其中 `num_latents` 在 Phase 1–3 固定为 1。

---

# 6. Gated Residual Fusion

对于当前 SID generation hidden state：

$$
h_t
$$

以及 latent：

$$
z
$$

计算 gate：

$$
g_t=
\sigma\left(W_g[h_t;z]+b_g\right)
$$

融合：

$$
h'_t
=
h_t+g_t\odot W_z z
$$

然后：

$$
logits_t=W_{LM}h'_t
$$

## Gate 初始化

不要使用 MIR 风格的精确 zero scalar gate：

~~~text
alpha = 0
~~~

推荐：

~~~text
gate.weight = 0
gate.bias = -3
~~~

此时：

$$
\sigma(-3)\approx0.047
$$

意味着模型初始接近原 SFT，但 latent pathway 仍有非零梯度。

---

# 7. Dataset 设计

不要修改已经用于 SFT baseline 的 `SFTDataset`。

新增：

~~~text
src/models/cure/dataset.py
~~~

主推荐样本至少返回：

~~~python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...,
    "history_positions": ...,
    "history_item_mask": ...,
    "user_id": ...,
    "target_item_id": ...,
    "history_len": ...,
    "latent_active": True,
}
~~~

其中 `history_positions` 明确标识 Qwen sequence 中属于历史 SID 的位置。

---

## 7.1 保留 SFT 的 item reconstruction auxiliary task

当前 SFT 使用 recommendation + item reconstruction mixture，因此 CURE 应保持相同训练协议，避免实验变量改变。

对于 item reconstruction 样本：

~~~text
latent_active = False
~~~

即：

$$
h'_t=h_t
$$

不要让 user latent 参与 item reconstruction。

这样：

~~~text
Recommendation sample
→ latent enabled

Item reconstruction sample
→ latent disabled
~~~

---

# 8. Model API

新增：

~~~text
src/models/cure/model.py
~~~

建议主类：

~~~text
CUREForCausalLM
~~~

至少提供：

~~~python
encode_latent(...)
forward(..., latent_z=None, latent_mode="normal")
prepare_inputs_for_generation(...)
~~~

`latent_mode` 后续统一支持：

~~~text
normal
zero
shuffle
matched
~~~

Phase 1 先实现：

~~~text
normal
zero
~~~

`shuffle / matched` 可以等 Phase 2–3 再完成。

---

# 9. Generation 路径

当前 SFT evaluator 使用：

~~~text
model.generate()
+ SidPrefixTree
+ constrained beam search
~~~

CURE 不应重写一套独立 reasoning inference。

正确路径：

~~~python
z = model.encode_latent(
    input_ids=prompt_ids,
    attention_mask=attention_mask,
    history_positions=history_positions,
)

model.generate(
    input_ids=prompt_ids,
    attention_mask=attention_mask,
    latent_z=z,
    ...
)
~~~

`prepare_inputs_for_generation()` 必须保证 beam expansion 后 `latent_z` 始终跟随对应样本。

训练 forward 和 generation forward 必须调用完全相同的 fusion implementation。

---

# 10. Loss Interface

训练器从第一天就统一支持三个 variant：

~~~yaml
method:
  variant: latent_ce
  lambda_util: 0.0
  lambda_anchor: 0.0
~~~

后续只通过配置切换：

### Latent-CE

$$
\lambda_u=0,\quad\lambda_a=0
$$

### CURE-U

$$
\mathcal{L}
=
\mathcal{L}_{rec}
+
\lambda_u\mathcal{L}_{util}
$$

### CURE-UA

$$
\mathcal{L}
=
\mathcal{L}_{rec}
+
\lambda_u\mathcal{L}_{util}
+
\lambda_a\mathcal{L}_{anchor}
$$

不要实现三套 model / trainer。

---

# 11. Utilization Loss（Phase 2）

对于目标三层 SID：

$$
y=(y_1,y_2,y_3)
$$

定义 target score：

$$
s(H,Z)
=
\frac{1}{3}
\sum_{j=1}^{3}
\log P(y_j\mid H,y_{<j},Z)
$$

只计算三个 SID token，不包括 EOS。

利用正确 latent 和 counterfactual latent：

$$
\mathcal{L}_{util}
=
\max
\left(
0,
\;m-[s(H,Z)-s(H,\tilde Z)]
\right)
$$

第一版 counterfactual donor 可只按：

- history length bin；
- target popularity bin；

进行 batch 内匹配。

不要一开始加入复杂 nearest-neighbor / category / difficulty matching。

---

# 12. Baseline Anchor（Phase 2）

如果 backbone 继续冻结，同一次 Qwen forward 可以构造：

~~~text
base branch:
h
→ logits_base

correct branch:
h + fuse(z)
→ logits_correct

counterfactual branch:
h + fuse(z_tilde)
→ logits_cf
~~~

因此不需要额外加载一份 SFT teacher。

推荐：

$$
\mathcal{L}_{anchor}
=
D_{KL}
\left(
P_{base}
\|\|
P_{cf}
\right)
$$

目标：

~~~text
Wrong latent
→ 回退到 SFT-like prediction

Correct latent
→ 提供额外收益
~~~

---

## 12.1 优先在 SID vocabulary 上计算 Util / Anchor

不建议比较整个 Qwen vocabulary。

每个 SID stage 只在对应 512 个合法 token 上归一化：

~~~text
stage 1: <a_0> ... <a_511>
stage 2: <b_0> ... <b_511>
stage 3: <c_0> ... <c_511>
~~~

定义：

$$
P_j(\cdot)
=
softmax(logits[\mathcal V_j])
$$

这样：

- 与 constrained decoding 一致；
- 减少无关自然语言 token 干扰；
- 更省显存；
- loss 更容易解释。

---

# 13. 文件结构

第一版控制在：

~~~text
src/models/cure/
├── __init__.py
├── dataset.py
├── reasoner.py
├── model.py
├── losses.py
└── trainer.py

src/inference/
└── cure_evaluator.py

configs/
└── cure.yaml

scripts/train/
└── run_cure.py

scripts/eval/
└── run_cure_eval.py

tests/
└── test_cure.py
~~~

职责：

| 文件 | 职责 |
|---|---|
| `reasoner.py` | K=1 history → latent `z` |
| `model.py` | Qwen + reasoner + gated residual fusion |
| `losses.py` | rec / util / anchor |
| `dataset.py` | history positions + recommendation / auxiliary samples |
| `trainer.py` | Latent-CE / CURE-U / CURE-UA 共用训练器 |
| `cure_evaluator.py` | CURE recommendation evaluation + intervention |

Phase 1 暂不新增复杂 `counterfactual.py`，等 Phase 2 真正需要 matched donor 后再加入。

---

# 14. 最小测试

第一批至少覆盖：

1. **Zero-Z recovers SFT behavior**  
   对同一输入，`latent_mode=zero` 的 logits 应与冻结 SFT logits 一致或在数值误差范围内接近。

2. **No target leakage**  
   保持 history 不变、改变 target teacher-forcing token，`encode_latent()` 输出必须不变。

3. **Frozen backbone has no gradients**  
   Latent-CE 第一阶段 Qwen backbone 参数梯度应为空。

4. **Reasoner / fusion receive gradients**  
   latent reasoner、projection、gate 必须有非零梯度。

5. **Checkpoint round-trip**  
   save → reload 后 logits 一致。

6. **Train / generation fusion parity**  
   teacher-forcing 的下一 SID logits 和 generation 同位置 logits 使用相同 fusion 路径。

这六个测试通过之前不跑正式 Beauty 实验。

---

# 15. Phase 1 实验顺序

## Step 1 — Architecture Sanity

加载 frozen SFT，不训练：

~~~text
SFT logits
vs
CURE model + Zero-Z logits
~~~

必须一致。

然后验证：

~~~text
Normal-Z logits != Zero-Z logits
~~~

保证 latent pathway 工程上实际生效。

---

## Step 2 — Latent-CE / Beauty

设置：

~~~yaml
variant: latent_ce
lambda_util: 0.0
lambda_anchor: 0.0
backbone_trainable: false
~~~

只训练：

~~~text
reasoner
latent projection
gate
~~~

主比较：

| Model | Backbone | Latent | Objective |
|---|---|---|---|
| SFT | frozen baseline | no | SFT CE |
| Latent-CE | frozen SFT | K=1 | rec CE |

---

## Step 3 — Go / No-Go

### Go

如果：

~~~text
Latent-CE >= SFT
~~~

或者 HR / NDCG 至少基本持平，同时 target log-prob / hard-sample performance 有明确改善，则进入 CURE-U。

### No-Go

如果 Latent-CE 明显低于 SFT：

优先检查：

1. history representation；
2. reasoner capacity；
3. fusion location；
4. gate initialization；
5. 是否需要 LoRA joint tuning。

不要立即加入更强 `L_util` 强迫模型依赖低质量 latent。

---

# 16. Phase 2 — CURE-U / CURE-UA

只有 Phase 1 通过后开始。

顺序严格为：

~~~text
Latent-CE
    ↓
CURE-U
    ↓
CURE-UA
~~~

核心比较：

| Variant | Rec | Util | Anchor |
|---|---:|---:|---:|
| SFT | ✓ | – | – |
| Latent-CE | ✓ | – | – |
| CURE-U | ✓ | ✓ | – |
| CURE-UA | ✓ | ✓ | ✓ |

主要指标仍然是：

- HR@K；
- NDCG@K。

只有性能成立后，才进入 Normal / Zero / Shuffle / Matched 机制验证。

---

# 17. 当前最重要的 Go / No-Go

当前阶段不要问：

> CURE 的 utilization loss 能不能把 NG / PG 做大？

先回答：

> **一个干净的 K=1 latent pathway，在不重新训练 SFT backbone 的情况下，能否提供额外有效推荐信息？**

如果答案是 yes，CURE-U / CURE-UA 才有可靠基础。

如果答案是 no，应先改 latent representation / fusion，而不是通过 regularization 人为制造 dependency。

---

# 18. 当前实施顺序

~~~text
[1] 创建 CURE model skeleton
        ↓
[2] 实现 history position extraction
        ↓
[3] 实现 K=1 latent reasoner
        ↓
[4] 实现 gated residual fusion
        ↓
[5] 打通 forward + generate
        ↓
[6] 完成 6 个关键 tests
        ↓
[7] Beauty Latent-CE
        ↓
[8] 对比 SFT HR/NDCG
        ↓
[9] Go → CURE-U
        ↓
[10] CURE-UA
        ↓
[11] CURE-specific interventions
~~~

在第 8 步之前，不实现 multi-intent、difficulty weighting 或复杂 matched counterfactual。
