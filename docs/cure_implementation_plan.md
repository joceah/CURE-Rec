# CURE v1 Implementation Plan

> 更新时间：2026-10-19  
> 当前阶段：**Phase 1 — Minimal Latent-CE**  
> 目标：在已经完成评估的 SFT/TIGER-style baseline 上实现一个最小、可验证、可自然扩展到 CURE-U / CURE-UA 的 latent reasoning framework。

---

# 1. 实现目标

CURE v1 不直接追求复杂的 multi-intent / multi-step reasoning，而先建立一个可靠的最小机制：

~~~text
Frozen SFT Qwen
      ↓
Contextualized History States
      ↓
K=1 Latent Reasoner
      ↓
Latent z
      ↓
Gated Residual Fusion
      ↓
SID Prediction
~~~

第一阶段只优化 recommendation CE：

$$
\mathcal{L}=\mathcal{L}_{rec}
$$

只有当 Latent-CE 能达到或超过 SFT 后，才依次加入：

$$
\mathcal{L}_{util}
$$

以及：

$$
\mathcal{L}_{anchor}
$$

因此代码从一开始就支持三种 variant，但 Phase 1 只运行 `latent_ce`：

| Variant | $\mathcal{L}_{rec}$ | $\mathcal{L}_{util}$ | $\mathcal{L}_{anchor}$ |
|---|---:|---:|---:|
| `latent_ce` | ✓ | – | – |
| `cure_u` | ✓ | ✓ | – |
| `cure_ua` | ✓ | ✓ | ✓ |

---

# 2. 核心工程原则

## 2.1 不使用 forward hook

CURE 不再复用 MIR 的中间层 hook 注入方式。

原因是 hook 容易让：

~~~text
training forward
!=
generation forward
~~~

从而再次出现 train / inference mismatch。

CURE 必须实现为正式的 model class，使训练、beam search、Zero-Z、Shuffle-Z 全部经过同一套 latent computation 与 fusion 逻辑。

## 2.2 Latent 只能依赖 history

任何情况下：

$$
Z = R(H)
$$

不能使用：

- ground-truth target；
- future SID token；
- validation / test target metadata；
- teacher explanation。

必须有 unit test 验证：改变 target 时，在相同 history 下得到的 $Z$ 不变。

## 2.3 v1 优先冻结 SFT backbone

第一轮 Latent-CE 推荐直接从已经完成评估的 merged SFT checkpoint 初始化，并冻结：

- Qwen backbone；
- LM head；
- SID token embeddings。

只训练：

- Latent Reasoner；
- latent projection；
- fusion gate。

这样如果出现：

~~~text
Latent-CE > SFT
~~~

就可以较干净地说明增益来自新增 latent pathway，而不是再次 fine-tune Qwen。

如果 frozen-backbone Latent-CE 明显弱于 SFT，再把 LoRA joint tuning 作为后续 ablation，而不是第一版默认设置。

---

# 3. 数据路径

现有 SFT 输入保持不变：

~~~text
用户历史：
<SID item 1>
<SID item 2>
...
请推荐下一个商品：
<target SID>
~~~

CURE 不修改已经冻结的 SFT dataset 实现，而新增：

~~~text
src/models/cure/dataset.py
~~~

CURE dataset 复用与 SFT 相同的 sample construction，但额外返回 history token 的明确位置。

建议 batch 字段：

~~~python
{
    "input_ids": ...,
    "labels": ...,
    "attention_mask": ...,
    "history_positions": ...,
    "history_item_mask": ...,
    "latent_active": ...,
    "user_id": ...,
    "target_item_id": ...,
}
~~~

`history_positions` 必须由 dataset 显式产生，不要在 model 内通过字符串长度或 magic offset 猜测。

---

# 4. 保留 SFT 的辅助任务

当前 SFT 使用 recommendation + item reconstruction 混合训练，因此 CURE 应保持同一 recipe，避免实验变量发生变化。

对于 recommendation sample：

~~~text
latent_active = True
~~~

对于 item reconstruction sample：

~~~text
latent_active = False
~~~

即 item reconstruction 继续走原始 SFT 路径：

$$
h'_t=h_t
$$

不引入 user latent。

这样可以保证：

~~~text
SFT
vs
Latent-CE
~~~

主要差异来自 latent pathway，而不是训练样本组成改变。

---

# 5. Contextualized History Representation

CURE reasoner 不直接读取静态 SID embedding，而读取 SFT Qwen 产生的 contextualized history hidden states。

对于 history 中第 $i$ 个 item，其三个 SID token hidden states 为：

$$
[h_i^a,h_i^b,h_i^c]
$$

先聚合为 item-level representation：

$$
e_i=
\frac{1}{3}
(h_i^a+h_i^b+h_i^c)
$$

得到：

$$
E=[e_1,e_2,\ldots,e_n]
$$

第一版只使用 mean pooling，不引入额外 SID hierarchy routing。

可选加入 recency positional embedding，使 reasoner 能区分较早和较新的交互。

---

# 6. Latent Reasoner v1

第一版保持小模型：

~~~text
item-level contextual states
        ↓
Linear(backbone_dim → 256)
        ↓
+ recency position embedding
        ↓
1 learned latent query
        ↓
1-layer Multi-Head Cross Attention
        ↓
FFN + LayerNorm
        ↓
z_r ∈ R^256
        ↓
Linear(256 → backbone_dim)
        ↓
z
~~~

建议默认配置：

~~~yaml
reasoner:
  num_latents: 1
  hidden_dim: 256
  num_heads: 4
  num_layers: 1
  dropout: 0.1
~~~

`backbone_dim` 从 Qwen config 动态读取，不在代码中硬编码。

Phase 1–3 不加入：

- K > 1；
- diversity loss；
- teacher CoT；
- GRPO；
- posterior-prior；
-复杂 routing。

---

# 7. Gated Residual Fusion

对于当前生成位置的 Qwen hidden state $h_t$ 与 latent $z$：

$$
g_t
=
\sigma
\left(
W_g[h_t;z]+b_g
\right)
$$

然后：

$$
h'_t
=
h_t
+
g_t \odot W_z z
$$

最后：

$$
\mathrm{logits}_t
=
W_{LM}h'_t
$$

v1 可以先使用 scalar gate；如果效果不足，再考虑 vector gate。

## 7.1 Gate 初始化

不要使用严格的 zero gate parameter：

~~~text
alpha = 0
~~~

因为这会削弱 latent branch 在初始化阶段的梯度。

推荐：

~~~python
gate.weight = 0
gate.bias = -3.0
~~~

因此初始：

$$
\sigma(-3)\approx0.047
$$

即模型一开始接近原 SFT，但 latent branch 仍然有非零梯度。

---

# 8. CURE Model API

建议实现：

~~~text
src/models/cure/model.py
~~~

核心接口：

~~~python
class CUREForCausalLM(...):
    def encode_latent(...):
        ...

    def fuse_hidden(...):
        ...

    def forward(..., latent_z=None, latent_mode="normal"):
        ...

    def prepare_inputs_for_generation(...):
        ...
~~~

`latent_mode` 至少预留：

~~~text
normal
zero
shuffle
matched
~~~

Phase 1 只要求 `normal` / `zero` / `shuffle` 可用；`matched` 在 Phase 2–3 再实现。

---

# 9. Generation 路径

现有 SFT evaluator 使用：

~~~text
model.generate()
+ SidPrefixTree
+ constrained beam search
~~~

CURE 保留完全相同的 SID prefix constraint。

推荐流程：

~~~python
latent_z = model.encode_latent(
    input_ids=prompt_ids,
    attention_mask=prompt_mask,
    history_positions=history_positions,
)

model.generate(
    input_ids=prompt_ids,
    attention_mask=prompt_mask,
    latent_z=latent_z,
    ...
)
~~~

必须 override `prepare_inputs_for_generation()`，确保 beam expansion / autoregressive decoding 时 latent state 始终被携带。

不能再额外写一套与训练 forward 不同的 reasoning inference。

---

# 10. Latent-CE

Phase 1 只训练：

$$
\mathcal{L}_{rec}
$$

即标准 SID causal CE。

配置：

~~~yaml
method:
  variant: latent_ce
  lambda_util: 0.0
  lambda_anchor: 0.0
~~~

第一轮：

~~~text
Frozen SFT
+ trainable reasoner
+ trainable projection
+ trainable gate
~~~

主实验只看：

~~~text
SFT
vs
Latent-CE
~~~

核心问题：

> 在不继续优化 SFT backbone 的情况下，latent pathway 能否提供额外 recommendation utility？

---

# 11. Utilization Loss

只有 Phase 1 通过 Go / No-Go 后才启用。

对三个目标 SID token：

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

不把 EOS 纳入 utilization score。

然后：

$$
\mathcal{L}_{util}
=
\max
\left(
0,
m-[s(H,Z)-s(H,\tilde Z)]
\right)
$$

即正确 latent 至少应比 counterfactual latent 提供一个 margin 的 target-score 增益。

---

# 12. Counterfactual Latent v1

Phase 2 第一版不做复杂 nearest-neighbor matching。

优先使用：

- history length bin；
- target popularity bin。

在同 batch 或 donor pool 内找到不同用户：

$$
\tilde Z_i = Z_j,\quad j\neq i
$$

并尽量满足相同 / 邻近 bin。

暂不加入：

- category；
- embedding NN；
- propensity score；
- baseline difficulty matching。

这些只在 Phase 4 需要时再扩展。

---

# 13. Baseline Anchor

如果 v1 使用 frozen SFT backbone，则不需要额外加载一个 SFT teacher model。

同一份 base hidden state $h$ 可以产生：

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

然后约束：

$$
\mathcal{L}_{anchor}
=
D_{KL}
\left(
P_{base}
\;\|\;
P_{cf}
\right)
$$

目标行为：

~~~text
Wrong / unhelpful latent
→ fall back toward SFT

Correct latent
→ improve beyond SFT
~~~

这样可以避免 utilization loss 通过“故意破坏 counterfactual branch”获得虚假的大 margin。

---

# 14. SID-Vocabulary Loss

CURE 的 utilization / anchor loss 建议只在当前 SID stage 的合法 SID vocabulary 上计算，而不是 Qwen 的完整语言词表。

例如：

~~~text
stage 1: <a_0> ... <a_511>
stage 2: <b_0> ... <b_511>
stage 3: <c_0> ... <c_511>
~~~

对于第 $j$ 个 stage：

$$
P_j
=
\operatorname{softmax}
\left(
\mathrm{logits}[\mathcal{V}_j]
\right)
$$

优点：

- 与真实 constrained decoding 一致；
- loss 更具有 recommendation 语义；
- KL 计算更便宜；
- 不受大量普通自然语言 token 的无关变化干扰。

---

# 15. Trainer 设计

建议所有 variant 共用：

~~~text
src/models/cure/trainer.py
~~~

统一目标：

$$
\mathcal{L}_{CURE}
=
\mathcal{L}_{rec}
+
\lambda_u\mathcal{L}_{util}
+
\lambda_a\mathcal{L}_{anchor}
$$

通过 config 决定：

~~~yaml
method:
  variant: latent_ce
  lambda_util: 0.0
  lambda_anchor: 0.0
~~~

后续：

~~~yaml
method:
  variant: cure_u
  lambda_util: 0.1
  lambda_anchor: 0.0
~~~

以及：

~~~yaml
method:
  variant: cure_ua
  lambda_util: 0.1
  lambda_anchor: 0.1
~~~

不要为三种 variant 写三份独立模型或 trainer。

---

# 16. 推荐文件结构

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

Phase 1 暂时不需要单独的 `counterfactual.py`；等 CURE-U 开始实现时再决定是否拆分。

---

# 17. Phase 1 必须测试

至少覆盖以下六项：

1. **Zero-Z 回退**  
   frozen SFT 设置下，Zero-Z logits 应与原始 SFT logits 一致或仅存在数值误差。

2. **无 target leakage**  
   相同 history、不同 target 时，`encode_latent()` 输出必须一致。

3. **Backbone frozen**  
   frozen setup 下 Qwen / LM head 不产生可训练 gradient。

4. **Latent branch gradient**  
   reasoner / projection / gate 必须获得非零 gradient。

5. **Checkpoint round-trip**  
   save → reload 后同输入 logits 一致。

6. **Train / generation path parity**  
   teacher-forcing forward 与 autoregressive generation 使用同一个 `fuse_hidden()` 实现。

额外建议：

- `latent_active=False` 时与 SFT 路径一致；
- beam expansion 后 latent batch dimension 正确复制；
- Zero / Shuffle mode 不改变 history / target sample identity。

---

# 18. 第一轮实验顺序

不要一次性实现完整 CURE。

严格按以下顺序：

### Step 1 — Architecture Sanity

~~~text
load frozen SFT
→ construct CURE model
→ no training
→ verify Zero-Z ≈ SFT
~~~

### Step 2 — Latent-CE

~~~text
Frozen SFT
+ K=1 reasoner
+ gated fusion
+ recommendation CE
~~~

只在 Beauty 先跑。

### Step 3 — Recommendation Evaluation

比较：

~~~text
SFT
vs
Latent-CE
~~~

主指标：

- HR@K；
- NDCG@K。

### Step 4 — CURE-U

只有 Latent-CE 至少不明显低于 SFT 时，加入：

$$
\lambda_u\mathcal{L}_{util}
$$

### Step 5 — CURE-UA

只有 CURE-U 有稳定信号时，再加入：

$$
\lambda_a\mathcal{L}_{anchor}
$$

### Step 6 — Mechanistic Validation

方法性能稳定后再跑：

- Normal；
- Zero-Z；
- Shuffle-Z；
- Matched-Z。

不要在架构尚未证明有效时投入大量时间做 intervention analysis。

---

# 19. Go / No-Go

## Phase 1 Go

如果：

~~~text
Latent-CE >= SFT
~~~

或 Latent-CE 在主要指标上只有非常小、可解释的下降，同时 target log-prob / difficult-sample behavior 显示 latent 有潜力，则进入 CURE-U。

更理想的是：

~~~text
Latent-CE > SFT
~~~

因为这说明 latent architecture 本身已经能够提供额外推荐信息。

## Phase 1 No-Go

如果：

~~~text
Latent-CE << SFT
~~~

不要直接通过增大 utilization penalty 强迫模型依赖 latent。

优先检查：

1. contextual history extraction 是否正确；
2. reasoner 是否过弱 / 过强；
3. fusion gate 是否饱和；
4. frozen backbone 是否限制过大；
5. history item pooling 是否损失过多信息。

必要时再尝试 LoRA joint tuning。

---

# 20. v1 暂不实现

Phase 1–2 明确不做：

- K > 1 multi-intent；
- diversity loss；
- explicit CoT；
- teacher distillation；
- GRPO；
- dynamic reasoning depth；
- complex routing；
- difficulty-aware utilization；
- 多种 matched-counterfactual sampler；
- CARE-specific intervention。

这些都只有在最小 CURE 已经证明有用后才考虑。

---

# 21. 当前最重要的开发目标

下一阶段真正需要完成的不是“完整论文系统”，而是一个未来可自然扩展到 CURE-U / CURE-UA、但目前只运行 Latent-CE 的稳定 skeleton：

~~~text
Frozen SFT
    ↓
Contextual History States
    ↓
K=1 Latent Reasoner
    ↓
z
    ↓
Gated Residual Fusion
    ↓
SID Prediction
~~~

第一条研究判据仍然是：

> **先证明 latent pathway 本身能够产生有价值的推荐信息，再训练模型必须更强地利用它。**
