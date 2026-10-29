# CURE Implementation Plan

> 更新时间：2026-10-26  
> 当前阶段：**Phase 1 — Safe Personalized Residual CURE**  
> 前提：Beauty/Sports 数据与 SID artifacts 已冻结，基础 Qwen SFT 已训练；Beauty baseline 已完成正式评估。

---

# 1. 实现目标

从冻结 SFT 出发实现一个新的 CURE。核心不是在 Qwen 后面再堆 latent depth，而是建立：

~~~text
Frozen SFT decision
        +
Baseline-aware latent evidence
        ↓
SID candidate-space residual correction
~~~

目标行为：

~~~text
Correct personalized latent
→ improves over frozen SFT

Matched wrong latent
→ loses the gain
→ but falls back near frozen SFT
~~~

数学关系：

$$
\text{Correct }Z > \text{SFT} \approx \text{Matched wrong }Z
$$

第一版 backbone 全冻结；只有新 reasoner、gate 和 residual scorer 可训练。

---

# 2. 不做什么

第一版明确不做：

- recurrent / multi-step latent depth；
- K>1 multi-intent；
- teacher CoT；
- GRPO；
- full-vocabulary correction head；
- 重新微调整个 Qwen；
- CARE-specific intervention；
- test-set driven hyperparameter tuning。

如果主方法没有正向 validation signal，不通过增加更多层来补救。

---

# 3. 建议文件结构

~~~text
src/models/cure/
├── __init__.py
├── dataset.py
├── reasoner.py
├── residual.py
├── losses.py
├── matching.py
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

旧实现不复用；按本计划重新创建。

---

# 4. Dataset 与训练样本

复用现有 frozen split、SID tokenizer 和 leave-two-out protocol。

推荐样本至少返回：

~~~python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...,
    "history_positions": ...,
    "user_id": ...,
    "target_item_id": ...,
    "history_len": ...,
    "target_popularity": ...,
}
~~~

要求：

- latent/reasoner 只能读取 history/prompt；
- target SID 只用于 teacher-forced scoring 和 loss；
- history positions 显式提供，不用 magic offset；
- 第一版 CURE 训练不混入 item reconstruction auxiliary task，因为 frozen SFT 已经完成该学习，而新 residual branch 只服务 recommendation correction。

---

# 5. Baseline Forward API

新增统一 helper，给定 history 与当前 SID prefix，返回：

~~~python
BaselineStepOutput(
    hidden=...,
    logits=...,
    candidate_ids=...,
    candidate_logits=...,
)
~~~

训练和 generation 必须使用同一 `SidPrefixTree` 逻辑得到合法 candidate set $\mathcal C_t$。

第 1/2/3 个 SID stage 的训练 scoring 必须与真实 constrained decoding 一致；不训练 EOS。

---

# 6. Baseline-Aware Reasoner

## 6.1 History representation

从冻结 Qwen 的 history contextual states 构造 item states：

$$
E=[e_1,\ldots,e_n]
$$

第一版可以把一个 item 的三个 SID hidden state做 masked mean；这只是输入压缩，不作为论文贡献。

## 6.2 Decision-conditioned query

取 recommendation prompt 末端的 frozen baseline hidden：

$$
B=h^0_{prompt}
$$

生成 query：

$$
q=W_qB
$$

reasoner：

$$
Z=\operatorname{CrossAttn}(q,E,E)
$$

建议默认：

~~~yaml
reasoner:
  hidden_dim: 256
  num_heads: 4
  dropout: 0.1
~~~

第一版只一层 cross-attention。不要增加 recurrent steps。

必须测试：改变 target teacher-forcing token 不得改变 $Z$。

---

# 7. Residual Candidate Scorer

对于 autoregressive stage $t$，冻结 SFT 当前 hidden 为 $h_t^0$。

构造：

$$
u_t=\operatorname{MLP}([h_t^0;Z])
$$

把 $u_t$ 映射到与 frozen LM-head embedding 相同的空间：

$$
q_t^r=W_r u_t
$$

对每个合法 candidate $v\in\mathcal C_t$：

$$
r_t(v)=\frac{(q_t^r)^\top e_v}{\sqrt d}
$$

其中 $e_v$ 直接取 frozen LM head / output embedding 对应 SID token row。

最终 candidate logits：

$$
\ell_t(v)=\ell_t^0(v)+\alpha_t r_t(v)
$$

其中：

$$
\alpha_t=\sigma(g([h_t^0;Z]))
$$

第一版 gate 为 scalar per stage，避免 hidden-dimension gate 带来难解释的自由度。

推荐初始化：gate bias 设为负值，让初始 correction 较小但非零。

结构 invariant：

~~~text
force alpha = 0
→ final candidate distribution == frozen SFT candidate distribution
~~~

这是第一条必须通过的真实模型测试。

---

# 8. Target Score Helper

实现统一函数：

~~~python
score_target(
    baseline_model,
    cure_model,
    batch,
    latent_mode,
) -> {
    "sequence_score": [B],
    "stage_log_probs": [B, 3],
    "stage_distributions": [...],
}
~~~

序列 score：

$$
S(Z)=\frac{1}{3}\sum_{t=1}^{3}\log P(y_t\mid H,y_{<t},Z)
$$

要求：

- 只算 3 个 SID；
- 不算 EOS；
- 每一步只在 prefix-tree 合法 children 上 softmax；
- baseline/correct/matched 三个 branch 必须调用同一个 scoring implementation。

---

# 9. Matched Donor

新增 `matching.py`。

第一版预先为训练样本计算三个离散 bucket：

~~~text
history_len_bin
target_popularity_bin
baseline_difficulty_bin
~~~

其中 baseline difficulty 使用 frozen SFT 的 target score 或 rank 派生，离线计算并缓存。

batch 内优先选择同 bucket donor；没有完全匹配时按预定义 relaxation 顺序放宽：

~~~text
all three
→ difficulty + history
→ difficulty
→ random non-self donor
~~~

必须保证：

~~~text
donor_index[i] != i
~~~

matching 只用于训练/分析，不进入正常 inference。

---

# 10. Losses

## 10.1 Recommendation

$$
L_{rec}=-\operatorname{mean}S(Z_i)
$$

## 10.2 Gain

$$
G_i=S(Z_i)-S^0_i
$$

$$
L_{gain}=\operatorname{mean}\max(0,m_g-G_i)
$$

## 10.3 Personalization

$$
A_i=S(Z_i)-S(\tilde Z_i)
$$

$$
L_{personal}=\operatorname{mean}\max(0,m_p-A_i)
$$

## 10.4 Safe fallback

对每个 stage 的合法 candidate distribution：

$$
L_{safe}
=\frac{1}{3}\sum_t KL(P_0^t\|P_{matched}^t)
$$

## 10.5 Full objective

$$
L
=L_{rec}
+\lambda_gL_{gain}
+\lambda_pL_{personal}
+\lambda_sL_{safe}
$$

代码从一开始支持 loss flags，但实验严格按 ablation 顺序启用。

---

# 11. Training Variants

~~~yaml
method:
  variant: residual_ce | cure_g | cure_gp | cure_full
  lambda_gain: ...
  lambda_personal: ...
  lambda_safe: ...
  margin_gain: ...
  margin_personal: ...
~~~

对应：

| Variant | Rec | Gain | Personal | Safe |
|---|---:|---:|---:|---:|
| residual_ce | ✓ | – | – | – |
| cure_g | ✓ | ✓ | – | – |
| cure_gp | ✓ | ✓ | ✓ | – |
| cure_full | ✓ | ✓ | ✓ | ✓ |

不要为每个 variant 写独立 model。

---

# 12. 第一批必须测试

1. **Baseline recovery**：`alpha=0` 时 candidate logits / probabilities 与 frozen SFT 一致；
2. **No target leakage**：同 history 改 target token，pre-decision $Z$ 不变；
3. **Prefix-tree parity**：训练 candidate set 与 evaluator 对同一 prefix 的合法 children 完全一致；
4. **Frozen backbone**：SFT 参数无 gradient；
5. **Residual gradients**：reasoner / gate / residual scorer 有非零 gradient；
6. **Matched non-self**：donor 永不等于当前样本；
7. **Loss sanity**：手工构造 `Correct > Base = Wrong` 时 gain/personal/safe 的数值符合预期；
8. **Generation parity**：teacher-forcing 某 stage 的 corrected candidate logits 与 generation 同 prefix 的 logits 一致；
9. **Checkpoint round-trip**：保存/加载后输出一致。

这些通过前不跑正式 Beauty 训练。

---

# 13. 实验顺序

## Step A — infrastructure

只做 frozen SFT scoring、prefix-tree candidate scoring、baseline recovery。

## Step B — Residual-CE

只训练新 branch：

~~~text
reasoner
gate
residual scorer
~~~

目的不是作为主结果，而是测新增 correction capacity 本身。

## Step C — CURE-G

加入 positive gain constraint。先看 validation target score 与 HR/NDCG 是否出现一致正向信号。

## Step D — CURE-GP

加入 matched personalized constraint；同时监控 matched branch 是否被明显破坏。

## Step E — CURE-Full

加入 safe fallback；目标关系：

~~~text
Correct > SFT ≈ Matched
~~~

---

# 14. Validation 与 checkpoint selection

方法开发只看 Beauty validation。

checkpoint selection 优先按：

1. validation NDCG@10；
2. validation HR@10；
3. 在主指标接近时，用 target-score gain 与 fallback degradation 作为 tie-breaker。

不要根据 test 结果选 loss weight、margin 或 reasoner capacity。

---

# 15. Evaluation Outputs

结果 JSON 至少记录：

~~~text
git_sha
model_variant
sft_checkpoint_identity
dataset_manifest
item_index_hash
seed
beam_size
sample_ids
HR/NDCG
RG
PG
FD
mean_target_score_gain
mean_personalized_score_gap
mean_safe_KL
mean_gate
mean_correction_norm
~~~

其中：

$$
RG=HR_{normal}-HR_{SFT}
$$

$$
PG=HR_{normal}-HR_{matched}
$$

$$
FD=HR_{SFT}-HR_{matched}
$$

理想：`RG > 0`, `PG > 0`, `FD ≈ 0`。

---

# 16. Go / No-Go

### Go

进入 test / Sports 前至少看到：

- CURE-Full 在 Beauty validation 的主要 HR/NDCG 指标优于 SFT；
- CURE-Full 不弱于 residual_ce；
- correct target score 明显高于 baseline；
- matched branch 接近 baseline；
- 多个 validation 子集/至少两个 seed 的方向一致。

### No-Go

如果：

~~~text
PG ↑
但 RG <= 0
~~~

说明只是制造 personalized dependency，没有产生推荐增益。

如果：

~~~text
Correct > Matched
但 Matched << SFT
~~~

说明 margin 主要来自破坏 wrong-latent branch，优先加强 safe constraint / 改 residual parameterization。

如果 residual_ce、CURE-G、CURE-Full 都无法超过 SFT，不继续单纯增加 latent depth；回到“reasoner 是否真的能找到 baseline 漏掉的信息”这个问题重新设计输入或 residual target。


---

# 17. 基础实现状态（2026-10-27）

已按上述结构新增独立 CURE 实现，使用说明见 `docs/cure_usage.md`。完成候选评分、无 target 泄漏的 reasoner、scalar-gated residual、四项 loss、离线 difficulty 缓存与 non-self matching、四种训练变体、validation checkpoint selection 和三分支机制评估。

验证结果：常规 CURE/SID/metrics 测试 32 项通过；独立真实 Beauty SFT invariant 测试通过；四个变体各用 8 条 train / 2 条 validation 完成一步训练 smoke。真实 SFT 在关闭 correction 后，与 candidate-normalized SFT 原生 beam 的 Top-10 完全一致。未运行正式方法训练、Beauty test 或 Sports。

历史 SFT beam 没有在屏蔽非法 SID 后重新归一化。保留该默认行为用于兼容，同时新增 `candidate_normalized=True` reference。CURE 统一使用 `sid-candidate-logsoftmax-3stage-v1`；后续主结果必须采用同协议、同样本的 SFT 配对 reference。新增代码复用原生 LM-head / beam / KV cache，避免替换 BF16 GEMM 或重写 beam 导致排序偏差。

下一步按 Step B–E 使用 Beauty validation 验证推荐增益与 safe personalized 行为；当前只完成实现与正确性验证，不代表 Go/No-Go 已满足。

2026-10-27 多卡补充：已接入 Accelerate DDP、梯度累积与可选 W&B，difficulty 缓存及三分支评估支持多卡无重复分片，全局评估 matching 保持相同样本池。共享输出只由主进程写入。常规与 CPU 双进程测试合计 34 passed / 1 opt-in skipped；另通过双 RTX 3090 NCCL + BF16 集成测试及真实 Beauty SFT 的多卡训练/独立评估 CLI smoke。只验证功能，没有启动正式方法实验。

2026-10-27 DeepSpeed 补充：已支持 ZeRO-2 / ZeRO-3 与 CPU parameter/optimizer offload，新增 `cure_zero2` / `cure_zero3` preset。ZeRO-3 分片加载、frozen SID basis 缓存、MHA external parameters、通信补齐评估和 branch-only gather checkpoint 均通过双 GPU 测试。真实 Beauty SFT 的 ZeRO-3 train/eval CLI 已跑通；同规模显存 smoke（microbatch=2、beam=10、8 train/3 valid）训练/评估阶段每卡 CUDA allocated peak 从 DDP 3.41–3.42 GiB 降至 ZeRO-3 + CPU parameter offload 0.916 GiB，非正式实验/显存上限承诺。
