
## 2026-10-04 — Phase 0B processed data audit and freeze

- Added reusable auditor `scripts/data/audit_freeze_processed.py`. It hashes source and processed artifacts, reconstructs each user timeline from raw review JSONL, checks saved train histories and leave-two-out targets, validates ID mappings, target coverage, k-core degrees, sequence lengths, and metadata missingness.
- Beauty 2018–2023 passed: 446,314 users, 148,338 items, 3,203,489 train interactions, 446,314 validation and 446,314 test samples. Sports passed: 169,680 users, 81,571 items, 1,081,335 train interactions, 169,680 validation and 169,680 test samples. All temporal histories and targets matched raw timestamp ordering exactly; all held-out targets had item and metadata coverage.
- Froze manifests and ordered validation/test sample IDs under `data/manifests/`; all six frozen-file checksums verified with `data/manifests/freeze.sha256`. Timestamp ties were resolved by stable source JSONL order, matching the preprocessing implementation.

## 2026-10-27 — Safe Personalized Residual CURE 基础实现与验证

- 阅读 `docs/research_plan.md` 与 `docs/cure_implementation_plan.md` 后从零新增 `src/models/cure/`，未复用旧 latent-residual implementation 或旧 checkpoint。实现 recommendation-only 数据与显式 history spans、冻结 Qwen baseline、单层 baseline-conditioned cross-attention、合法 SID candidate 的 scalar-gated residual、三个 stage 的统一 target scoring（不含 EOS）、rec/gain/personal/safe loss、四种训练变体。
- Matching 仅使用 history length、train-only target popularity 和 frozen baseline target-score bucket，按 all → difficulty+history → difficulty → random non-self 放宽。离线缓存绑定 artifact / 数据 / 样本 / 截断与 bucket 身份。训练合并 singleton 尾 batch；评估在完整选定样本池匹配并保存 donor IDs 与 relaxation counts。
- 新增 `configs/cure.yaml`、训练/评估 CLI 与 `docs/cure_usage.md`。checkpoint 仅保存新增 branch、结构和 provenance；按 validation NDCG@10、HR@10、target-score gain、较小 FD@10 选 best。评估输出三分支 HR/NDCG、RG/PG/FD、score gap、safe KL、gate、correction norm、beam-censored target-rank shift 及身份信息。加载数据核对 frozen manifest；评估核对 checkpoint 与预处理身份。
- 发现历史 SFT beam 没有在屏蔽非法 SID 后重新归一化。保留其默认行为，新增 `candidate_normalized=True` reference；CURE 统一采用 `sid-candidate-logsoftmax-3stage-v1`。历史 SFT test 指标不能直接充当新协议配对 reference。真实 BF16 验证还发现仅抽取 SID rows 做 GEMM及单独重写 beam 会改变接近候选的排序，最终复用原生 LM-head、beam、KV cache，通过 hook / logits processor 添加 residual。
- 验证：`tests/test_cure.py tests/test_prefix_tree.py tests/test_metrics.py` 常规 32 passed、1 opt-in skipped；启用 `CURE_REAL_SFT=1` 的真实 Beauty SFT invariant 测试单独通过（两条 validation，关闭 correction 的 Top-10 与同协议 SFT 原生 beam 精确一致，同时检查 target 隔离、冻结梯度和 checkpoint round-trip）。四种变体分别在 8 条 train / 2 条 validation 上完成一步真实模型 smoke，GP/Full 缓存创建与复用通过。独立 CURE-Full eval CLI 从保存的 checkpoint 加载并输出完整 metadata JSON 通过；语法编译、`git diff --check` 通过。
- 本机 Python 缺少 transformers 等依赖，验证时临时安装到 `/tmp/cure-rec-deps`，没有修改系统 Python；系统旧 ONNX/protobuf 导入冲突通过 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 绕过。临时依赖、smoke 模型与输出在收尾删除。
- 仅完成基础实现、正确性验证与 pipeline smoke，未启动正式方法训练，没有使用 Beauty test 或 Sports，也没有形成推荐性能增益结论。下一步按 Residual-CE → CURE-G → CURE-GP → CURE-Full 开展 Beauty validation 实验，再依据 Go/No-Go 决定是否冻结 recipe。

## 2026-10-27 — CURE Accelerate 多卡并行与 W&B

- 对照 SFT trainer 后确认原 CURE 仅支持单卡。新增 `src/models/cure/distributed.py`，接入 Accelerate DDP、rank device、实际 loader 分片、`accumulate/backward/clip_grad_norm_`，只同步可训练 branch 梯度；每卡保留完整 frozen SFT。默认 backbone BF16 / residual FP32，也支持 `distributed.mixed_precision=bf16`。当前支持 DDP，明确拒绝 DeepSpeed/FSDP 配置，依赖最低 Accelerate 版本更新为 1.10。
- `train.batch_size` 明确为 per-process microbatch。多卡训练使用完整 global batch、舍弃不足一组的随机尾样本；GP/Full donor 在本卡 batch 内 non-self matching，batch 至少 2；单卡保留 singleton merge。scheduler 根据 prepare 后的 per-rank loader 计算 optimizer steps，每次实际更新只 step 一次；partial accumulation window 补偿 Accelerate 的固定除数。
- difficulty 首次计算分片并行、主进程原子保存。评估按 sample index stride 分片，无 padding / 重复 / 丢失；先汇总全局 latent 和 difficulty、在完整样本池统一 matching，再并行生成三分支，按 sample index 聚合指标。checkpoint、配置、JSON、provenance hashing 和 W&B 初始化均由 global rank 0 执行；rank-0 文件操作错误会广播，避免其他 rank 等待不存在的结果。
- 配置增加 W&B enable/project/entity/run_name/mode/group/tags，默认关闭。启用后记录完整配置、runtime world size/effective batch、全 rank 平均 loss 分项、gate/correction、学习率和配对评估指标；online/offline 均每次训练一个 run。避免 Accelerate offline `store_init_configuration` 重启空 run 的行为，改为在 `wandb.init` kwargs 中一次传入配置。独立 eval CLI 也支持多卡、可选 tracking 与异常收尾。
- 验证：单卡 invariant、CPU 两进程 DDP、prefix tree、metrics 共 34 passed / 1 real-SFT opt-in skipped。独立双 RTX 3090 NCCL + BF16 测试 2 passed（Residual-CE / CURE-Full），逐次检查梯度和参数同步、backbone 冻结、训练样本分片、partial accumulation、主进程保存、单卡/多卡 global donor 与指标一致、cache 一致和单个 offline W&B run 的实际指标。
- 正式 `accelerate launch` CLI 另外用真实 Beauty frozen SFT，在 16 条 train / 5 条 validation 上完成 CURE-Full 一步双 GPU BF16 smoke，含并行 cache、checkpoint、一个 offline W&B run；独立双 GPU eval CLI 再加载该 checkpoint，sample IDs、donor IDs、三分支 HR/NDCG 与 RG/PG/FD 和训练时评估完全一致。未访问 Beauty test/Sports，未正式训练或调参；不据 smoke 判断性能增益。
- 修复本机自动选 MPI 失败和普通单进程不应强制初始化 process group 的问题：仅 launcher 环境显式使用 GPU NCCL / CPU Gloo。测试依赖与真实模型 smoke 均位于 `/tmp`，完成后清理；不改系统 Python，不上传 W&B。使用方式已补入 `docs/cure_usage.md`。

## 2026-10-27 — CURE DeepSpeed / ZeRO 与低显存执行

- 用户要求补齐 DeepSpeed/ZeRO。`configs/cure.yaml` 增加 `distributed.use_deepspeed`、`zero3_init` 及完整 DeepSpeed 配置；新增 `configs/cure_zero2.yaml` 和 `configs/cure_zero3.yaml`。后者默认 ZeRO-3 + CPU parameter offload，optimizer offload 可独立配置。默认普通配置仍用 DDP。依赖最低版本对齐已测试的 Accelerate 1.10.1 / DeepSpeed 0.16.9。
- `DeepSpeedPlugin` 根据 YAML 构造，batch/GAS/clipping 一致性在初始化前校验；DeepSpeed launcher 未显式开启 YAML 后端会报错。ZeRO-3 启用 Hugging Face 分片加载；新 reasoner / residual 支持 BF16 参数类型，并保留 FP32 candidate log-softmax、KL 与 gate sigmoid。loss 在模型 forward 内返回 Tensor 字典，满足 ZeRO backward hooks；DeepSpeed 的 backward 管理同步边界、clip 与 optimizer step，不使用不兼容的 no_sync，也不重复更新。scheduler 和 partial accumulation window 继续正确计步。
- frozen SID LM-head rows 在初始化时缓存为小型 nonpersistent basis buffer，避免在评分时从 sharded full LM head 直接取行；MHA functional out_proj 注册 external parameters。Residual 线性计算保留 batch 维度，修复 DeepSpeed memory-efficient linear 对 1D backward 的限制。保存 checkpoint 时所有 rank 只 gather 新增 branch，主进程写入；加载恢复 branch dtype，不聚合完整 frozen SFT 保存。
- ZeRO-3 cache / eval 按 rank stride 分片并补齐通信工作，各 rank 同步执行相同次数 forward；虚拟样本排除于 cache、global matching、指标和 sample IDs。生成使用 synced_gpus。DDP / ZeRO-1/2 无需该补齐；ZeRO-1/2 独立无 optimizer 推理直接使用复制模型，ZeRO-3 独立推理保持 sharding/offload。缓存身份增加 scoring engine/precision/config，避免误用其他后端的数值缓存。
- 验证：既有单卡/DDP/评分/metrics 与新配置保护合计常规测试 36 passed / 4 opt-in skipped；随后新增 launcher/YAML 冲突保护测试，配置单测 3 passed。CUDA ZeRO 套件单独 5 passed（2 配置单测 + ZeRO-2、ZeRO-3、ZeRO-3 CPU parameter/optimizer offload 3 个双 GPU 集成场景）；每个场景检查实际 engine update 次数含 partial window、非零 global grad norm、baseline 不变、rank 间参数一致、完整 branch 序列化及 5 条奇数 validation 样本无重复。ZeRO-3 各 rank frozen 参数 shard 元素数为总数的一半。
- 正式 `accelerate launch --use_deepspeed` 入口另用真实 Beauty frozen SFT 完成 8 train / 3 validation、一步 CURE-Full smoke；ZeRO-3 独立 eval CLI 载入 checkpoint 后，sample IDs、donor IDs、三分支指标与训练内评估完全一致。ZeRO-2 独立 eval CLI 也通过。没有读取 Beauty test / Sports 或开展正式方法实验。
- 真实显存对照：两张 RTX 3090、microbatch=2、GAS=2、beam=10、相同 8 train / 3 validation。从 prepare 后重置 CUDA peak，计 difficulty cache + train + validation：DDP 每卡峰值 allocated 3.410–3.418 GiB；ZeRO-3 + CPU parameter offload 每卡 0.916 GiB。不含 driver/NCCL 额外分配，不代表默认 batch/beam 的上限；CPU offload 明显增加传输开销。具体 metadata 与启动说明已写入 `docs/cure_usage.md`。
- 实现依据核对官方 Accelerate DeepSpeed guide 与 DeepSpeed ZeRO external-parameter / GatheredParameters 文档。测试依赖、模型、缓存及输出在 `/tmp` 创建并于收尾清理；不改系统 Python、不上传 W&B。本阶段追加记录，未读取历史 develop log。

## 2026-10-27 — CURE 公共配置命名统一

- 对照 `configs/sft.yaml` / `configs/eval.yaml`，统一 CURE：`train.batch_size → train.per_device_batch_size`、`train.log_steps → train.wandb_log_steps`、`train.cache_batch_size → train.cache_per_device_batch_size`；frozen SFT/tokenizer 路径改为 `model.llm.sft_model_path` / `model.llm.tokenizer_path`，并暴露与 SFT 同名的 `model.llm.local_files_only`。训练、DeepSpeed microbatch 插值、W&B runtime config、测试 worker 和文档同步使用新名称。
- 评估改为 `eval.model_path`、`eval.output_dir` + `eval.output_name`，新增显式 `eval.random_seed`（默认继承 train.seed）以及默认继承训练设置的 eval history/sequence 长度。验证/测试路径改为实际读取与 SFT 同名的 `data.valid_path` / `data.test_path`。独立评估与训练内 validation 都使用 eval.random_seed 做评估抽样和 donor matching；train seed 的职责不变。
- `configs/cure.yaml` 增加中文注释，说明单位、默认行为、有效 batch、optimizer update cadence、loss margin、分桶、gate 初始化、精度层次和 ZeRO/offload 配置。CURE 独有概念保留名称，不与语义不同的 SFT 参数混同。SFT/eval 仅补公共参数注释，未改默认值或其训练/评估行为。
- 旧 CLI 参数名不保留重复别名，迁移表写入 `docs/cure_usage.md`。旧 checkpoint 的参数格式不变；评估预处理核对优先读取保存的 eval 配置，历史 metadata 缺少对应字段时回退 train 配置，因此旧 checkpoint 无需重训。
- 验证：实际 Hydra CLI 覆盖与 cure/cure_zero2/cure_zero3 合并插值通过；常规 40 passed / 4 opt-in skipped（包括实际评估 CLI 使用新 model/output/seed 参数、eval seed 独立性、自定义 validation 路径和旧 checkpoint metadata 兼容测试）。独立 CUDA ZeRO suite 6 passed，其中 3 种双 GPU 场景覆盖 ZeRO-2 / ZeRO-3 / CPU parameter+optimizer offload。语法编译与 git diff --check 通过，没有正式训练或研究 test 访问。
- 临时验证依赖和 Triton 缓存置于 /tmp 并在完成后清理；未读取历史 develop log。


## 2026-10-27 — CURE / SFT step validation 与 early stopping

- 两个训练入口共用 `ValidationControl`：默认每 500 次 optimizer update 验证，距上次超过绝对 min_delta=0.0001 的提升达到 2500 次更新后，在 validation 时停止；epoch 结束补验证并避免同 step 重复。支持关闭早停或仅 epoch 验证，epochs/max_steps 仍作为上限。
- CURE 默认监控 correct NDCG@10/max，保留原四项 best checkpoint selection；SFT 监控 validation loss/min。min_delta 只影响 patience，不阻止保存更优 checkpoint。早停状态由 rank 0 广播；两者写 training_summary.json，记录实际步数、停止原因与 best checkpoint step，W&B 记录 early_stop/*。
- SFT 默认固定抽样 1000 条 validation，可用 eval.sample_size=null 改为全量；验证采用全局 target-token 加权 loss，去除分布式 sampler padding 的重复样本。SFT 所有 rank 共同保存并恢复 best_ckpt，再导出 merged，修复此前导出最后一步权重的问题；scheduler 与 step 计数按实际 optimizer update，支持 partial accumulation。
- 验证并修复 SFT ZeRO-3 的 PEFT modules_to_save hook 属性转发兼容问题；ZeRO-3 最优 checkpoint 在 CPU 重建完整模型后合并，避免全量模型聚合到 GPU。该导出需要完整模型所需主存。
- 验证：early-stop/SFT/CURE 单卡组合 25 passed、1 skipped（真实 SFT opt-in）；CURE 双进程 CPU DDP 两个变体、双 GPU ZeRO-2/3/CPU parameter+optimizer offload 及配置检查均通过；SFT 最终回归 10 passed，包含真实 tiny LoRA 更新、最优导出权重比对，以及双进程 DDP/ZeRO-2/3 的中途验证和同步早停。compileall 与 git diff --check 通过。
- 全部新增验证使用合成小模型/数据；未运行正式 Beauty 方法训练、test 或 Sports confirmation。配置和说明见 configs/cure.yaml、configs/sft.yaml、docs/cure_usage.md。


## 2026-10-27 — CURE 批量候选评分与固定子集开发流程

- 诊断原运行：ZeRO-3 + CPU parameter offload，每卡 batch=128、accumulation=4、有效 batch=1024，平均约 9.07 秒/update；原 residual 对每个样本三个 stage 分别调用小网络，并在候选/target 查找中反复触发 GPU 同步。
- 新增 PackedCandidates：合法候选 metadata 在 CPU 缓存/组装后整体传输到 GPU，训练每个分支一次批量 MLP/gate/SID projection；不物化 [B,3,K,H] embedding 副本。保留 full-vocab frozen LM-head、相同前缀合法 children 与候选归一化，ragged debug API 懒加载，branch 参数名/结构未变。
- beam 内 residual 同样批量化，safe KL 对 padding 做 mask，避免 -inf 相减。训练新增 baseline_rec 对照与排除 validation 耗时的 seconds_per_update 日志。
- 新增 configs/cure_dev.yaml：ZeRO-3，无 CPU parameter/optimizer offload；固定 train=32768、validation=200、seed=42、beam=10，3 epoch，默认 microbatch=128、accumulation=1。使用说明提供 microbatch 64/128 × lr 0.0001/0.0003 四组相同曝光量对照、结果判读及恢复全量配置命令。没有擅自执行完整四组方法试验或更改原 cure.yaml 参数。
- 验证：CURE/前缀树/指标/DDP/ZeRO-2/ZeRO-3/CPU parameter+optimizer offload 组合 47 passed、1 real-SFT opt-in skipped；另外真实 Beauty SFT invariant 测试 1 passed，关闭 correction 的 Top-10 精确恢复、target 隔离与 checkpoint round-trip 均通过。新增 residual_ce/full 与逐样本参考的 loss/输出/梯度数值对照，覆盖 ragged padding。
- 性能诊断：真实 frozen SFT、单卡、同一 microbatch=128，分支前向+反向中位耗时由参考 0.498 秒降为 0.00353 秒；frozen prepare 约 0.459 秒。真实双卡 ZeRO-3/无 offload CLI 四步 smoke，microbatch=128、accumulation=1，预热后 0.59–0.61 秒/update，每卡峰值 CUDA allocated 2.79 GiB，包含四条 validation/beam=10，不含 driver/NCCL。step 的有效 batch 改变，不能按 step 耗时直接声称整体 15 倍加速。
- 本次实际训练仅四步入口 smoke；未运行正式 train/validation 四组搜索、test 或 Sports，未宣称推荐性能提升。compileall 和 git diff --check 通过；专用临时基准、smoke artifacts 清理。


## 2026-10-27 — Beauty Residual-CE 四组开发 sweep 结果复核

- 读取用户已完成的 cure_dev_sweep 四组配置、三次 validation、best/summary 和日志；核对模型/数据/protocol/sample IDs/donor assignment 一致，donor 均非自身。没有运行新训练或额外评估，没有访问 test/Sports。
- lr=0.0003 的 b64/b128 均为 HR@10=0.020（4/200），baseline=0.015（3/200）；NDCG@10=0.011368528/0.011215300，baseline=0.008730707。lr=0.0001 两组为 HR@10=0.015、NDCG@10=0.009166667。best 均在 epoch 3。
- 全部 12 次验证的 correct/matched HR/NDCG 均一致，最优两组 personalized score gap 为约 -1.05e-5/-4.63e-6；尚未展现额外 Z 的个性化增益。聚合结果不能证明逐样本预测相同，也不能把 h 路径内的用户信息解释成完全非个性化。
- 最近一次完整运行的稳定训练吞吐：b64 约 329 samples/s、b128 约 422 samples/s。每组三 epoch 训练约 5.0/3.9 分钟；每次 validation 约 6 分钟，三次合计占 trainer 阶段约 78–82%，validation 逐样本三分支 beam generation 已成为主要耗时。
- 建议保留两个 lr=0.0003 checkpoint，优先优化 validation，再扩大 validation、检查 Z 敏感性及多 seed；尚不据 200 条单 seed 结果直接扩为全量长训练。完整记录见 docs/cure_dev_sweep_analysis.md。


## 2026-10-28 — 显式允许 CURE 原输出目录覆盖重训

- 新增 train.overwrite_output_dir=false；设为 true 时，由主进程删除本 run 的 best.pt/last.pt、旧 valid_step/valid_epoch JSON、best_validation.json 和 training_summary.json，重写 config；训练 CLI 重写本次日志。其他文件、W&B 历史目录与外部 difficulty cache 保留。
- 默认拒绝覆盖 checkpoint 或验证产物，并在报错中提示 train.overwrite_output_dir=true。覆盖代表重新训练，不恢复 optimizer/scheduler；显式从同目录 init_checkpoint 初始化时，所有 rank 先加载权重并同步，再清理旧结果。
- 验证：CURE 单卡组合 22 passed、1 real-SFT opt-in skipped；覆盖测试检查默认拒绝不删除、仅清理训练产物、保留其他文件以及先加载初始化权重。CPU 测试显式关闭 DeepSpeed，以兼容用户当前启用 DeepSpeed 的主配置。Hydra --cfg job 确认直接识别参数，compileall/git diff --check 通过。
- 没有覆盖或删除用户现有训练结果；本次仅添加可由用户显式启用的能力。
