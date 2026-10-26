"""SFT 模型评估高层 API：加载模型 + 推理 + 计算指标。"""

import csv
import json
import logging
import random
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.beam_search import generate_topk_sids
from inference.beam_search_diverse import generate_topk_sids_diverse, generate_topk_sids_sampling
from inference.metrics import compute_metrics
from inference.prefix_tree import SidPrefixTree

logger = logging.getLogger(__name__)


def _load_test_samples(
    test_csv_path: str,
    user_sequences_path: str,
    item_sid_ids: dict[int, list[int]],
    max_history_len: int,
    sample_size: int | None,
    random_seed: int,
    valid_csv_path: str | None = None,
) -> list[dict]:
    """读取评估集并构造样本。

    Leave-two-out 协议：
    - validation: valid_csv_path=None，history=train
    - test: valid_csv_path=valid.csv，history=train+valid
    """
    logger.info(f"读取评估集: {test_csv_path}")
    with open(user_sequences_path, "r", encoding="utf-8") as f:
        raw_sequences = json.load(f)
    user_sequences = {int(k): v for k, v in raw_sequences.items()}

    valid_targets: dict[int, int] = {}
    if valid_csv_path is not None:
        logger.info(f"读取 validation interaction 作为 test history: {valid_csv_path}")
        with open(valid_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                valid_targets[int(row["user_id"])] = int(row["item_id"])

    samples = []
    with open(test_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = int(row["user_id"])
            target_item = int(row["item_id"])

            if user_id not in user_sequences or len(user_sequences[user_id]) == 0:
                continue
            if target_item not in item_sid_ids:
                continue

            history = list(user_sequences[user_id])
            if user_id in valid_targets:
                history.append(valid_targets[user_id])
            history = history[-max_history_len:]

            samples.append({
                "user_id": user_id,
                "history": history,
                "target_sid": tuple(item_sid_ids[target_item]),
            })

    logger.info(f"评估样本总数: {len(samples)}")
    if sample_size is not None and sample_size < len(samples):
        random.seed(random_seed)
        samples = random.sample(samples, sample_size)
        logger.info(f"采样后样本数: {len(samples)}")
    return samples


def _build_prompt(
    history: list[int],
    item_sid_ids: dict[int, list[int]],
    tokenizer: AutoTokenizer,
) -> list[int]:
    """构造与 SFT 训练一致的 prompt。"""
    prefix_ids = tokenizer.encode("用户历史：", add_special_tokens=False)
    history_ids = []
    for iid in history:
        if iid in item_sid_ids:
            history_ids.extend(item_sid_ids[iid])
    suffix_ids = tokenizer.encode(" 请推荐下一个商品：", add_special_tokens=False)
    return prefix_ids + history_ids + suffix_ids


def evaluate(
    model_path: str,
    tokenizer_path: str,
    item_index_path: str,
    user_sequences_path: str,
    test_csv_path: str,
    valid_csv_path: str | None = None,
    max_history_len: int = 20,
    num_beams: int = 50,
    batch_size: int = 8,
    num_sid_layers: int = 3,
    sample_size: int | None = None,
    random_seed: int = 42,
    k_values: list[int] = None,
    output_path: str = "results.json",
    decode_strategy: str = "beam_search",
    diversity_penalty: float = 0.5,
    num_beam_groups: int = 5,
    temperature: float = 1.0,
) -> dict[str, float]:
    """主评估函数。

    流程：
    1. Accelerator 初始化
    2. 加载 model + tokenizer（fp16/bf16 + eval 模式）
    3. 加载 item.index.json → 构建 prefix_tree
    4. 读 test.csv 和 user_sequences.json，构造样本列表
    5. 按 process_index 切片 + 批量 inference
    6. accelerator.gather_for_metrics 收集所有结果
    7. 主进程算指标，写入 results.json

    Args:
        model_path: 合并后的 SFT 模型路径
        tokenizer_path: tokenizer 路径
        item_index_path: item.index.json 路径
        user_sequences_path: user_sequences.json 路径
        test_csv_path: test.csv 路径
        max_history_len: 最多取最后几个历史商品
        num_beams: beam search 数量
        batch_size: 推理 batch size
        num_sid_layers: SID 层数
        sample_size: 采样数量（None 表示全量）
        random_seed: 随机种子
        k_values: 要计算的 K 值列表
        output_path: 输出 JSON 路径

    Returns:
        metrics 字典
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 50]

    start_time = time.time()

    # ── 1. Accelerator 初始化 ─────────────────────────────────────────────
    accelerator = Accelerator()
    logger.info(f"Accelerator 初始化完成: {accelerator.num_processes} 个进程")

    # ── 2. 加载模型 + tokenizer ───────────────────────────────────────────
    if accelerator.is_main_process:
        logger.info(f"加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning(f"tokenizer 没有 pad_token_id，使用 eos_token_id={tokenizer.eos_token_id}")

    if accelerator.is_main_process:
        logger.info(f"加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    model = model.to(accelerator.device)
    logger.info(f"模型已加载到 {accelerator.device}（进程 {accelerator.process_index}）")

    # ── 3. 加载 item.index.json → 构建 prefix_tree ────────────────────────
    if accelerator.is_main_process:
        logger.info(f"加载 item.index.json: {item_index_path}")
    with open(item_index_path, "r", encoding="utf-8") as f:
        item_index = json.load(f)
    # item_index: {"item_id_str": ["<a_5>", "<b_12>", "<c_100>"], ...}

    item_sid_ids: dict[int, list[int]] = {}
    for item_id_str, sid_tokens in item_index.items():
        token_ids = tokenizer.convert_tokens_to_ids(sid_tokens)
        item_sid_ids[int(item_id_str)] = token_ids

    prefix_tree = SidPrefixTree(item_sid_ids)
    logger.info(f"前缀树构建完成: {len(item_sid_ids)} 个商品")

    # ── 4. 读测试样本 ─────────────────────────────────────────────────────
    if accelerator.is_main_process:
        all_samples = _load_test_samples(
            test_csv_path=test_csv_path,
            valid_csv_path=valid_csv_path,
            user_sequences_path=user_sequences_path,
            item_sid_ids=item_sid_ids,
            max_history_len=max_history_len,
            sample_size=sample_size,
            random_seed=random_seed,
        )
    else:
        all_samples = None

    # 广播样本到所有进程（单进程时直接使用）
    if accelerator.num_processes > 1:
        if hasattr(accelerator, 'broadcast_object_list'):
            all_samples = accelerator.broadcast_object_list([all_samples])[0]
        else:
            # 降级方案：所有进程都加载一遍
            if not accelerator.is_main_process:
                all_samples = _load_test_samples(
                    test_csv_path=test_csv_path,
                    user_sequences_path=user_sequences_path,
                    item_sid_ids=item_sid_ids,
                    max_history_len=max_history_len,
                    sample_size=sample_size,
                    random_seed=random_seed,
                )

    # 按进程切片
    local_samples = all_samples[accelerator.process_index::accelerator.num_processes]
    logger.info(f"进程 {accelerator.process_index} 处理 {len(local_samples)} 个样本")

    # ── 5. 批量推理 ───────────────────────────────────────────────────────
    local_predictions = []
    local_targets = []

    num_batches = (len(local_samples) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_samples = local_samples[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # 构造 prompts
        prompts = [
            _build_prompt(s["history"], item_sid_ids, tokenizer)
            for s in batch_samples
        ]

        # 选择推理策略
        if decode_strategy == "diverse_beam":
            topk_sids = generate_topk_sids_diverse(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                prefix_tree=prefix_tree,
                num_beams=num_beams,
                num_sid_layers=num_sid_layers,
                device=accelerator.device,
                diversity_penalty=diversity_penalty,
                num_beam_groups=num_beam_groups,
                temperature=temperature,
            )
        elif decode_strategy == "sampling":
            topk_sids = generate_topk_sids_sampling(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                prefix_tree=prefix_tree,
                num_samples=num_beams,
                num_sid_layers=num_sid_layers,
                device=accelerator.device,
                temperature=temperature,
            )
        else:  # default: beam_search
            topk_sids = generate_topk_sids(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                prefix_tree=prefix_tree,
                num_beams=num_beams,
                num_sid_layers=num_sid_layers,
                device=accelerator.device,
            )

        # 收集结果
        for i, s in enumerate(batch_samples):
            local_predictions.append(topk_sids[i])
            local_targets.append(s["target_sid"])

        # 每 50 batch 打印进度
        if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == num_batches:
            logger.info(
                f"进程 {accelerator.process_index}: "
                f"batch {batch_idx + 1}/{num_batches} 完成"
            )

    # ── 6. gather_for_metrics 收集结果 ────────────────────────────────────
    logger.info(f"进程 {accelerator.process_index} 推理完成，准备 gather")
    gathered_predictions = gather_object(local_predictions)
    gathered_targets = gather_object(local_targets)

    # ── 7. 主进程算指标 + 保存 ───────────────────────────────────────────
    if accelerator.is_main_process:
        # gathered_predictions/targets 在单进程时直接是数据，多进程时是嵌套列表
        # 单进程：直接使用
        if accelerator.num_processes == 1:
            all_predictions = list(gathered_predictions)
            all_targets = list(gathered_targets)
        else:
            # 多进程：拉平 list[list[...]]
            all_predictions = sum(list(gathered_predictions), [])
            all_targets = sum(list(gathered_targets), [])

        logger.info(f"收集到 {len(all_predictions)} 个预测结果")

        # 计算指标
        metrics = compute_metrics(all_predictions, all_targets, k_values)

        elapsed = time.time() - start_time

        # 构造输出
        result = {
            "config": {
                "model_path": model_path,
                "valid_csv_path": valid_csv_path,
                "sample_size": sample_size if sample_size else len(all_predictions),
                "num_beams": num_beams,
                "max_history_len": max_history_len,
                "batch_size": batch_size,
                "num_sid_layers": num_sid_layers,
                "random_seed": random_seed,
            },
            "metrics": metrics,
            "n_samples": len(all_predictions),
            "elapsed_seconds": round(elapsed, 2),
        }

        # 保存
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存到: {output_path}")

        # 打印指标
        logger.info("=" * 60)
        logger.info("评估结果:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info(f"总样本数: {len(all_predictions)}")
        logger.info(f"总耗时: {elapsed:.2f} 秒")
        logger.info("=" * 60)

        return metrics
    else:
        return {}
