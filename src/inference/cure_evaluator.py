"""复用 SFT 的 leave-two-out 样本与 SID 前缀树评估 CURE。"""

import json
import logging
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoTokenizer

from inference.evaluator import _build_prompt, _load_test_samples
from inference.metrics import compute_metrics
from inference.prefix_tree import ConstrainedLogitsProcessor, SidPrefixTree
from models.cure.model import CUREForCausalLM
from models.llm.tokenizer import get_item_sid_token_ids

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_cure_topk(model, tokenizer, prompts, history_positions, prefix_tree,
                       num_beams, device, latent_mode="normal"):
    """一批 prompt 生成 Top-K SID。

    prompts 是 B 条变长 token 列表；history_positions 是每条样本的 [N_i, 3]。
    左填充后 ids/mask: [B, T]，positions: [B, N, 3]，latent z: [B, H]。
    beam 输出 SID: [B, num_beams, 3]；返回 CPU Python tuple 列表。
    """
    width = max(map(len, prompts))
    batch_size = len(prompts)
    ids = torch.full((batch_size, width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    max_items = max(map(len, history_positions))
    positions = torch.zeros((batch_size, max_items, 3), dtype=torch.long)
    item_mask = torch.zeros((batch_size, max_items), dtype=torch.bool)
    # 生成采用左填充，历史位置必须加上各样本的左侧 pad offset。
    for i, (prompt, pos) in enumerate(zip(prompts, history_positions)):
        offset = width - len(prompt)
        ids[i, offset:] = torch.tensor(prompt)
        mask[i, offset:] = 1
        positions[i, :len(pos)] = torch.tensor(pos) + offset
        item_mask[i, :len(pos)] = True
    ids, mask = ids.to(device), mask.to(device)
    # 只在生成前编码一次历史；GenerationMixin 将 z 扩展到各 beam。
    z = model.encode_latent(ids, mask, positions.to(device), item_mask.to(device))
    processor = ConstrainedLogitsProcessor(prefix_tree, torch.full((batch_size,), width,
                                          device=device), 3, tokenizer.pad_token_id)
    output = model.generate(input_ids=ids, attention_mask=mask, latent_z=z,
                            latent_mode=latent_mode, max_new_tokens=3, min_new_tokens=3,
                            num_beams=num_beams, num_return_sequences=num_beams,
                            do_sample=False, early_stopping=False,
                            logits_processor=[processor], pad_token_id=tokenizer.pad_token_id)
    # 切掉 prompt；这里只保留 3 个 SID token，不包含 EOS。
    generated = output[:, width:].reshape(batch_size, num_beams, 3)
    return [[tuple(sid.tolist()) for sid in row] for row in generated]


def evaluate_cure(cfg):
    """按 train + valid -> test 协议评估，并在主进程写 HR/NDCG。"""
    start = time.time()
    accelerator = Accelerator()
    logger.info("rank=%s 开始评估: checkpoint=%s, sample_size=%s, batch_size=%s, num_beams=%s",
                accelerator.process_index, cfg.eval.model_path, cfg.eval.sample_size,
                cfg.eval.batch_size, cfg.eval.num_beams)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.sft_model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    item_sids = get_item_sid_token_ids(tokenizer, cfg.data.item_index_path)
    tree = SidPrefixTree(item_sids)
    model = CUREForCausalLM.from_cure_pretrained(cfg.model.sft_model_path,
                                                  cfg.eval.model_path).to(accelerator.device).eval()
    samples = _load_test_samples(cfg.data.test_path, cfg.data.user_sequences_path,
                                 item_sids, cfg.train.max_history_len, cfg.eval.sample_size,
                                 cfg.train.seed, cfg.data.valid_path)
    # 每个 rank 处理互不重叠的样本；预测列表在 CPU，不累计 GPU logits。
    local = samples[accelerator.process_index::accelerator.num_processes]
    predictions, targets = [], []
    prefix_len = len(tokenizer.encode("用户历史：", add_special_tokens=False))
    total_batches = (len(local) + cfg.eval.batch_size - 1) // cfg.eval.batch_size
    for batch_idx, start_idx in enumerate(range(0, len(local), cfg.eval.batch_size), start=1):
        batch = local[start_idx:start_idx + cfg.eval.batch_size]
        prompts = [_build_prompt(s["history"], item_sids, tokenizer) for s in batch]
        positions = []
        for sample in batch:
            cursor = prefix_len
            pos = []
            for item in sample["history"]:
                if item in item_sids:
                    pos.append(list(range(cursor, cursor + 3)))
                    cursor += 3
            positions.append(pos)
        predictions.extend(generate_cure_topk(model, tokenizer, prompts, positions, tree,
                                              cfg.eval.num_beams, accelerator.device,
                                              cfg.eval.latent_mode))
        targets.extend(s["target_sid"] for s in batch)
        if batch_idx % 50 == 0 or batch_idx == total_batches:
            logger.info("rank=%s 评估进度: %s/%s batch", accelerator.process_index,
                        batch_idx, total_batches)
    # 合并各 rank 的 Python 列表，K 值仅在最终 CPU 指标计算中使用。
    all_predictions = gather_object(predictions)
    all_targets = gather_object(targets)
    if accelerator.is_main_process:
        metrics = compute_metrics(all_predictions, all_targets, list(cfg.eval.k_values))
        output = Path(cfg.eval.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"metrics": metrics, "n_samples": len(all_targets),
                                      "latent_mode": cfg.eval.latent_mode,
                                      "elapsed_seconds": round(time.time() - start, 2)}, indent=2))
        logger.info("评估结果已写入: %s (%s 样本, %.1f 秒)",
                    output.resolve(), len(all_targets), time.time() - start)
        return metrics
    return {}
