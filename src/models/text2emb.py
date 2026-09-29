"""
Stage 2 Step 1：商品文本 → embedding 向量。

使用 Qwen2.5-0.5B（decoder-only）做 mean pooling，支持双卡 accelerate 并行。
输出：item_embeddings.npy，shape = (num_items, embedding_dim)，行序与 item_id 对齐。
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def build_item_texts(item_meta: dict, text_fields: list[str], separator: str = " ") -> tuple[list[int], list[str]]:
    """
    按 text_fields 顺序拼接商品文本。

    Returns:
        item_ids: 排序后的 item id 列表（int）
        texts:    对应的文本列表
    """
    item_ids = sorted(int(k) for k in item_meta.keys())
    texts = []
    for iid in item_ids:
        meta = item_meta[str(iid)]
        parts = [meta.get(f, "").strip() for f in text_fields]
        parts = [p for p in parts if p]  # 过滤空字段
        texts.append(separator.join(parts) if parts else "unknown")
    return item_ids, texts


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Decoder-only 模型没有 [CLS]，用 attention_mask 内的 mean pooling。"""
    mask = attention_mask.unsqueeze(-1).float()
    sum_emb = (last_hidden_state * mask).sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    return sum_emb / sum_mask


@torch.no_grad()
def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    accelerator: Accelerator,
) -> np.ndarray:
    """
    用 accelerate 多卡并行编码文本列表。
    每张卡处理 texts[process_index::num_processes] 的子集，最后 gather 合并。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = accelerator.prepare(model)
    model.eval()

    # 每张卡只处理自己的分片
    local_texts = texts[accelerator.process_index::accelerator.num_processes]
    logger.info(
        f"进程 {accelerator.process_index}/{accelerator.num_processes}: "
        f"处理 {len(local_texts)} 条文本"
    )

    local_embeddings = []
    for i in range(0, len(local_texts), batch_size):
        batch = local_texts[i: i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(accelerator.device)

        outputs = model(**encoded)
        emb = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
        local_embeddings.append(emb.float())  # 保持在 GPU，gather 需要

        if (i // batch_size) % 10 == 0:
            logger.info(f"  进程 {accelerator.process_index}: {i + len(batch)}/{len(local_texts)}")

    local_embeddings = torch.cat(local_embeddings, dim=0)  # (local_n, dim) GPU tensor

    # gather 所有进程的结果到主进程
    all_embeddings = accelerator.gather(local_embeddings)  # (total_n, dim) GPU tensor

    if accelerator.is_main_process:
        return all_embeddings.cpu().numpy()  # gather 后再转 CPU
    return None


def run_text2emb(cfg: DictConfig) -> None:
    """text2emb 主流程，由 Hydra 脚本调用。"""
    accelerator = Accelerator()

    if accelerator.is_main_process:
        logger.info(f"text_fields: {list(cfg.data.text_fields)}")
        logger.info(f"模型: {cfg.model.text2emb.model_name}")

    # 加载 item_meta
    with open(cfg.data.item_meta_path, "r") as f:
        item_meta = json.load(f)

    item_ids, texts = build_item_texts(
        item_meta,
        text_fields=list(cfg.data.text_fields),
        separator=cfg.data.field_separator,
    )

    if accelerator.is_main_process:
        logger.info(f"共 {len(texts)} 个商品需要编码")
        # 打印前 2 条示例
        for i in range(min(2, len(texts))):
            logger.info(f"  item {item_ids[i]}: {texts[i][:100]}")

    embeddings = encode_texts(
        texts=texts,
        model_name=cfg.model.text2emb.model_name,
        batch_size=cfg.model.text2emb.batch_size,
        max_length=cfg.model.text2emb.max_length,
        accelerator=accelerator,
    )

    if accelerator.is_main_process:
        out_path = Path(cfg.model.text2emb.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, embeddings)
        logger.info(f"embeddings 已保存: {out_path}, shape={embeddings.shape}")

        # 同时保存 item_id 顺序，确保后续 RQ-VAE 能对齐
        id_path = out_path.with_name("item_ids_order.json")
        with open(id_path, "w") as f:
            json.dump(item_ids, f)
        logger.info(f"item_ids_order.json 已保存: {id_path}")
