"""扩展 tokenizer，添加 RQ-VAE 生成的 SID special tokens。"""

import json
from pathlib import Path

from transformers import AutoTokenizer


LAYER_PREFIXES = ["a", "b", "c", "d"]


def build_sid_tokens(num_layers: int, codebook_size: int) -> list[str]:
    """生成所有 SID special tokens，如 <a_0>, <b_255> 等。"""
    tokens = []
    for layer in range(num_layers):
        prefix = LAYER_PREFIXES[layer]
        for code in range(codebook_size):
            tokens.append(f"<{prefix}_{code}>")
    return tokens


def load_tokenizer(
    model_name: str,
    num_layers: int,
    codebook_size: int,
    save_dir: str | None = None,
    tokenizer_path: str | None = None,
) -> AutoTokenizer:
    """加载并扩展 tokenizer。

    Args:
        model_name: 基础模型路径或名称
        num_layers: RQ-VAE 层数
        codebook_size: 每层码本大小
        save_dir: 扩展后 tokenizer 的保存路径，None 则不保存
        tokenizer_path: 已扩展的 tokenizer 路径，如提供则直接加载（优先级高于 model_name）
    """
    # 如果提供了 tokenizer_path，直接加载
    if tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    # 否则从 base model 动态扩展
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sid_tokens = build_sid_tokens(num_layers, codebook_size)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": sid_tokens})

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(save_dir)

    return tokenizer


def get_item_sid_token_ids(
    tokenizer: AutoTokenizer,
    item_index_path: str,
) -> dict[int, list[int]]:
    """返回 item_id -> SID token id 列表的映射。"""
    with open(item_index_path) as f:
        item_index = json.load(f)

    result = {}
    for item_id_str, sid_tokens in item_index.items():
        token_ids = tokenizer.convert_tokens_to_ids(sid_tokens)
        result[int(item_id_str)] = token_ids
    return result
