"""Beam search 推理：调用 model.generate() 生成 Top-K SID。"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from inference.prefix_tree import ConstrainedLogitsProcessor, SidPrefixTree


@torch.no_grad()
def generate_topk_sids(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[list[int]],
    prefix_tree: SidPrefixTree,
    num_beams: int = 50,
    num_sid_layers: int = 3,
    device: torch.device | str = "cuda",
) -> list[list[tuple[int, ...]]]:
    """对一批 prompt 做 beam search，每个 prompt 返回 Top-num_beams 个 SID。

    Args:
        model: 已 eval() 的 SFT 模型
        tokenizer: 扩展后的 tokenizer
        prompts: batch 个 prompt（每个是 token id 列表）
        prefix_tree: SID 前缀树（约束解码）
        num_beams: beam 数量（也是返回的 Top-K 数）
        num_sid_layers: SID 层数（决定生成长度）
        device: 推理设备

    Returns:
        长度为 batch 的列表。每项是该 prompt 的 Top-num_beams 个 SID
        （每个 SID 是 token id 的 tuple）。
    """
    batch_size = len(prompts)
    pad_id = tokenizer.pad_token_id
    max_len = max(len(p) for p in prompts)

    # left padding（对 generation 友好）
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    prompt_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, p in enumerate(prompts):
        plen = len(p)
        input_ids[i, max_len - plen:] = torch.tensor(p, dtype=torch.long)
        attention_mask[i, max_len - plen:] = 1
        prompt_lengths[i] = max_len  # left-pad 后所有 prompt 占满到 max_len

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    processor = ConstrainedLogitsProcessor(
        prefix_tree=prefix_tree,
        prompt_lengths=prompt_lengths.to(device),
        num_sid_layers=num_sid_layers,
        pad_token_id=pad_id,
    )

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=num_sid_layers,
        min_new_tokens=num_sid_layers,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        do_sample=False,
        early_stopping=False,
        logits_processor=[processor],
        pad_token_id=pad_id,
    )
    # outputs: [batch_size * num_beams, max_len + num_sid_layers]
    generated = outputs[:, max_len:]  # 切出 SID 部分
    generated = generated.view(batch_size, num_beams, num_sid_layers)

    result: list[list[tuple[int, ...]]] = []
    for b in range(batch_size):
        topk = [tuple(generated[b, k].tolist()) for k in range(num_beams)]
        result.append(topk)
    return result
