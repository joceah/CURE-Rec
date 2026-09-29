"""增强多样性的beam search：通过后处理和采样提升候选多样性"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from inference.prefix_tree import ConstrainedLogitsProcessor, SidPrefixTree


@torch.no_grad()
def generate_topk_sids_diverse(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[list[int]],
    prefix_tree: SidPrefixTree,
    num_beams: int = 50,
    num_sid_layers: int = 3,
    device: torch.device | str = "cuda",
    temperature: float = 1.2,
    top_p: float = 0.95,
    **kwargs,  # 忽略不支持的参数
) -> list[list[tuple[int, ...]]]:
    """使用更高temperature的beam search增加多样性

    策略：temperature > 1会flatten概率分布，让更多候选有机会
    """
    batch_size = len(prompts)
    pad_id = tokenizer.pad_token_id
    max_len = max(len(p) for p in prompts)

    # Left padding
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    prompt_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, p in enumerate(prompts):
        plen = len(p)
        input_ids[i, max_len - plen:] = torch.tensor(p, dtype=torch.long)
        attention_mask[i, max_len - plen:] = 1
        prompt_lengths[i] = max_len

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    processor = ConstrainedLogitsProcessor(
        prefix_tree=prefix_tree,
        prompt_lengths=prompt_lengths.to(device),
        num_sid_layers=num_sid_layers,
        pad_token_id=pad_id,
    )

    # Beam search with temperature
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
        temperature=temperature,
        top_p=top_p,
    )

    generated = outputs[:, max_len:]
    generated = generated.view(batch_size, num_beams, num_sid_layers)

    result: list[list[tuple[int, ...]]] = []
    for b in range(batch_size):
        topk = [tuple(generated[b, k].tolist()) for k in range(num_beams)]
        result.append(topk)
    return result


@torch.no_grad()
def generate_topk_sids_sampling(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[list[int]],
    prefix_tree: SidPrefixTree,
    num_samples: int = 50,
    num_sid_layers: int = 3,
    device: torch.device | str = "cuda",
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    **kwargs,
) -> list[list[tuple[int, ...]]]:
    """使用采样生成多样化候选"""
    batch_size = len(prompts)
    pad_id = tokenizer.pad_token_id
    max_len = max(len(p) for p in prompts)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    prompt_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, p in enumerate(prompts):
        plen = len(p)
        input_ids[i, max_len - plen:] = torch.tensor(p, dtype=torch.long)
        attention_mask[i, max_len - plen:] = 1
        prompt_lengths[i] = max_len

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    processor = ConstrainedLogitsProcessor(
        prefix_tree=prefix_tree,
        prompt_lengths=prompt_lengths.to(device),
        num_sid_layers=num_sid_layers,
        pad_token_id=pad_id,
    )

    # Sampling-based generation
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=num_sid_layers,
        min_new_tokens=num_sid_layers,
        num_return_sequences=num_samples,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        logits_processor=[processor],
        pad_token_id=pad_id,
    )

    generated = outputs[:, max_len:]
    generated = generated.view(batch_size, num_samples, num_sid_layers)

    result: list[list[tuple[int, ...]]] = []
    for b in range(batch_size):
        topk = [tuple(generated[b, k].tolist()) for k in range(num_samples)]
        result.append(topk)
    return result
