"""SID 前缀树和 ConstrainedLogitsProcessor。

约束 LLM 在生成 SID 时只能选择真实存在的 token 序列。
"""

import torch
from transformers import LogitsProcessor


class SidPrefixTree:
    """从 item.index.json 构建的 SID token 前缀树（trie）。

    用 dict 嵌套表示：
        {
            5: {
                12: {100: {}, 200: {}},
                99: {},
            },
            7: {
                12: {},
                50: {80: {}},
            },
        }
    """

    def __init__(self, item_sid_ids: dict[int, list[int]]):
        self._root: dict = {}
        for token_ids in item_sid_ids.values():
            node = self._root
            for tid in token_ids:
                if tid not in node:
                    node[tid] = {}
                node = node[tid]

    def get_allowed_tokens(self, prefix: tuple[int, ...]) -> set[int]:
        """给定 SID token 前缀，返回下一步合法的 token id 集合。

        prefix=()  → 第一层所有出现过的 token
        prefix=(5,) → 在 token 5 之后第二层出现过的 token
        prefix 已到叶子或不在 trie 中 → 空集
        """
        node = self._root
        for tid in prefix:
            if tid not in node:
                return set()
            node = node[tid]
        return set(node.keys())


class ConstrainedLogitsProcessor(LogitsProcessor):
    """根据 SID 前缀树，把非法 token 的 logit 设为 -inf。

    每个 beam 已生成的 SID 部分 = input_ids[i, prompt_lengths[i]:]
    根据该前缀查 trie，得到合法 token 集合，把其他位置 mask 掉。
    """

    NEG_INF = float("-inf")

    def __init__(
        self,
        prefix_tree: SidPrefixTree,
        prompt_lengths: torch.Tensor,
        num_sid_layers: int = 3,
        pad_token_id: int = 0,
    ):
        """
        Args:
            prefix_tree: 已构建的 SID 前缀树
            prompt_lengths: shape [batch_size]，每个样本 prompt 长度
                注意：beam search 时同一个样本的 num_beams 个 beam 共享
                同一个 prompt_length，所以传入时只需 batch 维度。
            num_sid_layers: SID 总层数，用于推断"是否已到 SID 末尾"
            pad_token_id: pad token 的 id，过滤生成序列中的 pad（默认 0，向后兼容）
        """
        self.tree = prefix_tree
        self.prompt_lengths = prompt_lengths
        self.num_sid_layers = num_sid_layers
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        input_ids: [batch_size * num_beams, seq_len]
        scores:    [batch_size * num_beams, vocab_size]
        """
        bsz_x_beams, _ = input_ids.shape
        batch_size = self.prompt_lengths.size(0)
        num_beams = bsz_x_beams // batch_size

        mask = torch.full_like(scores, self.NEG_INF)

        for i in range(bsz_x_beams):
            sample_idx = i // num_beams
            prompt_len = self.prompt_lengths[sample_idx].item()
            generated = input_ids[i, prompt_len:].tolist()
            # 过滤掉所有 pad token，得到实际生成的 SID token
            generated = [t for t in generated if t != self.pad_token_id]
            allowed = self.tree.get_allowed_tokens(tuple(generated))
            if not allowed:
                # 已到叶子或路径异常 —— 退化为允许所有（避免全 -inf 卡死）
                mask[i] = scores[i]
            else:
                allowed_idx = torch.tensor(list(allowed), dtype=torch.long, device=scores.device)
                mask[i].scatter_(0, allowed_idx, scores[i].gather(0, allowed_idx))

        return mask
