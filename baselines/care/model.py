"""CARE query reasoning and progressive history visibility on Qwen2.

The stage mask follows Lin et al., CARE, upstream commit
9af734f2b48a9643c5246ac68c1a9f16770bc2a8. This version uses the
public Qwen2Model attention-mask mapping supported by Transformers 4.57.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class CARE(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.query_list = list(getattr(config, "care_query_list", [1, 1, 1]))
        self.progressive_list = list(getattr(config, "care_progressive_list", [True] * len(self.query_list)))
        self.sid_layers = int(getattr(config, "care_sid_layers", len(self.query_list)))
        self.query_div_scale = float(getattr(config, "care_query_div_scale", 0.0))
        if len(self.query_list) != self.sid_layers or len(self.progressive_list) != self.sid_layers:
            raise ValueError("CARE query and progressive lists must match SID length")
        if any(count < 1 for count in self.query_list):
            raise ValueError("Each SID stage needs at least one query token")
        self.query_embeddings = nn.Embedding(sum(self.query_list), config.hidden_size)
        self.query_embeddings.reset_parameters()

    def _stage_inputs(self, input_ids, attention_mask, history_lengths, prefix_ids):
        """Append stage queries and previous gold/predicted SID codes."""
        batch_size, prompt_width = input_ids.shape
        n_stages = prefix_ids.shape[1] + 1
        if n_stages > self.sid_layers:
            raise ValueError("SID prefix is longer than the configured identifier")
        embed = self.model.embed_tokens
        parts = [embed(input_ids)]
        query_positions = []
        stage_positions = []
        cursor = prompt_width
        query_offset = 0
        for stage in range(n_stages):
            count = self.query_list[stage]
            query_ids = torch.arange(query_offset, query_offset + count, device=input_ids.device)
            parts.append(self.query_embeddings(query_ids).unsqueeze(0).expand(batch_size, -1, -1))
            query_positions.append(cursor + count - 1)
            stage_positions.append((cursor, count, stage < prefix_ids.shape[1]))
            cursor += count
            query_offset += count
            if stage < prefix_ids.shape[1]:
                parts.append(embed(prefix_ids[:, stage:stage + 1]))
                cursor += 1
        hidden_inputs = torch.cat(parts, dim=1)
        mask = self._stage_mask(attention_mask, history_lengths, stage_positions, hidden_inputs.dtype)
        return hidden_inputs, mask, query_positions

    def _stage_mask(self, prompt_mask, history_lengths, stage_positions, dtype):
        batch_size, prompt_width = prompt_mask.shape
        total_len = prompt_width + sum(count + int(has_code) for _, count, has_code in stage_positions)
        neg = torch.finfo(dtype).min
        mask = torch.full((batch_size, 1, total_len, total_len), neg,
                          dtype=dtype, device=prompt_mask.device)
        for row in range(batch_size):
            history_count = int(history_lengths[row])
            real_start = prompt_width - history_count * self.sid_layers - 1
            if real_start < 0 or prompt_mask[row, real_start:].sum() != prompt_width - real_start:
                raise ValueError("Prompt must be left padded SID history plus one answer token")
            for pos in range(prompt_width):
                mask[row, 0, pos, real_start:pos + 1] = 0 if pos >= real_start else neg
                if pos < real_start:
                    mask[row, 0, pos, pos] = 0
            for stage, (start, count, has_code) in enumerate(stage_positions):
                for offset in range(count):
                    pos = start + offset
                    if self.progressive_list[stage]:
                        for item in range(history_count):
                            first = real_start + item * self.sid_layers
                            mask[row, 0, pos, first:first + stage + 1] = 0
                        mask[row, 0, pos, prompt_width - 1] = 0
                    else:
                        mask[row, 0, pos, real_start:prompt_width] = 0
                    mask[row, 0, pos, prompt_width:pos + 1] = 0
                if has_code:
                    code_pos = start + count
                    mask[row, 0, code_pos, real_start:code_pos + 1] = 0
        return {"full_attention": mask}

    def _last_query_logits(self, input_ids, attention_mask, history_lengths, prefix_ids):
        embeds, mask, query_positions = self._stage_inputs(
            input_ids, attention_mask, history_lengths, prefix_ids
        )
        output = self.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False)
        return self.lm_head(output.last_hidden_state[:, query_positions, :])

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                history_lengths=None, **kwargs):
        if labels is None or history_lengths is None:
            raise ValueError("CARE training requires SID labels and history lengths")
        if labels.shape[1] != self.sid_layers:
            raise ValueError("Target SID length does not match CARE configuration")
        logits = self._last_query_logits(
            input_ids, attention_mask, history_lengths, labels[:, :-1]
        )
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        if self.query_div_scale and self.query_embeddings.num_embeddings > 1:
            queries = F.normalize(self.query_embeddings.weight.float(), dim=-1)
            similarities = queries @ queries.T
            off_diagonal = ~torch.eye(len(queries), device=queries.device, dtype=torch.bool)
            loss = loss + self.query_div_scale * similarities[off_diagonal].mean()
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def score_next(self, input_ids, attention_mask, history_lengths, prefix_ids):
        """Score one stage in the full training layout with masked future placeholders."""
        stage = prefix_ids.shape[1]
        if stage >= self.sid_layers:
            raise ValueError("Cannot score after the final SID stage")
        remaining = self.sid_layers - 1 - stage
        if remaining:
            placeholder = torch.zeros((len(prefix_ids), remaining),
                                      dtype=prefix_ids.dtype, device=prefix_ids.device)
            prefix_ids = torch.cat((prefix_ids, placeholder), dim=1)
        return self._last_query_logits(
            input_ids, attention_mask, history_lengths, prefix_ids
        )[:, stage, :]
