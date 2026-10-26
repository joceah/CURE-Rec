"""冻结 SFT Qwen，并让训练与生成共用同一条 latent fusion 路径。

记号：B 为 batch 大小，T 为 token 数，N 为历史商品数，H 为 Qwen hidden_size，
V 为词表大小。history_positions 指向每个历史商品的 3 个 SID token。
"""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .losses import cure_loss
from .reasoner import LatentReasoner


class CUREForCausalLM(PreTrainedModel, GenerationMixin):
    """冻结 backbone，只训练 reasoner、latent projection 和 fusion gate。"""

    _supports_sdpa = True  # attention 由内部 Qwen 实现；包装类沿用其 SDPA 支持。

    def __init__(self, backbone, reasoner_config=None):
        super().__init__(backbone.config)
        self.backbone = backbone
        config = dict(reasoner_config or {})
        if config.pop("num_latents", 1) != 1 or config.pop("num_layers", 1) != 1:
            raise ValueError("CURE v1 requires one latent and one attention layer")
        self.reasoner_config = config
        self.reasoner = LatentReasoner(backbone.config.hidden_size, **config)
        hidden = backbone.config.hidden_size
        self.latent_projection = nn.Linear(hidden, hidden)
        self.fusion_gate = nn.Linear(2 * hidden, hidden)
        # sigmoid(-3) 约为 0.047：初始接近 SFT，latent 路径仍可获得梯度。
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, -3.0)
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def train(self, mode=True):
        """训练新增模块时保持冻结的 backbone 为 eval 模式。"""
        super().train(mode)
        self.backbone.eval()
        return self

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def get_output_embeddings(self):
        return self.backbone.get_output_embeddings()

    def _history_latent(self, hidden, history_positions, history_item_mask):
        """从历史 SID hidden state 得到 z: [B, H]。

        B 是 batch 大小，H 是 Qwen hidden_size。
        T 是当前输入的 token 序列长度：训练时含目标 token，生成前编码时只有 prompt；
        两种情况都可能包含 batch padding。N 是本批填充后的历史商品数，
        history_item_mask 标明其中哪些商品有效。
        hidden: [B, T, H]；history_positions: [B, N, 3]；
        history_item_mask: [B, N]。位置只指向历史，不能包含目标 SID。
        """
        if history_positions is None or history_item_mask is None:
            raise ValueError("history_positions and history_item_mask are required")
        batch, items, width = history_positions.shape
        if width != 3:
            raise ValueError("CURE v1 expects three SID tokens per history item")
        if items == 0:
            return hidden.new_zeros((batch, hidden.size(-1)))
        if history_positions.max() >= hidden.size(1) or history_positions.min() < 0:
            raise ValueError("History position outside input sequence")
        # gather 后为 [B, N*3, H]，再把同一商品的 3 个 SID state 平均成 [B, N, H]。
        states = hidden.gather(1, history_positions.reshape(batch, -1).unsqueeze(-1)
                               .expand(-1, -1, hidden.size(-1)))
        states = states.reshape(batch, items, width, -1).mean(2)
        # 混合 batch 中的辅助重建样本没有历史；仅对有效推荐样本运行 reasoner。
        active = history_item_mask.any(1)
        latent = torch.zeros((batch, hidden.size(-1)), device=hidden.device,
                             dtype=self.reasoner.output_projection.weight.dtype)
        if active.any():
            latent[active] = self.reasoner(states[active].to(self.reasoner.input_projection.weight.dtype),
                                           history_item_mask[active])
        return latent

    def encode_latent(self, input_ids, attention_mask, history_positions,
                      history_item_mask, latent_active=None):
        """先编码 prompt，再提取历史 latent z: [B, H]。

        input_ids/attention_mask: [B, T]；history_positions: [B, N, 3]。
        生成前只调用一次；后续 beam 与解码步复用同一个 z。
        """
        with torch.no_grad():
            outputs = self.backbone.model(input_ids=input_ids,
                                          attention_mask=attention_mask, use_cache=False,
                                          return_dict=True)
        latent = self._history_latent(outputs.last_hidden_state, history_positions,
                                      history_item_mask)
        if latent_active is not None:
            latent = latent * latent_active.to(latent.dtype).unsqueeze(-1)
        return latent

    def _fuse(self, hidden, latent_z, latent_mode, latent_active=None):
        """同一 fusion 实现用于训练和生成。

        hidden: [B, T, H]（生成时 T=1）；latent_z: [B, H]；
        latent_active: [B]，辅助任务为 False；输出仍为 [B, T, H]。
        """
        if latent_mode not in ("normal", "zero"):
            raise ValueError(f"Unsupported latent_mode: {latent_mode}")
        if latent_mode == "zero" or latent_z is None:
            return hidden
        # 将 [B, H] 扩展为 [B, T, H]，与每个预测位置的 hidden 拼接求 gate。
        latent = self.latent_projection(latent_z.to(self.latent_projection.weight.dtype))
        latent = latent.to(hidden.dtype).unsqueeze(1).expand_as(hidden)
        gate_input = torch.cat((hidden, latent), dim=-1).to(self.fusion_gate.weight.dtype)
        gate = torch.sigmoid(self.fusion_gate(gate_input))
        residual = gate.to(hidden.dtype) * latent
        if latent_active is not None:
            residual = residual * latent_active.to(hidden.dtype)[:, None, None]
        return hidden + residual

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                history_positions=None, history_item_mask=None, latent_active=None,
                latent_z=None, latent_mode="normal", past_key_values=None,
                use_cache=None, cache_position=None, position_ids=None,
                inputs_embeds=None, return_dict=True, logits_to_keep=0, **kwargs):
        """训练返回 [B, T, V] logits；生成只返回 [B, 1, V]。

        labels: [B, T]，prompt 位置为 -100；生成时传入预计算的 latent_z: [B, H]。
        logits_to_keep=1 时只做最后位置的词表投影，避免 beam 展开后显存暴涨。
        """
        # 冻结 Qwen 不保存反传图；训练梯度仍会流向后面的 reasoner 和 fusion。
        with torch.no_grad():
            outputs = self.backbone.model(
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, past_key_values=past_key_values,
                use_cache=use_cache, cache_position=cache_position,
                position_ids=position_ids, return_dict=True, **kwargs)
        hidden = outputs.last_hidden_state
        if latent_z is None and latent_mode == "normal" and history_positions is not None:
            latent_z = self._history_latent(hidden, history_positions, history_item_mask)
        if labels is not None and logits_to_keep:
            raise ValueError("logits_to_keep cannot be used with labels")
        if logits_to_keep:
            # beam 只需下一 token 的 logits；先裁 hidden，再投影到整个词表。
            index = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            hidden = hidden[:, index, :]
        fused = self._fuse(hidden, latent_z, latent_mode, latent_active)
        logits = self.backbone.lm_head(fused.to(self.backbone.lm_head.weight.dtype)).float()
        loss = cure_loss(logits, labels) if labels is not None else None
        result = CausalLMOutputWithPast(loss=loss, logits=logits,
                                        past_key_values=outputs.past_key_values)
        return result if return_dict else result.to_tuple()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """让预计算的 z 随样本扩展到各 beam，并只输出最后位置 logits。

        HF GenerationMixin 负责对 latent_z 的 batch 维做 beam 扩展。
        """
        latent_z = kwargs.pop("latent_z", None)
        latent_mode = kwargs.pop("latent_mode", "normal")
        result = self.backbone.prepare_inputs_for_generation(input_ids, **kwargs)
        result["latent_z"] = latent_z
        result["latent_mode"] = latent_mode
        result["logits_to_keep"] = 1
        return result

    def save_cure(self, output_dir):
        """只保存新增模块；冻结的 SFT backbone 由原 checkpoint 提供。"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"reasoner": self.reasoner.state_dict(),
                    "latent_projection": self.latent_projection.state_dict(),
                    "fusion_gate": self.fusion_gate.state_dict()}, path / "cure.pt")
        (path / "cure_config.json").write_text(json.dumps(self.reasoner_config, indent=2))

    @classmethod
    def from_cure_pretrained(cls, backbone_path, cure_path, torch_dtype=torch.bfloat16):
        """加载 merged SFT backbone 与 CURE 新增权重。"""
        backbone = AutoModelForCausalLM.from_pretrained(backbone_path, torch_dtype=torch_dtype,
                                                         local_files_only=True)
        path = Path(cure_path)
        config = json.loads((path / "cure_config.json").read_text())
        model = cls(backbone, config)
        state = torch.load(path / "cure.pt", map_location="cpu", weights_only=True)
        for name, weights in state.items():
            getattr(model, name).load_state_dict(weights)
        return model
