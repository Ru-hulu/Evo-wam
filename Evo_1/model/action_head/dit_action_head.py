from types import SimpleNamespace
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_matching import CategorySpecificLinear, CategorySpecificMLP


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"`dim` must be even, got {dim}")
    sinusoid = torch.outer(
        position.to(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2),
        ),
    )
    embedding = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return embedding.to(position.dtype)
    # 两个[B, 128]->[B, 256] 前一半是cos(0...127) 后一半是 sin(128...256)

def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"`dim` must be even for RoPE, got {dim}")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64)[: dim // 2] / dim))
    # [1 / 10000^0, 1 / 10000^(2/128), ..., 1 / 10000^(126/128)] 一共64个频率通道
    freqs = torch.outer(torch.arange(end, dtype=torch.float64), freqs)
    # freqs.shape = [1024, 64] freqs[pos, i] = pos * 1 / 10000^(2 * i/128)
    return torch.polar(torch.ones_like(freqs), freqs)
    # [1024, 64] ，每个元素是angle，计算cos(angle) + i sin(angle)
    # 预先准备最多 1024 个 token 位置的 RoPE 编码。
    # 一个头中的token维度是128，这里频率token的维度只有64
def apply_rope(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    dtype = x.dtype # [B, H, 3072]
    bsz, seq_len, hidden = x.shape
    if hidden % num_heads != 0:
        raise ValueError(f"Hidden dim {hidden} is not divisible by num_heads {num_heads}")
    head_dim = hidden // num_heads
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head dim must be even, got {head_dim}")

    x = x.reshape(bsz, seq_len, num_heads, head_dim) # [B, H, 3072] -> [B, H, 24, 128]
    x_complex = torch.view_as_complex(x.float().reshape(bsz, seq_len, num_heads, head_dim // 2, 2))
    # [B, H, 24, 128] -> [B, H, 24, 64, 2] （H=50）->  [B, H, 24, 64] (复数)
    freqs = freqs.to(device=x.device).view(1, seq_len, 1, head_dim // 2) # [1 50 1 64]
    x_out = torch.view_as_real(x_complex * freqs).flatten(3)# -> [B, H, 24, 64] -> [B, H, 24, 64 2]
    return x_out.reshape(bsz, seq_len, hidden).to(dtype) # q: [B, H, 3072]
    # 本质上就是用正弦/余弦对每一对维度做二维旋转。复数只是把这个旋转写得更简洁。
    # 乘以了freq 以后，网络就知道了50个token 的先后关系。
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    k = k.to(dtype=q.dtype)
    v = v.to(dtype=q.dtype)
    bsz, q_len, hidden = q.shape
    head_dim = hidden // num_heads
    q = q.reshape(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(bsz, k.shape[1], num_heads, head_dim).transpose(1, 2)
    v = v.reshape(bsz, v.shape[1], num_heads, head_dim).transpose(1, 2)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return x.transpose(1, 2).reshape(bsz, q_len, hidden)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight.to(device=x.device, dtype=dtype)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.attn_hidden_dim = num_heads * attn_head_dim # 24 * 128
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps) # 1024 -> 3072
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = apply_rope(self.norm_q(self.q(x)), freqs, self.num_heads)
        # x: [B, H, 1024]->q(x)->[B, H, 3072] num_heads = 24  attn_head_dim = 128
        k = apply_rope(self.norm_k(self.k(x)), freqs, self.num_heads)
        v = self.v(x)
        x = scaled_dot_product_attention(q, k, v, self.num_heads, attn_mask=self_attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.attn_hidden_dim = num_heads * attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        x = scaled_dot_product_attention(q, k, v, self.num_heads, attn_mask=context_mask)
        return self.o(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        # hidden_dim = 1024 # num_heads = 24 # attn_head_dim = 128
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)

    def build_attention_io(
        self,
        x: torch.Tensor,  # action_tokens: [8, 50, 1024]
        t_mod: torch.Tensor,  # [8, 6, 1024]
        freqs: torch.Tensor,  # [50, 64]
    ) -> dict:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        q = apply_rope(self.self_attn.norm_q(self.self_attn.q(attn_input)), freqs, self.self_attn.num_heads)
        k = apply_rope(self.self_attn.norm_k(self.self_attn.k(attn_input)), freqs, self.self_attn.num_heads)
        v = self.self_attn.v(attn_input)

        return {
            "q": q,  # [B, 50, 3072]
            "k": k,  # [B, 50, 3072]
            "v": v,  # [B, 50, 3072]
            "residual_x": x,  # [B, 50, 1024]
            "gate_msa": gate_msa,  # [B, 1, 1024]
            "shift_mlp": shift_mlp,  # [B, 1, 1024]
            "scale_mlp": scale_mlp,  # [B, 1, 1024]
            "gate_mlp": gate_mlp,  # [B, 1, 1024]
        }

    def apply_post_attention(
        self,
        residual_x: torch.Tensor,  # [B, 50, 1024]
        mixed_attn_out: torch.Tensor,  # [B, 50, 3072]
        gate_msa: torch.Tensor,  # [B, 1, 1024]
        shift_mlp: torch.Tensor,  # [B, 1, 1024]
        scale_mlp: torch.Tensor,  # [B, 1, 1024]
        gate_mlp: torch.Tensor,  # [B, 1, 1024]
        context: torch.Tensor,  # [B, 1025, 1024]
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = residual_x + gate_msa * self.self_attn.o(mixed_attn_out)
        x = x + self.cross_attn(self.norm3(x), context, context_mask=context_mask)
        x = x + gate_mlp * self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward(
        self,
        x: torch.Tensor,      # action_tokens: [8, 50, 1024]
        context: torch.Tensor,# context: [8, 1025, 1024] （fused_tokens + state token）
        t_mod: torch.Tensor,  # t_mod:   [8, 6, 1024]
        freqs: torch.Tensor,  # freqs:   [50, 64]
        context_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_io = self.build_attention_io(x, t_mod, freqs)
        mixed_attn_out = scaled_dot_product_attention(
            action_io["q"],
            action_io["k"],
            action_io["v"],
            self.self_attn.num_heads,
            attn_mask=self_attn_mask,
        )
        return self.apply_post_attention(
            residual_x=action_io["residual_x"],
            mixed_attn_out=mixed_attn_out,
            gate_msa=action_io["gate_msa"],
            shift_mlp=action_io["shift_mlp"],
            scale_mlp=action_io["scale_mlp"],
            gate_mlp=action_io["gate_mlp"],
            context=context,
            context_mask=context_mask,
        )


class FlowmatchingDiTActionHead(nn.Module):
    def __init__(
        self,
        config=None,
        embed_dim: int = 896,
        hidden_dim: int = 1024,
        ffn_dim: Optional[int] = None,
        action_dim: int = 16 * 7,
        horizon: int = 16,
        per_action_dim: int = 7,
        num_heads: int = 8,
        attn_head_dim: Optional[int] = None,
        num_layers: int = 10,
        dropout: float = 0.0,
        num_inference_timesteps: int = 20,
        num_categories: int = 1,
        freq_dim: int = 256,
        eps: float = 1e-6,
        max_position: int = 1024,
    ):
        super().__init__()
        if config is not None:
            embed_dim = getattr(config, "embed_dim", embed_dim)
            hidden_dim = getattr(config, "hidden_dim", hidden_dim)
            ffn_dim = getattr(config, "ffn_dim", ffn_dim)
            action_dim = getattr(config, "action_dim", action_dim)
            horizon = getattr(config, "horizon", horizon)
            per_action_dim = getattr(config, "per_action_dim", per_action_dim)
            num_heads = getattr(config, "num_heads", num_heads)
            attn_head_dim = getattr(config, "attn_head_dim", attn_head_dim)
            num_layers = getattr(config, "num_layers", num_layers)
            dropout = getattr(config, "dropout", dropout)
            num_inference_timesteps = getattr(config, "num_inference_timesteps", num_inference_timesteps)
            num_categories = getattr(config, "num_categories", num_categories)
            freq_dim = getattr(config, "freq_dim", freq_dim)
            eps = getattr(config, "eps", eps)
            max_position = getattr(config, "max_position", max_position)
            self.config = config
        else:
            self.config = SimpleNamespace()

        if action_dim != horizon * per_action_dim:
            raise ValueError(
                f"action_dim ({action_dim}) must equal horizon ({horizon}) * per_action_dim ({per_action_dim})"
            )
        if attn_head_dim is None:
            if hidden_dim % num_heads != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}) "
                    "when attn_head_dim is not set"
                )
            attn_head_dim = hidden_dim // num_heads
        if attn_head_dim % 2 != 0:
            raise ValueError(f"attn_head_dim must be even for RoPE, got {attn_head_dim}")
        if freq_dim % 2 != 0:
            raise ValueError(f"freq_dim must be even, got {freq_dim}")

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim or hidden_dim * 4
        self.action_dim = action_dim
        self.horizon = horizon
        self.per_action_dim = per_action_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers
        self.num_inference_timesteps = num_inference_timesteps
        self.num_categories = num_categories
        self.freq_dim = freq_dim
        self.eps = eps

        self.context_embedding = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_encoder = CategorySpecificLinear(per_action_dim, hidden_dim, num_categories)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=self.ffn_dim,
                    eps=eps,
                ) # 投影的时候 hidden_dim 投影为 head_dim * num_heads
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.action_decoder = CategorySpecificLinear(hidden_dim, per_action_dim, num_categories)
        self.dropout = nn.Dropout(dropout)

        self.state_encoder = None
        if hasattr(self.config, "state_dim") and self.config.state_dim is not None:
            state_hidden_dim = getattr(self.config, "state_hidden_dim", hidden_dim)
            self.state_encoder = CategorySpecificMLP(
                input_dim=self.config.state_dim,
                hidden_dim=state_hidden_dim,
                output_dim=embed_dim,
                num_categories=num_categories,
            )

        rope_len = max(max_position, horizon)
        self.register_buffer("freqs", precompute_freqs_cis(attn_head_dim, end=rope_len), persistent=False)

    def forward(
        self,
        fused_tokens: torch.Tensor,  # [B, 1024, 896]
        state: torch.Tensor = None,  # [B, 24]
        actions_gt: torch.Tensor = None,  # [B, 50, 24]
        embodiment_id: torch.LongTensor = None,  # [B]
        state_mask: torch.Tensor = None,  # [B, 24]
        action_mask: torch.Tensor = None,  # [B, 50, 24]
    ):
        if actions_gt is None:
            return self.get_action(
                fused_tokens,
                state=state,
                embodiment_id=embodiment_id,
                action_mask=action_mask,
            )

        bsz = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(bsz, dtype=torch.long, device=device)

        actions_gt_seq = self._as_action_sequence(actions_gt)
        mask_seq = self._as_action_mask(action_mask, bsz, device, actions_gt_seq.dtype, required=False)
        noise = torch.rand_like(actions_gt_seq) * 2 - 1
        if mask_seq is not None:
            noise = noise * mask_seq

        t = torch.distributions.Beta(2, 2).sample((bsz,)).clamp(0.02, 0.98)
        t = t.to(device=device, dtype=actions_gt_seq.dtype)
        action_intermediate = (1 - t.view(bsz, 1, 1)) * noise + t.view(bsz, 1, 1) * actions_gt_seq

        pred_velocity = self._predict_velocity(
            action_seq=action_intermediate,
            timestep=t * 1000.0,
            fused_tokens=fused_tokens,
            state=state,
            embodiment_id=embodiment_id,
        )
        return pred_velocity.reshape(bsz, -1), noise

    @torch.no_grad()
    def get_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor = None,
        embodiment_id: torch.LongTensor = None,
        action_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        bsz = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(bsz, dtype=torch.long, device=device)

        action_seq = torch.rand(
            bsz,
            self.horizon,
            self.per_action_dim,
            dtype=fused_tokens.dtype,
            device=device,
        ) * 2 - 1
        mask_seq = self._as_action_mask(action_mask, bsz, device, action_seq.dtype, required=True)
        action_seq = action_seq * mask_seq

        num_steps = int(getattr(self.config, "num_inference_timesteps", self.num_inference_timesteps))
        dt = 1.0 / num_steps
        for i in range(num_steps):
            timestep = torch.full(
                (bsz,),
                (i / num_steps) * 1000.0,
                dtype=action_seq.dtype,
                device=device,
            )
            pred = self._predict_velocity(
                action_seq=action_seq * mask_seq,
                timestep=timestep,
                fused_tokens=fused_tokens,
                state=state,
                embodiment_id=embodiment_id,
            )
            action_seq = (action_seq + dt * pred) * mask_seq

        return action_seq.reshape(bsz, -1)

    def _predict_velocity(
        self,
        action_seq: torch.Tensor,  # [B, 50, 24]
        timestep: torch.Tensor,  # [B]
        fused_tokens: torch.Tensor,  # [B, 1024, 896]
        state: torch.Tensor,  # [B, 24]
        embodiment_id: torch.LongTensor,  # [B]
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_seq=action_seq,
            timestep=timestep,
            fused_tokens=fused_tokens,
            state=state,
            embodiment_id=embodiment_id,
        )
        action_tokens = self.run_dit_blocks(
            action_tokens=pre_state["action_tokens"],
            context=pre_state["context"],
            t_mod=pre_state["t_mod"],
            freqs=pre_state["freqs"],
        )
        return self.post_dit(action_tokens, pre_state)

    def pre_dit(
        self,
        action_seq: torch.Tensor,  # [B, 50, 24]
        timestep: torch.Tensor,  # [B]
        fused_tokens: torch.Tensor,  # [B, 1024, 896]
        state: torch.Tensor,  # [B, 24]
        embodiment_id: torch.LongTensor,  # [B]
    ) -> dict:
        context = self._build_context(fused_tokens, state, embodiment_id)
        # [B, 1024, 896] + state [B, 1, 896] -> [B, 1025, 896] -> [B, 1025, 1024]
        action_tokens = self._apply_category_linear(self.action_encoder, action_seq, embodiment_id)
        # [B, 50, 24] -> [B * 50, 24] -> [B * 50, 1024] -> [B, 50, 1024]
        action_tokens = self.dropout(action_tokens)

        time_emb = sinusoidal_embedding_1d(self.freq_dim, timestep).to(
            device=action_tokens.device,
            dtype=action_tokens.dtype,
        )
        # [B, 256] 前一半是cos(0...127) 后一半是 sin(128...256)
        time_emb = self.time_embedding(time_emb) # [B, 256] -> [B, 1024]
        t_mod = self.time_projection(time_emb).unflatten(1, (6, self.hidden_dim))
        # 把时间步信息转换成 DiT block 的条件调制参数。
        # 给每个 token 特征维度生成 6 组调制向量。
        # shift_msa scale_msa gate_msa shift_mlp scale_mlp gate_mlp
        # [B, 1024] -> [B, 6144] -> [B, 6, 1024]

        if action_seq.shape[1] > self.freqs.shape[0]:
            raise ValueError(
                f"Action horizon {action_seq.shape[1]} exceeds RoPE cache {self.freqs.shape[0]}"
            )
        # [1024, 64]  切一段出来，action 的H = 50， 所以切[50 64]
        freqs = self.freqs[: action_seq.shape[1]].to(action_tokens.device)
        # action_tokens 维度是1024
        return {
            "action_tokens": action_tokens,  # [B, 50, 1024]
            "context": context,  # [B, 1025, 1024]
            "t_mod": t_mod,  # [B, 6, 1024]
            "freqs": freqs,  # [50, 64]
            "embodiment_id": embodiment_id,  # [B]
        }

    def run_dit_blocks(
        self,
        action_tokens: torch.Tensor,  # [B, 50, 1024]
        context: torch.Tensor,  # [B, 1025, 1024] （fused_tokens + state token）
        t_mod: torch.Tensor,  # [B, 6, 1024]
        freqs: torch.Tensor,  # [50, 64]
        start_layer: int = 0,
        end_layer: Optional[int] = None,
        global_attention_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        end_layer = len(self.blocks) if end_layer is None else end_layer
        for layer_idx, block in enumerate(self.blocks[start_layer:end_layer], start=start_layer):
            if global_attention_fn is None:
                action_tokens = block(action_tokens, context, t_mod, freqs)
            else:
                action_io = block.build_attention_io(action_tokens, t_mod, freqs)
                mixed_attn_out = global_attention_fn(
                    layer_idx=layer_idx,
                    action_io=action_io,
                )
                action_tokens = block.apply_post_attention(
                    residual_x=action_io["residual_x"],
                    mixed_attn_out=mixed_attn_out,
                    gate_msa=action_io["gate_msa"],
                    shift_mlp=action_io["shift_mlp"],
                    scale_mlp=action_io["scale_mlp"],
                    gate_mlp=action_io["gate_mlp"],
                    context=context,
                )
        return action_tokens

    def build_layer_attention_io(
        self,
        layer_idx: int,
        action_tokens: torch.Tensor,  # [B, 50, 1024]
        t_mod: torch.Tensor,  # [B, 6, 1024]
        freqs: torch.Tensor,  # [50, 64]
    ) -> dict:
        return self.blocks[layer_idx].build_attention_io(action_tokens, t_mod, freqs)

    def apply_layer_post_attention(
        self,
        layer_idx: int,
        action_io: dict,
        mixed_attn_out: torch.Tensor,  # [B, 50, 3072]
        context: torch.Tensor,  # [B, 1025, 1024]
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        block = self.blocks[layer_idx]
        return block.apply_post_attention(
            residual_x=action_io["residual_x"],
            mixed_attn_out=mixed_attn_out,
            gate_msa=action_io["gate_msa"],
            shift_mlp=action_io["shift_mlp"],
            scale_mlp=action_io["scale_mlp"],
            gate_mlp=action_io["gate_mlp"],
            context=context,
            context_mask=context_mask,
        )

    def post_dit(
        self,
        action_tokens: torch.Tensor,  # [B, 50, 1024]
        pre_state: dict,
    ) -> torch.Tensor:
        action_tokens = self.norm_out(action_tokens)
        return self._apply_category_linear(self.action_decoder, action_tokens, pre_state["embodiment_id"])

    def _build_context(
        self,
        fused_tokens: torch.Tensor,
        state: Optional[torch.Tensor],
        embodiment_id: torch.LongTensor,
    ) -> torch.Tensor:
        if fused_tokens.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected fused token dim {self.embed_dim}, got {fused_tokens.shape[-1]}")

        context_tokens = fused_tokens
        if state is not None and self.state_encoder is not None:
            state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1) # 将机器人状态进行编码
            context_tokens = torch.cat([context_tokens, state_emb], dim=1) # 拼到 o text 的编码中去
        return self.context_embedding(context_tokens) # 再进行一次投影，得到fuse 编码

    def _apply_category_linear(
        self,
        layer: CategorySpecificLinear,
        x: torch.Tensor,
        embodiment_id: torch.LongTensor,
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        # x: [B, H, D_in]
        if embodiment_id.dim() == 0:
            cat_ids = embodiment_id.repeat(bsz * seq_len) # B * H
        else:
            cat_ids = embodiment_id.view(bsz, 1).expand(bsz, seq_len).reshape(bsz * seq_len)
        out = layer(x.reshape(bsz * seq_len, dim), cat_ids) # cat_ids 决定使用哪个线性层的参数，如果只有一个机器人类别，这里就等价于linear
        # [B H D_in] -> [B*H D_in] -> [B*H, D_out]
        return out.view(bsz, seq_len, -1) # [B*H, D_out] -> [B, H, D_out]

    def _as_action_sequence(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 3:
            if action.shape[1] != self.horizon or action.shape[2] != self.per_action_dim:
                raise ValueError(
                    f"Expected action shape [B, {self.horizon}, {self.per_action_dim}], got {tuple(action.shape)}"
                )
            return action
        if action.ndim == 2:
            if action.shape[1] != self.action_dim:
                raise ValueError(f"Expected flattened action dim {self.action_dim}, got {action.shape[1]}")
            return action.view(action.shape[0], self.horizon, self.per_action_dim)
        raise ValueError(f"Unsupported action shape {tuple(action.shape)}")

    def _as_action_mask(
        self,
        action_mask: Optional[torch.Tensor],
        bsz: int,
        device: torch.device,
        dtype: torch.dtype,
        required: bool,
    ) -> Optional[torch.Tensor]:
        if action_mask is None:
            if required:
                raise ValueError("action_mask must be provided for inference with flow matching.")
            return None

        mask = action_mask.to(device=device, dtype=dtype)
        if mask.ndim == 3:
            if mask.shape != (bsz, self.horizon, self.per_action_dim):
                raise ValueError(
                    f"Expected action_mask shape [B, {self.horizon}, {self.per_action_dim}], got {tuple(mask.shape)}"
                )
            return mask
        if mask.ndim == 2:
            if mask.shape == (bsz, self.action_dim):
                return mask.view(bsz, self.horizon, self.per_action_dim)
            if mask.shape == (bsz, self.per_action_dim):
                return mask.view(bsz, 1, self.per_action_dim).expand(bsz, self.horizon, self.per_action_dim)
        raise ValueError(
            f"Unsupported action_mask shape {tuple(mask.shape)}; expected [B,H,D], [B,H*D], or [B,D]"
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
