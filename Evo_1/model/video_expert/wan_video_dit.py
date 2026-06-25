import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Any, Dict, Tuple, Optional
from einops import rearrange
from .helpers.gradient import gradient_checkpoint_forward

logger = logging.getLogger(__name__)

    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, ctx_mask: Optional[torch.Tensor] = None, compatibility_mode=True):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return x
    else:
        raise NotImplementedError("Only compatibility mode is implemented for flash attention. Please set compatibility_mode=True.")



def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position): # posision B*360
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)  # dim=128 -> [1024, 22]
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)  # dim=128 -> [1024, 21]
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)  # dim=128 -> [1024, 21]
    return f_freqs_cis, h_freqs_cis, w_freqs_cis
    # 对 h w t三个维度分别进行旋转编码。


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim)) # [1/10000 ^(0),1/10000 ^(2/128),,1/10000 ^(64/128)]
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    # [0,1, ... , 1024] x freqs
    # freqs[p, i] = p * 1 / 10000^(2i / dim)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # torch.ones_like(freqs) 是 创建一个全1的矩阵作为r， 后半部分作为角度，这一行会计算得到一个复数
    # [1024,64] 1024 是 pos，64 是正余弦编码的维度
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads) # [B 360 3072] -> [B 360 24 128]
    # freqs[360 1 64]
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    #[B 360 24 64 2]->complex [B 360 24 64]
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    #广播乘法以后得到complex [B 360 24 64] 然后flatten->[B 360 24*64*2]
    # 这里做乘法的意义在于，让每个token 知道自己是 哪一个t 哪一个 h 哪一个 w
    # 把每个 video token 的 3D 位置信息注入到 q/k 向量里。
    return x_out.to(x.dtype)


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    """
    Creates a group-based attention mask for scaled dot-product attention with two modes:
    'causal' and 'group_diagonal'.

    Parameters:
    - num_temporal_groups (int): The number of temporal groups (e.g., frames in a video sequence).
    - num_query_per_group (int): The number of query tokens per temporal group. (e.g., latent tokens in a frame, H x W).
    - num_key_per_group (int): The number of key tokens per temporal group. (e.g., action tokens per frame).
    - mode (str): The mode of the attention mask. Options are:
        - 'causal': Query tokens can attend to key tokens from the same or previous temporal groups.
        - 'group_diagonal': Query tokens can attend only to key tokens from the same temporal group.

    Returns:
    - attn_mask (torch.Tensor): A boolean tensor of shape (L, S), where:
        - L = num_temporal_groups * num_query_per_group (total number of query tokens)
        - S = num_temporal_groups * num_key_per_group (total number of key tokens)
      The mask indicates where attention is allowed (True) and disallowed (False).

    Example:
    Input:
        num_temporal_groups = 3
        num_query_per_group = 4
        num_key_per_group = 2
    Output:
        Causal Mask Shape: torch.Size([12, 6])
        Group Diagonal Mask Shape: torch.Size([12, 6])
        if mode='causal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True]])

        if mode='group_diagonal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True]])

    """
    assert mode in ["causal", "group_diagonal"], f"Mode {mode} must be 'causal' or 'group_diagonal'"

    # Total number of query and key tokens
    total_num_query_tokens = num_temporal_groups * num_query_per_group  # Total number of query tokens (L)
    total_num_key_tokens = num_temporal_groups * num_key_per_group  # Total number of key tokens (S)

    # Generate time indices for query and key tokens (shape: [L] and [S])
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group)  # Shape: [L]
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group)  # Shape: [S]

    # Expand dimensions to compute outer comparison
    query_time_indices = query_time_indices.unsqueeze(1)  # Shape: [L, 1]
    key_time_indices = key_time_indices.unsqueeze(0)  # Shape: [1, S]

    if mode == "causal":
        # Causal Mode: Query can attend to keys where key_time <= query_time
        attn_mask = query_time_indices >= key_time_indices  # Shape: [L, S]
    elif mode == "group_diagonal":
        # Group Diagonal Mode: Query can attend only to keys where key_time == query_time
        attn_mask = query_time_indices == key_time_indices  # Shape: [L, S]

    assert attn_mask.shape == (total_num_query_tokens, total_num_key_tokens), "Attention mask shape mismatch"
    return attn_mask


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads # 24
        self.attn_hidden_dim = num_heads * attn_head_dim # 24 * 128 = 3072

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim) # 3072 -> 3072
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor] = None):
        # x: [B, 360, 3072]
        # freqs: [360, 1, 64], complex RoPE table for one 128-dim attention head
        q = self.norm_q(self.q(x))  # [B, 360, 3072] -> [B, 360, 3072]
        k = self.norm_k(self.k(x))  # [B, 360, 3072] -> [B, 360, 3072]
        v = self.v(x)  # [B, 360, 3072] -> [B, 360, 3072]
        q = rope_apply(q, freqs, self.num_heads)  # [B, 360, 3072]
        k = rope_apply(k, freqs, self.num_heads)  # [B, 360, 3072]
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)  # [B, 360, 3072]
        return self.o(x)  # [B, 360, 3072]


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6,):
        super().__init__()
        self.num_heads = num_heads
        self.attn_hidden_dim = num_heads * attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self,  hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask: Optional[torch.Tensor] = None):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1) # (B, 1, seq_len, context_len), 1 for heads
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        # modulation [1, 6, 3072] 是随机产生的
        # t_mod [B, 360, 6, 3072] 每个token 得到了6个调制参数。
        # 广播相加后变成[B, 360, 6, 3072] 有 chunk_dim = 2
        # 会被拆成6个[B, 360, 1, 3072]，就是等号左边的6个调制参数
        # 每一层 DiT block 有一组自己的 baseline 调制偏置，再叠加 timestep 条件调制。
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
            # 变成6个[B, 360, 3072]
            # x 自身的维度就是[B, 360, 3072]
        x = self.gate(x, gate_msa, self.self_attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            freqs,
            self_attn_mask=self_attn_mask,
        ))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        #[1 2 3072]维度的随机变量
    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            # [1, 2, 3072]->[1, 1, 2, 3072]
            # [B 360 3072]->[B, 360, 1, 3072]
            # 两者广播相加[B, 360, 2, 3072] 
            # 然后拆开为2个 [B, 360, 1, 3072] 
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
            # x进行缩放操作得到[B 360 3072]
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanVideoDiT(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,  #  default: 3072
        in_dim: int,  #  default: 48
        ffn_dim: int,  #  default: 14336
        out_dim: int,  #  default: 48
        text_dim: int,  #  default: 4096
        freq_dim: int,  #  default: 256
        eps: float,  #  default: 1e-6
        patch_size: Tuple[int, int, int],  #  default: (1, 2, 2)
        num_heads: int,  #  default: 24
        attn_head_dim: int,  #  default: 128
        num_layers: int,  #  default: 30
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,  # signature default: 7; FastWAM config uses dataset action dim
        action_group_causal_mask_mode = "causal",  # signature default: "causal"; FastWAM config: "group_diagonal"
        video_attention_mask_mode: str = "bidirectional",  # signature default: "bidirectional"; FastWAM config: "first_frame_causal"
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim  # 3072: video token hidden dimension
        # text_dim=4096: text encoder output dimension before projection to hidden_dim
        self.freq_dim = freq_dim  # 256: sinusoidal timestep embedding dimension
        self.patch_size = patch_size  # (1, 2, 2): Conv3d patch size/stride over T,H,W
        self.seperated_timestep = seperated_timestep  # True in FastWAM config
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents  # True in FastWAM config
        self.video_attention_mask_mode = str(video_attention_mask_mode)  # "first_frame_causal" in FastWAM config

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}"
            )
        
        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        if has_image_input:
            raise ValueError("This WanVideoDiT port does not support has_image_input=True.")
        if has_image_pos_emb:
            raise ValueError("This WanVideoDiT port does not support has_image_pos_emb=True.")
        if has_ref_conv:
            raise ValueError("This WanVideoDiT port does not support has_ref_conv=True.")
        if require_clip_embedding:
            raise ValueError("This WanVideoDiT port does not support require_clip_embedding=True.")
        if require_vae_embedding or not fuse_vae_embedding_in_latents:
            raise ValueError("Only fused VAE embedding in latents is supported.")

        self.patch_embedding = nn.Conv3d(
            in_dim,  # 48
            hidden_dim,  # 3072
            kernel_size=patch_size,  # (1, 2, 2)
            stride=patch_size,  # (1, 2, 2)
        )
        # Example: [B, 48, 3, 24, 20] -> [B, 3072, 3, 12, 10]
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)

        if self.action_conditioned:
            self.action_embedding = nn.Linear(action_dim, hidden_dim)
            self.action_group_causal_mask_mode = action_group_causal_mask_mode
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            logger.info("Using gradient checkpointing for DiT blocks. This will save memory but use more computation.")
            

    def patchify(self, x: torch.Tensor):
        return self.patch_embedding(x)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")
        num_latent_frames = x.shape[2]
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if self.action_conditioned:
            allow_text_only_single_frame = (num_latent_frames == 1 and action is None)
            if not allow_text_only_single_frame:
                assert action is not None, "Action input is required for action-conditioned model."
                if action.ndim != 3:
                    raise ValueError(f"`action` must be 3D [B, action_horizon, action_dim], got shape {tuple(action.shape)}")
                if action.shape[2] != self.action_dim:
                    raise ValueError(f"`action` last dimension must be {self.action_dim}, got {action.shape[2]}")
                if num_latent_frames <= 1:
                    raise ValueError(f"video length must be > 1 for action-conditioned model, got {num_latent_frames}")
                if action.shape[1] % (num_latent_frames - 1) != 0:
                    raise ValueError(
                        f"action horizon must be divisible by (num_latent_frames - 1), got action_horizon={action.shape[1]}"
                    )
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}")

        batch_size = x.shape[0]
        if batch_size != context.shape[0]:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(
                    f"Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}."
                )

        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            assert not self.training, "During training, timestep length must match batch_size."
            timestep = timestep.expand(batch_size)
        return x, timestep, context_mask

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def pre_dit(
        self,
        x: torch.Tensor,  # e.g. 9-frame 384x320 video -> [B, 48, 3, 24, 20]
        timestep: torch.Tensor,  # [B]
        context: torch.Tensor,  # [B, text_len, 4096], text_len usually <= 128 in FastWAM config
        context_mask: Optional[torch.Tensor] = None,  # [B, text_len]
        action: Optional[torch.Tensor] = None,  # [B, action_horizon, action_dim], only used when action_conditioned=True
        fuse_vae_embedding_in_latents: Optional[bool] = None,  # FastWAM/Wan2.2 default: True
    ) -> Dict[str, Any]:
        if fuse_vae_embedding_in_latents is None:
            fuse_vae_embedding_in_latents = self.fuse_vae_embedding_in_latents
        x, timestep, context_mask = self._validate_forward_inputs( # 在做简单校验，形式不变
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1]) # patch_size = [1 2 2] 
        patch_w = int(self.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)
        if self.seperated_timestep and fuse_vae_embedding_in_latents:
            if not hasattr(self, "patch_size") or len(self.patch_size) < 3:
                raise ValueError(f"Invalid dit.patch_size: {getattr(self, 'patch_size', None)}")

            token_timesteps = torch.ones(
                (batch_size, x.shape[2], tokens_per_frame),
                dtype=timestep.dtype,
                device=timestep.device,
            ) * timestep.view(batch_size, 1, 1) # (B 3 120) 3 代表一次处理3帧（当前帧，未来两个压缩帧） 120 代表每帧有多少token
            # (B 3 120) * (B 1 1) -> (B 3 120)
            token_timesteps[:, 0, :] = 0 # 第0帧的120个token的时间编码设置为0，后面240个时间编码是一样的
            token_timesteps = token_timesteps.reshape(batch_size, -1) # （B 360）
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            # [B*360, 128+128]，前128维是cos，后128维是sin，不是sin/cos交错。
            # 每360个token代表一条数据：0-119的timestep是0，120-239和240-359的timestep都是t。
            # 所以0-119的256维向量完全一样；120-359的256维向量也完全一样。
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim) # [B*360, 256]->[B*360, 3072][B 360 3072]
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
            # [B, 360, 3072] -> [B, 360, 18432] -> [B, 360, 6, 3072]，每个token得到6组3072维调制向量。
        else:
            raise NotImplementedError("Only support seperated_timestep with fuse_vae_embedding_in_latents for now.")
        x = self.patchify(x) # [B, 48, 3, 24, 20] -> [B, 3072, 3, 12, 10]一张图像经过patch 以后，只有120 个token
        f, h, w = x.shape[2:]

        context = self.text_embedding(context) # [B, text_len, 4096] -> (B, text_len, 3072)
        context_len = context.shape[1]
        if self.action_conditioned and action is not None:
            action_len = action.shape[1]
            action_emb = self.action_embedding(action) # (B, action_len, dim)
            action_pos_embed = sinusoidal_embedding_1d(self.hidden_dim, 
                torch.arange(action_len, device=action_emb.device)) # (action_len, dim)
            action_emb = action_emb + action_pos_embed.unsqueeze(0) # (B, action_len, dim)
            context = torch.cat([context, action_emb], dim=1) # (B, context_len + action_len, dim)

            # new mask
            num_temporal_groups = f - 1 # first latent frame do not attend to actions
            if num_temporal_groups <= 0:
                raise ValueError(
                    "Action-conditioned context mask requires at least 2 latent frames when `action` is provided."
                )
            assert action_emb.shape[1] % num_temporal_groups == 0, \
                f"Action embedding length {action_emb.shape[1]} must be divisible by number of temporal groups {num_temporal_groups}"
            # Each latent frame (from the 2nd one) attends to the corresponding group of action tokens
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(context.device) # ((f-1)*tokens_per_frame, action_len)

            seq_len = f * h * w # query length
            final_context_mask = torch.zeros((batch_size, seq_len, context.shape[1]), dtype=torch.bool, device=context.device) # (B, seq_len, L + action_len)
            # all latent frames attend to text tokens
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(-1, seq_len, -1) # (B, seq_len, L)
            # latent frames from the 2nd one attend to action tokens
            final_context_mask[:, tokens_per_frame:, context_len:] = action_group_mask.unsqueeze(0).expand(batch_size, -1, -1) # (B, seq_len, action_len)
            context_mask = final_context_mask
        elif self.action_conditioned and action is None:
            if f != 1:
                raise ValueError(
                    "Action-conditioned model requires `action` unless running single-frame text-only mode with num_latent_frames=1."
                )
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)
        else:
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        # [B, 360, 3072] 每120 个token 是一张图片
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1), 
            # self.freqs[0]: [1024, 22]->[3,22]->[3,1,1,22]->[3, 12, 10, 22] 
            # 不管你选的是哪一个h，w，只要选的时间是一样的，频率编码就是一样的。
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1), 
            # self.freqs[1]: [1024, 21]->[12,21]->[1,12,1,21]->[3, 12, 10, 21]
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            # self.freqs[2]: [1024, 21]->[10,21]->[1,1,10,21]->[3, 12, 10, 21]
        ], dim=-1).reshape(f * h * w, 1, -1).to(x_tokens.device) 
        # [3, 12, 10, 64] -> [360, 1, 64]
        # 例如这里如果选一个 [f h y] 则会对应一个64维度的复数向量，可以对128维度的向量进行旋转处理（cos sin 位置编码）
        # 复数维度0-21编码frame位置t，22-42编码height位置，43-63编码width位置。
        # 之所以这里是128维，是因为注意力头的向量维度是128维 （3072 token 的隐变量维度 = 24个头 * 128 每个头的维度）
        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(x_tokens, pre_state["t"]) # 输入两个维度都是 [B 360 3072]
        x = self.unpatchify(x, (f, h, w))
        return x

    def forward(
        self,
        x: torch.Tensor,  # 9-frame 384x320 video -> [B, 48, 3, 24, 20] 这里已经是VAE的输出了
        timestep: torch.Tensor,  # [B] 去噪过程的某一个时间步
        context: torch.Tensor,  # [B, text_len, 4096] 机器人状态、文本编码
        context_mask: Optional[torch.Tensor] = None,  # [B, text_len]
        action: Optional[torch.Tensor] = None,  # [B, action_horizon, action_dim], only used when action_conditioned=True
        fuse_vae_embedding_in_latents: Optional[bool] = None,  # FastWAM/Wan2.2 default: True
    ):
        # 先进行预处理
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = pre_state["tokens"]  # [B, 360, 3072]
        context_emb = pre_state["context"]  # [B, text_len, 3072]
        t_mod = pre_state["t_mod"]  # [B, 360, 6, 3072]
        freqs = pre_state["freqs"]  # [360, 1, 64]，对应attn_head_dim=128的复数RoPE表
        context_attn_mask = pre_state["context_mask"]  # [B, 360, text_len]
        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        ) if self.video_attention_mask_mode != "bidirectional" else None # special rule for faster speed

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask
                )
            else:
                x_tokens = block(x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask)

        return self.post_dit(x_tokens, pre_state)
