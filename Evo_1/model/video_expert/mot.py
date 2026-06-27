from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from .wan_video_dit import flash_attention


def _sum_counts(counts: int | Sequence[int]) -> int:
    return int(counts) if isinstance(counts, int) else sum(counts)


def _check_attention_dim(x: torch.Tensor, num_heads: int, name: str) -> None:
    if x.shape[-1] % num_heads != 0:
        raise ValueError(f"{name} attention dim {x.shape[-1]} must be divisible by num_heads={num_heads}")


def _check_video_action_attention_compat(
    video_q: torch.Tensor,
    action_q: torch.Tensor,
    num_heads: int,
) -> None:
    _check_attention_dim(video_q, num_heads, "Video")
    _check_attention_dim(action_q, num_heads, "Action")
    if video_q.shape[0] != action_q.shape[0]:
        raise ValueError(f"Batch size mismatch: video={video_q.shape[0]}, action={action_q.shape[0]}")
    if video_q.shape[-1] != action_q.shape[-1]:
        raise ValueError(f"Attention dim mismatch: video={video_q.shape[-1]}, action={action_q.shape[-1]}")


def build_first_table_attention_mask(
    current_obs_token_counts: int | Sequence[int],
    future_obs_token_counts: int | Sequence[int],
    action_token_counts: int | Sequence[int],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the MoT self-attention mask from the first table in att.md.

    Token order is expected to be:
    [current observation tokens, future observation tokens, action tokens].
    A True value means the query row can attend to the key column.
    """
    current_len = _sum_counts(current_obs_token_counts)  # [120, 120, 120] -> 360
    future_len = _sum_counts(future_obs_token_counts)
    action_len = _sum_counts(action_token_counts)
    total_len = current_len + future_len + action_len

    current = slice(0, current_len)
    future = slice(current_len, current_len + future_len)
    action = slice(current_len + future_len, total_len)

    mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)
    mask[current, current] = True
    mask[future, current] = True
    mask[future, future] = True
    mask[action, current] = True
    mask[action, action] = True
    return mask


class FirstTableMoTLayer(nn.Module):
    """Global attention layer for video/action experts using the first-table mask."""

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = int(num_heads)
        if self.num_heads <= 0:
            raise ValueError(f"`num_heads` must be positive, got {num_heads}")

    def forward(
        self,
        video_io: Dict[str, torch.Tensor], # 视频专家的所有 token
        action_io: Dict[str, torch.Tensor], # 动作专家的所有token
        current_obs_token_counts: int | Sequence[int], # oth otl otr 一共有多少token
        future_obs_token_counts: int | Sequence[int], # 未来预测的 oth 有多少token
        action_token_counts: int | Sequence[int], # 动作token
    ) -> Dict[str, torch.Tensor]:
        video_q, video_k, video_v = video_io["q"], video_io["k"], video_io["v"]
        action_q, action_k, action_v = action_io["q"], action_io["k"], action_io["v"]
        _check_video_action_attention_compat(video_q, action_q, self.num_heads)

        video_len = video_q.shape[1]
        action_len = action_q.shape[1]

        attention_mask = build_first_table_attention_mask(
            current_obs_token_counts=current_obs_token_counts,
            future_obs_token_counts=future_obs_token_counts,
            action_token_counts=action_token_counts,
            device=video_q.device,
        )

        total_len = video_len + action_len
        if attention_mask.shape != (total_len, total_len):
            raise ValueError(
                f"`attention_mask` must be [{total_len}, {total_len}], got {tuple(attention_mask.shape)}"
            )

        q = torch.cat([video_q, action_q], dim=1)
        k = torch.cat([video_k, action_k], dim=1)
        v = torch.cat([video_v, action_v], dim=1)
        mixed = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attention_mask)
        return {
            "video": mixed[:, :video_len],
            "action": mixed[:, video_len:],
        }

class SkipLayerVideoActionMoT(nn.Module):
    """Coordinate video/action experts with video layers 3, 6, ..., 30 paired to action layers."""

    def __init__(self, num_heads: int, video_layer_stride: int = 3):
        super().__init__()
        self.video_layer_stride = int(video_layer_stride)
        if self.video_layer_stride <= 0:
            raise ValueError(f"`video_layer_stride` must be positive, got {video_layer_stride}")
        self.global_attention = FirstTableMoTLayer(num_heads=num_heads)

    def forward(
        self,
        video_expert: nn.Module,
        action_expert: nn.Module,
        video_inputs: Dict[str, Any],
        action_inputs: Dict[str, Any],
        current_obs_token_counts: int | Sequence[int],
        future_obs_token_counts: int | Sequence[int],
        action_token_counts: int | Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        video_pre_state = video_expert.pre_dit(**video_inputs)
        action_pre_state = action_expert.pre_dit(**action_inputs)
        return self.forward_prepared(
            video_expert=video_expert,
            action_expert=action_expert,
            video_pre_state=video_pre_state,
            action_pre_state=action_pre_state,
            current_obs_token_counts=current_obs_token_counts,
            future_obs_token_counts=future_obs_token_counts,
            action_token_counts=action_token_counts,
        )

    def forward_prepared(
        self,
        video_expert: nn.Module,
        action_expert: nn.Module,
        video_pre_state: Dict[str, Any],
        action_pre_state: Dict[str, Any],
        current_obs_token_counts: int | Sequence[int],
        future_obs_token_counts: int | Sequence[int],
        action_token_counts: int | Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        video_tokens = video_pre_state["tokens"]
        action_tokens = action_pre_state["action_tokens"]
        video_context = video_pre_state["context"]
        action_context = action_pre_state["context"]
        video_t_mod = video_pre_state["t_mod"]
        action_t_mod = action_pre_state["t_mod"]
        video_freqs = video_pre_state["freqs"]
        action_freqs = action_pre_state["freqs"]
        video_context_mask = video_pre_state.get("context_mask")
        action_context_mask = action_pre_state.get("context_mask")

        video_self_attn_mask = self._build_video_self_attn_mask(video_expert, video_tokens, video_pre_state)
        # video_self_attn_mask = att.md
        action_layer_idx = 0

        for video_layer_idx in range(len(video_expert.blocks)):
            if not self._is_global_video_layer(video_layer_idx):
                video_tokens = video_expert.run_dit_blocks(
                    x_tokens=video_tokens,
                    context=video_context,
                    t_mod=video_t_mod,
                    freqs=video_freqs,
                    context_attn_mask=video_context_mask,
                    self_attn_mask=video_self_attn_mask,
                    start_layer=video_layer_idx,
                    end_layer=video_layer_idx + 1,
                )
                continue

            if action_layer_idx >= len(action_expert.blocks):
                raise ValueError(
                    f"Video global layer {video_layer_idx + 1} has no matching action layer. "
                    f"Check video_layer_stride={self.video_layer_stride} and action layer count."
                )
            video_io = video_expert.build_layer_attention_io(
                layer_idx=video_layer_idx,
                x_tokens=video_tokens,
                t_mod=video_t_mod,
                freqs=video_freqs,
            )
            action_io = action_expert.build_layer_attention_io(
                layer_idx=action_layer_idx,
                action_tokens=action_tokens,
                t_mod=action_t_mod,
                freqs=action_freqs,
            )
            mot_out = self.global_attention(
                video_io=video_io,
                action_io=action_io,
                current_obs_token_counts=current_obs_token_counts,
                future_obs_token_counts=future_obs_token_counts,
                action_token_counts=action_token_counts,
            )
            video_tokens = video_expert.apply_layer_post_attention(
                layer_idx=video_layer_idx,
                video_io=video_io,
                mixed_attn_out=mot_out["video"],
                context=video_context,
                context_mask=video_context_mask,
            )
            action_tokens = action_expert.apply_layer_post_attention(
                layer_idx=action_layer_idx,
                action_io=action_io,
                mixed_attn_out=mot_out["action"],
                context=action_context,
                context_mask=action_context_mask,
            )
            action_layer_idx += 1

        if action_layer_idx != len(action_expert.blocks):
            raise ValueError(
                f"Only consumed {action_layer_idx} action layers, expected {len(action_expert.blocks)}. "
                f"Check video_layer_stride={self.video_layer_stride} and video/action layer counts."
            )

        return {
            "video": video_expert.post_dit(video_tokens, video_pre_state),
            "action": action_expert.post_dit(action_tokens, action_pre_state),
            "video_tokens": video_tokens,
            "action_tokens": action_tokens,
        }

    def _is_global_video_layer(self, zero_based_layer_idx: int) -> bool:
        return (zero_based_layer_idx + 1) % self.video_layer_stride == 0

    def _build_video_self_attn_mask(
        self,
        video_expert: nn.Module,
        video_tokens: torch.Tensor,
        video_pre_state: Dict[str, Any],
    ) -> torch.Tensor | None:
        if str(video_expert.video_attention_mask_mode) == "bidirectional":
            return None
        return video_expert.build_video_to_video_mask(
            video_seq_len=video_tokens.shape[1],
            video_tokens_per_frame=int(video_pre_state["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
