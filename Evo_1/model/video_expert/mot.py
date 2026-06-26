from __future__ import annotations

from typing import Dict, Sequence

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
