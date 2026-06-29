from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from Evo_1.model.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


@dataclass
class JointInputAdapterOutput:
    video_inputs: dict[str, torch.Tensor | bool | None]
    action_inputs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor | None]
    context_inputs: dict[str, torch.Tensor]
    mot_counts: dict[str, int]
    debug: dict[str, torch.Tensor]


class FastWAMContextBuilder(nn.Module):
    """Append the current proprio/state as one extra context token."""

    def __init__(self, state_dim: int, context_dim: int):
        super().__init__()
        self.state_dim = int(state_dim)
        self.context_dim = int(context_dim)
        self.proprio_encoder = nn.Linear(self.state_dim, self.context_dim)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        proprio: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B, L, D_context], got {tuple(context.shape)}")
        if context.shape[-1] != self.context_dim:
            raise ValueError(f"`context` dim must be {self.context_dim}, got {context.shape[-1]}")
        if proprio.ndim == 3:
            current_state = proprio[:, 0, :]  # default [B, D_state]
        elif proprio.ndim == 2:
            current_state = proprio  # default [B, D_state]
        else:
            raise ValueError(f"`proprio` must be [B, T, D_state] or [B, D_state], got {tuple(proprio.shape)}")
        if current_state.shape[-1] != self.state_dim:
            raise ValueError(f"`proprio` dim must be {self.state_dim}, got {current_state.shape[-1]}")

        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)  # default [B, L]
        elif context_mask.ndim != 2:
            raise ValueError(f"`context_mask` must be [B, L], got {tuple(context_mask.shape)}")
        elif context_mask.shape != context.shape[:2]:
            raise ValueError(
                f"`context_mask` shape must match context [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape[:2])}"
            )

        state_token = self.proprio_encoder(current_state.to(device=context.device, dtype=context.dtype)).unsqueeze(1)
        # default [B, 1, D_context]
        state_mask = torch.ones((context.shape[0], 1), dtype=torch.bool, device=context.device)  # default [B, 1]
        return {
            "context": torch.cat([context, state_token], dim=1),  # default [B, L+1, D_context]
            "context_mask": torch.cat([context_mask.to(device=context.device, dtype=torch.bool), state_mask], dim=1),
            "state_token": state_token,
            "current_state": current_state,
        }


@torch.no_grad()
def encode_video_latents(
    vae: Any,
    video: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    tiled: bool = False,
) -> torch.Tensor:
    batch_size, num_views, C, T, H, W = video.shape  # default [B, V, 3, T, H, W]
    video = video.to(device=device, dtype=dtype, non_blocking=True)  # default [B, V, 3, T, H, W]
    video = video.reshape(batch_size * num_views, C, T, H, W)  # default [B*V, 3, T, H, W]
    latents = vae.encode(video, device=device, tiled=tiled)  # default [B*V, C_latent, T_latent, H_latent, W_latent]
    if isinstance(latents, list):
        latents = torch.stack(latents, dim=0)
    if latents.ndim != 5:
        raise ValueError(f"`vae.encode` must return [B, C, T, H, W], got {tuple(latents.shape)}")
    latents = latents.to(device=device, dtype=dtype)
    _, C_latent, T_latent, H_latent, W_latent = latents.shape
    return latents.reshape(batch_size, num_views, C_latent, T_latent, H_latent, W_latent)


def _build_flow_training_inputs(
    clean: torch.Tensor,
    scheduler: WanContinuousFlowMatchScheduler,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = clean.shape[0]
    noise = torch.randn_like(clean)
    timestep = scheduler.sample_training_t(batch_size, device=clean.device, dtype=clean.dtype)  # default [B]
    noised = scheduler.add_noise(clean, noise, timestep)
    target = scheduler.training_target(clean, noise, timestep)
    return noised, target, noise, timestep


def _infer_mot_counts(
    video_latents: torch.Tensor,
    action: torch.Tensor,
    video_patch_size: tuple[int, int, int],
) -> dict[str, int]:
    _, num_views, _, latent_t, latent_h, latent_w = video_latents.shape  # default [B, V, C_latent, T_latent, H_latent, W_latent]
    if action.ndim != 3:
        raise ValueError(f"`action` must be [B, H_action, D_action], got {tuple(action.shape)}")
    _, patch_h, patch_w = video_patch_size
    if latent_h % patch_h != 0 or latent_w % patch_w != 0:
        raise ValueError(
            f"Latent H/W must be divisible by video patch H/W, got {(latent_h, latent_w)} and {(patch_h, patch_w)}"
        )
    tokens_per_view_frame = (latent_h // patch_h) * (latent_w // patch_w)
    tokens_per_latent_frame = num_views * tokens_per_view_frame
    return {
        "current_obs_token_counts": tokens_per_latent_frame,
        "future_obs_token_counts": max(latent_t - 1, 0) * tokens_per_latent_frame,
        "action_token_counts": int(action.shape[1]),
        "num_video_views": int(num_views),
        "tokens_per_latent_frame": tokens_per_latent_frame,
        "tokens_per_view_frame": tokens_per_view_frame,
    }


def _future_video_latents(video_latents: torch.Tensor) -> torch.Tensor:
    return video_latents[:, :, :, 1:]  # default [B, V, C_latent, T_future, H_latent, W_latent]


def prepare_joint_model_inputs(
    batch: dict[str, Any],
    vae: Any,
    context_builder: FastWAMContextBuilder,
    video_scheduler: WanContinuousFlowMatchScheduler | None = None,
    action_scheduler: WanContinuousFlowMatchScheduler | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    video_patch_size: tuple[int, int, int] = (1, 2, 2),
    condition_video_on_action: bool = False,
    tiled_vae: bool = False,
) -> JointInputAdapterOutput:
    video_scheduler = video_scheduler or WanContinuousFlowMatchScheduler()
    action_scheduler = action_scheduler or WanContinuousFlowMatchScheduler()

    context = batch["context"].to(device=device, dtype=dtype, non_blocking=True)  # default [B, L, D_context]
    context_mask = batch["context_mask"].to(device=device, dtype=torch.bool, non_blocking=True)  # default [B, L]
    proprio = batch["proprio"].to(device=device, dtype=dtype, non_blocking=True)  # default [B, T_action, D_state]
    context_builder = context_builder.to(device=device, dtype=dtype)
    context_out = context_builder(context=context, context_mask=context_mask, proprio=proprio)
    cross_context = context_out["context"]  # default [B, L+1, D_context]
    cross_context_mask = context_out["context_mask"]  # default [B, L+1]

    selected_video = batch["video"]  # default [B, V, 3, T_video, H, W]
    input_latents = encode_video_latents(
        vae=vae,
        video=selected_video,
        device=device,
        dtype=dtype,
        tiled=tiled_vae,
    )  # default [B, V, C_latent, T_latent, H_latent, W_latent]
    noised_video_latents, target_video, noise_video, timestep_video = _build_flow_training_inputs(
        input_latents, video_scheduler
    )
    noised_video_latents[:, :, :, 0:1] = input_latents[:, :, :, 0:1]  # default [B, V, C_latent, 1, H_latent, W_latent]

    action = batch["action"].to(device=device, dtype=dtype, non_blocking=True)  # default [B, H_action, D_action]
    noised_action, target_action, noise_action, timestep_action = _build_flow_training_inputs(
        action, action_scheduler
    )

    video_inputs: dict[str, torch.Tensor | bool | None] = {
        "x": noised_video_latents,
        "timestep": timestep_video,
        "context": cross_context,
        "context_mask": cross_context_mask,
        "fuse_vae_embedding_in_latents": True,
    }
    if condition_video_on_action:
        video_inputs["action"] = action

    action_inputs = {
        "action_tokens": noised_action,
        "timestep": timestep_action,
        "context": cross_context,
        "context_mask": cross_context_mask,
    }

    return JointInputAdapterOutput(
        video_inputs=video_inputs,
        action_inputs=action_inputs,
        targets={
            "video": target_video,
            "future_video": _future_video_latents(target_video),
            "action": target_action,
        },
        masks={
            "image_is_pad": batch.get("image_is_pad"),
            "action_is_pad": batch.get("action_is_pad"),
            "proprio_is_pad": batch.get("proprio_is_pad"),
        },
        context_inputs={
            "context": cross_context,
            "context_mask": cross_context_mask,
            "state_token": context_out["state_token"],
            "current_state": context_out["current_state"],
        },
        mot_counts=_infer_mot_counts(input_latents, action, video_patch_size),
        debug={
            "selected_video": selected_video,
            "input_latents": input_latents,
            "noise_video": noise_video,
            "noise_action": noise_action,
        },
    )
