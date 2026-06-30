from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from Evo_1.model.action_head.dit_action_head import FlowmatchingDiTActionHead
from Evo_1.model.video_expert.mot import SkipLayerVideoActionMoT
from Evo_1.model.video_expert.wan_video_dit import WanVideoDiT
from Evo_1.training.model_input_adapter import JointInputAdapterOutput


@dataclass
class JointModelConfig:
    context_dim: int
    action_horizon: int
    action_dim: int
    vae_z_dim: int = 48
    hidden_dim: int = 64
    ffn_mult: int = 4
    num_heads: int = 4
    attn_head_dim: int = 24
    freq_dim: int = 16
    video_layers: int = 2
    action_layers: int = 1
    video_layer_stride: int = 2
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32


def build_joint_models(config: JointModelConfig) -> tuple[nn.Module, nn.Module, nn.Module]:
    video_expert = WanVideoDiT(
        hidden_dim=config.hidden_dim,
        in_dim=config.vae_z_dim,
        ffn_dim=config.hidden_dim * config.ffn_mult,
        out_dim=config.vae_z_dim,
        text_dim=config.context_dim,
        freq_dim=config.freq_dim,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=config.num_heads,
        attn_head_dim=config.attn_head_dim,
        num_layers=config.video_layers,
        has_image_input=False,
        has_image_pos_emb=False,
        has_ref_conv=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        action_conditioned=False,
        video_attention_mask_mode="first_frame_causal",
    ).to(device=config.device, dtype=config.dtype)

    action_expert = FlowmatchingDiTActionHead(
        embed_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        ffn_dim=config.hidden_dim * config.ffn_mult,
        action_dim=config.action_horizon * config.action_dim,
        horizon=config.action_horizon,
        per_action_dim=config.action_dim,
        num_heads=config.num_heads,
        attn_head_dim=config.attn_head_dim,
        num_layers=config.action_layers,
        freq_dim=config.freq_dim,
        max_position=max(1024, config.action_horizon),
    ).to(device=config.device, dtype=config.dtype)

    mot = SkipLayerVideoActionMoT(
        num_heads=config.num_heads,
        video_layer_stride=config.video_layer_stride,
    ).to(device=config.device)
    return video_expert, action_expert, mot


def joint_parameters(
    video_expert: nn.Module,
    action_expert: nn.Module,
    mot: nn.Module,
) -> list[nn.Parameter]:
    return list(video_expert.parameters()) + list(action_expert.parameters()) + list(mot.parameters())


def run_joint_forward(
    video_expert: nn.Module,
    action_expert: nn.Module,
    mot: nn.Module,
    prepared: JointInputAdapterOutput,
) -> dict[str, torch.Tensor]:
    # video_inputs.x: [B, C_latent, 3+T_future, H_latent, W_latent]
    # action_inputs.action_seq: [B, H_action, D_action]
    return mot(
        video_expert=video_expert,
        action_expert=action_expert,
        video_inputs=prepared.video_inputs,
        action_inputs=prepared.action_inputs,
        current_obs_token_counts=prepared.mot_counts["current_obs_token_counts"],
        future_obs_token_counts=prepared.mot_counts["future_obs_token_counts"],
        action_token_counts=prepared.mot_counts["action_token_counts"],
    )


def compute_joint_loss(
    prepared: JointInputAdapterOutput,
    outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    num_current_views = prepared.mot_counts["num_current_video_views"]
    pred_future_video = outputs["video"][:, :, num_current_views:]
    # default [B, C_latent, T_future, H_latent, W_latent]
    video_loss = F.mse_loss(pred_future_video.float(), prepared.targets["future_video"].float())
    action_loss = F.mse_loss(outputs["action"].float(), prepared.targets["action"].float())
    loss = video_loss + action_loss
    return {
        "loss": loss,
        "video_loss": video_loss,
        "action_loss": action_loss,
    }


def train_one_step(
    video_expert: nn.Module,
    action_expert: nn.Module,
    mot: nn.Module,
    prepared: JointInputAdapterOutput,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    video_expert.train()
    action_expert.train()
    mot.train()
    optimizer.zero_grad(set_to_none=True)

    outputs = run_joint_forward(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        prepared=prepared,
    )
    losses = compute_joint_loss(prepared, outputs)
    loss = losses["loss"]
    if not bool(torch.isfinite(loss).item()):
        raise ValueError(f"joint loss must be finite, got {float(loss.detach().cpu())}")
    loss.backward()
    optimizer.step()
    return {
        "outputs": outputs,
        "losses": losses,
    }


def detach_train_step_result(result: dict[str, Any]) -> dict[str, torch.Tensor]:
    outputs = result["outputs"]
    losses = result["losses"]
    return {
        "video": outputs["video"].detach(),
        "action": outputs["action"].detach(),
        "loss": losses["loss"].detach(),
        "video_loss": losses["video_loss"].detach(),
        "action_loss": losses["action_loss"].detach(),
    }
