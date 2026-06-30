#!/usr/bin/env python3
"""Smoke-test a minimal joint video/action training step."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "Evo_1" / "configs" / "data" / "robotwin_smoke.json"
sys.path.insert(0, str(REPO_ROOT))


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix == ".json":
            return json.load(f)

        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to read yaml configs. "
                "Install it or pass a .json config."
            ) from exc
        return yaml.safe_load(f)


def _import_object(target: str) -> Any:
    module_name, object_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _instantiate_transform(spec: dict[str, Any]) -> Any:
    target = spec["target"]
    cls = _import_object(target)
    kwargs = {k: v for k, v in spec.items() if k != "target"}
    return cls(**kwargs)


def _build_transforms(specs: list[dict[str, Any]] | dict[str, list[dict[str, Any]]] | None) -> Any:
    if specs is None:
        return None
    if isinstance(specs, dict):
        return {key: [_instantiate_transform(item) for item in value] for key, value in specs.items()}
    return [_instantiate_transform(item) for item in specs]


def _build_processor(dataset_cfg: dict[str, Any]) -> Any:
    from Evo_1.data.lerobot.processors.fastwam_processor import FastWAMProcessor
    from Evo_1.data.lerobot.transforms.action_state_merger import ConcatLeftAlign

    processor_cfg = dict(dataset_cfg["processor"])
    merger_cfg = dict(processor_cfg.pop("action_state_merger", {}))

    action_state_merger = ConcatLeftAlign(
        action_target_dim=merger_cfg.get("action_target_dim"),
        state_target_dim=merger_cfg.get("state_target_dim"),
    )

    return FastWAMProcessor(
        shape_meta=dataset_cfg["shape_meta"],
        num_obs_steps=dataset_cfg["num_frames"],
        num_output_cameras=processor_cfg["num_output_cameras"],
        action_output_dim=processor_cfg["action_output_dim"],
        proprio_output_dim=processor_cfg["proprio_output_dim"],
        action_state_transforms=processor_cfg.get("action_state_transforms"),
        use_stepwise_action_norm=processor_cfg.get("use_stepwise_action_norm", False),
        norm_default_mode=processor_cfg.get("norm_default_mode", "z-score"),
        norm_exception_mode=processor_cfg.get("norm_exception_mode"),
        action_state_merger=action_state_merger,
        train_transforms=_build_transforms(processor_cfg.get("train_transforms")),
        val_transforms=_build_transforms(processor_cfg.get("val_transforms")),
        delta_action_dim_mask=processor_cfg.get("delta_action_dim_mask"),
    )


def _build_dataset(dataset_cfg: dict[str, Any]) -> Any:
    from Evo_1.data.lerobot.robot_video_dataset import RobotVideoDataset

    processor = _build_processor(dataset_cfg)
    return RobotVideoDataset(
        dataset_dirs=dataset_cfg["dataset_dirs"],
        shape_meta=dataset_cfg["shape_meta"],
        num_frames=dataset_cfg.get("num_frames", 33),
        video_size=dataset_cfg.get("video_size", [384, 320]),
        camera_key=dataset_cfg.get("camera_key"),
        processor=processor,
        text_embedding_cache_dir=dataset_cfg.get("text_embedding_cache_dir"),
        context_len=dataset_cfg.get("context_len", 128),
        pretrained_norm_stats=dataset_cfg.get("pretrained_norm_stats"),
        val_set_proportion=dataset_cfg.get("val_set_proportion", 0.05),
        is_training_set=dataset_cfg.get("is_training_set", True),
        global_sample_stride=dataset_cfg.get("global_sample_stride", 1),
        action_video_freq_ratio=dataset_cfg.get("action_video_freq_ratio", 1),
        skip_padding_as_possible=dataset_cfg.get("skip_padding_as_possible", False),
        max_padding_retry=dataset_cfg.get("max_padding_retry", 3),
        concat_multi_camera=dataset_cfg.get("concat_multi_camera"),
        override_instruction=dataset_cfg.get("override_instruction"),
    )


def _describe_value(value: Any) -> str:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"shape={tuple(value.shape)}, dtype={value.dtype}"
    if isinstance(value, str):
        return f"str(len={len(value)})"
    return type(value).__name__


def _describe_batch_value(value: Any) -> str:
    if isinstance(value, list):
        if len(value) == 0:
            return "list(len=0)"
        first = value[0]
        if isinstance(first, str):
            return f"list[str](len={len(value)}, first_len={len(first)})"
        return f"list(len={len(value)}, first={type(first).__name__})"
    return _describe_value(value)


class ShapeOnlyWanVAE:
    """Shape-only VAE encoder for adapter smoke tests."""

    def __init__(self, z_dim: int = 48, temporal_downsample_factor: int = 4, upsampling_factor: int = 16):
        self.z_dim = int(z_dim)
        self.temporal_downsample_factor = int(temporal_downsample_factor)
        self.upsampling_factor = int(upsampling_factor)

    def encode(self, videos, device, tiled: bool = False):
        del tiled
        import torch

        if isinstance(videos, torch.Tensor):
            video = videos
        else:
            video = torch.stack(list(videos), dim=0)
        if video.ndim != 5:
            raise ValueError(f"`video` must be [B, 3, T, H, W], got {tuple(video.shape)}")
        batch_size, _, num_frames, height, width = video.shape
        if height % self.upsampling_factor != 0 or width % self.upsampling_factor != 0:
            raise ValueError(
                "Video H/W must be divisible by fake VAE upsampling factor "
                f"{self.upsampling_factor}, got {(height, width)}"
            )
        latent_t = (num_frames + self.temporal_downsample_factor - 1) // self.temporal_downsample_factor
        return torch.zeros(
            batch_size,
            self.z_dim,
            latent_t,
            height // self.upsampling_factor,
            width // self.upsampling_factor,
            device=device,
            dtype=video.dtype,
        )


def _describe_nested(prefix: str, value: Any) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            _describe_nested(f"{prefix}.{key}" if prefix else key, value[key])
        return
    print(f"{prefix}: {_describe_batch_value(value)}")


def _assert_adapter_contract(batch: dict[str, Any], prepared: Any) -> None:
    video = batch["video"]
    video_inputs = prepared.video_inputs["x"]
    action_inputs = prepared.action_inputs
    future_video = prepared.targets["future_video"]
    target_action = prepared.targets["action"]
    input_latents = prepared.debug["input_latents"]
    mot_counts = prepared.mot_counts

    assert video.ndim == 6, f"batch.video must be [B, V, 3, T, H, W], got {tuple(video.shape)}"
    assert video.shape[1] == 3, f"batch.video must have 3 views [h,l,r], got {video.shape[1]}"
    assert video_inputs.ndim == 5, f"video_inputs.x must be [B, C, S, H, W], got {tuple(video_inputs.shape)}"
    assert future_video.ndim == 5, f"targets.future_video must be [B, C, T_future, H, W], got {tuple(future_video.shape)}"
    assert input_latents.ndim == 6, f"debug.input_latents must be [B, V, C, T, H, W], got {tuple(input_latents.shape)}"
    assert input_latents.shape[1] == 3, f"debug.input_latents must have 3 views [h,l,r], got {input_latents.shape[1]}"
    assert video_inputs.shape[2] == 3 + future_video.shape[2], (
        "video_inputs.x token frames must be h0,l0,r0 plus future h frames, "
        f"got {video_inputs.shape[2]} and future {future_video.shape[2]}"
    )
    assert mot_counts["current_obs_token_counts"] == 3 * mot_counts["tokens_per_view_frame"], (
        "current_obs_token_counts must cover h0,l0,r0"
    )
    assert prepared.video_inputs["current_obs_token_counts"] == mot_counts["current_obs_token_counts"], (
        "video_inputs.current_obs_token_counts must match mot_counts"
    )
    assert mot_counts["future_obs_token_counts"] == future_video.shape[2] * mot_counts["tokens_per_view_frame"], (
        "future_obs_token_counts must cover only future h frames"
    )
    assert action_inputs["action_seq"].shape == target_action.shape, (
        f"action_inputs.action_seq must match target action shape, got {tuple(action_inputs['action_seq'].shape)} "
        f"and {tuple(target_action.shape)}"
    )
    assert action_inputs["fused_tokens"].shape == prepared.context_inputs["context"].shape, (
        "action_inputs.fused_tokens must reuse adapter context tokens"
    )
    assert action_inputs["state"].shape == prepared.context_inputs["current_state"].shape, (
        "action_inputs.state must reuse current proprio state"
    )
    assert action_inputs["embodiment_id"].shape == (target_action.shape[0],), (
        f"action_inputs.embodiment_id must be [B], got {tuple(action_inputs['embodiment_id'].shape)}"
    )
    print("adapter_contract: ok")


def _assert_mot_contract(prepared: Any, outputs: dict[str, Any]) -> None:
    assert outputs["video"].shape == prepared.targets["video"].shape, (
        f"mot video output must match video target shape, got {tuple(outputs['video'].shape)} "
        f"and {tuple(prepared.targets['video'].shape)}"
    )
    assert outputs["action"].shape == prepared.targets["action"].shape, (
        f"mot action output must match action target shape, got {tuple(outputs['action'].shape)} "
        f"and {tuple(prepared.targets['action'].shape)}"
    )
    assert outputs["video_tokens"].shape[1] == (
        prepared.mot_counts["current_obs_token_counts"] + prepared.mot_counts["future_obs_token_counts"]
    ), "mot video token count must match current + future video counts"
    assert outputs["action_tokens"].shape[1] == prepared.mot_counts["action_token_counts"], (
        "mot action token count must match action_token_counts"
    )
    print("mot_contract: ok")


def _build_tiny_joint_models(args: argparse.Namespace, model_cfg: dict[str, Any], dtype: Any):
    from Evo_1.training.joint_trainer import JointModelConfig, build_joint_models

    config = JointModelConfig(
        context_dim=int(model_cfg["context_dim"]),
        action_horizon=int(model_cfg["action_horizon"]),
        action_dim=int(model_cfg["action_dim"]),
        vae_z_dim=int(model_cfg["vae_z_dim"]),
        hidden_dim=args.mot_hidden_dim,
        ffn_mult=args.mot_ffn_mult,
        num_heads=args.mot_num_heads,
        attn_head_dim=args.mot_attn_head_dim,
        freq_dim=args.mot_freq_dim,
        video_layers=args.mot_video_layers,
        action_layers=args.mot_action_layers,
        video_layer_stride=args.mot_video_layer_stride,
        device=args.device,
        dtype=dtype,
    )
    return build_joint_models(config)


def _run_train_step_smoke(
    video_expert: Any,
    action_expert: Any,
    mot: Any,
    optimizer: Any,
    prepared: Any,
) -> dict[str, Any]:
    from Evo_1.training.joint_trainer import detach_train_step_result, train_one_step

    result = train_one_step(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        prepared=prepared,
        optimizer=optimizer,
    )
    _assert_mot_contract(prepared, result["outputs"])
    print("train_step_contract: ok")
    return detach_train_step_result(result)


def _prepare_inputs_for_batch(args: argparse.Namespace, batch: dict[str, Any], context_builder: Any, fake_vae: Any, dtype: Any):
    from Evo_1.training.model_input_adapter import prepare_joint_model_inputs

    prepared = prepare_joint_model_inputs(
        batch=batch,
        vae=fake_vae,
        context_builder=context_builder,
        device=args.device,
        dtype=dtype,
    )
    _assert_adapter_contract(batch, prepared)
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset-dir", action="append", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--vae-spatial-factor", type=int, default=16)
    parser.add_argument("--vae-temporal-factor", type=int, default=4)
    parser.add_argument("--mot-hidden-dim", type=int, default=64)
    parser.add_argument("--mot-ffn-mult", type=int, default=4)
    parser.add_argument("--mot-num-heads", type=int, default=4)
    parser.add_argument("--mot-attn-head-dim", type=int, default=24)
    parser.add_argument("--mot-freq-dim", type=int, default=16)
    parser.add_argument("--mot-video-layers", type=int, default=2)
    parser.add_argument("--mot-action-layers", type=int, default=1)
    parser.add_argument("--mot-video-layer-stride", type=int, default=2)
    parser.add_argument("--train-lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.split not in cfg:
        raise KeyError(f"Split `{args.split}` not found in {args.config}. Available: {sorted(cfg)}")

    dataset_cfg = cfg[args.split]
    if args.dataset_dir is not None:
        dataset_cfg["dataset_dirs"] = args.dataset_dir

    dataset = _build_dataset(dataset_cfg)

    print(f"dataset_len: {len(dataset)}")

    import torch
    from torch.utils.data import DataLoader

    from Evo_1.training.model_input_adapter import FastWAMContextBuilder

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    model_cfg = dataset_cfg["model"]
    state_dim = int(model_cfg["state_dim"])
    context_dim = int(model_cfg["context_dim"])
    vae_z_dim = int(model_cfg["vae_z_dim"])
    context_builder = FastWAMContextBuilder(state_dim=state_dim, context_dim=context_dim)
    fake_vae = ShapeOnlyWanVAE(
        z_dim=vae_z_dim,
        temporal_downsample_factor=args.vae_temporal_factor,
        upsampling_factor=args.vae_spatial_factor,
    )

    if args.max_steps <= 0:
        raise ValueError(f"`max_steps` must be positive, got {args.max_steps}")
    from Evo_1.training.joint_trainer import joint_parameters

    video_expert, action_expert, mot = _build_tiny_joint_models(args, model_cfg, dtype)
    optimizer = torch.optim.AdamW(joint_parameters(video_expert, action_expert, mot), lr=args.train_lr)
    dataloader_iter = iter(dataloader)
    for step_idx in range(args.max_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        prepared = _prepare_inputs_for_batch(args, batch, context_builder, fake_vae, dtype)

        outputs = _run_train_step_smoke(video_expert, action_expert, mot, optimizer, prepared)
        print(f"train_step_idx: {step_idx}")
        _describe_nested("outputs", outputs)
    print(f"batch_size: {args.batch_size}")
    print(f"max_steps: {args.max_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
