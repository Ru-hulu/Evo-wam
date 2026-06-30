#!/usr/bin/env python3
"""Smoke-test RobotVideoDataset and joint video/action adapter inputs."""

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


def _patch_fake_text_context(dataset: Any, context_len: int, context_dim: int) -> None:
    import torch

    def fake_context(_prompt: str):
        context = torch.zeros(context_len, context_dim)  # default [128, D_context]
        context_mask = torch.ones(context_len, dtype=torch.bool)  # default [128]
        return context, context_mask

    dataset._get_cached_text_context = fake_context  # noqa: SLF001


def _print_flat(item: dict[str, Any], prefix: str = "") -> None:
    for key in sorted(item):
        name = f"{prefix}.{key}" if prefix else key
        print(f"{name}: {_describe_batch_value(item[key])}")


def _print_sample(dataset: Any, idx: int) -> None:
    sample = dataset._get(idx)  # noqa: SLF001
    _print_flat(sample)


def _print_batches(
    dataset: Any,
    batch_size: int,
    num_workers: int,
    num_batches: int,
) -> None:
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(f"batch_size: {batch_size}")
    print(f"num_workers: {num_workers}")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        print(f"batch_idx: {batch_idx}")
        _print_flat(batch)


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
    future_video = prepared.targets["future_video"]
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
    print("adapter_contract: ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", choices=["adapter", "sample", "batch"], default="adapter")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--dataset-dir", action="append", default=None)
    parser.add_argument("--fake-context-dim", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--vae-z-dim", type=int, default=48)
    parser.add_argument("--vae-spatial-factor", type=int, default=16)
    parser.add_argument("--vae-temporal-factor", type=int, default=4)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.split not in cfg:
        raise KeyError(f"Split `{args.split}` not found in {args.config}. Available: {sorted(cfg)}")

    dataset_cfg = cfg[args.split]
    if args.dataset_dir is not None:
        dataset_cfg["dataset_dirs"] = args.dataset_dir

    dataset = _build_dataset(dataset_cfg)
    if args.fake_context_dim is not None:
        _patch_fake_text_context(
            dataset,
            context_len=dataset_cfg.get("context_len", 128),
            context_dim=args.fake_context_dim,
        )

    print(f"dataset_len: {len(dataset)}")

    if args.mode == "sample":
        _print_sample(dataset, args.idx)
        return 0

    if args.mode == "batch":
        _print_batches(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_batches=args.num_batches,
        )
        return 0

    import torch
    from torch.utils.data import DataLoader

    from Evo_1.training.model_input_adapter import (
        FastWAMContextBuilder,
        prepare_joint_model_inputs,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    batch = next(iter(dataloader))
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    state_dim = int(batch["proprio"].shape[-1])
    context_dim = int(batch["context"].shape[-1])
    context_builder = FastWAMContextBuilder(state_dim=state_dim, context_dim=context_dim)
    fake_vae = ShapeOnlyWanVAE(
        z_dim=args.vae_z_dim,
        temporal_downsample_factor=args.vae_temporal_factor,
        upsampling_factor=args.vae_spatial_factor,
    )

    prepared = prepare_joint_model_inputs(
        batch=batch,
        vae=fake_vae,
        context_builder=context_builder,
        device=args.device,
        dtype=dtype,
    )
    _assert_adapter_contract(batch, prepared)

    print(f"batch_size: {args.batch_size}")
    _print_flat(batch, prefix="batch")
    print("adapter:")
    _describe_nested("video_inputs", prepared.video_inputs)
    _describe_nested("action_inputs", prepared.action_inputs)
    _describe_nested("targets", prepared.targets)
    _describe_nested("context_inputs", prepared.context_inputs)
    _describe_nested("debug", prepared.debug)
    for key in sorted(prepared.mot_counts):
        print(f"mot_counts.{key}: {prepared.mot_counts[key]}")
    for key in sorted(prepared.masks):
        print(f"masks.{key}: {_describe_value(prepared.masks[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
