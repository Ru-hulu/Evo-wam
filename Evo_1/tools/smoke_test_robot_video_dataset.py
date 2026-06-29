#!/usr/bin/env python3
"""Instantiate RobotVideoDataset and print one FAST-WAM-style sample or batch."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "Evo_1" / "configs" / "data" / "robotwin_smoke.json"


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
        concat_multi_camera=dataset_cfg.get("concat_multi_camera", "horizontal"),
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


def _print_sample(dataset: Any, idx: int) -> None:
    # Call _get directly so missing text-cache/data errors stay precise.
    sample = dataset._get(idx)  # noqa: SLF001
    for key in sorted(sample):
        print(f"{key}: {_describe_value(sample[key])}")


def _print_batches(
    dataset: Any,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    shuffle: bool,
    drop_last: bool,
) -> None:
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    print(f"batch_size: {batch_size}")
    print(f"num_workers: {num_workers}")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        print(f"batch_idx: {batch_idx}")
        for key in sorted(batch):
            print(f"{key}: {_describe_batch_value(batch[key])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", choices=["sample", "batch"], default="sample")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--dataset-dir", action="append", default=None)
    parser.add_argument("--pretrained-norm-stats", default=None)
    parser.add_argument("--text-embedding-cache-dir", default=None)
    parser.add_argument(
        "--fake-context-dim",
        type=int,
        default=None,
        help="Use zero text context with this feature dim instead of reading the text embedding cache.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))

    cfg = _load_config(args.config)
    if args.split not in cfg:
        raise KeyError(f"Split `{args.split}` not found in {args.config}. Available: {sorted(cfg)}")
    dataset_cfg = cfg[args.split]
    if args.dataset_dir is not None:
        dataset_cfg["dataset_dirs"] = args.dataset_dir
    if args.pretrained_norm_stats is not None:
        dataset_cfg["pretrained_norm_stats"] = args.pretrained_norm_stats
    if args.text_embedding_cache_dir is not None:
        dataset_cfg["text_embedding_cache_dir"] = args.text_embedding_cache_dir

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
    else:
        _print_batches(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_batches=args.num_batches,
            shuffle=args.shuffle,
            drop_last=args.drop_last,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
