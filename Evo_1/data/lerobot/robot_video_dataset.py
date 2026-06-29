import hashlib
import logging
import os
from pathlib import Path
from typing import Optional
import time
import numpy as np
import traceback
import torch
from contextlib import contextmanager

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig = ()

    class OmegaConf:
        @staticmethod
        def to_container(value, resolve=True):
            return value

try:
    from hydra.utils import instantiate
except ImportError:
    def instantiate(_cfg):
        raise ImportError("hydra-core is required to instantiate processor configs.")

try:
    from accelerate import PartialState
except ImportError:
    class PartialState:
        is_main_process = True

from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
logger = logging.getLogger(__name__)


def _get_work_dir() -> str:
    work_dir = os.environ.get("EVO_WAM_WORK_DIR", "./runs/")
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    return work_dir


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: Optional[str] = None, # ignored: cameras are always kept independent
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )
    
        self.num_frames = num_frames # default 33
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio)) # default len=9

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        del concat_multi_camera
        self.override_instruction = override_instruction

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = _get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = _get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    def __len__(self):
        return len(self.lerobot_dataset)

    def _resize_crop_normalize_video(self, video: torch.Tensor) -> torch.Tensor:
        num_views, T_video, C, H, W = video.shape
        video = video.reshape(num_views * T_video, C, H, W)  # default [num_views*T_video, 3, H, W]
        video = self.resize_transform(video) # default [num_views*T_video, 3, H', W']
        video = self.crop_transform(video) # default [num_views*T_video, 3, H_out, W_out]
        video = self.normalize_transform(video)  # default [num_views*T_video, 3, H_out, W_out]
        _, C, H, W = video.shape
        return video.view(num_views, T_video, C, H, W)  # default [num_views, T_video, 3, H_out, W_out]

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]
            # RobotVideoDataset._get              # 最终整理成训练 batch：video/action/proprio/context
            # -> BaseLerobotDataset.__getitem__   # 组装单条机器人样本：images/state/action/masks
            # -> MultiLeRobotDataset.__getitem__  # 多数据集逻辑拼接：根据 idx 选择子 dataset
            # -> LeRobotDataset.__getitem__       # 读取标准 LeRobot：parquet 行 + video frame
            # -> BaseLerobotDataset._get_image    # 提取图像窗口：[33, 3, H, W]
            # -> BaseLerobotDataset._get_state    # 提取状态窗口：[33, raw_state_dim]
            # -> BaseLerobotDataset._get_action   # 提取动作窗口：[32, raw_action_dim]
            # -> FastWAMProcessor.preprocess      # 图像处理 + 归一化 + pad：pixel_values/action/proprio
            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"] # default [32]
            image_is_pad = sample["image_is_pad"] # default [33]
            proprio_is_pad = sample["proprio_is_pad"] # default [33]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"] # default [33] 长度为 33 的 bool 向量，表示 33 帧图像/视频里哪些是 padding。

        video = sample["pixel_values"]  # default [num_cameras, 33, 3, H, W]
        if video.ndim == 5: # 如果是多相机的情况
            video = video[:, self.video_sample_indices, :, :, :] # default [num_cameras, 9, 3, H, W]
        else:# 如果是单相机的情况
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :].unsqueeze(0) # default [1, 9, 3, H, W]
        num_cameras, T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices] # default [9] video_sample_indices 可能是0 4 8 这样的采样

        video = video.view(num_cameras, T_video, C, H, W)  # default [num_views, 9, 3, H, W]

        # final resize and normalization
        video = self._resize_crop_normalize_video(video)

        video = video.permute(0, 2, 1, 3, 4).contiguous() # default [num_views, 3, 9, H_out, W_out]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # default [32, D_action]
        proprio = sample["proprio"][:-1, :] # default [32, D_state]
        if video.shape[2] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[2] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[2] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction) # [128, D_context], [128]
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0 # 无效token全部置为0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video, # default [num_views, 3, 9, H_out, W_out]
            "action": action, # default [32, D_action]
            "proprio": proprio, # default [32, D_state]
            "prompt": instruction,
            "context": context, # default [128, D_context]
            "context_mask": context_mask, # default [128]
            "image_is_pad": image_is_pad, # default [9]
            "action_is_pad": sample["action_is_pad"], # default [32]
            "proprio_is_pad": sample["proprio_is_pad"], # default [33]
        }
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"] # default [128, D_context]
        context_mask = payload["mask"].bool() # default [128]
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
