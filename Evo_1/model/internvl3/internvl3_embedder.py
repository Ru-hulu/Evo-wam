# model/internvl3/internvl3_embedder.py
import torch
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers import GenerationConfig
from torchvision.transforms.functional import to_pil_image
from typing import Union, List
from torch import nn
import logging
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# === Image Transformations ===
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

# === Aspect Ratio Handling ===
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

class InternVL3Embedder(nn.Module):
    def __init__(self, model_name="OpenGVLab/InternVL3-1B", image_size=448, device="cuda"):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.max_text_length = 1024  # InternVL3 supports up to 1024 tokens
        self.transform = build_transform(image_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=True,
            low_cpu_mem_usage=True,
            _fast_init=False,
        ).to(self.device) 
        
        if hasattr(self.model.language_model, 'model'):
            layers = self.model.language_model.model.layers

        else:
            layers = self.model.language_model.layers
        layers = layers[:14]

        if hasattr(self.model.language_model, 'model'):
            self.model.language_model.model.layers = torch.nn.ModuleList(layers)
        else:
            self.model.language_model.layers = torch.nn.ModuleList(layers)
        self.model.language_model.lm_head = torch.nn.Identity()

        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
            self.model.vision_model.encoder.gradient_checkpointing = False
# 输入：2 张图片
#     图 1：切成 4 块
#     图 2：切成 9 块
# 输出：
#     pixel_values → shape = (9 + 4, 3, 448, 448)
#     num_tiles_list → [4, 9]
# 当前默认配置下，输入 [3, 3, 448, 448]
# 每张图片被切成一块（没有切） 
# num_tiles_list[1, 1, 1]
# pixel_values[3, 3, 448, 448]
    def _preprocess_images(
        self,
        image_tensors: List[Union[Image.Image, torch.Tensor]]
    ) -> (torch.Tensor, List[int]):

        pixel_values_list = []
        for i, image in enumerate(image_tensors):
            if isinstance(image, torch.Tensor):
                image = to_pil_image(image)
            tiles = dynamic_preprocess(image, image_size=self.image_size)
            tile_tensors = torch.stack([self.transform(t) for t in tiles])  # (T_i, 3, 448, 448)
            pixel_values_list.append(tile_tensors)

        pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype=torch.bfloat16, device=self.device)
        num_tiles_list = [pv.shape[0] for pv in pixel_values_list]
        return pixel_values, num_tiles_list
    # 根据每张图片被切了多少块（num_tiles_list），
    # 自动生成对应数量的 <IMG_CONTEXT> 图片占位 token，
    # 替换掉 prompt 里的 <image>，
    # 最终生成模型能直接输入的多模态 prompt。
    # 例如：
    # Image-1: <img> + <IMG_CONTEXT> * (256*i) + </img> i是图像被切成了多少块
    # Image-2: <img> + <IMG_CONTEXT> * (256*i) + </img> i是图像被切成了多少块 
    # Image-3: <img> + <IMG_CONTEXT> * (256*i) + </img> i是图像被切成了多少块 
    # text_prompt （2张图片里有什么？）
    def _build_multimodal_prompt(
        self,
        num_tiles_list: List[int],
        text_prompt: str
    ) -> str:

        prompt = ''
        for i in range(len(num_tiles_list)):
            prompt += f"Image-{i+1}: <image>\n"
        prompt += text_prompt.strip()
        # Image-1: <image>
        # Image-2: <image>
        # 2张图片里有什么？
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"

        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        for tile_count in num_tiles_list:
            token_count = self.model.num_image_token * tile_count
            # num_image_token只取决于 ViT 编码器的结构，不是我们可以调的参数，一般是256
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
            # <img> + <IMG_CONTEXT> * (256*1) + </img> 
            prompt = prompt.replace("<image>", image_tokens, 1)
        return prompt
            # prompt 在循环结束后变成
            # <img> + <IMG_CONTEXT> * (256*1) + </img> 
            # <img> + <IMG_CONTEXT> * (256*1) + </img> 
            # 2张图片里有什么？
    
    # 将prompt 转换为token向量。维度[1，1024, 896]，就是1024个token，每个维度896
    # 以及维度为[1, 1024]的mask，标记哪些token 是有效的
    # 注意，这里已经在处理一条数据，而不是一个batch 的数据了
    def _prepare_and_fuse_embeddings(
        self,
        prompt: str,
        vit_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        num_tiles_list: List[int]
    ) -> (torch.Tensor, torch.Tensor):
   
        untruncated_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        # 这里用tokenizer处理prompt，然后获得维度
        true_sequence_length = untruncated_ids.shape[1]
        # 一共有多少token

        if true_sequence_length > self.max_text_length:
            print("\n" + "="*80)
            print(f" WARNING: Input prompt was TRUNCATED!")
            print(f"   - Max Length Allowed    : {self.max_text_length}")
            print(f"   - Actual Length      : {true_sequence_length}")
            print(f"   - Truncated Prompt (first 100 chars): '{prompt[:100]}...'")
            print("="*80 + "\n")

        model_inputs = self.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_text_length).to(self.device)
        input_ids = model_inputs["input_ids"]
        # 这里用tokenizer处理prompt，然后获得 token_id
        attention_mask = model_inputs["attention_mask"]
        # 大概长这样，0表示当前token 不需要模型关注。[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        # 和 attention matrix 不是一个概念。
        img_token_mask = (input_ids == self.img_context_token_id)
        # input_ids：整个 prompt 编码后的数字序列
        # self.img_context_token_id：<IMG_CONTEXT> 对应的固定数字（比如 32100）
        img_token_locations = torch.where(img_token_mask)[1]
        # 把input_ids 中对应<IMG_CONTEXT>的 索引 提取出来
        # 例如：[5,6,7,...,1028] 这一长串就是图片占位符的位置

        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        # input_ids 是tocken的整数对应，找到每个id对应的token向量。
        # [1, 1024, 896] 注意，这我们已经是处理一个batch 中的一条数据了，所以B = 1，有1024个token ，维度是896
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C) # tocken 对应的向量，C是向量的维度
        input_ids = input_ids.reshape(B * N) # tocken 编码

        selected = (input_ids == self.img_context_token_id)

            
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            # 这里不要大语言模型对<IMG_CONTEXT>的编码，而是用vit先前对image tocken的编码替换
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

 
        tokens_per_tile = self.model.num_image_token  # 一张image的小块被映射为多少tocken，配置下为16*16 = 256
 
        torch.set_printoptions(profile="full", threshold=float('inf'))
   
        torch.set_printoptions(profile="default")
        current_token_idx = 0
        for i in range(len(image_mask)):
           
            num_tiles_for_this_image = num_tiles_list[i] ## 当前这张Image 被切成了几个块，但是当前配置下是1.
            num_tokens_for_this_image = num_tiles_for_this_image * tokens_per_tile # 256
       
            if not image_mask[i]:
                start_idx = img_token_locations[current_token_idx]
                end_idx = start_idx + num_tokens_for_this_image
                attention_mask[0, start_idx:end_idx] = 0
            ## 如果当前图像是无效的，则把它对应的token 位置打上mask
            current_token_idx += num_tokens_for_this_image

        input_embeds = input_embeds.reshape(B, N, C)
    
        torch.set_printoptions(profile="full", threshold=float('inf'))
     
        torch.set_printoptions(profile="default")
        return input_embeds, attention_mask


    def get_fused_image_text_embedding_from_tensor_images(
        self,
        image_tensors: list[Union[Image.Image, torch.Tensor]], #[3, 3, 448, 448]
        image_mask: torch.Tensor,
        text_prompt: str,
        return_cls_only: bool = True,
    ):

        # pixel_values → shape = (13, 3, 448, 448)
        # num_tiles_list → [4, 9]
        pixel_values, num_tiles_list = self._preprocess_images(image_tensors)

       
        if pixel_values.shape[0] == 0:
           
            print("Warning: No valid images to process after masking.")

        vit_embeds = self.model.extract_feature(pixel_values) 
        # 图像先过 Intern ViT 300M
        # vit_embeds.shape = [3, 256, 896]
        # 3   = 输入的 3 张图
        # 256 = 每张图变成 256 个视觉 token
        # 896 = 每个视觉 token 的 embedding 维度
        fused_embeds = vit_embeds # 图像被提取出来的特征信息
        # 图像ViT提取出来的特征 (总块数×256, 块数就是前面切的，256就是_build_multimodal_prompt中的num_image_token)
        prompt = self._build_multimodal_prompt(num_tiles_list, text_prompt) # 图像占位符 + 文字prompt
        inputs_embeds, attention_mask = self._prepare_and_fuse_embeddings(prompt, fused_embeds, image_mask, num_tiles_list)
        # 文本 token + 图像 token → 拼在一起 → 变成 LLM 能看懂的输入 embedding
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        ) # LLM进行推理，就是过Qwen2.5 0.5B
        fused_hidden = outputs.hidden_states[-1].to(torch.float32)

        return fused_hidden[:, 0, :] if return_cls_only else fused_hidden
