# Evo-WAM 联合训练接口约定

这份文档只用来固定后续增量修改时的输入输出形状和模块边界。它不是完整实现方案，也不提前展开太多代码细节。后面每一步改代码时，都可以拿它当对照表。

## 目标

新的训练流程需要把当前仓库里已经有的「视频专家」和「动作专家」通过现有 MoT attention 设计串起来。

和 FAST-WAM 的主要区别：

- 为了先跑通训练闭环，视频专家只使用主视角观测；左/右手视角暂时不进入视频专家。
- 未来帧监督只预测主视角。
- 视频和动作的交互使用当前仓库已有的 skip-layer 设计：
  视频专家 30 层，动作专家 10 层，每 3 个视频层和 1 个动作层交互一次。
- FAST-WAM 的 VAE、text context、continuous flow-matching scheduler 可以迁移过来，但 attention token 顺序以当前仓库为准。

## Batch 约定

当前动作训练已有字段先保持不变，这样旧训练路径在迁移过程中还能继续跑：

- `images`：当前时刻的多视角图像，用于旧的 InternVL/action-only 训练。
- `image_mask`：`images` 中哪些视角有效。
- `prompt`：任务文本指令。
- `state`：当前时刻归一化后的机器人状态。
- `state_mask`：哪些 state 维度是真实维度，哪些是 padding。
- `action`：归一化后的动作序列，形状 `[B, H, Da]`。
- `action_mask`：哪些 action 维度是真实维度，形状 `[B, H, Da]`。
- `embodiment_id`：机器人类型或 embodiment 类别 id。

联合训练路径会新增这些字段：

- `current_main_video`：当前时刻主视角帧，形状 `[B, C, Himg, Wimg]`。
- `future_main_video`：未来主视角帧序列，形状 `[B, T_future, C, Himg, Wimg]`。
- `video_is_pad`：视频监督的有效帧 mask，形状 `[B, 1 + T_future]`；第一个位置对应当前主视角帧。
- `context`：统一的文本/state context token，形状 `[B, L, Dc]`。
- `context_mask`：哪些 context token 有效，形状 `[B, L]`。

图像尺寸、视频帧数、动作和视频频率比例都应该是配置项，不应该硬编码在训练循环里。

## Context 约定

视频专家和动作专家统一使用同一种 context 风格：

- 文本 context 参考 FAST-WAM/Wan 的方式，优先从 cache 中读取。
- 当前 state/proprio 需要投影到同一个 context 维度，并作为一个额外有效 token 拼到文本 context 后面。
- 视频专家和动作专家内部可以各自有 context projection，但外部输入接口统一为 `context/context_mask`。

联合训练路径里，动作专家不再依赖 InternVL 的 fused tokens。动作专家通过 MoT attention 从当前主视角观测 token 中获取视觉信息。

## 视频专家约定

视频专家接收的 VAE latent 顺序为：

1. 当前主视角。
2. 未来主视角 latent 帧。

视频专家只对未来主视角 denoising target 计算 loss。当前主视角是条件观测，不参与 video loss。

MoT self-attention 的 token 分组为：

- `current_obs`：当前主视角 token。
- `future_obs`：未来主视角 token。
- `action`：动作 token。

attention mask 必须保持 `Evo_1/model/video_expert/att.md` 里的语义：

- 当前观测只能看当前观测内部。
- 未来观测可以看当前观测和未来观测。
- 动作可以看当前观测和动作。
- 动作不能看未来观测。

## 动作专家约定

动作专家接收加噪后的归一化动作，形状 `[B, H, Da]`。

动作专家输出预测的动作 velocity/noise target，形状 `[B, H, Da]`。

动作 loss 必须使用 `action_mask`，padding 出来的动作维度不能参与 loss。

## Loss 约定

联合训练返回：

- `loss_video`：只在未来主视角 latent target 上计算 masked MSE。
- `loss_action`：只在有效动作 step 和有效动作维度上计算 masked MSE。
- `loss_total = lambda_video * loss_video + lambda_action * loss_action`。

模型封装层建议暴露一个统一方法：

```python
loss, loss_dict = model.training_loss(batch)
```

训练脚本主要负责 dataloader、optimizer、scheduler、日志和 checkpoint，不把复杂的 joint loss 逻辑塞进训练循环。

## 增量迁移顺序

1. 保持这份接口约定稳定。
2. 迁移 scheduler、VAE、text-context 工具，但暂时不接旧 trainer。
3. 扩展 dataset，让它新增联合训练字段，同时保留旧字段。
4. 加入统一 context 准备逻辑，并先验证两个专家的 `pre_dit` shape。
5. 新增 joint model wrapper，再把它接入新的训练模式。
