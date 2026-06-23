第一优先级：Evo_1/scripts/Evo1.py
文件：Evo1.py (line 12)

这是整个模型的“装配入口”，非常值得看，因为它回答了三个关键问题：

InternVL3 的多模态特征是怎么接到 action head 上的
FlowmatchingActionHead 的 config 到底从哪来
训练和推理时，外层到底怎么调用 action head
尤其值得看：

init (line 13)：模型组装
predict_action (line 91)：训练/推理分流
run_inference (line 106)：真实推理入口
set_finetune_flags (line 140)：哪些模块被冻结、哪些被训练
如果你现在已经比较理解 flow_matching.py，那下一个最该看这个文件。因为它能帮你建立“整个系统是怎么串起来的”这一层视角。

第二优先级：Evo_1/model/internvl3/internvl3_embedder.py
文件：internvl3_embedder.py (line 71)

这是另一个核心模块。
flow matching 决定“怎么生成动作”，但这个文件决定“动作生成依赖的条件特征是什么”。

它值得看的原因：

图像怎么预处理、切块、送进 VLM
文本 prompt 怎么和图像 token 拼起来
最终给 action head 的 fused_tokens 到底是什么
最值得看：

_preprocess_images (line 105)
_build_multimodal_prompt (line 123)
_prepare_and_fuse_embeddings (line 146)
get_fused_image_text_embedding_from_tensor_images (line 224)
如果你想真正理解 fused_tokens 的来源，这个文件必须看。

第三优先级：Evo_1/dataset/lerobot_dataset_pretrain_mp.py
文件：lerobot_dataset_pretrain_mp.py (line 145)

这个文件很重要，因为它决定了模型到底学到什么格式的数据。
很多 action model 的理解难点，最后都不是模型本身，而是数据组织方式。

你会在这里看到：

多机器人 embodiment 是怎么编码的
state/action/image 是怎么 pad 到统一维度的
action_mask/state_mask/image_mask 是怎么来的
action_horizon 对应的序列是怎么切出来的
视频帧是怎么按时间戳读取的
最值得看：

init (line 146)
_load_metadata (line 203)
_pad_tensor (line 317)
getitem (line 402)
如果你后面想弄清楚 action_mask、embodiment_id、state_dim 这些变量为什么长这样，这个文件几乎一定要看。

第四优先级：Evo_1/scripts/train.py
文件：train.py (line 328)

这个文件不是模型结构，但它决定训练逻辑，是理解整个项目闭环的关键。

它值得看的点：

loss 到底怎么定义
pred_velocity 和 target_velocity 是怎么对上的
optimizer / scheduler / grad clip 怎么接进去
dataloader 出来的 batch 是怎样喂给模型的
最关键的位置：

prepare_dataset (line 142)
train (line 328)
pred_velocity, noise = model(...) (line 451)
target_velocity 与 loss (line 453)
backward + optimizer.step (line 481)
如果你已经在研究 flow matching 的数学意义，这个文件会让你看到它是怎么真正落地训练的。

第五优先级：Evo_1/scripts/Evo1_server.py
文件：Evo1_server.py (line 63)

这个文件适合在你已经理解训练后再看。它回答的是：

checkpoint 加载后怎么部署
state/action 的归一化与反归一化怎么做
实际推理输入输出是什么格式
run_inference() 在真实系统里怎么被调用
尤其如果你关心机器人实际控制链路，这个文件很有价值。