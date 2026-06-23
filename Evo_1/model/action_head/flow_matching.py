import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class SinusoidalPositionalEncoding(nn.Module): ## 给动作序列的时间步位置编码
    def __init__(self, dim: int, max_len: int = 1000): ## dim 896 max_len 1000
        super().__init__()
        pe = torch.zeros(max_len, dim) # 1. 创建一个全 0 的位置编码矩阵，形状 [max_len, dim]
        position = torch.arange(0, max_len).unsqueeze(1) # 2. 生成位置索引：0,1,2,...,999 [max_len] -> 形状变成 [max_len, 1]
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)) # 1/10000^{2i/dim} 形状[dim/2]
        pe[:, 0::2] = torch.sin(position * div_term) ## 2i的位置编码
        pe[:, 1::2] = torch.cos(position * div_term) ## 2i+1的位置编码
        # pe的第i行内容为 sin(ai) cos(ai) ... sin(ai) cos(ai),其中ai仅与i相关
        pe = pe.unsqueeze(0)  ## [max_len, dim]->[1, max_len, dim]
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int): ## 如果传入的维度大于了pe的max_len，那么位置编码是不够长的，会把 pe维度从 [1,1000,896] -> [1,more,896]
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len)
        return self.pe[:, :seq_len, :]

    def _extend_pe(self, new_max_len):
        old_max_len, dim = self.pe.size(1), self.pe.size(2) ## size(0) = 1 size(1) = 1000（旧的最大长度） size(2) = 896（编码维度）
        if new_max_len <= old_max_len:
            return
        extra_positions = torch.arange(old_max_len, new_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        extra_pe = torch.zeros(new_max_len - old_max_len, dim)
        extra_pe[:, 0::2] = torch.sin(extra_positions * div_term)
        extra_pe[:, 1::2] = torch.cos(extra_positions * div_term)
        extra_pe = extra_pe.unsqueeze(0)
        new_pe = torch.cat([self.pe, extra_pe.to(self.pe.device)], dim=1)
        self.pe = new_pe

class CategorySpecificLinear(nn.Module):  ## 可以理解为一个线性层,作了线性变换 x->ax+b->out
    def __init__(self, in_dim: int, out_dim: int, num_categories: int = 1):
        super().__init__()
        self.num_categories = num_categories
        if num_categories <= 1:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            # 形状：[num_categories, in_dim, out_dim] 每个类别都有一个独立的 [in_dim, out_dim] 矩阵
            self.weight = nn.Parameter(torch.randn(num_categories, in_dim, out_dim))
            # 形状：[num_categories, out_dim] 每个类别都有一个独立的 [out_dim] 偏置
            self.bias = nn.Parameter(torch.randn(num_categories, out_dim))

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):

        if self.num_categories <= 1:
            return self.linear(x)

        orig_shape = x.shape ## 例如 x.shape = [Batch, 序列长度, 维度] [8, 100, 896]
        x_flat = x.reshape(-1, orig_shape[-1]) # x.reshape(-1, 896)->[8*100 896]        
        if category_id.dim() == 0:       
            cid = category_id.item()
            out = x_flat @ self.weight[cid] + self.bias[cid]
        else:
            # x_flat.shape = [B, in_dim]
            # category_id.shape = [B]（每个样本对应 1 个类别）
            # self.weight.shape = [num_categories, in_dim, out_dim]
            # self.bias.shape = [num_categories, out_dim]
            category_id = category_id.view(-1)  # 把类别 ID 张量展平成一维，为[B]
            weight_selected = self.weight[category_id] # [B, in_dim, out_dim]
            bias_selected = self.bias[category_id] # [B, out_dim]
            out = torch.bmm(x_flat.unsqueeze(1), weight_selected).squeeze(1) + bias_selected
            # bmm([B, in_dim]->[B, 1, in_dim], [B, in_dim, out_dim]) ->[B, 1, out_dim]->squeeze(1) -> [B, out_dim]
        out_shape = orig_shape[:-1] + (out.shape[-1],)
        return out.view(out_shape)

class CategorySpecificMLP(nn.Module): ## 堆了一个两层 CategorySpecificLinear: x->线性变换->ReLU->线性变换->out

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_categories: int = 1):
        super().__init__()
        self.fc1 = CategorySpecificLinear(input_dim, hidden_dim, num_categories)
        self.fc2 = CategorySpecificLinear(hidden_dim, output_dim, num_categories)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):
        out = self.activation(self.fc1(x, category_id))
        out = self.fc2(out, category_id)
        return out

class MultiEmbodimentActionEncoder(nn.Module): ## 把动作序列 action_seq 编码成 token 序列。

    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int, horizon: int, num_categories: int = 1):
        super().__init__()
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.num_categories = num_categories
        
        self.W1 = CategorySpecificLinear(action_dim, hidden_dim, num_categories)
        self.W2 = CategorySpecificLinear(hidden_dim, hidden_dim, num_categories)
        self.W3 = CategorySpecificLinear(hidden_dim, embed_dim, num_categories)
   
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=horizon) ## 输出为[1, horizon, hidden_dim]
        self.activation = nn.ReLU(inplace=True)

    def forward(self, action_seq: torch.Tensor, category_id: torch.LongTensor):

        B, H, D = action_seq.shape
        assert H == self.horizon, "Action sequence length must match horizon"
        
        x = action_seq.reshape(B * H, D)
        # 对于tensor dim = 0 不是列表，不是数组，就是一个单独的数字 不能用 category_id[0] 取值
        if category_id.dim() == 0:
            cat_ids = category_id.repeat(H * B) # -> [H*B]，从0维变1维。不是[1,H*B]
        else:
            cat_ids = category_id.unsqueeze(1).repeat(1, H).reshape(B * H) # [B]->[B,1]->[B,H]->[B*H]
        out = self.activation(self.W1(x, cat_ids)) # 输入 [B*H, D] [B*H] 输出 [B*H, hidden_dim]
        pos_enc = self.pos_encoding(H).to(out.device) ## 等价于 self.pos_encoding.forward(H)->[1, H, hidden_dim]
        pos_enc = pos_enc.repeat(B, 1, 1).reshape(B * H, -1) # 变成 [B*H, hidden_dim]
        out = out + pos_enc # 数值相加
        out = self.activation(self.W2(out, cat_ids)) # 输入 [B*H, hidden_dim] [B*H] 输出 [B*H, hidden_dim]
        out = self.W3(out, cat_ids) # 输入 [B*H, hidden_dim] [B*H] 输出 [B*H, embed_dim]               
        out = out.view(B, H, self.embed_dim) # [B, H, embed_dim]
        return out

class BasicTransformerBlock(nn.Module): ## 动作 token 对上下文 token 的 cross-attention。

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, action_tokens: torch.Tensor, context_tokens: torch.Tensor, time_emb: torch.Tensor):

        x = self.norm1(action_tokens)
        attn_out, _ = self.attn(x, context_tokens, context_tokens)
        ##Q = x K = context_tokens V = context_tokens
        x = action_tokens + attn_out
        x2 = self.norm2(x)
        # transformer 中 Feed-Forward部分
        if time_emb is not None:
            x2 = x2 + time_emb.unsqueeze(1)
        ff_out = self.ff(x2)
        x = x + ff_out
        return x

class FlowmatchingActionHead(nn.Module):

    # 没有什么作用，只是做了一些模块定义
    def __init__(self, config=None,
                 embed_dim: int = 896, 
                 hidden_dim: int = 1024,
                 action_dim: int = 16*7,
                 horizon: int = 16,
                 per_action_dim: int = 7,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 dropout: float = 0.0,
                 num_inference_timesteps: int = 20,
                 num_categories: int = 1):
        super().__init__()

        if config is not None:
      
            embed_dim = getattr(config, "embed_dim", embed_dim)
            hidden_dim = getattr(config, "hidden_dim", hidden_dim)
            action_dim = getattr(config, "action_dim", action_dim)
            horizon = getattr(config, "horizon", horizon)
            num_heads = getattr(config, "num_heads", num_heads)
            num_layers = getattr(config, "num_layers", num_layers)
            dropout = getattr(config, "dropout", dropout)
            num_inference_timesteps = getattr(config, "num_inference_timesteps", num_inference_timesteps)
            num_categories = getattr(config, "num_categories", num_categories)
            self.config = config
        else:
            from types import SimpleNamespace
            self.config = SimpleNamespace(embed_dim=embed_dim, hidden_dim=hidden_dim,
                                          action_dim=action_dim, horizon=horizon,
                                          num_heads=num_heads, num_layers=num_layers,
                                          dropout=dropout, num_inference_timesteps=num_inference_timesteps,
                                          num_categories=num_categories)
        print(f"num_inference_timesteps {num_inference_timesteps}")
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.per_action_dim = config.per_action_dim
        self.action_dim = config.action_dim

        # get("embed_dim", 896),    
        # .get("hidden_dim", 1024),
        # get("state_dim", 7),
        # config.get("state_hidden_dim", 1024),
        # g.get("num_heads", 8),
        # ig.get("num_layers", 8),
        # get("dropout", 0.0),
        # ("num_inference_timesteps", 50),
        # ("num_categories", 1)

        self.time_pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=1000) ## 默认参数 896

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(embed_dim=embed_dim, num_heads=num_heads,
                                   hidden_dim=embed_dim*4, dropout=dropout)
            for _ in range(num_layers)
        ])
       
        self.norm_out = nn.LayerNorm(embed_dim)
        self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim)

        self.mlp_head = CategorySpecificMLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                                            output_dim=action_dim, num_categories=num_categories)

        self.state_encoder = None
        if hasattr(self.config, "state_dim") and self.config.state_dim is not None:
       
            state_hidden = getattr(self.config, "state_hidden_dim", embed_dim)
        
            self.state_encoder = CategorySpecificMLP(input_dim=self.config.state_dim,
                                                    hidden_dim=state_hidden,
                                                    output_dim=embed_dim,
                                                    num_categories=num_categories)
            # (num_categories, embed_dim)
        self.action_encoder = None
        if horizon > 1:
          
            per_action_dim = getattr(self.config, "per_action_dim", None)
            if per_action_dim is None:
                per_action_dim = action_dim // horizon if action_dim % horizon == 0 else action_dim ## 这么写 // 是为了保证是整数
            self.action_encoder = MultiEmbodimentActionEncoder(action_dim=per_action_dim,
                                                               embed_dim=embed_dim,
                                                               hidden_dim=embed_dim,  
                                                               horizon=horizon,
                                                               num_categories=num_categories)

    # 前向传播（训练用）
    ## noise 是随即生成的噪声位置
    ## pre_velocity 是 noise 和 target之间若干个随机位置，预测的速度向量。
    ## 假设当前state 对应的帧是t
    # fused_tokens: prompt + t时刻的多视角image融合 [8, 1024, 896]
    # states.shape = [8, 24]
    # actions_gt: t 时刻开始的 H 时刻内的所有  action， 列为 action_dim, 行为H [8, 50, 24]
    # state_emb: t 时刻的 状态编码
    # 随机变量 noise：actions_gt 同维度随机噪声
    # actions_m: 随机权重 (1-r)noise + r*actions_gt 加权
    # 映射: actions_m, state_emb, fused_tokens -> target_v(actions_gt - noise)
    # 可以理解为给定 fused_tokens 和 state_emb 的条件下，时变向量场action的建模:对任意给定的一个点action_t,都可以映射出一个对应的速度
    # v_theta(action_t, t | fused_tokens, state)
    def forward(self, fused_tokens: torch.Tensor, state: torch.Tensor = None,
                actions_gt: torch.Tensor = None, embodiment_id: torch.LongTensor = None, 
                state_mask: torch.Tensor = None, action_mask: torch.Tensor = None):

        if actions_gt is None:
            return self.get_action(fused_tokens, state=state, embodiment_id=embodiment_id)
        B = fused_tokens.size(0) # 批次大小 8
        device = fused_tokens.device
        # 如果没有传入类别ID，默认全0
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        context_tokens = fused_tokens # [8, 1024, 896]
        if state is not None and self.state_encoder is not None:
            state_emb = self.state_encoder(state, embodiment_id)  # [B, 896]
            state_emb = state_emb.unsqueeze(1) # [8, 896]->[8, 1, 896]
            context_tokens = torch.cat([context_tokens, state_emb], dim=1) ## state 编码以后和context_tokens拼接
            # [8, 1024+1, 896]
        t = torch.distributions.Beta(2, 2).sample((B,)).clamp(0.02, 0.98).to(device).to(dtype=self.dtype)
        # 维度是[B],每个数都在[0.02,0.98]之间
        # t = [t1, .... , t8]
        time_index = (t * 1000).long() # 得到的是[B],但是数值在[2,980]之间，可以理解为B个索引值
        time_emb = self.time_pos_enc(1000)[:, time_index, :].squeeze(0) ## [B, 896]
        # 为 0~999 这 1000 个离散时间步，分别生成一个 896 维的时间向量[1, 1000, 896] 
        # ->依据索引选->[1,B,embed_dim]->[B,embed_dim] 筛选出了这B个时间步的对应时间向量
        # 对于896这个向量，计算公式就是transformer 中的计算公式，只不过把dmodel替换为896，把pos替换为对应的时间步
        action_shape = actions_gt.shape[1]  # B H D
        actions_gt_seq = actions_gt # 真实动作（标签）
        noise = torch.rand_like(actions_gt) * 2 - 1 #[8, 50, 24]，每个值[-1,1]，
        # B H D
        if action_mask is not None:
            action_mask = action_mask.to(dtype=noise.dtype, device=noise.device) # type 是统一数据类型，例如float32
            assert action_mask.shape == noise.shape, f"action_mask shape {action_mask.shape} != noise shape {noise.shape}"
            noise = noise * action_mask # 对noise 进行mask操作，让噪声只在部分位置生效

        if self.horizon > 1:
            noise_seq = noise.view(B, self.horizon, self.per_action_dim) #维度不变，还是[8, 50, 24]，这里只是做一个显示约束
            t_broadcast = t.view(B, 1, 1) ## 随机时间，维度从[8]->[8, 1, 1]
        else:
            noise_seq = noise.unsqueeze(1)
            t_broadcast = t.view(B, 1)

        action_intermediate_seq = (1 - t_broadcast) * noise_seq + t_broadcast * actions_gt_seq  
        # 对噪声和真实动作进行加权，得到加噪的噪声 
        # [B,1,1] * [B, self.horizon, self.per_action_dim] + [B,1,1] * [B, self.horizon, self.per_action_dim]
        # 采样一条从噪声到真实动作的路径上的中间点。
        #      |noise_seq1|    |actions_gt_seq1|   |action_intermediate_seq1|
        # (1-t)|noise_seq2| + t|actions_gt_seq2| = |action_intermediate_seq2|
        # t 属于 (0.02,0.98)  noise_seq1 随机产生
        if self.horizon > 1 and self.action_encoder is not None: 
            action_tokens = self.action_encoder(action_intermediate_seq, embodiment_id)  
            # [8, 50, 24] -> [8, 50, 896]
        else:
            if not hasattr(self, "single_action_proj"):
                self.single_action_proj = nn.Linear(self.per_action_dim, self.embed_dim).to(device)
            action_tokens = self.single_action_proj(action_intermediate_seq) 

        x = action_tokens
        for block in self.transformer_blocks:
            x = block(x, context_tokens, time_emb) ## 中间态 真实态对应的上下文信息 中间态对应的时间点
        # context_tokens [8, 1024+1, 896] image prompt state
        x = self.norm_out(x)
        # x.shape = [8, 50, 896]
        if self.horizon > 1:
            x_flat = x.reshape(B, -1)  
            if not hasattr(self, "seq_pool_proj"):
                self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim).to(device)
            x_pooled = self.seq_pool_proj(x_flat)
        # [8, 50 * 896] -> [8, 896] 
        else:          
            x_pooled = x.squeeze(1) 
        pred_velocity = self.mlp_head(x_pooled, embodiment_id) ## 预测速度
        # [8, 896] -> [8, 24] 
        return pred_velocity, noise
        # noise 是起点
        # actions_gt 是终点
        # action_intermediate_seq 是路上的某个位置
        # pred_velocity 是在这个位置应该走的方向和速度
    # 动作生成 (部署用)
    def get_action(self, fused_tokens: torch.Tensor, state: torch.Tensor = None, embodiment_id: torch.LongTensor = None, action_mask: torch.Tensor = None):

        print(f"action_mask shape: {action_mask.shape if action_mask is not None else 'None'}")
        print(f"one sample action_mask: {action_mask[0] if action_mask is not None else 'None'}")

        B = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        context_tokens = fused_tokens
        if state is not None and self.state_encoder is not None:

            state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1) 
            context_tokens = torch.cat([context_tokens, state_emb], dim=1)

        action_dim_total = getattr(self.config, "action_dim", None)
        if action_dim_total is None:
          
            action_dim_total = self.action_dim
       
        if self.horizon > 1:
            per_action_dim = getattr(self.config, "per_action_dim", action_dim_total // self.horizon)
        else:
            per_action_dim = action_dim_total

        action = (torch.rand(B, action_dim_total, device=device) * 2 - 1) ## 随即产生的动作
        print(f"action shape: {action.shape}")
        print(f"one sample action: {action[0]}")

        if self.horizon > 1:
            action_seq = action.view(B, self.horizon, per_action_dim)

        else:
            action_seq = action.view(B, 1, per_action_dim)

        action_mask = action_mask.view(B, 1, per_action_dim).repeat(1,self.horizon,1)

        print(f"action_mask: {action_mask}")
        print(f"one sample action_mask: {action_mask[0]}")

        if action_mask is not None:
            action_mask = action_mask.to(dtype=action_seq.dtype, device=action_seq.device)
            assert action_mask.shape == action_seq.shape, f"action_mask shape {action_mask.shape} != noise shape {action_seq.shape}"
            action_seq = action_seq * action_mask
        else:
            raise ValueError("action_mask must be provided for inference with flow matching.")
        print(f"action shape: {action_seq.shape}")
        print(f"one sample action: {action_seq[0]}")

        N = int(getattr(self.config, "num_inference_timesteps", 32))
        dt = 1.0 / N
        for i in range(N):
            t = i / N 

            time_index = int(t * 1000)
            time_emb = self.time_pos_enc(1000)[:, time_index, :].to(device).squeeze(0)  
            time_emb = time_emb.unsqueeze(0).repeat(B, 1)  


            if self.horizon > 1 and self.action_encoder is not None:
                action_seq = action_seq * action_mask
                action_tokens = self.action_encoder(action_seq, embodiment_id) 
            else:
                if hasattr(self, "single_action_proj"):
                    action_tokens = self.single_action_proj(action_seq)  
                else:
                    self.single_action_proj = nn.Linear(per_action_dim, self.embed_dim).to(device)
                    action_tokens = self.single_action_proj(action_seq)

            x = action_tokens  ## 循环一次
            for block in self.transformer_blocks:
                x = block(x, context_tokens, time_emb)
            x = self.norm_out(x)

            if self.horizon > 1:
                x_flat = x.reshape(B, -1)
                if hasattr(self, "seq_pool_proj"):
                    x_pooled = self.seq_pool_proj(x_flat)
                else:
                    self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim).to(device)
                    x_pooled = self.seq_pool_proj(x_flat)
            else:
                x_pooled = x.squeeze(1)
         
            pred = self.mlp_head(x_pooled, embodiment_id)  
  
            action = action + dt * pred  ## 计算一次
          
            if self.horizon > 1:
                action_seq = action.view(B, self.horizon, per_action_dim) ## 更新一次
            else:
                action_seq = action.view(B, 1, per_action_dim)
      
        return action

    @property
    def device(self):
      
        return next(self.parameters()).device

    @property
    def dtype(self):
        
        return next(self.parameters()).dtype




# 遗留问题：flow matching 在训练和测试的时候，对时间编码分别是如何利用的？