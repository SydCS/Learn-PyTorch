# 单个专家的架构，就是经典的 FFN
class MixtralBLockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        # FFNSize，一般是 HidSize x4
        self.ffn_dim = config.intermediate_size
        # HidSize，隐藏状态的向量尺寸
        self.hidden_dim = config.hidden_size

        # 用于隐藏状态扩张的线性层
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        # 用于隐藏状态收缩的线性层
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        # 用于计算隐藏状态门控的线性层
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        # 输入隐藏状态的形状为 [BatchSize, SeqLen, HidSize]、
        # 输入经过第三个线性层并激活，得到门控
        # 输入经过第一个线性层，乘以门控，经过第二个线性层，得到输出
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(
            hidden_states
        )
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


# MOE 的架构
class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        # HidSize，隐藏状态的向量尺寸
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        # NExp，专家数量
        self.num_experts = config.num_local_experts
        # TopK，激活的专家数量
        self.top_k = config.num_experts_per_tok

        # 门控线性层
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # 专家模块列表，每个都是 FFN
        self.experts = nn.ModuleList(
            [MixtralBLockSparseTop2MLP(config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        # 输入尺寸：[BatchSize, SeqLen, HidSize]
        # 获取 BatchSize（批量大小）
        #     SeqLen（序列长度）
        #     HidSize（隐藏状态尺寸）
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # 将输入前两维合并，[BatchSize * SeqLen, HidSize]
        hidden_states = hidden_states.view(-1, hidden_dim)
        # 将隐藏状态传入门控线性层得到专家得分
        # 每个样本的每个单词都有一组得分
        # [BatchSize * SeqLen, NExp]
        router_logits = self.gate(hidden_states)
        # 专家得分经过 Softmax 得到专家概率
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # 计算每个得分的 TOPK，得到专家索引
        # routing_weights：TOPK 专家概率，[BatchSize * SeqLen, TopK]
        # selected_experts：TOPK 专家索引，[BatchSize * SeqLen, TopK]
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        # 专家概率归一化，使每组得分和为一
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # 转换为输入的数据类型
        routing_weights = routing_weights.to(hidden_states.dtype)
        # 将最终的隐藏状态初始化为零，用于累加
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # 将专家索引单热化，交换前后两维，得到专家的掩码
        # [NExp, TopK, BatchSize * SeqLen]
        # mask[i, j, k] 表示第 k 个单词的第 j 个专家是不是专家 i
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # 遍历每个专家，expert_idx 为专家索引
        for expert_idx in range(self.num_experts):
            # 获取当前专家模块
            expert_layer = self.experts[expert_idx]
            # 使用索引来索引掩码，得到当前专家的掩码矩阵
            # [TopK, BatchSize * SeqLen]
            # 它的元素 [i, j] 表示第 j 个样本的第 i 个专家是不是当前专家
            # where 计算调用该专家的单词序号（top_x），以及该专家的排名（idx）
            idx, top_x = torch.where(expert_mask[expert_idx])

            # 如果没有单词调用该专家，转到下一个
            if top_x.shape[0] == 0:
                continue

            # 转 Python 列表
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # 获取调用该专家的单词的隐藏状态，[NHid, HidSize]
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            # 将隐藏状态传入当前专家，得到专家输出，[NHid, HidSize]
            # 获取调用该专家的单词的专家概率，[NHid, 1]
            # 二者相乘
            current_hidden_states = (
                expert_layer(current_state)
                * routing_weights[top_x_list, idx_list, None]
            )

            # 将隐藏状态加到最终隐藏状态
            # 即 final_hidden_states[top_x[i]] += current_hidden_states[i]
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        # 拆分第一维，[BatchSize, SeqLen, HidSize]
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits
