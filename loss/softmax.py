import torch
import torch.nn as nn

# 假设我们有3个类别
num_classes = 3
# 假设我们的模型输出未经过 softmax 的原始预测值
raw_predictions = torch.tensor([[1.2, 0.5, -0.3], [-0.1, 2.0, -0.5], [0.3, -0.2, 1.5]])

# 使用 PyTorch 自带的 CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# 计算 CrossEntropyLoss
loss = criterion(raw_predictions, torch.tensor([0, 1, 2]))  # 真实标签分别为0, 1, 2

print("CrossEntropyLoss:", loss.item())

# 分步计算等式中的 Softmax + log + NLLLoss
softmax = nn.Softmax(dim=1)
log_softmax = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()

# 计算 Softmax
# softmax_output = softmax(raw_predictions)

# 计算 log
log_softmax_output = log_softmax(raw_predictions)

# 计算 NLLLoss
nll_loss_value = nll_loss(log_softmax_output, torch.tensor([0, 1, 2]))

print("NLLLoss:", nll_loss_value.item())
