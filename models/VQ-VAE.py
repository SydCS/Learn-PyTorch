import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings  # embedding 的数量
        self.D = embedding_dim  # embedding 维度
        self.beta = beta  # commitment loss 的权重

        self.embedding = nn.Embedding(self.K, self.D)  # 创建 dictionary
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)  # 初始化

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(
            0, 2, 3, 1
        ).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(flat_latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_indexes = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_indexes.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_indexes, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(
            encoding_one_hot, self.embedding.weight
        )  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        # 两个 L2 损失直接用 mse_loss 进行实现
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        # 这里巧妙地实现了梯度的连接回传，在数值上进行了变化但是梯度上进行了保留
        quantized_latents = latents + (quantized_latents - latents).detach()

        return (
            quantized_latents.permute(0, 3, 1, 2).contiguous(),
            vq_loss,
        )  # [B x D x H x W]
