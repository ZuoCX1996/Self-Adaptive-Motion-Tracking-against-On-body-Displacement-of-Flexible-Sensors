import torch
from torch import nn
def min_max_scale(x, epsilon=1):
    min = torch.min(x, dim=0).values.detach()
    max = torch.max(x, dim=0).values.detach()
    x = (x-min) / (max-min+epsilon)
    return x

def SD_loss(x, alpha=0.5, epsilon=1e-2):
    # x: seq_len x seq_num x feat_dim
    seq_num = x.shape[1]
    x_scaled = min_max_scale(x, epsilon=epsilon)
    x_mean = x.mean(dim=0, keepdim=True)


    x_scaled_avg = x_scaled.mean(dim=1).unsqueeze(dim=1).detach()
    x_mean_avg = x_mean.mean(dim=1).unsqueeze(dim=1).detach()

    x_scaled_avg = x_scaled_avg.repeat(1, seq_num, 1)
    x_mean_avg = x_mean_avg.repeat(1, seq_num, 1)

    mean_loss = torch.pow(x_scaled - x_scaled_avg, 2).mean()
    shape_loss = torch.pow(x_mean - x_mean_avg, 2).mean()

    loss = alpha * mean_loss + (1 - alpha) * shape_loss

    return loss

# # -----unit test-----
# x = torch.randn(size=[128, 10, 2])
# print(SD_loss(x))