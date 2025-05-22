import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    softmax_result = exp_x / sum_exp_x
    return softmax_result