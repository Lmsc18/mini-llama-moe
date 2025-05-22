import torch
import torch.nn.functional as F
import torch.nn as nn
from model.modules.linear import Linear
from model.modules.swiglu import SWIGLU

class MoeLayer(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.k = config.k
        self.n_exp = config.num_experts

        self.experts = nn.ModuleList(
            [SWIGLU(config) for _ in range(config.num_experts)]
        )
        self.shared_expert = SWIGLU(config)
        self.gate = Linear(config.d_model, config.num_experts)
        self.aux_loss=0.0
        self.z_loss=0.0

    def forward(self, inputs):
        B, T, D = inputs.shape
        x = inputs.view(-1, D)  # [B*T, D]
        logits = self.gate(x)  # [B*T, n_exp]
        topk_scores, topk_indices = torch.topk(logits, self.k, dim=-1)  # [B*T, k]
        probs = F.softmax(topk_scores, dim=-1, dtype=torch.float).type_as(inputs)  # [B*T, k]

        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            batch_pos, expert_pos = torch.where(topk_indices == i)
            if batch_pos.numel() == 0:
                continue
            selected_inputs = x[batch_pos]
            expert_output = expert(selected_inputs)
            weight = probs[batch_pos, expert_pos].unsqueeze(-1)
            output[batch_pos] += weight * expert_output

        # Shared expert is always used
        output += self.shared_expert(x)

        # Compute losses
        aux_loss = self.compute_aux_loss(probs, topk_indices, self.n_exp)
        z_loss = self.compute_router_z_loss(logits)
        self.aux_loss=aux_loss
        self.z_loss=z_loss
        return output.view(B, T, D)
    @staticmethod
    def compute_aux_loss(expert_probs: torch.Tensor, indices: torch.Tensor, n_exp: int):
        """
        Switch Transformer auxiliary loss (eq 4-6)
        """
        with torch.no_grad():
            one_hot = F.one_hot(indices, num_classes=n_exp).sum(dim=1).float()  # [B*T, n_exp]
            tokens_per_expert = one_hot.mean(dim=0)  # [n_exp]
        prob_per_expert = expert_probs.new_zeros(expert_probs.size(0), n_exp).scatter_add(
            1, indices, expert_probs
        ).mean(dim=0)  # [n_exp]
        return n_exp * torch.sum(prob_per_expert * tokens_per_expert)

    @staticmethod
    def compute_router_z_loss(logits: torch.Tensor):
        """
        ST-MoE router z-loss (eq 5 in ST-MoE)
        """
        return torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
