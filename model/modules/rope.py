import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, seq_len: int, device=None):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, freqs)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.register_buffer("cos", cos)  
        self.register_buffer("sin", sin)
        self.d_k = d_k
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        cos = self.cos[token_positions]  
        sin = self.sin[token_positions]
        x_reshaped = x.view(*x.shape[:-1], -1, 2)  
        x_even = x_reshaped[..., 0]  
        x_odd = x_reshaped[..., 1]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_odd * cos + x_even * sin
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.reshape_as(x)
        return x_rotated