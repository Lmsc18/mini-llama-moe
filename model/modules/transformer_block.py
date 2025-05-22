import torch
from .multihead_attention import MultiHeadAttention
from .rms_norm import RMSNorm
from .moe import MoeLayer   


class TransformerBlock(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.d_model=config.d_model
        self.num_heads=config.num_heads
        self.norm=RMSNorm(config)
        self.mha=MultiHeadAttention(config)
        self.moe=MoeLayer(config)
        self.aux_loss=0.0
        self.z_loss=0.0
    def forward(self,x,freqs_cis):
        mha_output=x+self.mha(self.norm(x),freqs_cis)
        out=mha_output+self.moe(self.norm(mha_output))
        self.aux_loss=self.moe.aux_loss
        self.z_loss=self.moe.z_loss
        return out
