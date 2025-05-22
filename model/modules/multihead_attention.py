import torch
from .linear import Linear
#from .rope import RotaryPositionalEmbedding
from .rope_llama import *
from .scaled_dot_product_attention import scaled_dot_product_attention

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        #assert config.d_model % config.num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model=config.d_model
        self.num_heads=config.num_heads
        self.d_k=config.d_model//config.num_heads
        self.d_v=config.d_model//config.num_heads
        self.qkv_proj=Linear(config.d_model,3*config.d_model)
        #self.rope=RotaryPositionalEmbedding(theta=10000.0,d_k=self.d_k,seq_len=config.seq_len)
        self.out_proj=Linear(config.d_model,config.d_model)
    
    def forward(self,x,freqs_cis,mask=None):
        B,T,C=x.size()
        out=self.qkv_proj(x)
        q,k,v=out.split(self.d_model,dim=2)
        q=q.view(B,T,self.num_heads,self.d_k).transpose(1,2)
        k=k.view(B,T,self.num_heads,self.d_k).transpose(1,2)
        v=v.view(B,T,self.num_heads,self.d_v).transpose(1,2)
        # token_positions = torch.arange(T, device=x.device)
        # q=self.rope(q,token_positions)
        # k=self.rope(k,token_positions)
        q=apply_rotary_emb(q,freqs_cis)
        k=apply_rotary_emb(k,freqs_cis)
        if mask is None:
           causal_mask = torch.tril(torch.ones(T,T)).to(x.device)
           mask = causal_mask.bool()
        out=scaled_dot_product_attention(q,k,v,mask)
        out=out.transpose(1,2).contiguous().view(B,T,C)
        out=self.out_proj(out)
        return out