from dataclasses import dataclass

@dataclass
class GPTConfig:
    seq_len: int = 1024 
    vocab_size: int = 50257 
    num_layer: int = 12 
    num_heads: int = 12 
    d_model: int = 768
    d_ff:int=None


@dataclass
class TSConfig:
    seq_len: int = 256 
    vocab_size: int = 10000
    num_layer: int = 4
    num_heads: int = 16
    d_model: int = 512
    d_ff:int=1344