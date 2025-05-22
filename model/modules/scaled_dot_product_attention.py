import torch
from .softmax import softmax

def scaled_dot_product_attention(query,key,value,mask=None):
    d_k=query.size(-1)
    scores=query@key.transpose(-2,-1)/torch.sqrt(torch.tensor(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    attention_scores=softmax(scores,-1)
    out=attention_scores@value
    return out
