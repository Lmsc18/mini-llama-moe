import torch

class Embedding(torch.nn.Module):
    def __init__(self,num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.vocab_size=num_embeddings
        self.d_model=embedding_dim
        self.device=device
        self.dtype=dtype
        self.embedding_table=torch.nn.Parameter(torch.empty((num_embeddings,embedding_dim),device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.embedding_table)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings=self.embedding_table[token_ids]
        return embeddings