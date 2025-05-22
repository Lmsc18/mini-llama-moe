import torch 


class RMSNorm(torch.nn.Module):
    def __init__(self, config, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.hidden_dim=config.d_model
        self.eps=eps
        self.device=device
        self.dtype=dtype
        self.gamma=torch.nn.Parameter(torch.ones(config.d_model,device=device,dtype=dtype))
    def forward(self,x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean=torch.mean(x**2,-1,keepdim=True)
        rms=torch.sqrt(mean+self.eps)
        normed=x/rms
        result=normed*self.gamma
        result=result.to(in_dtype)
        return result