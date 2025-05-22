import torch

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), **kwargs)
        )
        torch.nn.init.trunc_normal_(self.weight)
    
    def forward(self, x):
        #print(self.weight.shape)
        return x @ self.weight.T