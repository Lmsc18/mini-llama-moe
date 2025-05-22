import torch 
from .linear import Linear
class SWIGLU(torch.nn.Module):
    def __init__(self,config,device=None,dtype=None):
        super().__init__()
        self.d_model=config.d_model
        self.d_ff = config.d_ff if config.d_ff is not None else int((8/3)* config.d_model)
        assert self.d_ff % 64 == 0, f"{self.d_ff} is not a multiple of 64"
        # self.w1=Linear(self.d_ff,self.d_ff,device=device,dtype=dtype)
        # self.w2=Linear(self.d_ff,self.d_ff,device=device,dtype=dtype)
        # self.w3=Linear(self.d_ff,self.d_ff,device=device,dtype=dtype)

        self.w1=Linear(self.d_model,self.d_ff,device=device,dtype=dtype)
        self.w2=Linear(self.d_ff,self.d_model,device=device,dtype=dtype)
        self.w3=Linear(self.d_model,self.d_ff,device=device,dtype=dtype)

    def forward(self,x):
        gate=torch.sigmoid(self.w1(x))
        hidden=self.w3(x)
        out=gate*hidden
        out=self.w2(out)
        return out