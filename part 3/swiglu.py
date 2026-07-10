import torch.nn as nn

class SwiGLU(nn.Module):
    """SwiGLU FFN: (xW1) ⊗ swish(xW2) W3  with expansion factor `mult`."""
    def __init__(self,dim:int , mult :int = 4 , dropout : float = 4):
        super().__init__()
        inner = mult*dim
        self.w1 = nn.Linear(dim,inner,bias=False)
        self.w2 = nn.Linear(dim,inner,bias=False)
        self.w3 = nn.Linear(inner , dim , bias=False)
        self.act=nn.SILU()
        self.drop=nn.Dropout(dropout)

    def forward(self,x):
        a=self.w1(x)
        b=self.act(self.w2(x))

        return self.drop(self.w3(a*b)) 