import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_mask import causal_mask

class MultiHeadSelfAttention(nn.Module):

    """Multi head attention with explicit shape tracing
    
    Dimensions (before masking)
    x: B,T,d_model
    qkv : B,T,3*d_model
    view : B,T,3,n_head,d_head where d head = d_model // n_head
    split : q, k ,v each (B,T,n_head,d_head)
    swap : B,n_head,T,d_head
    scores : (B,n_head,T,T) = q@kT / sqrt(d_head)
    weights : (B,n_head , T , T) = softmax(scores)
    ctx : (B , n_head , T , d_head) = weights @ values
    merge : (B,T,n_head*d_head)
    """

    def __init__(self, d_model:int , n_head : int , dropout : float = 0.0 , trace_shapes : bool = True ) : 
        super().__init__()
        assert d_model % n_head == 0 , "d_model must be divisibe by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model , 3*d_model , bias = False)
        self.proj = nn.Linear(d_model , d_model , bias = False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self , x:torch.Tensor): #(B,T,d_model)
        B , T , C = x.shape
        qkv= self.qkv(x) #(B,T,3*C)
        qkv = qkv.view(B,T,3,self.n_head,self.d_head) #(B,T,3, n_head , d_head)

        if self.trace_shapes : 
            print ("qkv view = " , qkv.shape) 

        q,k,v = qkv.unbind(dim=2) #each (B,T , n_head , d_head)
        q = q.transpose(1,2) # ( B , n_head , T , d_head)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        if self.trace_shapes :
            print (" q : " , q.shape , " k : " , k.shape , " v : " , v.shape)

        scale = 1.0 / math.sqrt(self.d_head)

        attn = torch.matmul(q,k.transpose(-2,-1)) * scale  # (B, n_head , T , T)
        mask = causal_mask( T , device=x.device)
        attn = attn.masked_fill(mask , float('-inf'))
        w=F.softmax(attn, dim = -1)
        w=self.dropout(w)
        ctx = torch.matmul(w,v) #(B , n_head , T , d_head)

        if self.trace_shapes :
            print (" weights : " , w.shape  , " ctx : " , ctx.shape)

        out = ctx.transpose(1,2).contiguous().view(B,T,C)  # (B,T,A+)
        out = self.proj(out)

        if self.trace_shapes :
            print( " out : " , out.shape)

        return out , w
    
    