import torch
import math
import sys
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, reduce, repeat
from typing import IO, Any, BinaryIO, Optional
from jaxtyping import Float, Int,Bool
from torch import Tensor

from cs336_basics.layers import Linear,Embedding, RMSNorm,PositionwiseFeedforward
from cs336_basics.layers import RotaryPositionalEmbedding,MultiheadSelfAttention
from cs336_basics.layers import softmax, scaled_dot_product_attention

class PrenormBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int,
                  max_seq_len:int, theta:float, device=None, dtype=None):
        super().__init__()
        # norm layer
        self.ln1 = RMSNorm(d_model,device=device,dtype=dtype)
        # mhsa with rope
        self.attn = MultiheadSelfAttention(d_model,num_heads,max_seq_len,theta,device,dtype)
        # add step
        # norm layer
        self.ln2 = RMSNorm(d_model,device=device,dtype=dtype)
        # positionwise feed forward
        self.ffn = PositionwiseFeedforward(d_model,d_ff,device,dtype)
        # add to output

    def forward(self, x: Float[Tensor, " ..."], token_positions:Optional[Int[Tensor, " ..."]]=None)-> Float[Tensor, "..."]:
        
        # first Tx operation, Norm + MHSA w/ RoPE
        norm1_out = self.ln1.forward(x)
        # we may have to define token_positions if it is not given
        attn_out = self.attn.forward(norm1_out,token_positions)
        
        # ensure no broadcasting, elementwise addition on [batch seq d_model]
        assert(x.shape == attn_out.shape)
        resid1_out = attn_out + x

        # second Tx operation, Norm + SwiGLU
        norm2_out = self.ln2.forward(resid1_out)
        ffn_out = self.ffn.forward(norm2_out)

        # ensure no broadcasting, elementwise addition
        assert(ffn_out.shape == resid1_out.shape)
        final_out = resid1_out + ffn_out
        return final_out


class Transformer(nn.Module):
    def __init__(
            self, vocab_size: int, 
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device=None, dtype=None):
       super().__init__()
       self.token_embeddings = Embedding(vocab_size,d_model,device=device,dtype=dtype)
       self.layers = nn.ModuleList([PrenormBlock(d_model,num_heads,d_ff,context_length,rope_theta,device,dtype) for _ in range(num_layers)])
       self.ln_final = RMSNorm(d_model,device=device,dtype=dtype)
       self.lm_head = Linear(d_model,vocab_size,device=device,dtype=dtype)

    def forward(self,x:Int[Tensor, "..."]) -> Float[Tensor, "..."]:
        # 1. token embed step
        x = self.token_embeddings.forward(x)

        # 2. prenorm blocks step
        for layer in self.layers:
            x = layer.forward(x)
        
        # 3. Final norm
        x = self.ln_final.forward(x)

        # 4. Vocab projection or lm_head
        x = self.lm_head(x)

        return x




    
    