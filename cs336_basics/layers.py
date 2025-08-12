import torch
import math
import sys
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, reduce, repeat

# y = Wx (no bias terms!)
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        # initialize weights matrix
        weights = torch.empty(out_features,in_features,dtype=dtype, device=device)
        std = math.sqrt(2/(in_features+out_features))
        
        # mean = 0, std=std, lower bound -3*std, upper bound 3*std
        weights = nn.init.trunc_normal_(weights,mean=0.0,std=std, a=-3*std, b=3*std)

        # assign as instance variable
        self.weight = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # on input side of expression, d_in is last dim of x so "... d_in"
        # on output side of einsum expression, so "... d_out" follows convention
        # to put the output dim last
        return einsum(self.weight,x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        # initialize a matrix of vocab_size x embedding_dim
        embeddings = torch.empty(num_embeddings, embedding_dim,dtype=dtype, device=device)

        # normalize the embeddings to spec
        embeddings = nn.init.trunc_normal_(embeddings,mean=0.0,std=1.0,a=-3,b=3)

        # save and enroll as torch param
        self.embeddings = nn.Parameter(embeddings)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # for every id, we need to pull the row vector associated
        return self.embeddings[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        # initialize gain factor
        gain = torch.empty(d_model, dtype=dtype, device=device)
        self.gain = nn.Parameter(gain) #learnable
        self.d_model = d_model
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upcast input to torch.float32
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # calculate the RMS scalar 
        # scalar for every ex. in batch, for every emb in sequence
        mean_squared_sum = (1/self.d_model)*einsum(x,x, "b seq d, b seq d -> b seq")
        rms = torch.sqrt(mean_squared_sum+self.eps)

        # normalize with the rms and gain
        gain_product = einsum(x,self.gain, "b seq d, d -> b seq d")

        # divide by rms (elementwise, so input shape preserved)
        rms_norm = einsum(gain_product,1/rms, "b seq d, b seq -> b seq d")

        # return result to original dtype
        return rms_norm.to(in_dtype)
    
class positionwise_feedforward(nn.Module):
    def __init__(self, d_model:int, d_ff:int,device=None, dtype=None):
        super().__init__()
        
        # initialize parameters of SWiGLU FFN
        self.w1_weight = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2_weight = Linear(d_ff, d_model,device=device, dtype=dtype)
        self.w3_weight = Linear(d_model, d_ff, device=device, dtype=dtype) 

    def forward(self,x: torch.Tensor)-> torch.Tensor:
        # FFN = W2*(SiLU(W1*X) dot W3X)
        silu_in = self.w1_weight.forward(x)
        silu_out = silu_in * torch.sigmoid(silu_in)
        gate = self.w3_weight.forward(x)
        gated_prod = silu_out * gate
        final_prod = self.w2_weight.forward(gated_prod)
        return final_prod

class rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted 
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        rotations = torch.empty(max_seq_len,d_k//2,2,2,device=device)
        
        # initialize rotation matrix
        for i in range(max_seq_len):
            for k in range(d_k//2):
                angle = i/(theta**(2*k/d_k))
                rot = torch.tensor([[math.cos(angle), -math.sin(angle)],
                                    [math.sin(angle), math.cos(angle)]])
                rotations[i,k,:] = rot

        self.register_buffer("rotations",rotations,persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        self.rotations shape: (seq_dim, feature_dim, 2, 2)
        x: tensor of shape (..., seq_dim, feature_dim)
        token_positions: tensor of shape (..., seq_dim)
        """
        # get the correct rotation matrices 
        # by default, 0'th dim of array_indexed is index dim, last dim of input is feature dim
        rot = self.rotations[token_positions] # shape (..., seq_dim feature_dim, 2, 2)
       
        # rearrange by every two elements along feature dim of input x
        x_pairs = rearrange(x, "... seq_dim (feature_dim pair) -> ... seq_dim feature_dim pair",pair=2)
        
        # apply rotations to these. for each pairwise position: (ixj)@(j,)->(i,)
        y_pairs = einsum(rot,x_pairs,"... seq_dim feature_dim i j, ... seq_dim feature_dim j -> ... seq_dim feature_dim i")

        # reshape y_pairs back to original shape
        y = rearrange(y_pairs, "... seq_dim feature_dim i -> ... seq_dim (feature_dim i)")

        return y

