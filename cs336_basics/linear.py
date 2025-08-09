import torch
import math
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
        self.weights = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights,x, "d_out d_in, ... d_in -> ... d_out")
