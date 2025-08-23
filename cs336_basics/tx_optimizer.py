import torch
import math
import sys
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable, Iterable
from einops import rearrange, einsum, reduce, repeat
from typing import IO, Any, BinaryIO, Optional
from jaxtyping import Float, Int,Bool
from torch import Tensor
import numpy.typing as npt

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr,
                    "betas":betas,
                    "eps":eps,
                    "weight_decay":weight_decay
                    }
        # automatically initializes self.state
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # get state related information or initialize it
                t = state.get("t",1)
                grad = p.grad.data

                # update first and second moment
                m = beta1*state.get("m",torch.zeros_like(p.data))+(1-beta1)*grad
                v = beta2*state.get("v",torch.zeros_like(p.data))+(1-beta2)*grad**2                

                # calculate lr for step t
                lr_t = lr*(math.sqrt(1-beta2**t))/(1-beta1**t)
                
                # update parameters
                p.data -= lr_t*m/(v**(1/2)+eps)
                
                # apply weight decay
                p.data -= lr*weight_decay*p.data

                # save state for next optimizer step
                state["m"] = m
                state["v"] = v
                state["t"] = t+1
        
        return loss

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it/warmup_iters*max_learning_rate 
    if warmup_iters <= it <= cosine_cycle_iters:
        alpha = min_learning_rate+0.5*(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))*(max_learning_rate-min_learning_rate)
        return alpha  
    return min_learning_rate
