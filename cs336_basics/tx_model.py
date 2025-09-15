import torch
import math
import sys
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable, Iterable
from einops import rearrange, einsum, reduce, repeat
from typing import IO, Any, BinaryIO, Optional
from jaxtyping import Float, Int,Bool
from cs336_basics.tx_utils import softmax
from torch import Tensor

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

    def forward(self, x: Tensor) -> Tensor:
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
        self.weight = nn.Parameter(embeddings)

    def forward(self, token_ids: Tensor) -> Tensor:
        # for every id, we need to pull the row vector associated
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        # initialize gain factor
        gain = torch.ones(d_model, dtype=dtype, device=device)
        self.weight = nn.Parameter(gain) #learnable
        self.d_model = d_model
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        # upcast input to torch.float32
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # calculate the RMS scalar 
        # scalar for every ex. in batch, for every emb in sequence
        mean_squared_sum = (1/self.d_model)*einsum(x,x, "b seq d, b seq d -> b seq")
        rms = torch.sqrt(mean_squared_sum+self.eps)

        # normalize with the rms and gain
        gain_product = einsum(x,self.weight, "b seq d, d -> b seq d")

        # divide by rms (elementwise, so input shape preserved)
        rms_norm = einsum(gain_product,1/rms, "b seq d, b seq -> b seq d")

        # return result to original dtype
        return rms_norm.to(in_dtype)
    
class PositionwiseFeedforward(nn.Module):
    # SwiGLU(x) = W2(SiLU(W1x)⊙W3x)
    def __init__(self, d_model:int, d_ff:int,device=None, dtype=None):
        super().__init__()
        
        # initialize parameters of SWiGLU FFN
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model,device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype) 

    def forward(self,x: Tensor)-> Tensor:
        # FFN = W2*(SiLU(W1*X) dot W3X)
        silu_in = self.w1.forward(x)
        silu_out = silu_in * torch.sigmoid(silu_in)
        gate = self.w3.forward(x)
        gated_prod = silu_out * gate
        final_prod = self.w2.forward(gated_prod)
        return final_prod

class SiLUFeedforward(nn.Module):
    # SiLU FFN(x) = W2@SiLU(W1@x)
    def __init__(self, d_model:int, d_ff:int,device=None, dtype=None):
        super().__init__()
        
        # initialize parameters of SWiGLU FFN
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model,device=device, dtype=dtype)

    def forward(self,x: Tensor)-> Tensor:
        # FFN = W2*(SiLU(W1*X) dot W3X)
        silu_in = self.w1.forward(x)
        silu_out = silu_in * torch.sigmoid(silu_in)
        out = self.w2.forward(silu_out)
        return out


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        """
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted 
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        rotations = torch.empty(max_seq_len,d_k//2,2,2,device=device,dtype=dtype)
        
        # initialize rotation matrix
        for i in range(max_seq_len):
            for k in range(d_k//2):
                angle = i/(theta**(2*k/d_k))
                rot = Tensor([[math.cos(angle), -math.sin(angle)],
                                    [math.sin(angle), math.cos(angle)]])
                rotations[i,k,:] = rot

        self.register_buffer("rotations",rotations,persistent=False)


    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        self.rotations shape: (seq_dim, feature_dim, 2, 2)
        x: tensor of shape (..., seq_dim, feature_dim)
        token_positions: tensor of shape (..., seq_dim)
        """
        # get the correct rotation matrices 
        # by default, 0'th dim of array_indexed is index dim, last dim of indices is feature dim
        rot = self.rotations[token_positions] # shape (..., seq_dim feature_dim, 2, 2)
       
        # rearrange by every two elements along feature dim of input x
        x_pairs = rearrange(x, "... seq_dim (feature_dim i) -> ... seq_dim feature_dim i",i=2)
        
        # apply rotations to these. for each pairwise position is A@x->y : (ixj)@(j,)->(i,)
        y_pairs = einsum(rot,x_pairs,"... seq_dim feature_dim i j, ... seq_dim feature_dim j -> ... seq_dim feature_dim i")

        # reshape y_pairs back to original shape
        y = rearrange(y_pairs, "... seq_dim feature_dim i -> ... seq_dim (feature_dim i)")

        return y


def scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
        ) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        let m be seq length of inputs, n be seq length of outputs
        d_k is look-up dim, d_v is value dim
        Q (Float[Tensor, "batch ... n d_k"]): Query tensor
        K (Float[Tensor, "batch ... m d_k"]): Key tensor
        V (Float[Tensor, "batch ... m d_v"]): Values tensor
        mask (Float[Tensor, " ... n m"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... n d_v"]: Output of SDPA
    """

    # get the key feature dim (should be last dim of Q and K)
    d_k = Q.shape[-1]
    assert d_k == K.shape[-1]

    # calculate the weighted scores (similarity product)
    scores = einsum(Q,K,"... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)

    # apply the mask if there is one
    if mask is not None:
        bool_mask = mask.bool() # compatible if somehow, input is mask bool or if float
        attn_mask = torch.where(bool_mask,0.0, float('-inf')) #torch.where for boolean tensors
        scores = scores+attn_mask

    # calculate the weighted
    weights = softmax(scores, dim=-1) # the softmax should be taken over the m inputs at an i'th output pos. 

    # return weights@V
    return einsum(weights,V,"... n m, ... m d_v -> ... n d_v")

class MultiheadSelfAttention(nn.Module):
    """
    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    def __init__(self, d_model:int, num_heads:int, max_seq_len:int=None, theta:float=None, device=None, dtype=None):    
        super().__init__()
        
        # initialize the multi-head self attention weights as 1 large matrix (which will be sliced)
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if max_seq_len:
            causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=dtype, device=device))
            self.register_buffer("causal_mask", causal_mask, persistent=False)
        else:
            self.register_buffer("causal_mask", None, persistent=False)
        
        assert theta is None or max_seq_len is not None, "max_seq_len must be provided when theta is given for multi-head self attention with RoPE."
        
        if theta:
            d_k = d_model//num_heads
            self.rope = RotaryPositionalEmbedding(theta,d_k, max_seq_len,device, dtype)
        else:
            self.rope = None

    def forward(self, x: Float[Tensor, " ..."], token_positions: Optional[Int[Tensor, "..."]]= None) -> Float[Tensor, " ..."]:
        # get Q, K, V matrices
        Q = self.q_proj.forward(x) # output shape is [batch seq d_model]
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)

        # create causal mask intepreting the second to last dim as seq dim
        if self.causal_mask is None:    
            seq_dim = x.shape[-2]
            cmask = torch.tril(torch.ones(seq_dim, seq_dim, dtype=x.dtype, device=x.device))
        else:
            # Slice the pre-computed mask to match actual sequence length (could be < than max_seq_len)
            seq_dim = x.shape[-2]
            cmask = self.causal_mask[:seq_dim, :seq_dim]

        # get slice size for multi-head self attention
        d_k = self.d_model // self.num_heads
        d_v = self.d_model // self.num_heads

        q_heads = rearrange(Q,"batch seq (heads d_k) -> batch heads seq d_k", d_k=d_k)
        k_heads = rearrange(K,"batch seq (heads d_k) -> batch heads seq d_k", d_k=d_k)

        # apply RoPE to q_heads and k_heads
        if self.rope:
            if token_positions is None:
                token_positions = torch.arange(seq_dim,device=x.device)
                token_positions = rearrange(token_positions, "seq -> 1 seq") # 1 seq allows broadcast across batch dim
            
            q_heads = self.rope.forward(q_heads,token_positions)
            k_heads = self.rope.forward(k_heads,token_positions)

        v_heads = rearrange(V, "batch seq (heads d_v) -> batch heads seq d_v", d_v=d_v)

        mha_heads = scaled_dot_product_attention(q_heads, k_heads, v_heads, cmask)
        mha = rearrange(mha_heads, "batch heads seq d_v -> batch seq (heads d_v)")

        # apply o_proj_weight to the concatenated multi-head attention product
        out = self.output_proj.forward(mha)

        return out

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

class PostnormBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int,
                  max_seq_len:int, theta:float, device=None, dtype=None):
        super().__init__()
        # mhsa with rope
        self.attn = MultiheadSelfAttention(d_model,num_heads,max_seq_len,theta,device,dtype)
        # norm layer
        self.ln1 = RMSNorm(d_model,device=device,dtype=dtype)
        # positionwise feed forward
        self.ffn = PositionwiseFeedforward(d_model,d_ff,device,dtype)
        # norm layer
        self.ln2 = RMSNorm(d_model,device=device,dtype=dtype)

    def forward(self, x: Float[Tensor, " ..."], token_positions:Optional[Int[Tensor, " ..."]]=None)-> Float[Tensor, "..."]:
        
        # MHSA block
        attn_out = self.attn.forward(x,token_positions)
        pre_norm_sum1 = attn_out+x

        # post norm
        post_norm1 = self.ln1.forward(pre_norm_sum1)

        # SWIGLU FFN
        swiglu_out = self.ffn.forward(post_norm1)
        pre_norm_sum2 = swiglu_out+post_norm1

        # final norm
        post_norm2 = self.ln2.forward(pre_norm_sum2)
        return post_norm2

class PrenormBlockNoRMS(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int,
                  max_seq_len:int, theta:float, device=None, dtype=None):
        super().__init__()
        # mhsa with rope
        self.attn = MultiheadSelfAttention(d_model,num_heads,max_seq_len,theta,device,dtype)
        # positionwise feed forward
        self.ffn = PositionwiseFeedforward(d_model,d_ff,device,dtype)

    def forward(self, x: Float[Tensor, " ..."], token_positions:Optional[Int[Tensor, " ..."]]=None)-> Float[Tensor, "..."]:
        
        # attention without prenorm
        attn_out = self.attn.forward(x,token_positions)
        
        # ensure no broadcasting, elementwise addition on [batch seq d_model]
        assert(x.shape == attn_out.shape)
        resid1_out = attn_out + x

        # second Tx operation, SwiGLU FFN without prenorm
        ffn_out = self.ffn.forward(resid1_out)

        # final residual concatenation
        assert(ffn_out.shape == resid1_out.shape)
        final_out = resid1_out + ffn_out
        return final_out

class PrenormBlockSiLU(nn.Module):
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
        self.ffn = SiLUFeedforward(d_model,d_ff,device,dtype)
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

class NoRMSTransformer(nn.Module):
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
       self.layers = nn.ModuleList([PrenormBlockNoRMS(d_model,num_heads,d_ff,context_length,rope_theta,device,dtype) for _ in range(num_layers)])
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

class NoPETransformer(nn.Module):
    def __init__(
            self, vocab_size: int, 
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: None,
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

class PostNormTransformer(nn.Module):
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
       self.layers = nn.ModuleList([PostnormBlock(d_model,num_heads,d_ff,context_length,rope_theta,device,dtype) for _ in range(num_layers)])
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

class SiluTransformer(nn.Module):
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
       self.layers = nn.ModuleList([PrenormBlockSiLU(d_model,num_heads,d_ff,context_length,rope_theta,device,dtype) for _ in range(num_layers)])
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