import torch
import sys
from einops import rearrange
from cs336_basics.tx_model import Transformer
from cs336_basics.tx_tokenizer import Tokenizer
from cs336_basics.tx_utils import cross_entropy_loss,softmax
from jaxtyping import Float, Int,Bool
from torch import Tensor
from typing import IO, Any, BinaryIO, Optional


def softmax_temp(
        logits: Float[Tensor, " ..."],
        dim: int, temperature: float,) -> Float[Tensor, " ..."]:
    
    # scale the logits by temperature
    scaled_logits = logits/temperature

    # get max values over specified dimension
    max_values = torch.max(scaled_logits,dim=dim,keepdim=True).values

    # subtract max_values from x so max element is 0
    shifted = scaled_logits-max_values # broadcast should work

    # get exp of shifted terms
    shifted_exps = torch.exp(shifted)

    # get sum of shifted terms
    shifted_exp_sums = torch.sum(shifted_exps, dim=dim, keepdim=True)

    # calculate product
    product = shifted_exps / shifted_exp_sums

    return product

def decode(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.8,
    sampling: bool = False,
    end_token: str = "<|endoftext|>"
) -> str:
    """
    Generate text from a trained language model using autoregressive sampling.
    
    Args:
        model (Transformer): The trained language model to generate from.
        tokenizer (Tokenizer): The tokenizer for encoding/decoding text.
        prompt (str): The starting text prompt to continue from.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Higher values make output more random. Defaults to 1.0.
        top_p (float, optional): Top-p sampling threshold for nucleus sampling. Defaults to 1.0.
        end_token (str, optional): Special token that signals the end of generation. Defaults to "<|endoftext|>".
    
    Returns:
        str: The generated text continuation.
    """
    # get end_token id
    eos_token_id = tokenizer.encode(end_token)
    # get token ids
    token_ids = tokenizer.encode(prompt) # list of ints

    def decode_step(prefix_tokens: list[int], model: Transformer, tokenizer: Tokenizer, temperature: float, top_p: float, sampling:bool=False) -> list[int]:
        
        # to tensor, add batch dim then get logits
        prefix_tokens = torch.tensor(prefix_tokens)
        prefix_tokens = rearrange(prefix_tokens, "seq -> 1 seq")
        logits = model.forward(prefix_tokens) # 1, seq, vocab
        
        # calculate probs with softmax
        probs = softmax_temp(logits,temperature=temperature,dim=-1) # (1, seq,vocab)
        probs = rearrange(probs, "1 seq vocab -> seq vocab")

        # select the last position for the vocab probs
        next_token_probs = probs[-1:].flatten() # (vocab,) shape
        if sampling:
            # sort from greatest to least, but need to keep original indices. 
            sorted_probs,vocab_indices = torch.sort(next_token_probs,descending=True)

            # calculate cumulative prob > top_p
            summed_probs = torch.cumsum(sorted_probs,dim=0)
            cutoff_indices = torch.where(summed_probs >= top_p,1.0,0)            
            first_ref_index = torch.argmax(cutoff_indices).item()+1 # gets first index,+1 to include index
            
            # out of bounds check
            first_ref_index = len(vocab_indices) if first_ref_index>len(vocab_indices) else first_ref_index

            # get indices to sample 
            valid_indices = vocab_indices[:first_ref_index]
            valid_sorted_probs = sorted_probs[:first_ref_index]

            # normalize valid_sorted_probs by elementwise division by sum
            valid_sorted_probs = valid_sorted_probs/(torch.sum(valid_sorted_probs))

            # sample indices and sorted probs (with automatic normalization)
            sampled_idx = torch.multinomial(valid_sorted_probs,num_samples=1)
            selected_vocab_idx = valid_indices[sampled_idx.item()].item()

            return [selected_vocab_idx]
        else: 
            _ ,next_token_id = torch.max(next_token_probs,dim=-1)
            return [next_token_id.item()]

    gen_ids = []
    while len(gen_ids) < max_tokens:
        next_token_id = decode_step(token_ids,model,tokenizer,temperature,top_p,sampling=sampling)
        if next_token_id == eos_token_id: break
        token_ids.extend(next_token_id)
        gen_ids.extend(next_token_id)

    # decode string to output
    gen_string = tokenizer.decode(gen_ids)
    return gen_string

    