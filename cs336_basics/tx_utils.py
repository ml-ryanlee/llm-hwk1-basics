import torch
import math
import sys
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable, Iterable
from einops import rearrange, einsum, reduce, repeat
from typing import IO, Any, BinaryIO, Optional
import os
from jaxtyping import Float, Int,Bool
from torch import Tensor
import numpy as np
import numpy.typing as npt

def softmax(logits: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    # get max values over specified dimension
    max_values = torch.max(logits,dim=dim,keepdim=True).values

    # subtract max_values from x so max element is 0
    shifted = logits-max_values # broadcast should work

    # get exp of shifted terms
    shifted_exps = torch.exp(shifted)

    # get sum of shifted terms
    shifted_exp_sums = torch.sum(shifted_exps, dim=dim, keepdim=True)

    # calculate product
    product = shifted_exps / shifted_exp_sums

    return product

def cross_entropy_loss(logits: Float[Tensor, ""], targets: Int[Tensor, ""])->Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples and positions.
    Args:
        logits (Float[Tensor, "batch_size context_length vocab_size"]): logits[i][j][k] is the
            unnormalized logit of kth class for the jth position in the ith example.
        targets (Int[Tensor, "batch_size context_length"]): Tensor of shape (batch_size, context_length) with the index of the correct class at each position.
            Each value must be between 0 and `vocab_size - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples and positions.
    """
    # subtract the max value along the vocab dim dimension (logits are. batch x seq x vocab)
    max_dim_value = torch.max(logits,dim=-1,keepdim=True).values
    shifted_logits = logits - max_dim_value

    # get sum of exp(logits)
    exp_shifted_logits = torch.exp(shifted_logits) # batch, seq, vocab
    sum_exp_shifted_logits = torch.sum(exp_shifted_logits,dim=-1, keepdim=True) #torch.sum-> batch,seq,1

    # get log of (sum(exp(logits)))
    vocab_logit_sum = torch.log(sum_exp_shifted_logits) # (batch, seq, 1) shape
    
    # get the logit for the target at a batch position with torch.gather 
    reshaped_targets = rearrange(targets,"... -> ... 1")
    target_logits = torch.gather(shifted_logits, dim=-1,index=reshaped_targets)

    assert target_logits.shape == vocab_logit_sum.shape

    # get score by subtracting target_logits from vocab_logit sum
    batch_scores = vocab_logit_sum-target_logits # (batch, seq, 1)

    # get average across batch
    return reduce(batch_scores, "... -> ","mean")

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6) -> None:
    
    # reshape the grads into one tensor to take L2 norm
    p_grads_list = []
    for p in parameters:
        if p.grad is not None:
            p_grads_list.append(torch.flatten(p.grad))
    
    # check that list is not empty
    if not p_grads_list:
        return
    
    # get grads into 1D list, and calculate l2 norm
    grads = torch.cat(p_grads_list)    
    l2_norm = torch.norm(grads)

    # return early if l2_norm < max, we don't need to clip gradients
    if l2_norm < max_l2_norm:
        return
    
    # apply scaling to each parameter in place
    scale = max_l2_norm/(l2_norm+eps)
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(scale)
    return

def data_loader(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling input sequences and next-token targets from dataset.

    Args:
        dataset (npt.NDArray): 1D array of token IDs.
        batch_size (int): Number of sequences to sample.
        context_length (int): Length of each sequence.
        device (str): PyTorch device (e.g., 'cpu', 'cuda:0').

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (inputs, targets) both shape (batch_size, context_length).
    """
    # initialize lists to capture randomly sampled sequences and targets
    seqs = []
    targets = []
    
    # designate valid start indices to randomly sample
    max_start_idx = len(dataset)-context_length #non-inclusive upper bound

    for _ in range(batch_size):
        # choose random start index within valid range
        start_idx = np.random.randint(0,max_start_idx)
        end_idx = start_idx+context_length

        # sample numpy array from start index to context_length, convert to tensor of ints
        seq_array = dataset[start_idx:end_idx]
        seq_tensor = torch.tensor(seq_array,device=device,dtype=torch.long)

        # for targets, sample from start_index+1, convert to tensor of ints
        target_array = dataset[(start_idx+1):(end_idx+1)]
        target_tensor = torch.tensor(target_array,device=device,dtype=torch.long)

        # add to lists
        seqs.append(seq_tensor)
        targets.append(target_tensor)

    # stack list of seqs and targets into batch_size x context_length and return 
    seq_batched = rearrange(seqs,"batch context_length -> batch context_length")
    targets_batched = rearrange(targets,"batch context_length -> batch context_length")

    return (seq_batched, targets_batched)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    config: dict = None,
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
        config (dict): Optional configuration dictionary to save with the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # Add configuration if provided
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']


def load_checkpoint_config(
    src: str | os.PathLike | BinaryIO | IO[bytes],
) -> dict:
    """
    Load only the configuration from a checkpoint file.
    
    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
    Returns:
        dict: The model configuration dictionary, or None if not found.
    """
    checkpoint = torch.load(src, map_location='cpu')  # Load on CPU to avoid GPU memory issues
    return checkpoint.get('config', None)


def cleanup_old_checkpoints(checkpoint_path: str, keep_count: int):
    """
    Remove old checkpoints, keeping only the most recent ones.
    
    Args:
        checkpoint_path (str): Base path for checkpoint files (e.g., './results/checkpoint')
        keep_count (int): Number of recent checkpoints to keep (0 to keep all)
    """
    if keep_count <= 0:
        return
    
    import glob
    import re
    
    # Get all checkpoint files for this base path
    pattern = f"{checkpoint_path}_step_*.ckpt"
    checkpoint_files = glob.glob(pattern)
    
    # Extract step numbers and sort by step
    step_files = []
    for file in checkpoint_files:
        match = re.search(r'_step_(\d+)', file)
        if match:
            step = int(match.group(1))
            step_files.append((step, file))
    
    step_files.sort(reverse=True)  # Sort by step number, newest first
    
    # Remove old checkpoints
    for step, file in step_files[keep_count:]:
        try:
            os.remove(file)
            print(f"Removed old checkpoint: {file}")
        except OSError as e:
            print(f"Warning: Could not remove {file}: {e}")
    