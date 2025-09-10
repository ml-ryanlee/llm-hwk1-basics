import os
import numpy as np
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int, Bool
from cs336_basics.tx_train_bpe import train_bpe
from cs336_basics.tx_tokenizer import Tokenizer
from cs336_basics.tx_model import Linear,Embedding, RMSNorm,PositionwiseFeedforward,PrenormBlock, Transformer
from cs336_basics.tx_model import RotaryPositionalEmbedding,MultiheadSelfAttention,scaled_dot_product_attention
from cs336_basics.tx_utils import softmax, cross_entropy_loss, gradient_clipping, data_loader, save_checkpoint, load_checkpoint, cleanup_old_checkpoints, load_checkpoint_config
from cs336_basics.tx_optimizer import AdamW,lr_cosine_schedule
import numpy.typing as npt
import torch
from torch import Tensor
import argparse

from cs336_basics.tx_inference import decode
MODEL_PATH = "/project/jonmay_1426/ryantlee/llm-hwk1-basics/cs336_basics/tx_experiments/results/lr_0.001_checkpoint_step_002000_final.ckpt"
TOKENIZER_PATH = "/project/jonmay_1426/ryantlee/llm-hwk1-basics/results/TinyStories-train-results"

def main():
    # Load model checkpoint
    model_path = MODEL_PATH
   
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint file not found at {model_path}")
        return
    
    try:
        # Load configuration from checkpoint
        config = load_checkpoint_config(model_path)
        if config is None:
            print("Error: No configuration found in checkpoint. Using default values.")
            # Fall back to default configuration
            config = {
                'vocab_size': 10000,
                'context_length': 256,
                'd_model': 512,
                'num_layers': 4,
                'num_heads': 16,
                'd_ff': 1344,
                'rope_theta': 10000.0,
                'device': 'cpu'
            }
        else:
            print(f"Loaded configuration from checkpoint: {config}")
        
        # Initialize model with configuration from checkpoint
        model = Transformer(
            vocab_size=config['vocab_size'],
            context_length=config['context_length'],
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            rope_theta=config['rope_theta'],
            device=config['device']
        )
        
        # Initialize optimizer (needed for load_checkpoint, but won't be used for inference)
        # Use optimizer config if available, otherwise use defaults
        if 'optimizer_config' in config:
            opt_config = config['optimizer_config']
            optimizer = AdamW(
                model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config['betas'],
                eps=opt_config['eps'],
                weight_decay=opt_config['weight_decay']
            )
        else:
            optimizer = AdamW(model.parameters(), lr=0.001)
        
        # Load the checkpoint using the existing function
        iteration = load_checkpoint(model_path, model, optimizer)
        print(f"Successfully loaded model from checkpoint at iteration {iteration}")
        
        # Set model to evaluation mode
        model.eval()
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # load tokenizer
    # You'll need to load the tokenizer that was used during training
    # This would typically be from the same directory as the training data
    tokenizer_dir = TOKENIZER_PATH
    vocab_path = os.path.join(tokenizer_dir, "vocab.pkl")
    merges_path = os.path.join(tokenizer_dir, "merges.pkl")
    
    try:
        # Load the tokenizer using the from_files class method
        if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
            print(f"Error: Tokenizer files not found. Expected {vocab_path} and {merges_path}")
            return
            
        tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
        print("Successfully loaded tokenizer")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # example string
    example_text = ""
    
    # run decode
    try:
        print("\nRun 1: Unconditional Generation\n")
        generated_text = decode(model, tokenizer,example_text,max_tokens=512,sampling=True)
        print(f"Input: {example_text}")
        print(f"Generated: {generated_text}")
    except Exception as e:
        print(f"Error during generation: {e}")

     # example string
    example_text = "Once upon a time"

    # run decode
    try:
        print("\nRun 2: Conditional Generation\n")
        generated_text = decode(model, tokenizer,example_text,max_tokens=512,sampling=True)
        print(f"Input: {example_text}")
        print(f"Generated: {generated_text}")
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()