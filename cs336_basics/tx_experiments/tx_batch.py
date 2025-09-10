import os
import numpy as np
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int, Bool
from cs336_basics.tx_train_bpe import train_bpe
from cs336_basics.tx_tokenizer import Tokenizer
from cs336_basics.tx_model import Linear,Embedding, RMSNorm,PositionwiseFeedforward,PrenormBlock, Transformer
from cs336_basics.tx_model import RotaryPositionalEmbedding,MultiheadSelfAttention,scaled_dot_product_attention
from cs336_basics.tx_utils import softmax, cross_entropy_loss, gradient_clipping, data_loader, save_checkpoint, load_checkpoint, cleanup_old_checkpoints
from cs336_basics.tx_optimizer import AdamW
from keys import WANDB_API_KEY

import numpy.typing as npt
import torch
import wandb
from torch import Tensor

os.environ["WANDB_API_KEY"] = WANDB_API_KEY

def train_single_batch(config):
    # Extract parameters from config
    lr = config['lr']
    train_path = config['train_path']
    val_path = config['val_path']
    batch_size = config['batch_size']
    context_length = config['context_length']
    vocab_size = config['vocab_size']
    d_model = config['d_model']
    d_ff = config['d_ff']
    rope_theta = config['rope_theta']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    beta1 = config['beta1']
    beta2 = config['beta2']
    eps = config['eps']
    weight_decay = config['weight_decay']
    device = config['device']
    steps = config['steps']
    save_every = config['save_every']
    eval_every = config['eval_every']
    checkpoint_path = config['checkpoint_path']
    resume_from = config.get('resume_from', None)
    keep_checkpoints = config['keep_checkpoints']
  
    # check dataset file path
    if not train_path.endswith(('.npy','.npz')):
        raise ValueError(f"Dataset file must be .npy or .npz, got {train_path}")
    if not os.path.exists(train_path):
        raise ValueError(f"Dataset file does not exist: {train_path}")

    if not val_path.endswith(('.npy','.npz')):
        raise ValueError(f"Validation file must be .npy or .npz, got {val_path}")
    if not os.path.exists(val_path):
        raise ValueError(f"Validation file does not exist: {val_path}")    
    
    # efficiently load numpy array from dataset_path 
    trainset = np.load(train_path,mmap_mode='r')
    valset = np.load(val_path,mmap_mode='r')
    print(f"Loaded dataset with {len(trainset)} tokens")

    # Initialize model
    model = Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device
    )
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on device: {device}")

    # Create configuration dictionary for saving
    save_config = {
        'vocab_size': vocab_size,
        'context_length': context_length,
        'd_model': d_model,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'd_ff': d_ff,
        'rope_theta': rope_theta,
        'device': device,
        # Also save optimizer config for reference
        'optimizer_config': {
            'lr': lr,
            'betas': (beta1, beta2),
            'eps': eps,
            'weight_decay': weight_decay
        }
    }

    # resume from checkpoint if specified
    if resume_from:
        if not os.path.exists(resume_from):
            raise ValueError(f"Checkpoint file does not exist: {resume_from}")
        resume_step = load_checkpoint(resume_from, model, optimizer)
        print(f"Resumed training from step {resume_step}")
        
        # Adjust total steps to account for already completed steps
        remaining_steps = steps - resume_step
        if remaining_steps <= 0:
            print(f"Training already completed! Checkpoint shows {resume_step} steps completed, but only {steps} requested.")
            return
        print(f"Continuing for {remaining_steps} more steps...")
    else:
        resume_step = 0

    # train loop
    print(f"Starting training for {steps} steps (resuming from step {resume_step})...")
    print(f"Batch size: {batch_size}, Context length: {context_length}")
    print("-" * 60)
    
    for step in range(resume_step, steps):
        # zero grad
        optimizer.zero_grad()
        
        # get batches
        inputs, targets = data_loader(trainset, batch_size, context_length, device)

        # forward (tensors)
        logits = model.forward(inputs)

        # calculate loss
        loss = cross_entropy_loss(logits,targets)

        # backward on loss
        loss.backward()

        # step optimizer
        optimizer.step()
        
        # checkpointing
        if checkpoint_path and step % save_every == 0 and step > 0:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_file = f"{checkpoint_path}_step_{step:06d}.ckpt"
            save_checkpoint(model, optimizer, step, checkpoint_file, save_config)
            print(f"Checkpoint saved: {checkpoint_file}")
            
            # Clean up old checkpoints
            cleanup_old_checkpoints(checkpoint_path, keep_checkpoints)

        # logging
        if step % 10 == 0:  # Log every 10 steps
            wandb.log({
                "train_loss": loss.item()
            }, step=step)
    
        # validation
        if step % eval_every == 0: 
            model.eval()
            with torch.no_grad():
                val_inputs,val_targets = data_loader(valset, batch_size, context_length, device)
                val_logits = model.forward(val_inputs)
                val_loss = cross_entropy_loss(val_logits,val_targets)
                print(f"Step {step:4d}/{steps}: Validation Loss = {val_loss.item():.4f}")
                
                # Log validation loss to wandb
                wandb.log({
                    "val_loss": val_loss.item()
                }, step=step)
            model.train()
    
    print("-" * 60)
    print(f"Training completed! Final loss: {loss.item():.4f}")
    
    # save final checkpoint
    if checkpoint_path:
        final_checkpoint_file = f"{checkpoint_path}_step_{steps:06d}_final.ckpt"
        save_checkpoint(model, optimizer, steps, final_checkpoint_file, save_config)
        print(f"Final checkpoint saved to {final_checkpoint_file}")
    
def main():
    # Fixed configuration - modify these as needed
    base_config = {
        # I/O settings
        'train_path': '/project/jonmay_1426/ryantlee/llm-hwk1-basics/data/tinystories_train_tokens.npy',
        'val_path': '/project/jonmay_1426/ryantlee/llm-hwk1-basics/data/tinystories_valid_tokens.npy',
        
        # Data settings (lr will be varied)
        'context_length': 256,
        
        # Model settings
        'vocab_size': 10000,
        'd_model': 512,
        'd_ff': 1344,
        'rope_theta': 10000.0,
        'num_layers': 4,
        'num_heads': 16,
        
        # Optimizer settings 
        'lr':1e-3,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        'weight_decay': 0.01,
        
        # Training settings - auto-detect best device
        'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        'steps': 2000,
        'save_every': 10,
        'eval_every': 100,
        'keep_checkpoints': 3,
    }
    
    # Learning rates to test (centered around 0.001)
    batch_sizes = [64,126,256,1024,2048,4096]
    
    print("Starting batch size experiments...")
    print(f"Testing batch sizes: {batch_sizes}")
    print("=" * 80)
    
    wandb.login()
    
    for batch_size in batch_sizes:
        # Create config for this specific run
        config = base_config.copy()
        config['batch_size'] = batch_size
        config['checkpoint_path'] = f'./results/batch_size_{batch_size}_checkpoint'
        
        # Initialize wandb for this run
        wandb.init(
            project="cs336-hwk1-batch-experiments",
            name=f"batch_{batch_size}",
            config=config,
            reinit=True
        )
        
        print(f"\n{'='*20} Training with batch={batch_size} {'='*20}")
        try:
            train_single_batch(config)
            print(f"Completed training with batch={batch_size}")
        except Exception as e:
            print(f"Error with batch={batch_size}: {e}")
        finally:
            wandb.finish()
        print("=" * 60)

if __name__ == "__main__":
    # You can either run all experiments or a single one
    # For all learning rate experiments:
    main()