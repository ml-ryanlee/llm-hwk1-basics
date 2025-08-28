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
from cs336_basics.tx_optimizer import AdamW,lr_cosine_schedule
import numpy.typing as npt
import torch
from torch import Tensor
import argparse

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(prog="model training",
    description="Train a model given a tokenized dataset given as a numpy array.")

    # I/O settings
    parser.add_argument('--train_path',
                        default='/Users/ryanlee/code/llm-hwk1-basics/data/tinystories_train_tokens.npy',
                        help='path to dataset for model training, must be .npy file')
    
    parser.add_argument('--val_path',
                    default='/Users/ryanlee/code/llm-hwk1-basics/data/tinystories_valid_tokens.npy',
                    help='path to dataset for model training, must be .npy file')

    parser.add_argument('--output_dir', 
                        default='./results',
                        help='directory to save the trained model')

    # Data settings
    parser.add_argument('--batch_size', 
                       type=int, 
                       default=64, 
                       help='number of examples in batch (default: 64)')

    parser.add_argument('--context_length', 
                       type=int, 
                       default=256, 
                       help='context length for training sequences (default: 256)')                           


    # Model settings
    parser.add_argument('--vocab_size',
                       type=int,
                       default=10000,
                       help='vocabulary size (default: 10000)')

    parser.add_argument('--d_model', 
                       type=int, 
                       default=512, 
                       help='dimension of the hidden model dimension (default: 512)')        
    
    parser.add_argument('--d_ff', 
                       type=int, 
                       default=1344, 
                       help='dimension of the feed-forward dimension (default: 1344)')     
                            
    parser.add_argument('--rope_theta', 
                       type=float, 
                       default=10000.0, 
                       help='RoPE theta parameter (default: 10000.0)')    
    
    parser.add_argument('--num_layers', 
                       type=int, 
                       default=4, 
                       help='Number of pre-norm transformer layers (default: 4)')
    
    parser.add_argument('--num_heads', 
                       type=int, 
                       default=16, 
                       help='Number of attention heads (default: 16)')

    # Optimizer Settings
    parser.add_argument('--lr', 
                       type=float, 
                       default=0.001, 
                       help='learning rate (default: 0.001)')
    
    parser.add_argument('--lr_warmup_iters', 
                       type=int, 
                       default=100, 
                       help='learning rate warmup iterations (default: 100)')
    
    parser.add_argument('--lr_min', 
                       type=float, 
                       default=0.0001, 
                       help='minimum learning rate (default: 0.0001)')
    
    parser.add_argument('--beta1', 
                       type=float, 
                       default=0.9, 
                       help='AdamW beta1 parameter (default: 0.9)')
    
    parser.add_argument('--beta2', 
                       type=float, 
                       default=0.999, 
                       help='AdamW beta2 parameter (default: 0.999)')
    
    parser.add_argument('--eps', 
                       type=float, 
                       default=1e-8, 
                       help='AdamW epsilon parameter (default: 1e-8)')
    
    parser.add_argument('--weight_decay', 
                       type=float,
                       default=0.01, 
                       help='weight decay (default: 0.01)')

    # Training settings
    parser.add_argument('--device',
                       type=str,
                       default='mps',
                       help='device to train on (default: mps)')

    parser.add_argument('--steps',
                       type=int,
                       default=50,
                       help='Optimizer steps to perform in training')

    parser.add_argument('--max_grad_norm',
                       type=float,
                       default=1.0,
                       help='maximum gradient norm for clipping (default: 1.0)')
    
    parser.add_argument('--save_every',
                       type=int,
                       default=10,
                       help='save checkpoint every N iterations (default: 1000)')
    
    parser.add_argument('--eval_every',
                       type=int,
                       default=100,
                       help='evaluate every N iterations (default: 100)')
    
    parser.add_argument('--checkpoint_path',
                       type=str,
                       default='./results/checkpoint',
                       help='path to save checkpoints (default: ./results/checkpoint, will append step number)')
    
    parser.add_argument('--resume_from',
                       type=str,
                       default=None,
                       help='path to checkpoint to resume training from (default: None)')
    
    parser.add_argument('--keep_checkpoints',
                       type=int,
                       default=3,
                       help='number of recent checkpoints to keep (default: 3, set to 0 to keep all)')

    args = parser.parse_args()
  
    # check dataset file path
    train_path = args.train_path
    if not train_path.endswith(('.npy','.npz')):
        parser.error(f"Dataset file must be .npy or .npz, got {train_path}")
    if not os.path.exists(train_path):
        parser.error(f"Dataset file does not exist: {train_path}")

    valid_path = args.val_path
    if not valid_path.endswith(('.npy','.npz')):
        parser.error(f"Validation file must be .npy or .npz, got {train_path}")
    if not os.path.exists(valid_path):
        parser.error(f"Dataset file does not exist: {train_path}")    
    
    # efficiently load numpy array from dataset_path 
    trainset = np.load(train_path,mmap_mode='r')
    valset = np.load(valid_path,mmap_mode='r')
    print(f"Loaded dataset with {len(trainset)} tokens")
    
    # set up dataloader
    batch_size = args.batch_size
    context_length = args.context_length
    device = args.device

    # Initialize model
    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device
    )
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on device: {device}")

    # Create configuration dictionary for saving
    config = {
        'vocab_size': args.vocab_size,
        'context_length': args.context_length,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'd_ff': args.d_ff,
        'rope_theta': args.rope_theta,
        'device': device,
        # Also save optimizer config for reference
        'optimizer_config': {
            'lr': args.lr,
            'betas': (args.beta1, args.beta2),
            'eps': args.eps,
            'weight_decay': args.weight_decay
        }
    }

    # resume from checkpoint if specified
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            parser.error(f"Checkpoint file does not exist: {args.resume_from}")
        resume_step = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed training from step {resume_step}")
        
        # Adjust total steps to account for already completed steps
        remaining_steps = args.steps - resume_step
        if remaining_steps <= 0:
            print(f"Training already completed! Checkpoint shows {resume_step} steps completed, but only {args.steps} requested.")
            return
        print(f"Continuing for {remaining_steps} more steps...")
    else:
        resume_step = 0

    # train loop
    print(f"Starting training for {args.steps} steps (resuming from step {resume_step})...")
    print(f"Batch size: {batch_size}, Context length: {context_length}")
    print("-" * 60)
    
    for step in range(resume_step, args.steps):
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
        if args.checkpoint_path and step % args.save_every == 0 and step > 0:
            checkpoint_dir = os.path.dirname(args.checkpoint_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_file = f"{args.checkpoint_path}_step_{step:06d}.ckpt"
            save_checkpoint(model, optimizer, step, checkpoint_file, config)
            print(f"Checkpoint saved: {checkpoint_file}")
            
            # Clean up old checkpoints
            cleanup_old_checkpoints(args.checkpoint_path, args.keep_checkpoints)

        # logging
        if step % 10 == 0:  # Print every 10 steps
            print(f"Step {step:4d}/{args.steps}: Loss = {loss.item():.4f} | "
                  f"Progress: {step/args.steps*100:.1f}%")
        
        # validation
        if step % args.eval_every == 0: 
            model.eval()
            with torch.no_grad():
                val_inputs,val_targets = data_loader(valset, batch_size, context_length, device)
                val_logits = model.forward(val_inputs)
                val_loss = cross_entropy_loss(val_logits,val_targets)
                print(f"Step {step:4d}/{args.steps}: Validation Loss = {val_loss.item():.4f}")
            model.train()
    
    print("-" * 60)
    print(f"Training completed! Final loss: {loss.item():.4f}")
    
    # save final checkpoint
    if args.checkpoint_path:
        final_checkpoint_file = f"{args.checkpoint_path}_step_{args.steps:06d}_final.ckpt"
        save_checkpoint(model, optimizer, args.steps, final_checkpoint_file, config)
        print(f"Final checkpoint saved to {final_checkpoint_file}")
    
    # save model

if __name__ == "__main__":
    main()