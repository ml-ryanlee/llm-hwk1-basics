
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from typing import List, Tuple
import time
from pathlib import Path
from tqdm import tqdm

from cs336_basics.tx_tokenizer import Tokenizer
from cs336_basics.tx_train_bpe import find_chunk_boundaries

def encode_text_chunk(args: Tuple[str, int, int, str, str, List[str]]) -> List[int]:
    """
    Encode a chunk of text from a file.
    
    Args:
        args: Tuple containing (filepath, start_byte, end_byte, vocab_path, merges_path, special_tokens)
    
    Returns:
        List of token IDs for this chunk
    """
    filepath, start_byte, end_byte, vocab_path, merges_path, special_tokens = args
    
    # Initialize tokenizer in each worker process
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    
    # Read the chunk
    with open(filepath, 'rb') as f:
        f.seek(start_byte)
        chunk_bytes = f.read(end_byte - start_byte)
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
    
    # Encode the chunk
    token_ids = tokenizer.encode(chunk_text)
    return token_ids

def encode_dataset_parallel(
    data_path: str,
    vocab_path: str, 
    merges_path: str,
    output_path: str,
    special_tokens: List[str] = None,
    num_processes: int = None
) -> None:
    """
    Encode a dataset in parallel and save as NumPy uint16 array.
    
    Args:
        data_path: Path to the text data file
        vocab_path: Path to the vocabulary pickle file
        merges_path: Path to the merges pickle file  
        output_path: Path to save the encoded token IDs
        special_tokens: List of special tokens
        num_processes: Number of processes to use (defaults to CPU count)
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
        
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Encoding {data_path} using {num_processes} processes...")
    start_time = time.time()
    
    # Find chunk boundaries based on special tokens
    with open(data_path, 'rb') as f:
        split_token = special_tokens[0].encode('utf-8')
        chunk_boundaries = find_chunk_boundaries(f, num_processes, split_token)
    
    print(f"Split file into {len(chunk_boundaries)-1} chunks")
    
    # Prepare arguments for each chunk
    chunk_args = []
    for i in range(len(chunk_boundaries) - 1):
        start_byte = chunk_boundaries[i]
        end_byte = chunk_boundaries[i + 1]
        chunk_args.append((data_path, start_byte, end_byte, vocab_path, merges_path, special_tokens))
    
    # Process chunks in parallel
    all_token_ids = []
    with Pool(processes=num_processes) as pool:
        print("Processing chunks...")
        # Use tqdm to show progress as chunks complete
        chunk_results = []
        with tqdm(total=len(chunk_args), desc="Encoding chunks", unit="chunk") as pbar:
            # Use imap for progress tracking
            for result in pool.imap(encode_text_chunk, chunk_args):
                chunk_results.append(result)
                pbar.update(1)
        
        # Combine results
        print("Combining chunk results...")
        for i, token_ids in enumerate(tqdm(chunk_results, desc="Combining results", unit="chunk")):
            all_token_ids.extend(token_ids)
    
    # Convert to NumPy array with uint16 datatype
    print(f"Converting {len(all_token_ids)} tokens to NumPy array...")
    token_array = np.array(all_token_ids, dtype=np.uint16)
    
    # Save the array
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, token_array)
    
    elapsed_time = time.time() - start_time
    print(f"Encoding complete! Saved {len(all_token_ids)} tokens to {output_path}")
    print(f"Array shape: {token_array.shape}, dtype: {token_array.dtype}")
    print(f"Total time: {elapsed_time:.2f} seconds")

def main():
    """Main function to encode TinyStories datasets"""
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results" / "TinyStories-train-results"
    
    train_data_path = str(data_dir / "TinyStoriesV2-GPT4-train.txt")
    valid_data_path = str(data_dir / "TinyStoriesV2-GPT4-valid.txt")
    vocab_path = str(results_dir / "vocab.pkl")
    merges_path = str(results_dir / "merges.pkl")
    
    # Output paths
    train_output = str(data_dir / "tinystories_train_tokens.npy")
    valid_output = str(data_dir / "tinystories_valid_tokens.npy")
    
    special_tokens = ["<|endoftext|>"]
    
    
    # Encode validation set
    if os.path.exists(valid_data_path):
        encode_dataset_parallel(
            valid_data_path, vocab_path, merges_path,
            valid_output, special_tokens
        )
    else:
        print(f"Validation data not found at {valid_data_path}")
    
    # Encode training set
    # if os.path.exists(train_data_path):
    #     encode_dataset_parallel(
    #         train_data_path, vocab_path, merges_path, 
    #         train_output, special_tokens
    #     )
    # else:
    #     print(f"Training data not found at {train_data_path}")

if __name__ == "__main__":
    main()
