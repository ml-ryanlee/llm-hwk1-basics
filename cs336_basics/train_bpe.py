import os
import regex as re
import multiprocessing as mp

from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from collections import Counter


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess byte position
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))




def pre_tokenize(start: int, end: int, filepath: str, special_tokens: list[str]) -> list[dict[tuple[int], int]]:
    """
    Pre-tokenize a chunk of text and return the counts for each pre-token per document.
    Returns a list of dicts, where each dict maps (byte1, byte2, ...) -> count.
    Each token is represented as a tuple of byte values (0-255) from UTF-8 encoding.
    """

    with open(filepath, "rb") as f:
        f.seek(start)
        file_chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        # split file_chunk by special_tokens
        escaped_special_tokens = [re.escape(token) for token in special_tokens]
        special_token_pattern = "|".join(escaped_special_tokens)
        document_chunks = re.split(special_token_pattern, file_chunk)

        # run pre-tokenization on each document chunk
        token_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # process each document chunk separately
        document_token_counts = []
        for doc_chunk in document_chunks:
            if doc_chunk.strip():  # Skip empty chunks
                token_counts = {}
                for match in token_pattern.finditer(doc_chunk):
                    token = match.group()
                    token_bytes = token.encode("utf-8")
                    # Convert bytes object to tuple of byte values (0-255)
                    token_tuple = tuple(token_bytes)
                    token_counts[token_tuple] = token_counts.get(token_tuple, 0) + 1
                document_token_counts.append(token_counts)
        
        return document_token_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    num_processes = kwargs.get("num_processes", mp.cpu_count())

    # find chunk boundaries in text to parallelize pre-tokenization
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        
        # parallelize pre-tokenization
        start_end_pairs =list(zip(boundaries[:-1], boundaries[1:]))
        num_pairs = len(start_end_pairs)

        # parallelize pre-tokenization
        argslist = [(start, end, input_path,special_tokens) for start, end in start_end_pairs]

        with Pool(min(num_processes, num_pairs)) as p:
            results =p.starmap(pre_tokenize, argslist)
    
        print(len(results))
        
        # TO DO: apply bpe training algorithm.

        # After pre-tokenization, add this to test just that part:
        # print(f"Pre-token counts: {len(pre_token_counts)}")
        # print(f"Sample tokens: {list(pre_token_counts.keys())[:10]}")
    
        # Return early to test just pre-tokenization
        raise NotImplementedError("Testing pre-tokenization only")

# Test script 

