import os
import sys
import regex as re
import multiprocessing as mp

from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict


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


def pre_tokenize(start: int, end: int, filepath: str, special_tokens: list[str]) -> list[dict[tuple[bytes,...], int]]:
    """
    Pre-tokenize a chunk of text and return the counts for each pre-token per document.
    Returns a list of dicts, where each dict maps (byte1, byte2, ...) -> count.
    Each token is represented as a tuple of byte values (0-255) from UTF-8 encoding.
    
    Args:
        start (int): The start index of the chunk to pre-tokenize.
        end (int): The end index of the chunk to pre-tokenize.
        filepath (str): The path to the file to pre-tokenize.
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.
    
    Returns:
        list[dict[tuple[bytes], int]]:
            A list of dicts, where each dict maps (byte1, byte2, ...) -> count, e.g. {(l,o,w): 5 ...}
            each element in the list represents the pre-token counts for a document
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
        document_pretoken_counts = []
        for doc_chunk in document_chunks:
            if doc_chunk.strip():  # Skip empty chunks
                token_counts = Counter()
                for match in token_pattern.finditer(doc_chunk):
                    token = match.group() # pre-token string
                    token_bytes = token.encode("utf-8") # bytes
                    token_bytes_tuple = tuple(bytes([b]) for b in token_bytes)
                    token_counts[token_bytes_tuple] += 1
                document_pretoken_counts.append(token_counts)
        
        return document_pretoken_counts

def bpe_merges(
        vocab: dict[int, bytes],
        pretoken_counts: Counter[tuple[bytes,...]],
        vocab_size: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Perform BPE merges until vocab_size is reached.
    
    Args:
        vocab: Initial vocabulary (0-255 single bytes)
        document_pretoken_counts: Pre-token counts per document
        vocab_size: Target vocabulary size
        special_tokens: Special tokens to preserve
    
    Returns:
        (vocab, merges): Final vocabulary and merge operations
    """
    merges = []
    bp = Counter() #byte pairs to occurrence count
    leading_token_id = len(vocab)

    # data structures to keep track of byte sequences and byte pairs
    bp_counts = Counter()
    bp2pretokens = defaultdict(list) # tracks what pretokens bp came from
    pretoken2bps = defaultdict(list) # tracks the bps in a pretoken
    pretoken_idx = 0

    for byte_tuple, counts in pretoken_counts.items():
        for i, byte in enumerate(byte_tuple):
            if i>=1:
                # update bp counts
                bp = (byte_tuple[i-1],byte_tuple[i])
                bp_counts[bp]+=counts

                # update map from bp to tokens
                bp2pretokens[bp].append((pretoken_idx,counts,i-1,i))

                # track the indexes of the bp associated with the pretoken
                pretoken2bps[pretoken_idx].append((bp,i-1, i))

        pretoken_idx+=1
    
    # loop to add to vocabulary
    while len(vocab) < vocab_size:
        # sort by greatest co-occurance, then lexicographic bytes
        max_bp = max(bp_counts.items(), key=lambda x: (x[1], x[0]))[0]

        # add max_bp to merges
        merges.append(max_bp)

        # create a merged byte object and add it to vocab
        merged_bp = max_bp[0]+max_bp[1]
        vocab[leading_token_id] = merged_bp
        leading_token_id+=1

        # delete the max_bp from the bp_counts 
        del bp_counts[max_bp]

        # iterate through impacted pretokens to get new bp with merged tokens and update
        impacted_pretokens = bp2pretokens[max_bp]
        for pretoken_tuple in impacted_pretokens:
            pretoken_ref_idx, pretoken_ref_counts, left, right = pretoken_tuple

            # logic on adjacent indices
            bps_in_pretoken = pretoken2bps[pretoken_ref_idx]

            # Collect changes to apply after the loop
            to_remove = []
            to_add = []

            # iterate through other bps 
            bps_in_pretoken.remove((max_bp, left, right))
            for bp_tuple in bps_in_pretoken:
                adj_bp, adj_left, adj_right = bp_tuple
                if left == adj_right:
                    new_bp = (adj_bp[0], merged_bp)
                    to_remove.append((adj_bp, adj_left, adj_right))  # Original BP with original indices
                    to_add.append((new_bp, adj_left, right))
                elif right == adj_left:
                    new_bp = (merged_bp,adj_bp[1])
                    to_remove.append((adj_bp, adj_left, adj_right))  # Original BP with original indices
                    to_add.append((new_bp,left,adj_right))

            # Apply changes after the loop
            for bp_tuple in to_remove:
                # pass by reference modification to pretoken2bps
                bps_in_pretoken.remove(bp_tuple)
                # Also remove from bp2pretokens and update bp_counts
                old_bp = bp_tuple[0]
                entry_to_remove = (pretoken_ref_idx, pretoken_ref_counts, bp_tuple[1], bp_tuple[2])
                bp2pretokens[old_bp].remove(entry_to_remove)
                bp_counts[old_bp] -= pretoken_ref_counts
                if bp_counts[old_bp] <= 0:
                    del bp_counts[old_bp]

            for new_bp_tuple in to_add:
                bps_in_pretoken.append(new_bp_tuple)
                # Also add to bp2pretokens and update bp_counts
                new_bp = new_bp_tuple[0]
                new_left, new_right = new_bp_tuple[1], new_bp_tuple[2]
                bp2pretokens[new_bp].append((pretoken_ref_idx, pretoken_ref_counts, new_left, new_right))
                bp_counts[new_bp] += pretoken_ref_counts

        # Clear bp2pretokens for the merged BP since it no longer exists
        del bp2pretokens[max_bp]


    return vocab, merges
    

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
    vocab = {}
    # give special tokens token IDS first, following convention expected
    for idx, special_token in enumerate(special_tokens):
        vocab[idx] = special_token.encode("utf-8")

    # after, token ids assigned to all single byte values (0-256)
    offset = len(special_tokens)
    for idx in range(256):
        vocab[idx+offset] = bytes([idx])

    # pretokenization: find chunk boundaries, create arglist for parallel input, run mp
    num_processes = kwargs.get("num_processes", mp.cpu_count())

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
    
    start_end_pairs = list(zip(boundaries[:-1], boundaries[1:]))
    num_pairs = len(start_end_pairs)
    argslist = [(start, end, input_path, special_tokens) for start, end in start_end_pairs]
    with Pool(min(num_processes, num_pairs)) as p:
        results = p.starmap(pre_tokenize, argslist)

    # combine parallel results by unrolling into a single list of dicts, then sum the dicts
    all_docs_pretoken_counts = []
    for result in results:
        all_docs_pretoken_counts.extend(result)
    pretoken_counts = sum(all_docs_pretoken_counts,Counter())

    # perform bpe_merges until vocab size is reached.
    vocab, merges = bpe_merges(vocab, pretoken_counts, vocab_size)

    # Return vocab and merges
    return tuple(vocab,merges)

# Test script 

