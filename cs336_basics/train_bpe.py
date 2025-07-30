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


def remove_pretoken_pairs(pretok_bytes, pretok_counts, pair_counts, pair2pretoks, pretok_id):
    """Remove all pairs from this pretoken from global counts"""
    for i in range(len(pretok_bytes)-1):
        # "erase" pair counts for old pretoken
        pair = (pretok_bytes[i],pretok_bytes[i+1])
        pair_counts[pair]-= pretok_counts
        if pair_counts[pair] <= 0:
            del pair_counts[pair]

        # remove pair from pair2pretoks for old pretoken
        pair2pretoks[pair].discard(pretok_id)
        if not pair2pretoks[pair]:
            del pair2pretoks[pair] # delete entry if pair has no pretokens associated


def apply_merge(pretok_bytes, merge_pair):
    """Apply merge with simple left-to-right scan (handles overlaps correctly)"""
    idx=0
    new_pretok_bytes = []
    new_merged_byte = merge_pair[0]+merge_pair[1]
    while idx < len(pretok_bytes):
        if (idx < len(pretok_bytes)-1 and
             (pretok_bytes[idx],pretok_bytes[idx+1])== merge_pair):
            new_pretok_bytes.append(new_merged_byte)
            idx+=2 #skip over the merged right byte
        else:
            new_pretok_bytes.append(pretok_bytes[idx])
            idx+=1
    return new_pretok_bytes

def add_pretoken_pairs(new_pretok_bytes, pretok_counts, pair_counts, pair2pretoks, pretok_id):
    """Add pairs from new pretoken (with max pair merged) from global counts"""
    for i in range(len(new_pretok_bytes)-1):
        # "erase" pair counts for old pretoken
        pair = (new_pretok_bytes[i],new_pretok_bytes[i+1])
        pair_counts[pair] += pretok_counts

        # add pair from pair2pretoks for old pretoken
        pair2pretoks[pair].add(pretok_id)


def bpe_merges(
        vocab: dict[int, bytes],
        pretoken_counts: Counter[tuple[bytes,...]],
        vocab_size: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Perform BPE merges until vocab_size is reached.
    Simpler incremental: only update impacted pretokens, but with cleaner logic.
    
    Args:
        vocab: Initial vocabulary (0-255 single bytes)
        pretoken_counts: Pre-token counts 
        vocab_size: Target vocabulary size
    
    Returns:
        (vocab, merges): Final vocabulary and merge operations
    """
    merges = []
    leading_token_id = len(vocab)
    
    # initialize local data structures to keep track of pretok byte lists, pretok counts (static), pair counts, bp to pretokens 
    pretoks = {i:list(byte_tuple) for i,byte_tuple in enumerate(pretoken_counts.keys())} # pretok id : byte list
    pretok_counts = {i:counts for i, (_,counts) in enumerate(pretoken_counts.items())} # pretok id : pretoken count (static)
    pair_counts = Counter() # bp counts
    pair2pretoks = defaultdict(set) # bp to pretok ids

    # initial pass to count byte pairs in corpus
    for pretok_id, pretok_bytes in pretoks.items():
        for i in range(len(pretok_bytes)-1):
            # update byte pair counts
            pair = (pretok_bytes[i],pretok_bytes[i+1])
            pair_counts[pair] += pretok_counts[pretok_id]
            
            # add map from bp to pretoken it occurred in
            pair2pretoks[pair].add(pretok_id)

    # main loop for be merges
    while len(vocab) < vocab_size:
        # find the max bp, update merges and vocab
        max_pair = max(pair_counts.items(),key=lambda x: (x[1], x[0]))[0]
        merges.append(max_pair)
        max_pair_merged = max_pair[0]+max_pair[1]
        vocab[leading_token_id] = max_pair_merged
        leading_token_id+=1
    
        # find impacted pretokens
        impacted_pretoks = pair2pretoks[max_pair].copy()

        for pretok_id in impacted_pretoks:
            pretok_bytes = pretoks[pretok_id]
            impacted_pretok_counts = pretok_counts[pretok_id]
            # for each impacted pretoken remove all counts for bps in these pretokens
            remove_pretoken_pairs(pretok_bytes,impacted_pretok_counts,pair_counts,pair2pretoks,pretok_id)
            
            # remake the pretoken bytes list with merged token, overwrite old pretoken bytes
            new_pretok_bytes = apply_merge(pretok_bytes,max_pair)
            pretoks[pretok_id]=new_pretok_bytes

            # recount the pretoks with new byte list
            add_pretoken_pairs(new_pretok_bytes, impacted_pretok_counts, pair_counts,pair2pretoks, pretok_id)
    
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
    return vocab, merges
