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
        document_pretoken_counts: list[dict[tuple[bytes,...], int]],
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
    adj = defaultdict(Counter) # track and count overlapping byte-pairs
    leading_token_id = len(vocab)

    # first pass to build up byte pairs counts and adjacency counts
    for pretoken_counts in document_pretoken_counts:
        # count byte pairs in a single document
        for byte_tuple, counts in pretoken_counts.items():
            # for every pretoken, get the byte pair
            prev = None
            for byte_idx in range(len(byte_tuple)-1):
                pair = (byte_tuple[byte_idx],byte_tuple[byte_idx+1])
                bp[pair]+=counts
                if prev:
                    # counts how many times bp specifically overlapped with adj bps
                    adj[pair][prev] +=counts
                    adj[prev][pair] +=counts
                prev = pair

    # loop to add to vocabulary
    while len(vocab) < vocab_size:
        # sort by greatest co-occurance, then lexicographic bytes
        most_frequent_pair = max(bp.items(), key=lambda x: (x[1], x[0]))[0]

        # add byte pair to merges and vocab
        merges.append(most_frequent_pair)
        merged_bp = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[leading_token_id] = merged_bp
        leading_token_id+=1
        del bp[most_frequent_pair] # remove the byte pair from bp count dict

        # update adjacency dict and byte pair counts after merge
        adj_list = adj[most_frequent_pair].items()
        for adj_bp,adj_count in adj_list:
            # update bp count for that neighbor
            bp[adj_bp]-=adj_count
            if bp[adj_bp] <= 0:
                assert(bp[adj_bp]==0), 'ERROR: bp counter should never be negative! Check bp counter update'
                del bp[adj_bp]
        
            # create new bp entries based on neighbor
            if most_frequent_pair[1] == adj_bp[0]:
                # neighbor overlaps on last byte of most frequent bp
                new_bp = (merged_bp,adj_bp[1])
                # TODO: update the new adjacent pairs in adj dict
                adj[adj_bp[1]]+= adj_count
            else:
                # neighbor overlaps on first byte of most frequent bp
                new_bp = (adj_bp[0],merged_bp)
            bp[new_bp] = adj_count

           
            


            # update adj counts associated with new_bp
            del adj[adj_bp][most_frequent_pair]
            if not adj[adj_bp]:
                del adj[adj_bp]

            
            






            # get those neighbors, new neighbors of merged_bp (exclude merged_bp)
            




    
    # bp.items() returns ((bytes_tuple),count). Find count, then max bytes tuple (lexicographic)
    most_frequent_pair = max(bp.items(), key=lambda x: (x[1], x[0]))[0]
    print("DEBUG")
    print(f"Type of most_frequent_pair: {type(most_frequent_pair)}")
    print(f"most_frequent_pair: {most_frequent_pair}")
    print(f"Length: {len(most_frequent_pair)}")
    print(f"First element type: {type(most_frequent_pair[0])}")
    print(f"Second element type: {type(most_frequent_pair[1])}")
    sys.exit()
    # while len(vocab) < vocab_size:
    #     # add merged byte to vocabulary and list of merges
    #     vocab[leading_token_id] = most_frequent_pair
    #     leading_token_id+=1
    #     merges.append(most_frequent_pair)

        # update pretoken counts by removing individual bytes
        # bp_update_locations = bp_locations[most_frequent_pair]  # TODO: implement location tracking
        # for location in bp_update_locations:
        #     doc_idx, pretoken_byte_tuple, pretoken_byte_idx = location
        #     # assign value of old key with new key
        #     # delete old key




        # update bp by removing the bp entry for the merged





    
    # while len(vocab) < vocab_size:
    #     # logic to update counts after merge
    #     # get the byte pairs 
        
    #     # get byte pairs per document
    #     for pretoken_counts in document_pretoken_counts:
    #         for byte_tuple, counts in pretoken_counts:
    #             for idx, byte in enumerate(byte_tuple):

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

    # combine parallel results by unrolling into a single list of dicts
    all_docs_pretoken_counts = []
    for result in results:
        all_docs_pretoken_counts.extend(result)

    # perform bpe_merges until vocab size is reached.
    vocab, merges = bpe_merges(vocab, all_docs_pretoken_counts, vocab_size)

    # Return vocab and merges
    return tuple(vocab,merges)

# Test script 

