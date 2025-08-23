import cProfile
import pstats
import io

from .tx_train_bpe import train_bpe, save_vocab_and_merges


def main():
    # directories
    input_path = "data/owt_train.txt"
    save_dir = "results/owt-train-results"

    # initialize profiling
    pr = cProfile.Profile()
    pr.enable()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    pr.disable()

    # Save vocab and merges to results folder
    save_vocab_and_merges(vocab, merges, save_dir)

    # get the longest token in vocabulary by byte length
    _, longest_token_by_bytes = max(vocab.items(), key=lambda x: len(x[1]))
    
    # get the longest token in vocabulary by character length (only valid UTF-8 tokens)
    def safe_char_length(token_bytes):
        try:
            return len(token_bytes.decode('utf-8'))
        except UnicodeDecodeError:
            return 0  # Skip invalid UTF-8 tokens
    
    _, longest_token_by_chars = max(vocab.items(), key=lambda x: safe_char_length(x[1]))
    
    print("\nBPE Training on TinyStories:")
    print(f"(i) Longest token by byte length: '{longest_token_by_bytes.decode('utf-8', errors='replace')}' ({len(longest_token_by_bytes)} bytes)")
    print(f"(ii) Longest token by character length: '{longest_token_by_chars.decode('utf-8', errors='replace')}' ({safe_char_length(longest_token_by_chars)} characters)")

    # Print top 20 functions by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    main()
