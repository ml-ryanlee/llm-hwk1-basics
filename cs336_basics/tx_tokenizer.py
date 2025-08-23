import pickle
import json
import regex as re
from typing import Iterable, Iterator, Union
from collections import defaultdict, Counter

from .tx_train_bpe import apply_merge

class Tokenizer: 

    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab = vocab
        self.vocab_length = len(vocab.items())
        self.vocab2id = {v:k for k,v in vocab.items()}
        self.special_tokens = special_tokens
        if special_tokens: 
            self.special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
        self.merges = merges
        self.pretoken2ids = defaultdict(list)

    @classmethod #used to create instance of class
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab,merges,special_tokens) 

    def encode(self, text: str) -> list[int]:
        
        # get pretokens
        pretoken_list = self.encode_pretokenize(text)
        
        # use vocab to get IDs in a particular order
        token_list = self.get_token_ids(pretoken_list)
        
        return token_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            if not text:
                continue
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        byte_parts = []

        # build the byte seq from the IDs
        for token_id in ids:
            token_bytes = self.vocab[token_id]
            byte_parts.append(token_bytes)
        
        # join byte list to byte s
        byte_seq = b"".join(byte_parts)
        decoded_str = byte_seq.decode("utf-8",errors="replace")

        return decoded_str

    """ Helper Functions """
    def encode_pretokenize(self, text: str)->list[Union[list[bytes],bytes]]:
        
        def pretokenize_text(text_segment: str):
            token_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

            pretoken_bytes_list=[] # list of pretokens, each pretoken is a list of bytes
            for match in token_pattern.finditer(text_segment):
                pretoken = match.group() # pre-token string
                pretoken_bytes = pretoken.encode("utf-8")

                curr_pretoken_bytes_list = [bytes([b]) for b in pretoken_bytes] # a pretoken represented as a list of bytes
                pretoken_bytes_list.append(curr_pretoken_bytes_list)

            return pretoken_bytes_list

        # result list
        pretokens = []

        if self.special_tokens:
            escaped_special_tokens = [re.escape(token) for token in self.special_tokens]
            # sort largest to smallest so we find longest special tokens first to catch overlaps
            escaped_special_tokens = sorted(escaped_special_tokens,key=len,reverse=True)
            special_token_pattern = re.compile(f"({'|'.join(escaped_special_tokens)})")

            last_end = 0
            for match in special_token_pattern.finditer(text):
                # get normal list of pretokens
                if match.start()> last_end:
                    normal_segment = text[last_end:match.start()]
                    pretoken_bytes_part = pretokenize_text(normal_segment)
                    pretokens.extend(pretoken_bytes_part)
                    
                # add special token at end of sequence of pretokens
                special_token = match.group().encode("utf-8")
                pretokens.append(special_token)

                # update last end for next special token
                last_end = match.end()

            # finish up any text remaining after last special token match
            if last_end < len(text):
                remaining_text = text[last_end:]
                pretoken_bytes_part = pretokenize_text(remaining_text)
                pretokens.extend(pretoken_bytes_part)

        else:
            pretokens = pretokenize_text(text)

        return pretokens

    def find_soonest_merge(self, byte_seq)->tuple[bytes,bytes]:
        # set lowest idx to beyond list idx
        lowest_idx = len(self.merges)
        soonest_bp = None
        
        for idx in range(1,len(byte_seq),1):
            bp = (byte_seq[idx-1], byte_seq[idx])
            if bp in self.merges:
                bp_index = self.merges.index(bp)
                if bp_index < lowest_idx:
                    lowest_idx = bp_index
                    soonest_bp = bp
            else:
                continue
        

        return soonest_bp
    
    def get_token_ids(self, pretokens: list[Union[list[bytes]],bytes]) -> list[int]:
        token_ids = []

        for pretoken in pretokens:
            # Check if this is a special token
            if self.special_tokens and pretoken in self.special_tokens_bytes:
                token_ids.append(self.vocab2id[pretoken])
            else:
                if tuple(pretoken) in self.pretoken2ids:
                    curr_token_ids = self.pretoken2ids[tuple(pretoken)] # list of ints
                    token_ids.extend(curr_token_ids)
                else:
                    # Regular token processing
                    soonest_bp_merge = self.find_soonest_merge(pretoken)
                    bp_seq = pretoken.copy()

                    while(soonest_bp_merge):
                        # print(f"\nDEBUG:\nSoonest_bp_merge: {soonest_bp_merge}")
                        bp_seq = apply_merge(bp_seq,soonest_bp_merge)
                        # print(f"\nDEBUG:\nBP seq in while loop: {bp_seq} ")
                        soonest_bp_merge = self.find_soonest_merge(bp_seq)
                    
                    curr_token_ids = []
                    for byte_object in bp_seq:
                        curr_token_ids.append(self.vocab2id[byte_object])
                        token_ids.append(self.vocab2id[byte_object])

                    # cache answer for future pretokens
                    self.pretoken2ids[tuple(pretoken)] = curr_token_ids
        
        return token_ids

