import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        words = set()
        for txt in texts:
            words.update(txt.split())

        for i, w in enumerate(sorted(words), start=4):
            self.word_to_id[w] = i

        for word, ids in self.word_to_id.items():
            self.id_to_word[ids] = word
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        enc = []
        for t in text.split():
            if t in self.word_to_id:
                enc.append(self.word_to_id[t])
            else:
                enc.append(self.word_to_id[self.unk_token])
        return enc

    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        dec = []
        for i in ids:
            if i in self.id_to_word:
                dec.append(self.id_to_word[i])
        return " ".join(dec)
