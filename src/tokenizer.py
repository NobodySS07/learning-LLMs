import os
import json
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer

class CustomBPETokenizer:
    def __init__(self, pre_tokenizer_name: str = "gpt2"):
        """Initializes the backend pre-tokenizer and state variables."""
        self.backend_tokenizer = AutoTokenizer.from_pretrained(pre_tokenizer_name)
        
        self.vocab: List[str] = []
        self.merges: Dict[Tuple[str, str], str] = {}
        self.special_tokens = ["<|endoftext|>"]

    def _get_word_frequencies(self, corpus: List[str]) -> Dict[str, int]:
        word_frequencies = defaultdict(int)
        for text in corpus:
            word_wt_offsets = self.backend_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in word_wt_offsets]
            for word in new_words:
                word_frequencies[word] += 1
        return word_frequencies

    def _get_alphabet(self, word_frequencies: Dict[str, int]) -> List[str]:
        alphabets = []
        for word in word_frequencies.keys():
            for letter in word:
                if letter not in alphabets:
                    alphabets.append(letter)
        return alphabets 

    def _compute_pair_frequencies(self, splits: Dict[str, List[str]], word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pair_freqs = defaultdict(int)
        for word, freqs in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freqs 
        return pair_freqs

    def _merge_pair(self, a: str, b: str, splits: Dict[str, List[str]], word_freqs: Dict[str, int]) -> Dict[str, List[str]]:
        for word in word_freqs.keys():
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < (len(split) - 1):
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split
        return splits 

    def save(self, file_path: str):
        """Saves the learned vocabulary and merges to a JSON file."""
        data = {
            "vocab": self.vocab,
            # Convert dictionary with tuple keys into a list of [char1, char2, merged_result] for JSON compatibility
            "merges": [[pair[0], pair[1], merge] for pair, merge in self.merges.items()]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer state successfully saved to {file_path}")

    def load(self, file_path: str):
        """Loads the vocabulary and merges from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.vocab = data["vocab"]
        # Reconstruct the dictionary with tuple keys
        self.merges = {(p[0], p[1]): p[2] for p in data["merges"]}
        print(f"Tokenizer state loaded from {file_path}. Vocabulary size: {len(self.vocab)}")

    def train(self, corpus: List[str], vocab_size: int = 12000, save_path: str = None):
        """Learns the BPE merges from a given text corpus and auto-saves the result."""
        if save_path is None:
            from pathlib import Path
            save_path = str(Path(__file__).parent.parent / "data" / "models" / "tokenizer_state.json")
        
        print("Extracting word frequencies and base alphabet...")
        word_freqs = self._get_word_frequencies(corpus)
        alphabets = self._get_alphabet(word_freqs)
        
        self.vocab = self.special_tokens.copy() + alphabets.copy()
        splits = {word: [c for c in word] for word in word_freqs.keys()}
        self.merges = {}

        initial_vocab_size = len(self.vocab)
        total_merges = vocab_size - initial_vocab_size

        with tqdm(total=total_merges, desc="Learning BPE Merges") as pbar:
            while len(self.vocab) < vocab_size:
                pair_freqs = self._compute_pair_frequencies(splits, word_freqs)
                
                if not pair_freqs:
                    break
                    
                best_pair = max(pair_freqs, key=pair_freqs.get)
                
                splits = self._merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
                merged_token = best_pair[0] + best_pair[1]
                
                self.merges[best_pair] = merged_token
                self.vocab.append(merged_token)
                
                pbar.update(1)

        print(f"Training complete. Final vocabulary size: {len(self.vocab)}")
        # Auto-save immediately after training concludes
        self.save(save_path)

    def load_or_train(self, file_path: str, corpus: List[str], vocab_size: int = 12000):
        """Attempts to load a saved tokenizer; if it doesn't exist, trains from scratch."""
        if os.path.exists(file_path):
            self.load(file_path)
        else:
            print(f"'{file_path}' not found. Initiating training...")
            self.train(corpus=corpus, vocab_size=vocab_size, save_path=file_path)

    def tokenize(self, text: str) -> List[str]:
        """Converts raw text into a list of subword strings based on learned merges."""
        pre_tokenize_result = self.backend_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        
        splits = [[l for l in word] for word in pre_tokenized_text]
        
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split

        return sum(splits, [])

    def _build_mapping(self):
        """Creates a fast lookup dictionary for O(1) token-to-ID conversion."""
        # Only build it if it doesn't exist or if the vocab has grown
        if not hasattr(self, "token_to_id") or len(self.token_to_id) != len(self.vocab):
            self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
            self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        """Converts raw text directly into a list of integer IDs for PyTorch."""
    
        tokens = self.tokenize(text.lower())
 
        self._build_mapping() 

        unk_id = self.token_to_id.get("<|endoftext|>", 0)

        input_ids = [self.token_to_id.get(token, unk_id) for token in tokens]
        
        return input_ids

    def decode(self, ids: List[int]) -> str:
        """Converts integer IDs back into readable text."""
        self._build_mapping()
        tokens = [self.id_to_token.get(idx, "") for idx in ids]
        return "".join(tokens).replace("Ġ", " ")