# Move these functions and utilities from libraries.py
import re
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer

def normalize_review(example):
    """Normalize IMDB review text"""
    text = example["text"]
    text = re.sub(r"<br\s*/>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    example["text"] = text
    return example

def get_word_frequencies(corpus: List[str], backend_tokenizer) -> Dict[str, int]:
    """Extract word frequencies from corpus"""
    word_frequencies = defaultdict(int)
    for text in corpus:
        word_wt_offsets = backend_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in word_wt_offsets]
        for word in new_words:
            word_frequencies[word] += 1
    return word_frequencies