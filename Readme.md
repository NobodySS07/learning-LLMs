# Custom BPE Tokenizer

A Python implementation of **Byte Pair Encoding (BPE)** tokenizer trained on the IMDB dataset. This project provides a lightweight, modular tokenizer that can be trained and used for NLP tasks.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Architecture](#project-architecture)
- [License](#license)

---

## ✨ Features

- **Custom BPE Implementation**: Build your own tokenizer using Byte Pair Encoding algorithm
- **Dataset Support**: Pre-configured for IMDB dataset, extensible to other datasets
- **Text Normalization**: Automatic HTML tag removal and text cleaning
- **Model Persistence**: Save and load tokenizer vocabulary and merge rules as JSON
- **Progress Tracking**: Real-time progress bar during tokenizer training
- **Modular Design**: Clean separation between core tokenizer, utilities, and notebooks

---

## 📁 Project Structure

```
LLMs/
├── src/
│   ├── __init__.py              # Module exports (CustomBPETokenizer, utilities)
│   ├── tokenizer.py             # CustomBPETokenizer class
│   └── utils.py                 # Shared utilities (normalize_review, etc.)
│
├── notebooks/
│   └── script.ipynb             # Main training and evaluation notebook
│
├── data/
│   ├── raw/                     # Raw datasets (auto-downloaded)
│   ├── processed/               # Cleaned datasets
│   └── models/
│       └── tokenizer_state.json # Saved tokenizer vocabulary & merges
│
├── config/
│   └── .env                     # Environment variables (git-ignored)
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git configuration
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

1. **Clone the repository** (if using git):
```bash
git clone <repository-url>
cd LLMs
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment variables** (optional):
```bash
cp config/.env.example config/.env
# Edit config/.env with your settings (HF_TOKEN, CUDA_PATH, etc.)
```

---

## 🎯 Quick Start

### Basic Usage in Python

```python
from src import CustomBPETokenizer
from pathlib import Path

# Initialize tokenizer
tokenizer = CustomBPETokenizer(pre_tokenizer_name="gpt2")

# Train from corpus or load existing model
model_path = Path("data/models/tokenizer_state.json")
corpus = ["your text here", "more training text"]

tokenizer.load_or_train(
    file_path=str(model_path),
    corpus=corpus,
    vocab_size=10000
)

# Tokenize text
tokens = tokenizer.tokenize("This is a test sentence.")
print(tokens)  # ['This', ' is', ' a', ' test', ' sentence', '.']
```

### Running the Full Pipeline (Notebook)

1. Open `notebooks/script.ipynb` in Jupyter:
```bash
jupyter notebook
```

2. Execute cells in order to:
   - Load IMDB dataset
   - Normalize review text
   - Train BPE tokenizer
   - Test tokenization

---

## 📖 Usage

### Training a New Tokenizer

```python
from src import CustomBPETokenizer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("stanfordnlp/imdb")
corpus = dataset["train"]["text"]

# Create and train
tokenizer = CustomBPETokenizer()
tokenizer.train(
    corpus=corpus,
    vocab_size=10000,
    save_path="data/models/my_tokenizer.json"
)
```

### Loading a Trained Tokenizer

```python
from src import CustomBPETokenizer

tokenizer = CustomBPETokenizer()
tokenizer.load("data/models/tokenizer_state.json")

# Use for tokenization
tokens = tokenizer.tokenize("Hello, world!")
print(tokens)
```

### Text Normalization

```python
from src.utils import normalize_review

example = {"text": "<br/> This is a TEST!!!"}
normalized = normalize_review(example)
print(normalized["text"])
# Output: "this is a test"
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# HuggingFace cache location (models will be downloaded here)
HF_HOME="C:\Hf"

# Optional: HuggingFace API token for private models
HF_TOKEN=hf_xxxxxxxxxxxxx

# CUDA configuration (if using GPU)
CUDA_PATH=C:\Users\pc\anaconda3\envs\drishti
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# User agent for web scraping
SCRAPER_CONTACT=your.email@example.com
```

### Tokenizer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 12000 | Size of final vocabulary |
| `pre_tokenizer_name` | "gpt2" | Backend tokenizer for pre-tokenization |
| `file_path` | "tokenizer_state.json" | Path to save/load model |

---

## 🏗️ Project Architecture

### CustomBPETokenizer Class

**Core Methods:**
- `__init__()` - Initialize with pre-tokenizer
- `train()` - Train BPE on corpus
- `load()` - Load tokenizer from disk
- `load_or_train()` - Smart loading (load if exists, else train)
- `tokenize()` - Tokenize text using learned merges
- `save()` - Save vocabulary and merges to JSON

**Private Methods:**
- `_get_word_frequencies()` - Extract word frequencies from corpus
- `_get_alphabet()` - Extract base character alphabet
- `_compute_pair_frequencies()` - Find most common adjacent pairs
- `_merge_pair()` - Merge token pairs

### Import Hierarchy

```
notebooks/script.ipynb
    ↓
from src import CustomBPETokenizer
    ↓
src/__init__.py → src/tokenizer.py (class)
              → src/utils.py (functions)
```

---

## 🔧 Common Tasks

### Update tokenizer vocabulary size
Edit `notebooks/script.ipynb` Cell 8:
```python
vocab_size = 10000  # Change this value
```

### Switch to different dataset
In `notebooks/script.ipynb` Cell 3:
```python
raw_dataset = load_dataset("wikitext", "wikitext-2")  # or any Hugging Face dataset
```

### Debug tokenization
```python
tokenizer = CustomBPETokenizer()
tokenizer.load("data/models/tokenizer_state.json")

# View vocabulary size
print(f"Vocab size: {len(tokenizer.vocab)}")

# View first 20 tokens
print(tokenizer.vocab[:20])

# Check merge rules count
print(f"Merge rules: {len(tokenizer.merges)}")
```

---

## 📊 Performance Notes

- **Training time**: ~5-15 minutes for vocab_size=10000 on standard hardware
- **Memory**: ~500MB-1GB depending on corpus size
- **Storage**: ~2-5MB for tokenizer_state.json (vocab + merges)
- **Tokenization speed**: ~1000 tokens/second

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "ModuleNotFoundError: No module named 'src'" | Ensure `sys.path` includes parent directory in notebook (Cell 1) |
| Slow dataset loading | Use `cache_dir` in `load_dataset()` to avoid re-downloading |
| Memory errors during training | Reduce `vocab_size` or use smaller corpus subset |
| FileNotFoundError for tokenizer_state.json | Ensure `data/models/` directory exists or let `load_or_train()` create it |

---

## 📝 Notes

- The tokenizer uses **GPT-2 pre-tokenizer** from Hugging Face by default
- Trained models are saved in **JSON format** for portability
- All text normalization follows the **IMDB preprocessing pipeline**
- The project uses `tqdm` for progress visualization

---

## 📚 References

- [Byte Pair Encoding (BPE) Paper](https://arxiv.org/abs/1508.07909)
- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/)
- [IMDB Dataset](https://www.imdb.com/)
- [Transformers Library](https://huggingface.co/docs/transformers/)

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 👤 Author Notes

- Project initialized with focus on modular, production-ready code
- Suitable for learning tokenization concepts
- Extensible for other datasets and applications
