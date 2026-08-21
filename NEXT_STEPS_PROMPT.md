# Prompt: Continue Building the IMDB Sentiment Language Model

You are helping me continue a small language-model project. First inspect the entire repository and use the existing code and notebook as the source of truth. Do not replace the custom tokenizer with a pretrained sentiment model unless you clearly explain why and receive permission.

## Project summary

The repository is a Python project at `c:\Users\pc\Desktop\LLMs` with this structure:

```text
src/
  __init__.py
  tokenizer.py
  utils.py
notebooks/
  script.ipynb
data/models/
  tokenizer_state.json
config/.env
requirements.txt
Readme.md
```

The project goal is to train a small language model for binary sentiment analysis on the IMDB dataset (`stanfordnlp/imdb`). The current work has focused on implementing and testing a custom BPE-style tokenizer.

## What has already been done

1. `src/tokenizer.py` contains `CustomBPETokenizer`.
2. The tokenizer uses the GPT-2 backend pre-tokenizer from `transformers` to split text before learning custom character-level merges.
3. It builds word frequencies, an alphabet, pair frequencies, and merges the most frequent adjacent pair until the target vocabulary size is reached.
4. It can save and load its vocabulary and merge rules as JSON.
5. It supports `load_or_train`, `tokenize`, `encode`, and `decode`.
6. The default tokenizer vocabulary size is 12,000; the notebook trains or loads it with vocabulary size 10,000.
7. `src/utils.py` contains `normalize_review`, which removes HTML tags, lowercases text, keeps common punctuation, and collapses whitespace.
8. `src/__init__.py` exports `CustomBPETokenizer`, `normalize_review`, and `get_word_frequencies`.
9. `notebooks/script.ipynb` currently:
   - imports the local source package;
   - loads `stanfordnlp/imdb` with `load_dataset`;
   - splits the original training set into 80% train and 20% validation using `test_size=0.2` and `seed=42`;
   - keeps the official IMDB test split as test data;
   - normalizes all three splits with `num_proc=4`;
   - builds the tokenizer corpus from the normalized training reviews only;
   - trains or loads `data/models/tokenizer_state.json` with `vocab_size=10000`;
   - encodes and decodes one sample review to inspect the result.
10. The notebook cells have not been executed in the current saved notebook state. There is no neural network, embedding layer, attention implementation, training loop, checkpoint, or sentiment evaluation pipeline yet.

## Your task

Act as a technical mentor and give me a sequential roadmap for completing this project myself. Inspect the entire repository first, then tell me exactly what I should learn, decide, build, run, and verify at each stage, starting from the tokenizer and ending with a trained sentiment model.

Do not write code. Do not edit or create files. Do not implement anything for me. Do not give me a large code example or a finished notebook. I want to perform the work myself and use your response as a checklist and learning guide.

Explain the purpose, dependencies, decisions, and verification criteria for each of the following stages:

1. **Tokenizer contract and data preparation**
   - Verify that the custom tokenizer can represent all required inputs.
   - Identify and fix issues that would make training unreliable, especially unknown-token handling, padding, truncation, special tokens, deterministic vocabulary mappings, and encode/decode behavior.
   - Avoid data leakage: fit the tokenizer and any statistics using training data only.
   - Tell me when and how I should create reproducible train, validation, and test data loaders or equivalent batching code.

2. **Token IDs and embeddings**
   - Explain how token IDs become vectors through a learned `nn.Embedding` layer.
   - Tell me how to choose a small, justified model configuration that fits a normal laptop or modest GPU.
   - Tell me how to decide sequence-length handling, padding masks, and an appropriate way to represent a review for classification.

3. **Self-attention and Transformer blocks**
   - Explain query, key, and value projections, scaled dot-product attention, multi-head attention, causal versus bidirectional attention, positional information, residual connections, layer normalization, and the feed-forward network.
   - Choose the correct attention behavior for sentiment classification. Explain whether the model is an encoder-style bidirectional classifier or a causal language model with a classification head, and why.
   - Prefer PyTorch's well-tested primitives where appropriate, and explain the tensor shapes I should expect at each point.

4. **Sentiment model**
   - Build a small Transformer-based classifier using the custom tokenizer.
   - Use the IMDB labels (`0` and `1`) and an appropriate classification loss.
   - Handle padding correctly so padded positions do not affect attention or pooling.
   - Explain how to keep the eventual code modular and consistent with the existing `src/` layout.

5. **Training and evaluation**
   - Build a reproducible training loop with optimizer, learning-rate choice or scheduler, gradient handling, validation, checkpoint saving, and early stopping or a clear stopping rule.
   - Report training loss and validation loss/accuracy after each epoch.
   - Evaluate once on the untouched test set and report accuracy, precision, recall, F1, and a confusion matrix.
   - Add inference for new review text.
   - Save enough configuration with the checkpoint to reload the model safely.

6. **Notebook and tests**
   - Decide whether to extend `notebooks/script.ipynb` or create focused Python modules plus notebook cells that run in order.
   - Plan lightweight tests for tokenizer round trips, padding and truncation, attention masks, tensor dimensions, one training step, and inference.
   - Make the first run practical on CPU by exposing small-data/debug settings, while retaining a full-dataset configuration.

## Important engineering requirements

- Use the current repository rather than inventing a disconnected example.
- Do not modify any repository files.
- Use ASCII-compatible source where possible and preserve the existing style.
- Point out risks and likely failure modes instead of hiding them with implementation details.
- Do not silently download or use a pretrained sentiment classifier.
- Do not claim results until I have run the relevant experiments.
- Give me commands or actions to run only when they are necessary as checkpoints; do not run them yourself.
- Clearly separate concepts I need to understand from tasks I need to perform.
- If the custom tokenizer has a design flaw that could prevent reliable model training, explain the problem, why it matters, and what I should investigate or change myself.

## Required response format

Return only a practical, ordered learning and execution plan:

1. A concise assessment of the current repository and what is missing.
2. A numbered sequence of steps in dependency order. For every step include:
   - the goal;
   - what concept I need to understand first;
   - the task I should perform myself;
   - what should exist or be true when the step is complete;
   - a small check that can confirm or disprove my work;
   - what could go wrong and how I should diagnose it.
3. The recommended architecture and the decisions I need to make, without implementing it.
4. Suggested milestones from tokenizer readiness, to embeddings, to attention, to the model, to training, to evaluation and inference.
5. A final checklist of prerequisites and completion criteria.
6. A short explanation of the full data flow:

```text
raw IMDB review -> normalization -> custom tokenizer -> token IDs
-> embeddings + positional information -> Transformer self-attention blocks
-> masked pooling or classification token -> logits -> sentiment prediction
```

Start by inspecting the repository and confirming which of these statements are true. Then give me the sequence only. Do not implement, edit, or generate the project files for me.