# Offline Tokenizer for gpt-4o-mini

This folder contains the complete tokenizer data for gpt-4o-mini, saved for offline use.

## Files:
- `tokenizer.pkl`: Complete tiktoken encoding object (pickle format)
- `vocabulary.json`: Full vocabulary mapping (token_id -> token_text)
- `reverse_vocabulary.json`: Reverse mapping (token_text -> token_id)
- `special_tokens.json`: Special tokens and their IDs
- `metadata.json`: Tokenizer metadata and configuration
- `offline_tokenizer.py`: Python script to load and use the tokenizer offline
- `README.md`: This file

## Usage:

### Method 1: Use the offline loader
```python
from offline_tokenizer import OfflineTokenizer

tokenizer = OfflineTokenizer()
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)
details = tokenizer.tokenize_with_details("Hello, world!")
```

### Method 2: Load the pickle directly
```python
import pickle

with open("tokenizer.pkl", 'rb') as f:
    encoding = pickle.load(f)

tokens = encoding.encode("Hello, world!")
text = encoding.decode(tokens)
```

### Method 3: Use the vocabulary JSON
```python
import json

with open("vocabulary.json", 'r') as f:
    vocab = json.load(f)
    vocab = {int(k): v for k, v in vocab.items()}  # Convert keys to int

# Now you can look up token meanings
print(vocab[9906])  # Should print 'Hello'
```

## Metadata:
- Model: gpt-4o-mini
- Encoding: cl100k_base
- Vocabulary size: 100,277 tokens
- Max token value: 100,276
- Special tokens: 5
- tiktoken version: 0.8.0
