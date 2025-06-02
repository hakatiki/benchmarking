import tiktoken
import json
import pickle
from pathlib import Path
import base64

def create_tokenizers_folder():
    """Create tokenizers folder if it doesn't exist."""
    tokenizers_path = Path("tokenizers")
    tokenizers_path.mkdir(exist_ok=True)
    print(f"✅ Created/verified tokenizers folder: {tokenizers_path.absolute()}")
    return tokenizers_path

def save_complete_tokenizer(model_name, tokenizers_path):
    """Save complete tokenizer data for offline use."""
    print(f"\n🔄 Saving complete tokenizer data for {model_name}...")
    
    try:
        # Get the tokenizer
        if model_name in ["gpt-4o", "gpt-4o-mini"]:
            encoding = tiktoken.get_encoding("cl100k_base")
            encoding_name = "cl100k_base"
        else:
            encoding = tiktoken.encoding_for_model(model_name)
            encoding_name = f"{model_name}_encoding"
        
        model_folder = tokenizers_path / model_name
        model_folder.mkdir(exist_ok=True)
        
        # 1. Save the encoding object itself (pickle)
        pickle_file = model_folder / "tokenizer.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(encoding, f)
        print(f"  ✅ Saved tokenizer pickle: {pickle_file}")
        
        # 2. Extract and save vocabulary (token_id -> token_text mapping)
        print("  🔄 Extracting vocabulary...")
        vocab = {}
        max_token = encoding.max_token_value
        
        # This might take a while for large vocabularies
        print(f"  📊 Processing {max_token + 1:,} tokens...")
        for token_id in range(max_token + 1):
            try:
                token_bytes = encoding.decode_single_token_bytes(token_id)
                # Try to decode as UTF-8, fallback to base64 for non-UTF-8 bytes
                try:
                    token_text = token_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    token_text = f"<BYTES:{base64.b64encode(token_bytes).decode('ascii')}>"
                vocab[token_id] = token_text
            except Exception as e:
                vocab[token_id] = f"<ERROR:{str(e)}>"
            
            # Progress indicator
            if (token_id + 1) % 10000 == 0:
                print(f"    Progress: {token_id + 1:,}/{max_token + 1:,} tokens processed")
        
        # Save vocabulary as JSON
        vocab_file = model_folder / "vocabulary.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Saved vocabulary: {vocab_file}")
        
        # 3. Save reverse vocabulary (token_text -> token_id mapping)
        reverse_vocab = {v: k for k, v in vocab.items() if not v.startswith('<')}
        reverse_vocab_file = model_folder / "reverse_vocabulary.json"
        with open(reverse_vocab_file, 'w', encoding='utf-8') as f:
            json.dump(reverse_vocab, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Saved reverse vocabulary: {reverse_vocab_file}")
        
        # 4. Save special tokens
        special_tokens = getattr(encoding, 'special_tokens_set', set())
        special_tokens_data = {
            'special_tokens': list(special_tokens),
            'special_tokens_dict': {}
        }
        
        # Try to get special token IDs
        for token in special_tokens:
            try:
                token_id = encoding.encode(token, allowed_special={token})[0]
                special_tokens_data['special_tokens_dict'][token] = token_id
            except:
                pass
        
        special_tokens_file = model_folder / "special_tokens.json"
        with open(special_tokens_file, 'w', encoding='utf-8') as f:
            json.dump(special_tokens_data, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Saved special tokens: {special_tokens_file}")
        
        # 5. Save tokenizer metadata
        metadata = {
            'model_name': model_name,
            'encoding_name': encoding_name,
            'max_token_value': encoding.max_token_value,
            'vocab_size': len(vocab),
            'special_tokens_count': len(special_tokens),
            'tiktoken_version': tiktoken.__version__
        }
        
        metadata_file = model_folder / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✅ Saved metadata: {metadata_file}")
        
        # 6. Create a simple offline loader script
        loader_script = f'''#!/usr/bin/env python3
"""
Offline tokenizer loader for {model_name}
Generated automatically - loads the saved tokenizer data for offline use.
"""

import pickle
import json
from pathlib import Path

class OfflineTokenizer:
    def __init__(self, tokenizer_folder="{model_name}"):
        self.folder = Path(tokenizer_folder)
        
        # Load the tokenizer
        with open(self.folder / "tokenizer.pkl", 'rb') as f:
            self.encoding = pickle.load(f)
        
        # Load metadata
        with open(self.folder / "metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load vocabulary
        with open(self.folder / "vocabulary.json", 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            # Convert string keys back to integers
            self.vocab = {{int(k): v for k, v in self.vocab.items()}}
        
        print(f"🚀 Loaded offline tokenizer for {{self.metadata['model_name']}}")
        print(f"   Vocabulary size: {{self.metadata['vocab_size']:,}} tokens")
    
    def encode(self, text, **kwargs):
        """Encode text to tokens."""
        return self.encoding.encode(text, **kwargs)
    
    def decode(self, tokens):
        """Decode tokens to text."""
        return self.encoding.decode(tokens)
    
    def decode_single_token_bytes(self, token_id):
        """Get the raw bytes for a single token."""
        return self.encoding.decode_single_token_bytes(token_id)
    
    def get_token_text(self, token_id):
        """Get the text representation of a token."""
        return self.vocab.get(token_id, f"<UNK:{token_id}>")
    
    def tokenize_with_details(self, text):
        """Tokenize text and return detailed information."""
        tokens = self.encode(text)
        details = []
        for i, token_id in enumerate(tokens):
            details.append({{
                'position': i,
                'token_id': token_id,
                'token_text': self.get_token_text(token_id)
            }})
        return {{
            'text': text,
            'tokens': tokens,
            'token_count': len(tokens),
            'details': details
        }}

# Example usage:
if __name__ == "__main__":
    tokenizer = OfflineTokenizer()
    
    # Test tokenization
    test_text = "Hello, world! This is a test."
    result = tokenizer.tokenize_with_details(test_text)
    
    print(f"\\nTest text: '{{test_text}}'")
    print(f"Token count: {{result['token_count']}}")
    print(f"Tokens: {{result['tokens']}}")
    print("\\nToken breakdown:")
    for detail in result['details']:
        print(f"  {{detail['position']+1:2d}}: {{detail['token_id']:5d}} -> '{{detail['token_text']}}'")
'''
        
        loader_file = model_folder / "offline_tokenizer.py"
        with open(loader_file, 'w', encoding='utf-8') as f:
            f.write(loader_script)
        print(f"  ✅ Created offline loader: {loader_file}")
        
        # 7. Create a README
        readme_content = f'''# Offline Tokenizer for {model_name}

This folder contains the complete tokenizer data for {model_name}, saved for offline use.

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
    vocab = {{int(k): v for k, v in vocab.items()}}  # Convert keys to int

# Now you can look up token meanings
print(vocab[9906])  # Should print 'Hello'
```

## Metadata:
- Model: {model_name}
- Encoding: {encoding_name}
- Vocabulary size: {len(vocab):,} tokens
- Max token value: {encoding.max_token_value:,}
- Special tokens: {len(special_tokens)}
- tiktoken version: {tiktoken.__version__}
'''
        
        readme_file = model_folder / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"  ✅ Created README: {readme_file}")
        
        print(f"  🎉 Complete tokenizer data saved for {model_name}!")
        print(f"     Total files: {len(list(model_folder.iterdir()))}")
        print(f"     Folder size: ~{sum(f.stat().st_size for f in model_folder.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error saving tokenizer for {model_name}: {e}")
        return False

def main():
    print("💾 Complete Tokenizer Offline Saver")
    print("=" * 50)
    print("This will save the complete tokenizer data for offline use.")
    print("⚠️  Warning: This will take several minutes and use significant disk space!")
    
    # Create tokenizers folder
    tokenizers_path = create_tokenizers_folder()
    
    # Models to save
    models = ["gpt-4o", "gpt-4o-mini"]
    
    success_count = 0
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"💾 Saving {model_name} tokenizer data...")
        print(f"{'='*60}")
        
        if save_complete_tokenizer(model_name, tokenizers_path):
            success_count += 1
    
    print(f"\n✅ Completed! Successfully saved {success_count}/{len(models)} tokenizers.")
    print(f"📁 Check the '{tokenizers_path}' folder for your offline tokenizer data.")
    print("\n💡 To use offline:")
    print("   cd tokenizers/gpt-4o")
    print("   python offline_tokenizer.py")

if __name__ == "__main__":
    main() 