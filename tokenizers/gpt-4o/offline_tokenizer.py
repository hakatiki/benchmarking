#!/usr/bin/env python3
"""
Offline tokenizer loader for gpt-4o
Generated automatically - loads the saved tokenizer data for offline use.
"""

import pickle
import json
from pathlib import Path

class OfflineTokenizer:
    def __init__(self, tokenizer_folder="gpt-4o"):
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
            self.vocab = {int(k): v for k, v in self.vocab.items()}
        
        print(f"🚀 Loaded offline tokenizer for {self.metadata['model_name']}")
        print(f"   Vocabulary size: {self.metadata['vocab_size']:,} tokens")
    
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
        return self.vocab.get(token_id, f"<UNK:100260>")
    
    def tokenize_with_details(self, text):
        """Tokenize text and return detailed information."""
        tokens = self.encode(text)
        details = []
        for i, token_id in enumerate(tokens):
            details.append({
                'position': i,
                'token_id': token_id,
                'token_text': self.get_token_text(token_id)
            })
        return {
            'text': text,
            'tokens': tokens,
            'token_count': len(tokens),
            'details': details
        }

# Example usage:
if __name__ == "__main__":
    tokenizer = OfflineTokenizer()
    
    # Test tokenization
    test_text = "Hello, world! This is a test."
    result = tokenizer.tokenize_with_details(test_text)
    
    print(f"\nTest text: '{test_text}'")
    print(f"Token count: {result['token_count']}")
    print(f"Tokens: {result['tokens']}")
    print("\nToken breakdown:")
    for detail in result['details']:
        print(f"  {detail['position']+1:2d}: {detail['token_id']:5d} -> '{detail['token_text']}'")
