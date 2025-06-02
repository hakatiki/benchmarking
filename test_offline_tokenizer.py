#!/usr/bin/env python3
"""
Test script for the offline tokenizer
"""

import pickle
import json
from pathlib import Path

def test_offline_tokenizer(model_name="gpt-4o"):
    """Test the offline tokenizer functionality."""
    print(f"🧪 Testing offline tokenizer for {model_name}")
    print("=" * 50)
    
    model_folder = Path("tokenizers") / model_name
    
    if not model_folder.exists():
        print(f"❌ Model folder not found: {model_folder}")
        return False
    
    try:
        # Load the tokenizer
        print("📁 Loading tokenizer files...")
        
        with open(model_folder / "tokenizer.pkl", 'rb') as f:
            encoding = pickle.load(f)
        print("  ✅ Loaded tokenizer.pkl")
        
        with open(model_folder / "metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print("  ✅ Loaded metadata.json")
        
        with open(model_folder / "vocabulary.json", 'r', encoding='utf-8') as f:
            vocab_raw = json.load(f)
            vocab = {int(k): v for k, v in vocab_raw.items()}
        print("  ✅ Loaded vocabulary.json")
        
        with open(model_folder / "special_tokens.json", 'r', encoding='utf-8') as f:
            special_tokens = json.load(f)
        print("  ✅ Loaded special_tokens.json")
        
        print(f"\n📊 Tokenizer Information:")
        print(f"  Model: {metadata['model_name']}")
        print(f"  Encoding: {metadata['encoding_name']}")
        print(f"  Vocabulary size: {metadata['vocab_size']:,}")
        print(f"  Max token value: {metadata['max_token_value']:,}")
        print(f"  Special tokens: {len(special_tokens['special_tokens'])}")
        
        # Test tokenization
        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "🎉 Testing emojis and special characters!",
            "This is a test of the offline tokenizer."
        ]
        
        print(f"\n🔍 Testing tokenization:")
        for i, text in enumerate(test_texts, 1):
            print(f"\n  Test {i}: '{text}'")
            
            # Encode
            tokens = encoding.encode(text)
            print(f"    Tokens: {tokens}")
            print(f"    Token count: {len(tokens)}")
            
            # Decode back
            decoded = encoding.decode(tokens)
            print(f"    Decoded: '{decoded}'")
            
            # Show individual tokens
            print(f"    Breakdown:")
            for j, token_id in enumerate(tokens):
                token_text = vocab.get(token_id, f"<UNK:{token_id}>")
                print(f"      {j+1:2d}: {token_id:5d} -> '{token_text}'")
            
            # Verify roundtrip
            if text == decoded:
                print(f"    ✅ Roundtrip successful")
            else:
                print(f"    ❌ Roundtrip failed!")
                return False
        
        # Test special tokens
        print(f"\n🔧 Testing special tokens:")
        for token_name, token_id in special_tokens['special_tokens_dict'].items():
            print(f"  {token_name}: {token_id}")
            # Try to get the token text from vocab
            token_text = vocab.get(token_id, f"<NOT_FOUND:{token_id}>")
            print(f"    -> '{token_text}'")
        
        print(f"\n✅ All tests passed! Offline tokenizer is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error testing offline tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_online(model_name="gpt-4o"):
    """Compare offline tokenizer with online tiktoken."""
    print(f"\n🔄 Comparing offline vs online tokenization for {model_name}")
    print("=" * 60)
    
    try:
        import tiktoken
        
        # Load offline tokenizer
        model_folder = Path("tokenizers") / model_name
        with open(model_folder / "tokenizer.pkl", 'rb') as f:
            offline_encoding = pickle.load(f)
        
        # Load online tokenizer
        online_encoding = tiktoken.get_encoding("cl100k_base")
        
        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming everything.",
        ]
        
        all_match = True
        for text in test_texts:
            offline_tokens = offline_encoding.encode(text)
            online_tokens = online_encoding.encode(text)
            
            match = offline_tokens == online_tokens
            status = "✅ MATCH" if match else "❌ MISMATCH"
            print(f"'{text[:30]}...': {status}")
            print(f"  Offline: {offline_tokens}")
            print(f"  Online:  {online_tokens}")
            
            if not match:
                all_match = False
        
        if all_match:
            print(f"\n🎉 Perfect! Offline tokenizer matches online behavior exactly.")
        else:
            print(f"\n⚠️ Some mismatches found between offline and online tokenizers.")
        
        return all_match
        
    except Exception as e:
        print(f"❌ Error comparing tokenizers: {e}")
        return False

def main():
    print("🚀 Offline Tokenizer Test Suite")
    print("=" * 50)
    
    # Test both models
    models = ["gpt-4o", "gpt-4o-mini"]
    
    for model in models:
        success = test_offline_tokenizer(model)
        if success:
            compare_with_online(model)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 