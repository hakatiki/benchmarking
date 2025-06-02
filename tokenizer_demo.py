import tiktoken
import os
from pathlib import Path

def create_tokenizers_folder():
    """Create tokenizers folder if it doesn't exist."""
    tokenizers_path = Path("tokenizers")
    tokenizers_path.mkdir(exist_ok=True)
    print(f"✅ Created/verified tokenizers folder: {tokenizers_path.absolute()}")
    return tokenizers_path

def get_model_tokenizer(model_name):
    """Get the tokenizer for a specific model."""
    try:
        # For GPT-4o and GPT-4o-mini, they use the cl100k_base encoding
        if model_name in ["gpt-4o", "gpt-4o-mini"]:
            encoding = tiktoken.get_encoding("cl100k_base")
            print(f"✅ Loaded tokenizer for {model_name} (cl100k_base encoding)")
            return encoding
        else:
            # Try to get model-specific encoding
            encoding = tiktoken.encoding_for_model(model_name)
            print(f"✅ Loaded tokenizer for {model_name}")
            return encoding
    except Exception as e:
        print(f"❌ Error loading tokenizer for {model_name}: {e}")
        return None

def tokenize_text(tokenizer, text, model_name):
    """Tokenize text and show results."""
    if not tokenizer:
        print(f"❌ No tokenizer available for {model_name}")
        return
    
    print(f"\n🔍 Tokenizing with {model_name}:")
    print(f"Text: '{text}'")
    
    # Encode text to tokens
    tokens = tokenizer.encode(text)
    print(f"Token count: {len(tokens)}")
    print(f"Token IDs: {tokens}")
    
    # Decode tokens back to text to verify
    decoded_text = tokenizer.decode(tokens)
    print(f"Decoded text: '{decoded_text}'")
    
    # Show individual tokens
    print("Individual tokens:")
    for i, token_id in enumerate(tokens):
        token_text = tokenizer.decode([token_id])
        print(f"  {i+1:2d}: {token_id:5d} -> '{token_text}'")
    
    return tokens

def save_tokenizer_info(tokenizers_path, model_name, tokenizer, sample_tokens):
    """Save tokenizer information to a file."""
    if not tokenizer:
        return
    
    info_file = tokenizers_path / f"{model_name}_tokenizer_info.txt"
    
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Tokenizer Information for {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Encoding: cl100k_base\n")
        f.write(f"Max token value: {tokenizer.max_token_value}\n")
        f.write(f"Special tokens: {getattr(tokenizer, 'special_tokens_set', 'N/A')}\n\n")
        
        f.write("Sample tokenization:\n")
        f.write(f"Sample text: 'Hello, world! How are you today?'\n")
        sample_text = "Hello, world! How are you today?"
        sample_tokens = tokenizer.encode(sample_text)
        f.write(f"Token count: {len(sample_tokens)}\n")
        f.write(f"Tokens: {sample_tokens}\n\n")
        
        f.write("Token breakdown:\n")
        for i, token_id in enumerate(sample_tokens):
            token_text = tokenizer.decode([token_id])
            f.write(f"  {i+1:2d}: {token_id:5d} -> '{token_text}'\n")
    
    print(f"💾 Saved tokenizer info to: {info_file}")

def main():
    """Main function to demonstrate tokenizer usage."""
    print("🚀 GPT-4o/GPT-4o-mini Tokenizer Demo")
    print("=" * 50)
    
    # Create tokenizers folder
    tokenizers_path = create_tokenizers_folder()
    
    # Models to test
    models = ["gpt-4o", "gpt-4o-mini"]
    
    # Sample texts to tokenize
    sample_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "🎉 Emojis and special characters: $100, 50% off!",
        "Miután április közepén megjelent a cikk, rengeteg levelet kaptunk.",  # Hungarian text
    ]
    
    # Process each model
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"🤖 Processing {model_name}")
        print(f"{'='*60}")
        
        # Get tokenizer
        tokenizer = get_model_tokenizer(model_name)
        
        if tokenizer:
            # Tokenize sample texts
            for i, text in enumerate(sample_texts, 1):
                print(f"\n📝 Sample {i}:")
                tokens = tokenize_text(tokenizer, text, model_name)
            
            # Save tokenizer info
            save_tokenizer_info(tokenizers_path, model_name, tokenizer, None)
            
            # Show vocabulary size
            try:
                vocab_size = tokenizer.n_vocab
                print(f"\n📊 Vocabulary size: {vocab_size:,} tokens")
            except:
                print(f"\n📊 Max token value: {tokenizer.max_token_value:,}")
    
    print(f"\n✅ Demo complete! Check the '{tokenizers_path}' folder for saved tokenizer information.")

if __name__ == "__main__":
    main() 