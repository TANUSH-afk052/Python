#!/usr/bin/env python3
"""
Vocabulary Size Checker
Check and verify the vocabulary size of your trained tokenizer and model.
"""

import os
import sys
from pathlib import Path

def check_tokenizer_vocab():
    """Check tokenizer vocabulary size"""
    tokenizer_path = "tokenizer/python_tokenizer.pkl"
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found at: {tokenizer_path}")
        return None
    
    try:
        from train_tokenizer import BPETokenizer
        tokenizer = BPETokenizer()
        tokenizer.load(tokenizer_path)
        
        vocab_size = len(tokenizer.vocab)
        print(f"✅ Tokenizer vocabulary size: {vocab_size}")
        return vocab_size
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return None

def check_config_vocab():
    """Check config vocabulary size"""
    try:
        from config import ModelConfig
        config = ModelConfig()
        print(f"📋 Config target vocabulary size: {config.vocab_size}")
        return config.vocab_size
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def check_model_checkpoints():
    """Check model checkpoint vocabulary sizes"""
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        print("❌ No checkpoints directory found")
        return
    
    # Check latest checkpoint
    latest_checkpoint = checkpoint_dir / "checkpoint_latest.pt"
    if latest_checkpoint.exists():
        try:
            import torch
            checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
            config = checkpoint.get('config')
            if config and hasattr(config, 'vocab_size'):
                print(f"📦 Latest checkpoint vocab size: {config.vocab_size}")
            else:
                print("⚠️  Latest checkpoint has no vocab size info")
        except Exception as e:
            print(f"❌ Error reading latest checkpoint: {e}")
    
    # Check final epoch checkpoint
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoints:
        # Get the highest numbered checkpoint
        latest_epoch_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        try:
            import torch
            checkpoint = torch.load(latest_epoch_checkpoint, map_location='cpu', weights_only=False)
            config = checkpoint.get('config')
            if config and hasattr(config, 'vocab_size'):
                print(f"📦 {latest_epoch_checkpoint.name} vocab size: {config.vocab_size}")
            else:
                print(f"⚠️  {latest_epoch_checkpoint.name} has no vocab size info")
        except Exception as e:
            print(f"❌ Error reading {latest_epoch_checkpoint.name}: {e}")

def check_generation_compatibility():
    """Check if generation script will work with current vocab size"""
    try:
        # Try to load model and tokenizer like generate.py does
        model_path = "checkpoints/checkpoint_epoch_10.pt"
        tokenizer_path = "tokenizer/python_tokenizer.pkl"
        
        if not os.path.exists(model_path):
            print("⚠️  Default model checkpoint not found for generation")
            # Find any epoch checkpoint
            checkpoint_dir = Path("checkpoints")
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
                if checkpoints:
                    model_path = str(max(checkpoints, key=lambda x: int(x.stem.split('_')[-1])))
                    print(f"📦 Found checkpoint: {Path(model_path).name}")
                else:
                    print("❌ No epoch checkpoints found")
                    return False
        
        if not os.path.exists(tokenizer_path):
            print(f"❌ Tokenizer not found at: {tokenizer_path}")
            return False
        
        # Load and check compatibility
        import torch
        from train_tokenizer import BPETokenizer
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        tokenizer = BPETokenizer()
        tokenizer.load(tokenizer_path)
        
        model_vocab_size = None
        config = checkpoint.get('config')
        if config and hasattr(config, 'vocab_size'):
            model_vocab_size = config.vocab_size
        
        tokenizer_vocab_size = len(tokenizer.vocab)
        
        if model_vocab_size and model_vocab_size == tokenizer_vocab_size:
            print(f"✅ Model and tokenizer vocab sizes match: {model_vocab_size}")
            return True
        elif model_vocab_size:
            print(f"❌ Vocab size mismatch! Model: {model_vocab_size}, Tokenizer: {tokenizer_vocab_size}")
            return False
        else:
            print(f"⚠️  Cannot determine model vocab size, tokenizer has: {tokenizer_vocab_size}")
            return None
            
    except Exception as e:
        print(f"❌ Error checking generation compatibility: {e}")
        return False

def main():
    print("="*60)
    print("🔍 VOCABULARY SIZE CHECKER")
    print("="*60)
    
    # Check config
    print("\n1. Configuration:")
    config_vocab_size = check_config_vocab()
    
    # Check tokenizer
    print("\n2. Tokenizer:")
    tokenizer_vocab_size = check_tokenizer_vocab()
    
    # Check model checkpoints
    print("\n3. Model Checkpoints:")
    check_model_checkpoints()
    
    # Check generation compatibility
    print("\n4. Generation Compatibility:")
    compat = check_generation_compatibility()
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY:")
    if config_vocab_size and tokenizer_vocab_size:
        if config_vocab_size == tokenizer_vocab_size:
            print(f"✅ Everything looks good! Vocab size: {config_vocab_size}")
        else:
            print(f"❌ MISMATCH DETECTED:")
            print(f"   Config wants: {config_vocab_size}")
            print(f"   Tokenizer has: {tokenizer_vocab_size}")
            print("\n💡 SOLUTION:")
            print("   Run: python train.py --retrain-tokenizer")
            print("   This will force retrain the tokenizer to match config.")
    
    if compat is False:
        print("\n🚨 GENERATION WILL FAIL!")
        print("   Model and tokenizer vocab sizes don't match.")
        print("   You need to retrain with: python train.py --retrain-tokenizer")
    elif compat is True:
        print("\n🎉 GENERATION SHOULD WORK!")
        print("   Run: python generate.py --interactive")
    
    print("="*60)

if __name__ == "__main__":
    main()