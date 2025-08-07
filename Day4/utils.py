import os
import json
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

def create_directories():
    """Create necessary project directories"""
    dirs = ['src', 'src/model', 'src/tokenizer', 'tokenizer', 'model', 'data', 'db', 'logs', 'checkpoints']
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)

def setup_logging():
    """Setup logging configuration without unicode characters"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'code_ai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def save_config(config, path: str):
    """Save model configuration to JSON file"""
    config_dict = {
        'vocab_size': config.vocab_size,
        'd_model': config.d_model,
        'n_heads': config.n_heads,
        'n_layers': config.n_layers,
        'd_ff': config.d_ff,
        'max_seq_len': config.max_seq_len,
        'dropout': config.dropout,
        'pad_token_id': config.pad_token_id,
        'eos_token_id': config.eos_token_id,
        'bos_token_id': config.bos_token_id
    }
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config(path: str):
    """Load model configuration from JSON file"""
    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    from config import ModelConfig
    config = ModelConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    return config

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_device():
    """Setup and return the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }, path)

def load_checkpoint(path, model, optimizer=None):
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('loss', 0)

def read_training_data(data_path):
    """Read training data from file or create sample data"""
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return [f.read()]
    else:
        # Create sample training data from python_qa_data.py
        from python_qa_data import PYTHON_QA_PAIRS
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Write sample data to file
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(PYTHON_QA_PAIRS)
        
        return [PYTHON_QA_PAIRS]

class CustomDataset:
    """Custom dataset for training"""
    def __init__(self, encoded_texts, max_length=512):
        self.data = []
        for text in encoded_texts:
            if len(text) > max_length:
                # Split long sequences
                for i in range(0, len(text), max_length):
                    chunk = text[i:i+max_length]
                    if len(chunk) > 10:  # Only keep reasonable chunks
                        self.data.append(chunk)
            else:
                self.data.append(text)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, pad_token_id=0, max_length=512):
    """Collate function for DataLoader"""
    # Pad sequences to same length
    max_len = min(max(len(seq) for seq in batch), max_length)
    
    padded_batch = []
    for seq in batch:
        if len(seq) >= max_len:
            padded_seq = seq[:max_len]
        else:
            padded_seq = seq + [pad_token_id] * (max_len - len(seq))
        padded_batch.append(padded_seq)
    
    return torch.tensor(padded_batch, dtype=torch.long)

def save_model(model, tokenizer, config, model_path):
    """Save model, tokenizer and config"""
    os.makedirs(model_path, exist_ok=True)
    
    # Save model
    model_file = os.path.join(model_path, 'python_code_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer.vocab)
    }, model_file)
    
    # Save tokenizer
    tokenizer_file = os.path.join(model_path, 'python_tokenizer.pkl')
    tokenizer.save(tokenizer_file)
    
    # Save config
    config_file = os.path.join(model_path, 'config.json')
    save_config(config, config_file)
    
    print(f"Model saved to {model_path}")