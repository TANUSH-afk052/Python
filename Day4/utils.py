import os
import json
import torch

def create_directories():
    dirs = ['tokenizer', 'model', 'data', 'db']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

def save_config(config, path):
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

def load_config(path):
    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    from model.config import ModelConfig
    config = ModelConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    return config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

if __name__ == "__main__":
    create_directories()
    print("Project structure created!")