import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from tokenizer.train_tokenizer import BPETokenizer, train_tokenizer
from model.transformer import CodeTransformer
from model.config import ModelConfig
from data.qa_data import QA_PAIRS

class CodeDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length - 1:
                tokens = tokens[:max_length - 1]
            tokens.append(tokenizer.token_to_id['<eos>'])
            
            while len(tokens) < max_length:
                tokens.append(tokenizer.token_to_id['<pad>'])
            
            self.data.append(tokens)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = torch.tensor(self.data[idx], dtype=torch.long)
        return tokens[:-1], tokens[1:]

def load_data(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    qa_pairs = content.split('\n\n')
    return [pair.strip() for pair in qa_pairs if pair.strip()]

def train_model():
    data_file = 'data/code_questions.txt'
    
    # Use imported Q&A data from separate Python file
    qa_data = QA_PAIRS
    
    os.makedirs('data', exist_ok=True)
    with open(data_file, 'w') as f:
        f.write(qa_data)
    
    # Train tokenizer
    os.makedirs('tokenizer', exist_ok=True)
    tokenizer = train_tokenizer(data_file, 'tokenizer/tokenizer.pkl', vocab_size=1000)
    
    # Load data
    texts = load_data(data_file)
    
    # Create dataset
    dataset = CodeDataset(texts, tokenizer, max_length=256)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model
    config = ModelConfig()
    config.vocab_size = len(tokenizer.vocab)
    model = CodeTransformer(config)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Training loop
    model.train()
    for epoch in range(50):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/50')
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            optimizer.zero_grad()
            
            logits, loss = model(input_ids, target_ids)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    
    # Save model
    os.makedirs('model', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, 'model/trained_model.pt')
    
    print("Training completed and model saved!")

if __name__ == "__main__":
    train_model()