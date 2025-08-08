import os
import torch
import torch.nn as nn
import logging
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from local files (fixed imports)
from config import ModelConfig
from train_tokenizer import BPETokenizer, train_tokenizer
from transformer import CodeTransformer

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
        print("Using CPU (optimized for your laptop)")
    
    return device
def read_training_data(data_path):
    """Read training data from file or from python_qa_data.py"""
    try:
        from python_qa_data import PYTHON_QA_PAIRS
        # Ensure we have a list of separate tasks/functions
        if isinstance(PYTHON_QA_PAIRS, str):
            samples = PYTHON_QA_PAIRS.strip().split("\n\n")
        elif isinstance(PYTHON_QA_PAIRS, list):
            samples = PYTHON_QA_PAIRS
        else:
            samples = [str(PYTHON_QA_PAIRS)]
    except ImportError:
        samples = ["# No training data found"]

    # Save to file for persistence
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(samples))

    return samples





class CustomDataset:
    """Custom dataset for training"""
    def __init__(self, encoded_texts, max_length=128):
        self.data = []
        for text in encoded_texts:
            if len(text) > max_length:
                # Split long sequences
                for i in range(0, len(text), max_length):
                    chunk = text[i:i+max_length]
                    if len(chunk) > 5:  # Only keep reasonable chunks
                        self.data.append(chunk)
            else:
                if len(text) > 5:  # Only keep sequences longer than 5 tokens
                    self.data.append(text)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, pad_token_id=0, max_length=128):
    """Collate function for DataLoader with proper tensor handling"""
    # Filter out any None or empty sequences
    valid_batch = []
    for seq in batch:
        if seq is not None and len(seq) > 1:
            if isinstance(seq, torch.Tensor):
                valid_batch.append(seq.tolist())
            else:
                valid_batch.append(seq)
    
    if not valid_batch:
        # Return a minimal valid batch
        dummy_seq = [pad_token_id] * 10
        return torch.tensor([dummy_seq], dtype=torch.long)
    
    # Ensure all sequences are lists of integers
    clean_batch = []
    for seq in valid_batch:
        if isinstance(seq, (list, tuple)):
            # Convert to integers and limit length
            clean_seq = [int(token) for token in seq[:max_length]]
            if len(clean_seq) > 1:  # Ensure sequence has at least 2 tokens
                clean_batch.append(clean_seq)
    
    if not clean_batch:
        dummy_seq = [pad_token_id] * 10
        return torch.tensor([dummy_seq], dtype=torch.long)
    
    # Pad sequences to same length
    max_len = min(max(len(seq) for seq in clean_batch), max_length)
    if max_len < 2:
        max_len = 10
    
    padded_batch = []
    for seq in clean_batch:
        if len(seq) >= max_len:
            padded_seq = seq[:max_len]
        else:
            padded_seq = seq + [pad_token_id] * (max_len - len(seq))
        padded_batch.append(padded_seq)
    
    return torch.tensor(padded_batch, dtype=torch.long)

def save_checkpoint(model, optimizer, epoch, loss, config, checkpoint_dir):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    config = checkpoint.get('config', None)
    
    logger.info(f"Loaded checkpoint from epoch {epoch}, loss: {loss:.4f}")
    return epoch, loss, config

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    latest_path = checkpoint_dir / 'checkpoint_latest.pt'
    if latest_path.exists():
        return str(latest_path)
    
    # Find highest numbered checkpoint
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    if not checkpoints:
        return None
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return str(checkpoints[-1])

def train_model(epochs=None, resume=False, checkpoint_path=None):
    """Train the model with improved error handling"""
    logger.info("Starting training process - CPU optimized")
    
    try:
        # Setup
        device = setup_device()
        config = ModelConfig()
        
        # Override epochs if specified
        if epochs is not None:
            config.num_epochs = epochs
            logger.info(f"Training for {epochs} epochs")
        
        # Create necessary directories
        for directory in ['tokenizer', 'model', 'data', 'checkpoints']:
            Path(directory).mkdir(exist_ok=True)
        
        # Paths
        data_path = "data/train.txt"
        tokenizer_save_path = "tokenizer/python_tokenizer.pkl"
        
        # Step 1: Load/train tokenizer
        logger.info("Setting up tokenizer...")
        
        if os.path.exists(tokenizer_save_path):
            tokenizer = BPETokenizer()
            tokenizer.load(tokenizer_save_path)
            logger.info(f"Loaded existing tokenizer with vocab size: {len(tokenizer.vocab)}")
        else:
            logger.info("Training new tokenizer...")
            texts = read_training_data(data_path)
            tokenizer = train_tokenizer(data_path, tokenizer_save_path, vocab_size=config.vocab_size)
            logger.info(f"Trained new tokenizer with vocab size: {len(tokenizer.vocab)}")
        
        # Update config with actual vocab size
        config.vocab_size = len(tokenizer.vocab)
        
        # Step 2: Prepare data
        logger.info("Preparing training data...")
        texts = read_training_data(data_path)
        encoded_texts = []
        
        for i, text in enumerate(texts):
            try:
                encoded = tokenizer.encode(text)
                if encoded and len(encoded) > 5:  # Only keep sequences with reasonable length
                    encoded_texts.append(torch.tensor(encoded, dtype=torch.long))
                    logger.info(f"Encoded text {i+1}: {len(encoded)} tokens")
            except Exception as e:
                logger.warning(f"Failed to encode text {i+1}: {e}")
                continue
        
        if not encoded_texts:
            raise ValueError("No valid encoded texts found. Please check your training data.")
        
        dataset = CustomDataset(encoded_texts, max_length=config.max_seq_len)
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty after filtering. Please check your data.")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, pad_token_id=config.pad_token_id, max_length=config.max_seq_len)
        )
        
        logger.info(f"Dataset ready: {len(dataset)} samples, {len(dataloader)} batches")
        
        # Step 3: Initialize model
        model = CodeTransformer(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized with {total_params:,} parameters (CPU optimized)")
        
        # Step 4: Resume from checkpoint if requested
        start_epoch = 0
        best_loss = float('inf')
        
        if resume:
            if checkpoint_path is None:
                checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                start_epoch, best_loss, _ = load_checkpoint(checkpoint_path, model, optimizer, device)
                logger.info(f"Resuming from epoch {start_epoch}")
            else:
                logger.warning("No checkpoint found, starting from scratch")
        
        # Step 5: Training loop
        logger.info(f"Starting training from epoch {start_epoch} to {start_epoch + config.num_epochs}")
        model.train()
        
        for epoch in range(start_epoch, start_epoch + config.num_epochs):
            total_loss = 0
            num_batches = 0
            successful_batches = 0
            
            logger.info(f"Starting epoch {epoch + 1}/{start_epoch + config.num_epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    batch = batch.to(device)
                    
                    # Validate batch
                    if batch.size(0) == 0 or batch.size(1) <= 1:
                        logger.warning(f"Skipping invalid batch {batch_idx}: shape {batch.shape}")
                        continue
                    
                    # Create targets (shifted input)
                    inputs = batch[:, :-1]
                    targets = batch[:, 1:]
                    
                    # Ensure inputs and targets have the same sequence length
                    if inputs.size(1) != targets.size(1):
                        min_len = min(inputs.size(1), targets.size(1))
                        inputs = inputs[:, :min_len]
                        targets = targets[:, :min_len]
                    
                    if inputs.size(1) == 0:
                        continue
                    
                    # Forward pass
                    optimizer.zero_grad()
                    logits, loss = model(inputs, targets)
                    
                    # Validate loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Invalid loss in batch {batch_idx}: {loss}")
                        continue
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    successful_batches += 1
                    
                    # Log progress
                    if (batch_idx + 1) % 3 == 0:
                        avg_loss = total_loss / successful_batches if successful_batches > 0 else 0
                        logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}, Avg Loss: {avg_loss:.4f}")
                
                except Exception as e:
                    logger.warning(f"Error in batch {batch_idx}: {e}")
                    logger.debug(traceback.format_exc())
                    continue
                
                num_batches += 1
            
            # Calculate epoch loss
            if successful_batches > 0:
                epoch_loss = total_loss / successful_batches
                logger.info(f"Epoch {epoch + 1} completed - Loss: {epoch_loss:.4f} (from {successful_batches} successful batches)")
                
                # Save checkpoint every config.save_every epochs
                if (epoch + 1) % config.save_every == 0:
                    save_checkpoint(model, optimizer, epoch + 1, epoch_loss, config, config.checkpoint_dir)
                
                # Update best loss and save best model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    logger.info(f"New best loss: {best_loss:.4f}")
                    
                    # Save best model
                    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'tokenizer_vocab': tokenizer.vocab,
                        'config': config,
                        'loss': best_loss,
                        'vocab_size': len(tokenizer.vocab)
                    }, best_model_path)
                    
                    # Also save to model directory for generation
                    model_dir = Path('model')
                    model_dir.mkdir(exist_ok=True)
                    final_model_path = model_dir / 'python_code_model.pt'
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'best_loss': best_loss,
                        'vocab_size': len(tokenizer.vocab)
                    }, final_model_path)
            else:
                logger.warning(f"No successful batches in epoch {epoch + 1}")
        
        # Final save
        save_checkpoint(model, optimizer, start_epoch + config.num_epochs, best_loss, config, config.checkpoint_dir)
        
        logger.info(f"Training completed! Best loss: {best_loss:.4f}")
        logger.info("Model ready for generation. Run: python generate.py")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def main():
    """Main training interface with command line arguments"""
    parser = argparse.ArgumentParser(description='Train Python Code AI')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Specific checkpoint to resume from')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Python Code AI - Training (CPU Optimized)")
    print("="*60)
    print(f"Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Resume: {args.resume}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    print("="*60)
    
    try:
        train_model(epochs=args.epochs, resume=args.resume, checkpoint_path=args.checkpoint)
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("You can now run: python generate.py --interactive")
        print("="*60)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        logger.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()


