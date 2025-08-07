class ModelConfig:
    """CPU-optimized model configuration for laptop training"""
    def __init__(self):
        # Model architecture - optimized for CPU training
        self.vocab_size = 500           # Reduced for faster training
        self.d_model = 64               # Smaller model dimension
        self.n_heads = 2                # Fewer attention heads
        self.n_layers = 2               # Fewer transformer layers
        self.d_ff = 256                 # Smaller feed-forward dimension
        self.max_seq_len = 128          # Shorter sequences
        self.dropout = 0.1
        
        # Special tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2
        self.bos_token_id = 3
        
        # Training parameters - CPU optimized
        self.learning_rate = 0.001
        self.batch_size = 2             # Smaller batches for CPU
        self.num_epochs = 10            # Default 10 epochs as requested
        self.warmup_steps = 50          # Reduced warmup
        self.weight_decay = 0.01
        
        # Checkpoint settings
        self.save_every = 2             # Save every 2 epochs
        self.checkpoint_dir = "checkpoints"
        
        # Generation parameters
        self.max_generation_length = 100
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9