class ModelConfig:
    """Simplified model configuration with controllable training"""
    def __init__(self):
        # Model architecture - smaller for faster training
        self.vocab_size = 1000      # Smaller vocab
        self.d_model = 128          # Smaller model
        self.n_heads = 4            # Fewer heads
        self.n_layers = 3           # Fewer layers
        self.d_ff = 512             # Smaller FF
        self.max_seq_len = 256      # Shorter sequences
        self.dropout = 0.1
        
        # Special tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2
        self.bos_token_id = 3
        
        # Training parameters - CONTROLLABLE
        self.learning_rate = 0.001
        self.batch_size = 4         # Smaller batches
        self.num_epochs = 5         # DEFAULT: Only 5 epochs at a time
        self.warmup_steps = 100
        self.weight_decay = 0.01
        
        # Checkpoint settings
        self.save_every = 1         # Save after every epoch
        self.checkpoint_dir = "checkpoints"
        
        # Generation parameters
        self.max_generation_length = 100
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9