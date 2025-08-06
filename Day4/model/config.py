class ModelConfig:
    def __init__(self):
        self.vocab_size = 1000
        self.d_model = 128
        self.n_heads = 4
        self.n_layers = 4
        self.d_ff = 512
        self.max_seq_len = 256
        self.dropout = 0.1
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 3
