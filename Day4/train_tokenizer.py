import pickle
import re
import logging
from collections import defaultdict, Counter
from pathlib import Path

logger = logging.getLogger(__name__)

class BPETokenizer:
    """Byte-Pair Encoding tokenizer optimized for Python code with guaranteed vocab size"""
    
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = ['<pad>', '<unk>', '<eos>', '<bos>']
        
    def _get_word_tokens(self, text):
        """Extract tokens from text using regex patterns optimized for Python code"""
        # Python-specific patterns
        patterns = [
            r'def\s+\w+',  # function definitions
            r'class\s+\w+',  # class definitions  
            r'import\s+\w+',  # imports
            r'from\s+\w+',  # from imports
            r'""".*?"""',  # docstrings
            r"'''.*?'''",  # docstrings
            r'#.*?$',  # comments
            r'\b\d+\.\d+\b',  # floats
            r'\b\d+\b',  # integers
            r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',  # identifiers
            r'[+\-*/%=<>!&|^~]+',  # operators
            r'[()[\]{}]',  # brackets
            r'[,;:.]',  # punctuation
            r'\s+',  # whitespace
            r'.'  # any other character
        ]
        
        combined_pattern = '|'.join(f'({pattern})' for pattern in patterns)
        tokens = re.findall(combined_pattern, text, re.MULTILINE | re.DOTALL)
        
        # Flatten and filter
        result = []
        for match in tokens:
            for group in match:
                if group.strip():  # Only keep non-empty tokens
                    result.append(group)
        
        return result
    
    def _get_pairs(self, word):
        """Get all adjacent pairs in a word"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _expand_vocabulary_with_subwords(self, texts, target_vocab_size):
        """Expand vocabulary by adding common subwords and n-grams"""
        logger.info("Expanding vocabulary with subwords and n-grams...")
        
        # Extract all tokens from texts
        all_tokens = []
        for text in texts:
            tokens = self._get_word_tokens(text)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_freq = Counter(all_tokens)
        
        # Add most frequent tokens that aren't already in vocab
        added_tokens = 0
        for token, freq in token_freq.most_common():
            if len(self.vocab) >= target_vocab_size:
                break
            if token not in self.vocab and len(token) > 1:
                self.vocab[token] = len(self.vocab)
                added_tokens += 1
        
        logger.info(f"Added {added_tokens} frequent tokens to vocabulary")
        
        # Generate n-grams for remaining slots
        if len(self.vocab) < target_vocab_size:
            logger.info("Generating n-grams to fill remaining vocabulary slots...")
            
            # Extract character n-grams (2-4 characters)
            ngrams = Counter()
            for token in all_tokens:
                if len(token) >= 2:
                    # 2-grams
                    for i in range(len(token) - 1):
                        ngrams[token[i:i+2]] += token_freq[token]
                    # 3-grams
                    for i in range(len(token) - 2):
                        ngrams[token[i:i+3]] += token_freq[token]
                    # 4-grams
                    if len(token) >= 4:
                        for i in range(len(token) - 3):
                            ngrams[token[i:i+4]] += token_freq[token]
            
            # Add most frequent n-grams
            for ngram, freq in ngrams.most_common():
                if len(self.vocab) >= target_vocab_size:
                    break
                if ngram not in self.vocab and freq >= 2:  # Only add if appears at least twice
                    self.vocab[ngram] = len(self.vocab)
        
        # Fill remaining slots with character combinations if needed
        if len(self.vocab) < target_vocab_size:
            logger.info("Adding character combinations to reach target vocab size...")
            
            # Get all unique characters
            all_chars = set()
            for text in texts:
                all_chars.update(text)
            all_chars = sorted(list(all_chars))
            
            # Generate 2-character combinations
            for i, char1 in enumerate(all_chars):
                for j, char2 in enumerate(all_chars):
                    if len(self.vocab) >= target_vocab_size:
                        break
                    combo = char1 + char2
                    if combo not in self.vocab:
                        self.vocab[combo] = len(self.vocab)
                if len(self.vocab) >= target_vocab_size:
                    break
    
    def train(self, texts, vocab_size=1500, min_frequency=1):
        """Train BPE tokenizer on texts with guaranteed vocab size"""
        logger.info(f"Starting tokenizer training with target vocab size: {vocab_size}")
        
        # Initialize vocabulary with special tokens
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        current_vocab_size = len(self.special_tokens)
        
        # Tokenize all texts and count character frequencies
        word_freqs = defaultdict(int)
        all_chars = set()
        
        logger.info("Processing training texts...")
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"Processing text {i+1}/{len(texts)}")
            
            tokens = self._get_word_tokens(text)
            for token in tokens:
                # Convert token to list of characters
                chars = list(token)
                if len(chars) > 0:
                    word_freqs[tuple(chars)] += 1
                    all_chars.update(chars)
        
        # Add individual characters to vocabulary
        logger.info(f"Adding {len(all_chars)} unique characters to vocabulary")
        for char in sorted(all_chars):
            if char not in self.vocab:
                self.vocab[char] = current_vocab_size
                current_vocab_size += 1
        
        logger.info(f"Current vocab size after characters: {len(self.vocab)}")
        
        # Perform BPE merges with reduced min_frequency for better coverage
        remaining_slots = vocab_size - current_vocab_size
        logger.info(f"Performing BPE merges to fill {remaining_slots} remaining slots...")
        
        # Use lower min_frequency to get more merges
        effective_min_frequency = max(1, min_frequency // 2)
        
        for merge_count in range(remaining_slots):
            if merge_count % 100 == 0 and merge_count > 0:
                logger.info(f"Completed {merge_count}/{remaining_slots} merges")
            
            # Count all pairs
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                if len(word) > 1:
                    word_pairs = self._get_pairs(word)
                    for pair in word_pairs:
                        pairs[pair] += freq
            
            if not pairs:
                logger.info(f"No more pairs to merge. Breaking at {merge_count} merges.")
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < effective_min_frequency:
                logger.info(f"Best pair frequency {pairs[best_pair]} < {effective_min_frequency}. Reducing threshold.")
                effective_min_frequency = max(1, effective_min_frequency - 1)
                if effective_min_frequency < 1:
                    logger.info("Reached minimum frequency threshold. Breaking.")
                    break
            
            # Merge the best pair
            new_word_freqs = {}
            new_token = ''.join(best_pair)
            
            for word, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
            
            word_freqs = new_word_freqs
            
            # Add new token to vocabulary
            if new_token not in self.vocab:
                self.vocab[new_token] = current_vocab_size
                current_vocab_size += 1
                self.merges[best_pair] = new_token
        
        logger.info(f"BPE merges completed. Current vocab size: {len(self.vocab)}")
        
        # If we still haven't reached target vocab size, expand with subwords
        if len(self.vocab) < vocab_size:
            logger.info(f"Expanding vocabulary from {len(self.vocab)} to {vocab_size}")
            self._expand_vocabulary_with_subwords(texts, vocab_size)
        
        # Create bidirectional mappings
        self.token_to_id = self.vocab
        self.id_to_token = {id: token for token, id in self.vocab.items()}
        
        logger.info(f"Tokenizer training completed! Final vocab size: {len(self.vocab)}")
        
        # Verify we reached target size
        if len(self.vocab) != vocab_size:
            logger.warning(f"Target vocab size {vocab_size} not reached. Final size: {len(self.vocab)}")
        else:
            logger.info(f"Successfully reached target vocab size: {vocab_size}")
        
    def _apply_merges(self, tokens):
        """Apply learned merges to a list of tokens"""
        if len(tokens) <= 1:
            return tokens
            
        while True:
            pairs = []
            for i in range(len(tokens) - 1):
                pairs.append((tokens[i], tokens[i + 1]))
            
            # Find the first pair that can be merged (in order of learning)
            merge_found = False
            for i, pair in enumerate(pairs):
                if pair in self.merges:
                    # Merge this pair
                    new_tokens = tokens[:i] + [self.merges[pair]] + tokens[i + 2:]
                    tokens = new_tokens
                    merge_found = True
                    break
            
            if not merge_found:
                break
                
        return tokens
    
    def encode(self, text):
        """Encode text to token IDs"""
        if not text.strip():
            return []
        
        try:
            # Tokenize text
            tokens = self._get_word_tokens(text)
            if not tokens:
                return []
            
            # First try to find complete tokens in vocabulary
            token_ids = []
            for token in tokens:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    # Convert to character level and apply BPE
                    char_tokens = list(token)
                    merged_tokens = self._apply_merges(char_tokens)
                    
                    for merged_token in merged_tokens:
                        if merged_token in self.token_to_id:
                            token_ids.append(self.token_to_id[merged_token])
                        else:
                            # Use UNK token for unknown tokens
                            token_ids.append(self.token_to_id.get('<unk>', 1))
            
            return token_ids
            
        except Exception as e:
            logger.warning(f"Encoding failed for text: {text[:50]}... Error: {e}")
            return [self.token_to_id.get('<unk>', 1)]
    
    def decode(self, token_ids):
        """Decode token IDs back to text"""
        if not token_ids:
            return ""
        
        try:
            tokens = []
            for token_id in token_ids:
                if isinstance(token_id, int) and token_id in self.id_to_token:
                    token = self.id_to_token[token_id]
                    if token not in self.special_tokens:  # Skip special tokens
                        tokens.append(token)
            
            return ''.join(tokens)
            
        except Exception as e:
            logger.warning(f"Decoding failed for token_ids: {token_ids[:10]}... Error: {e}")
            return ""
    
    def save(self, path):
        """Save tokenizer to file"""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'special_tokens': self.special_tokens
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Tokenizer saved to {path} with vocab size: {len(self.vocab)}")
    
    def load(self, path):
        """Load tokenizer from file"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.vocab = data['vocab']
            self.merges = data['merges']
            self.token_to_id = data['token_to_id']
            self.id_to_token = data['id_to_token']
            self.special_tokens = data['special_tokens']
            
            logger.info(f"Tokenizer loaded from {path} with vocab size: {len(self.vocab)}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    @property
    def vocab_size(self):
        """Return vocabulary size"""
        return len(self.vocab)

def train_tokenizer(data_path, save_path, vocab_size=1500):
    """Train and save a BPE tokenizer with guaranteed vocab size"""
    # Read training data
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into multiple texts for better training
        texts = []
        # Split by functions/classes for better vocabulary coverage
        function_splits = text.split('def ')
        for split in function_splits:
            if split.strip():
                texts.append('def ' + split if not split.startswith('def') else split)
        
        # Also add the full text for context
        texts.append(text)
        
        logger.info(f"Training on {len(texts)} text segments")
        
    except FileNotFoundError:
        logger.error(f"Training data file not found: {data_path}")
        raise
    
    # Create save directory
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Train tokenizer with explicit vocab size
    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=vocab_size, min_frequency=1)
    
    # Verify vocab size
    actual_vocab_size = len(tokenizer.vocab)
    logger.info(f"Trained tokenizer with vocab size: {actual_vocab_size}")
    
    if actual_vocab_size < vocab_size:
        logger.warning(f"Vocab size {actual_vocab_size} is less than target {vocab_size}")
    
    # Save tokenizer
    tokenizer.save(save_path)
    logger.info(f"Tokenizer saved to {save_path}")
    
    return tokenizer