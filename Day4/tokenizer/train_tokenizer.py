import re
import json
import pickle
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_freqs = {}
        self.splits = {}
        self.merges = []
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
        
    def pre_tokenize(self, text):
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def compute_pair_freqs(self, splits):
        pair_freqs = defaultdict(int)
        for word, freq in splits.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair_freqs[(symbols[i], symbols[i + 1])] += freq
        return pair_freqs
    
    def merge_vocab(self, pair, splits):
        new_splits = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in splits:
            new_word = word.replace(bigram, replacement)
            new_splits[new_word] = splits[word]
        return new_splits
    
    def train(self, texts):
        for text in texts:
            tokens = self.pre_tokenize(text)
            for token in tokens:
                self.word_freqs[token] = self.word_freqs.get(token, 0) + 1
        
        for word, freq in self.word_freqs.items():
            self.splits[' '.join(word) + ' </w>'] = freq
        
        special_tokens = ['<pad>', '<unk>', '<eos>', '<bos>']
        vocab = {token: i for i, token in enumerate(special_tokens)}
        
        for word in self.word_freqs:
            for char in word:
                if char not in vocab:
                    vocab[char] = len(vocab)
        
        vocab['</w>'] = len(vocab)
        
        num_merges = self.vocab_size - len(vocab)
        
        for i in range(num_merges):
            pair_freqs = self.compute_pair_freqs(self.splits)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.splits = self.merge_vocab(best_pair, self.splits)
            self.merges.append(best_pair)
            vocab[''.join(best_pair)] = len(vocab)
        
        self.vocab = vocab
        self.token_to_id = vocab
        self.id_to_token = {i: token for token, i in vocab.items()}
    
    def encode(self, text):
        tokens = self.pre_tokenize(text)
        encoded = []
        
        for token in tokens:
            word = ' '.join(token) + ' </w>'
            
            for merge in self.merges:
                bigram = ' '.join(merge)
                replacement = ''.join(merge)
                word = word.replace(bigram, replacement)
            
            word_tokens = word.split()
            for w_token in word_tokens:
                if w_token in self.token_to_id:
                    encoded.append(self.token_to_id[w_token])
                else:
                    encoded.append(self.token_to_id['<unk>'])
        
        return encoded
    
    def decode(self, token_ids):
        tokens = [self.id_to_token[id_] for id_ in token_ids if id_ in self.id_to_token]
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()
    
    def save(self, path):
        data = {
            'vocab_size': self.vocab_size,
            'merges': self.merges,
            'vocab': self.vocab,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vocab_size = data['vocab_size']
        self.merges = data['merges']
        self.vocab = data['vocab']
        self.token_to_id = data['token_to_id']
        self.id_to_token = data['id_to_token']

def train_tokenizer(data_file, save_path, vocab_size=1000):
    with open(data_file, 'r') as f:
        text = f.read()
    
    tokenizer = BPETokenizer(vocab_size)
    tokenizer.train([text])
    tokenizer.save(save_path)
    print(f"Tokenizer trained and saved to {save_path}")
    return tokenizer
