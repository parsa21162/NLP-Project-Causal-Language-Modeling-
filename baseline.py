"""
Baseline: Simple BiGram Language Model
A statistical baseline for comparison with neural model
"""

import numpy as np
from collections import defaultdict, Counter
import pickle
import json


class BiGramLanguageModel:
    """
    Simple statistical bigram language model
    Uses Maximum Likelihood Estimation with add-k smoothing
    """
    def __init__(self, k=1.0):
        """
        Args:
            k: Smoothing parameter (k=1 is Laplace smoothing)
        """
        self.k = k
        self.vocab = set()
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.total_tokens = 0
        
        # Special tokens
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
    def train(self, texts, tokenizer_type='char'):
        """
        Train the bigram model on texts
        
        Args:
            texts: List of text strings
            tokenizer_type: 'char' or 'word'
        """
        print(f"Training BiGram model with {len(texts)} samples...")
        
        for text in texts:
            # Tokenize
            if tokenizer_type == 'char':
                tokens = [self.bos_token] + list(text) + [self.eos_token]
            else:
                tokens = [self.bos_token] + text.split() + [self.eos_token]
            
            # Update vocabulary
            self.vocab.update(tokens)
            
            # Count unigrams and bigrams
            for i in range(len(tokens)):
                self.unigram_counts[tokens[i]] += 1
                self.total_tokens += 1
                
                if i > 0:
                    prev_token = tokens[i-1]
                    curr_token = tokens[i]
                    self.bigram_counts[prev_token][curr_token] += 1
        
        # Add special tokens to vocab
        self.vocab.add(self.unk_token)
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Total tokens: {self.total_tokens}")
        print(f"Unique unigrams: {len(self.unigram_counts)}")
        print(f"Unique bigrams: {sum(len(v) for v in self.bigram_counts.values())}")
    
    def get_probability(self, prev_token, curr_token):
        """
        Calculate P(curr_token | prev_token) with add-k smoothing
        
        P(w_i | w_{i-1}) = (Count(w_{i-1}, w_i) + k) / (Count(w_{i-1}) + k*V)
        """
        # Handle unknown tokens
        if prev_token not in self.vocab:
            prev_token = self.unk_token
        if curr_token not in self.vocab:
            curr_token = self.unk_token
        
        numerator = self.bigram_counts[prev_token][curr_token] + self.k
        denominator = self.unigram_counts[prev_token] + self.k * len(self.vocab)
        
        return numerator / denominator
    
    def calculate_perplexity(self, texts, tokenizer_type='char'):
        """
        Calculate perplexity on test texts
        
        Perplexity = exp(-1/N * sum(log P(w_i | w_{i-1})))
        """
        total_log_prob = 0
        total_tokens = 0
        
        for text in texts:
            if not text.strip():
                continue
            
            # Tokenize
            if tokenizer_type == 'char':
                tokens = [self.bos_token] + list(text) + [self.eos_token]
            else:
                tokens = [self.bos_token] + text.split() + [self.eos_token]
            
            # Calculate log probability
            for i in range(1, len(tokens)):
                prev_token = tokens[i-1]
                curr_token = tokens[i]
                
                prob = self.get_probability(prev_token, curr_token)
                total_log_prob += np.log(prob)
                total_tokens += 1
        
        avg_log_prob = total_log_prob / total_tokens
        perplexity = np.exp(-avg_log_prob)
        
        return perplexity
    
    def generate(self, prompt, max_length=50, tokenizer_type='char'):
        """
        Generate text from a prompt using the bigram model
        
        Args:
            prompt: Starting text
            max_length: Maximum number of tokens to generate
            tokenizer_type: 'char' or 'word'
            
        Returns:
            Generated text string
        """
        # Tokenize prompt
        if tokenizer_type == 'char':
            tokens = list(prompt) if prompt else [self.bos_token]
        else:
            tokens = prompt.split() if prompt else [self.bos_token]
        
        # Generate
        for _ in range(max_length):
            prev_token = tokens[-1]
            
            # Get possible next tokens and their probabilities
            if prev_token not in self.bigram_counts:
                prev_token = self.unk_token
            
            next_token_counts = self.bigram_counts[prev_token]
            
            if not next_token_counts:
                # If no bigrams found, sample from unigram distribution
                candidates = list(self.vocab - {self.bos_token, self.unk_token})
                probs = [self.unigram_counts[t] / self.total_tokens for t in candidates]
            else:
                candidates = list(next_token_counts.keys())
                probs = [self.get_probability(prev_token, t) for t in candidates]
            
            # Normalize probabilities
            probs = np.array(probs)
            probs = probs / probs.sum()
            
            # Sample next token
            next_token = np.random.choice(candidates, p=probs)
            
            # Stop if EOS token
            if next_token == self.eos_token:
                break
            
            tokens.append(next_token)
        
        # Convert back to text
        if tokenizer_type == 'char':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def save(self, path):
        """Save model to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'k': self.k,
                'vocab': self.vocab,
                'unigram_counts': self.unigram_counts,
                'bigram_counts': dict(self.bigram_counts),
                'total_tokens': self.total_tokens
            }, f)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load model from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(k=data['k'])
        model.vocab = data['vocab']
        model.unigram_counts = data['unigram_counts']
        model.bigram_counts = defaultdict(Counter, data['bigram_counts'])
        model.total_tokens = data['total_tokens']
        
        print(f"Model loaded from {path}")
        return model


def main():
    """
    Demo: Train and evaluate BiGram baseline
    """
    # Sample data
    train_texts = [
        "This is a sample text for training.",
        "We need more data to train models.",
        "Language modeling is important in NLP.",
        "Natural language processing deals with text.",
        "Machine learning models learn from data.",
    ] * 20
    
    test_texts = [
        "This is test data for evaluation.",
        "Models should generalize to new text.",
    ] * 10
    
    # Train model
    print("="*60)
    print("Training BiGram Baseline Model")
    print("="*60)
    
    model = BiGramLanguageModel(k=1.0)
    model.train(train_texts, tokenizer_type='word')
    
    # Calculate perplexity
    print(f"\n{'='*60}")
    print("Evaluation")
    print("="*60)
    
    train_ppl = model.calculate_perplexity(train_texts, tokenizer_type='word')
    test_ppl = model.calculate_perplexity(test_texts, tokenizer_type='word')
    
    print(f"Train Perplexity: {train_ppl:.2f}")
    print(f"Test Perplexity: {test_ppl:.2f}")
    
    # Generate some text
    print(f"\n{'='*60}")
    print("Text Generation")
    print("="*60)
    
    prompts = ["This is", "Language modeling", "Machine learning"]
    
    for prompt in prompts:
        generated = model.generate(prompt, max_length=15, tokenizer_type='word')
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated}'")
    
    # Save model
    model.save('./checkpoints/bigram_baseline.pkl')
    
    print(f"\n{'='*60}")
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
