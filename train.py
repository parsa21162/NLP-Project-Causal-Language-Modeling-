"""
Training script for Causal Language Model
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
import json
from pathlib import Path

from model import CausalLanguageModel, count_parameters


class TextDataset(Dataset):
    """
    Simple text dataset for causal language modeling
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        
        # Pad if necessary
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        tokens = torch.tensor(tokens[:self.max_length], dtype=torch.long)
        
        # For causal LM, input is tokens[:-1] and target is tokens[1:]
        # But we'll handle this in the model
        return {
            'input_ids': tokens,
            'labels': tokens  # Model will shift internally
        }


class SimpleTokenizer:
    """
    Character-level or word-level tokenizer
    """
    def __init__(self, vocab=None, tokenizer_type='char'):
        self.tokenizer_type = tokenizer_type
        self.vocab = vocab or {}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        if not vocab:
            self.vocab = {
                self.pad_token: self.pad_token_id,
                self.unk_token: self.unk_token_id,
                self.bos_token: self.bos_token_id,
                self.eos_token: self.eos_token_id,
            }
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self, texts, min_freq=1):
        """Build vocabulary from texts"""
        from collections import Counter
        
        counter = Counter()
        for text in texts:
            if self.tokenizer_type == 'char':
                counter.update(list(text))
            else:  # word-level
                counter.update(text.split())
        
        # Add tokens to vocab
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"Vocabulary size: {len(self.vocab)}")
        return self.vocab
    
    def encode(self, text, max_length=None):
        """Convert text to token ids"""
        if self.tokenizer_type == 'char':
            tokens = list(text)
        else:
            tokens = text.split()
        
        # Add BOS token
        ids = [self.bos_token_id]
        
        for token in tokens:
            ids.append(self.vocab.get(token, self.unk_token_id))
        
        # Add EOS token
        ids.append(self.eos_token_id)
        
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
        
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """Convert token ids back to text"""
        tokens = []
        special_ids = {self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id}
        
        for id in ids:
            if skip_special_tokens and id in special_ids:
                continue
            token = self.reverse_vocab.get(id, self.unk_token)
            tokens.append(token)
        
        if self.tokenizer_type == 'char':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def save(self, path):
        """Save tokenizer"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'tokenizer_type': self.tokenizer_type
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(vocab=data['vocab'], tokenizer_type=data['tokenizer_type'])


class Trainer:
    """
    Training loop for causal language model
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        output_dir='./checkpoints',
        log_interval=100
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.log_interval = log_interval
        
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            
            # For causal LM: input = tokens[:-1], target = tokens[1:]
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            # Forward pass
            logits, loss = self.model(inputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Logging
            if batch_idx % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                ppl = np.exp(avg_loss)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{ppl:.2f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(self.device)
            
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            logits, loss = self.model(inputs, targets)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = np.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'global_step': self.global_step
        }
        
        path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def train(self, num_epochs):
        """Full training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch()
            train_ppl = np.exp(train_loss)
            
            # Validate
            val_loss, val_ppl = self.validate()
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)


def main():
    # Configuration
    config = {
        'vocab_size': 5000,  # Will be set based on data
        'd_model': 256,
        'n_layers': 4,
        'n_heads': 4,
        'd_ff': 1024,
        'max_seq_len': 256,
        'dropout': 0.1,
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'tokenizer_type': 'char',  # 'char' or 'word'
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # For demo purposes, create dummy data
    # In practice, load from file
    print("\nGenerating dummy data for demonstration...")
    train_texts = [
        "This is a sample text for training the language model.",
        "We need more data to train a good model.",
        "Language modeling is an important task in NLP.",
    ] * 100  # Repeat for demo
    
    val_texts = [
        "This is validation data.",
        "Testing the model performance.",
    ] * 50
    
    # Build tokenizer
    print("\nBuilding tokenizer...")
    tokenizer = SimpleTokenizer(tokenizer_type=config['tokenizer_type'])
    tokenizer.build_vocab(train_texts + val_texts, min_freq=1)
    config['vocab_size'] = len(tokenizer.vocab)
    
    # Save tokenizer
    os.makedirs('./checkpoints', exist_ok=True)
    tokenizer.save('./checkpoints/tokenizer.json')
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, config['max_seq_len'])
    val_dataset = TextDataset(val_texts, tokenizer, config['max_seq_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    print("\nCreating model...")
    model = CausalLanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'] * len(train_loader)
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir='./checkpoints'
    )
    
    # Train
    trainer.train(config['num_epochs'])
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
