"""
Inference script for text generation using trained Causal Language Model
"""

import torch
import argparse
from pathlib import Path

from model import CausalLanguageModel
from train import SimpleTokenizer


class TextGenerator:
    """
    Text generator using trained causal language model
    """
    def __init__(self, model_path, tokenizer_path, device='cpu'):
        self.device = torch.device(device)
        
        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = SimpleTokenizer.load(tokenizer_path)
        
        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config from checkpoint or use defaults
        self.model = CausalLanguageModel(
            vocab_size=len(self.tokenizer.vocab),
            d_model=256,
            n_layers=4,
            n_heads=4,
            d_ff=1024,
            max_seq_len=256,
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("Model loaded successfully!")
        print(f"Vocabulary size: {len(self.tokenizer.vocab)}")
        print(f"Device: {self.device}")
    
    def generate(
        self, 
        prompt, 
        max_new_tokens=50,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        num_return_sequences=1
    ):
        """
        Generate text from a prompt
        
        Args:
            prompt: Input text string
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated texts
        """
        # Encode prompt
        token_ids = self.tokenizer.encode(prompt, max_length=self.model.max_seq_len)
        input_ids = torch.tensor([token_ids] * num_return_sequences, 
                                dtype=torch.long, 
                                device=self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Decode
        generated_texts = []
        for ids in generated_ids:
            text = self.tokenizer.decode(ids.cpu().tolist(), skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def interactive_generation(self):
        """
        Interactive text generation loop
        """
        print("\n" + "="*60)
        print("Interactive Text Generation")
        print("="*60)
        print("Enter your prompt (or 'quit' to exit)")
        print("Settings: max_tokens=50, temperature=0.8, top_k=50, top_p=0.9")
        print("="*60 + "\n")
        
        while True:
            prompt = input("\nPrompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("\nGenerating...\n")
            
            try:
                generated_texts = self.generate(
                    prompt,
                    max_new_tokens=50,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    num_return_sequences=1
                )
                
                print("="*60)
                print("Generated Text:")
                print("="*60)
                print(generated_texts[0])
                print("="*60)
                
            except Exception as e:
                print(f"Error during generation: {e}")


def calculate_perplexity(model_path, tokenizer_path, text_file, device='cpu'):
    """
    Calculate perplexity on a text file
    """
    device = torch.device(device)
    
    # Load model and tokenizer
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    checkpoint = torch.load(model_path, map_location=device)
    
    model = CausalLanguageModel(
        vocab_size=len(tokenizer.vocab),
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        max_seq_len=256,
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Read text
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    total_loss = 0
    total_tokens = 0
    
    print(f"Calculating perplexity on {len(texts)} samples...")
    
    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            
            # Tokenize
            token_ids = tokenizer.encode(text.strip(), max_length=model.max_seq_len)
            if len(token_ids) < 2:
                continue
            
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            
            # Forward pass
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            logits, loss = model(inputs, targets)
            
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"\nResults:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    
    return perplexity


def main():
    parser = argparse.ArgumentParser(description='Text generation with Causal Language Model')
    parser.add_argument('--model', type=str, default='./checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='./checkpoints/tokenizer.json',
                       help='Path to tokenizer')
    parser.add_argument('--mode', type=str, choices=['generate', 'interactive', 'perplexity'],
                       default='interactive', help='Mode of operation')
    parser.add_argument('--prompt', type=str, default='',
                       help='Prompt for generation (only in generate mode)')
    parser.add_argument('--max_tokens', type=int, default=50,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p (nucleus) sampling')
    parser.add_argument('--num_sequences', type=int, default=1,
                       help='Number of sequences to generate')
    parser.add_argument('--text_file', type=str, default='',
                       help='Text file for perplexity calculation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.mode in ['generate', 'interactive']:
        generator = TextGenerator(args.model, args.tokenizer, args.device)
        
        if args.mode == 'interactive':
            generator.interactive_generation()
        else:
            if not args.prompt:
                print("Error: --prompt is required for generate mode")
                return
            
            generated_texts = generator.generate(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_return_sequences=args.num_sequences
            )
            
            print(f"\nPrompt: {args.prompt}\n")
            for i, text in enumerate(generated_texts, 1):
                print(f"Generated {i}:")
                print("="*60)
                print(text)
                print("="*60 + "\n")
    
    elif args.mode == 'perplexity':
        if not args.text_file:
            print("Error: --text_file is required for perplexity mode")
            return
        
        calculate_perplexity(args.model, args.tokenizer, args.text_file, args.device)


if __name__ == "__main__":
    main()
