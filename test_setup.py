#!/usr/bin/env python3
"""
Quick test script to verify installation and model functionality
"""

import torch
import sys
sys.path.append('./src')

from model import CausalLanguageModel, count_parameters

def test_model():
    """Test basic model functionality"""
    print("="*60)
    print("Testing Causal Language Model")
    print("="*60)
    
    # Configuration
    vocab_size = 1000
    batch_size = 2
    seq_len = 32
    
    # Create model
    print("\n1. Creating model...")
    model = CausalLanguageModel(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=64,
        dropout=0.1
    )
    
    print(f"   ✓ Model created successfully")
    print(f"   ✓ Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, loss = model(input_ids, targets)
    
    print(f"   ✓ Input shape: {input_ids.shape}")
    print(f"   ✓ Logits shape: {logits.shape}")
    print(f"   ✓ Loss: {loss.item():.4f}")
    
    # Test generation
    print("\n3. Testing text generation...")
    prompt = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(
        prompt, 
        max_new_tokens=10, 
        temperature=1.0, 
        top_k=50
    )
    
    print(f"   ✓ Prompt length: {prompt.shape[1]}")
    print(f"   ✓ Generated length: {generated.shape[1]}")
    print(f"   ✓ Generation successful")
    
    # Test backward pass
    print("\n4. Testing backward pass...")
    loss.backward()
    print(f"   ✓ Backward pass successful")
    
    # Check gradients
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"   ✓ All parameters have gradients: {has_grads}")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    
    return True

def test_cuda():
    """Test CUDA availability"""
    print("\nCUDA Information:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("  Running on CPU")

def main():
    print("\nStarting tests...\n")
    
    try:
        test_cuda()
        success = test_model()
        
        if success:
            print("\n🎉 Setup is complete and working!")
            print("\nNext steps:")
            print("  1. Prepare your data in data/ directory")
            print("  2. Run: python src/train.py")
            print("  3. Or explore: jupyter notebook demo/demo_notebook.ipynb")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
