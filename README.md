# Causal Language Modeling

<div dir="rtl">

## 📋 خلاصه پروژه

این پروژه یک پیاده‌سازی کامل از **Causal Language Modeling** است - یکی از اساسی‌ترین وظایف در NLP که پایه مدل‌های زبانی مدرن مانند GPT را تشکیل می‌دهد.

### ویژگی‌های کلیدی:
- ✅ پیاده‌سازی کامل معماری Transformer Decoder
- ✅ مکانیزم Causal Self-Attention با ماسک‌گذاری
- ✅ آموزش و ارزیابی مدل
- ✅ تولید متن با روش‌های مختلف (Greedy, Top-k, Top-p)
- ✅ مقایسه با Baseline (BiGram Model)
- ✅ Jupyter Notebook جامع برای تحلیل و تجسم
- ✅ مستندات کامل و کد قابل اجرا

</div>

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd causal-lm-project

# Install dependencies
pip install torch numpy matplotlib seaborn tqdm pandas jupyter python-docx
```

### Training

```bash
# Train the model
cd src
python train.py
```

### Inference

```bash
# Interactive text generation
python inference.py --mode interactive

# Generate from a prompt
python inference.py --mode generate --prompt "Machine learning is"

# Calculate perplexity
python inference.py --mode perplexity --text_file data.txt
```

### Demo Notebook

```bash
# Launch Jupyter
jupyter notebook demo/demo_notebook.ipynb
```

---

## 📁 Project Structure

```
causal-lm-project/
├── src/
│   ├── model.py           # Transformer-based Causal LM implementation
│   ├── train.py           # Training script and utilities
│   ├── inference.py       # Text generation and evaluation
│   └── baseline.py        # BiGram baseline model
├── demo/
│   ├── demo_notebook.ipynb  # Complete demo with analysis
│   └── results.json         # Saved results
├── models/
│   ├── best_model.pt        # Best trained model
│   └── tokenizer.json       # Tokenizer vocabulary
├── docs/
│   └── paper_summary.md     # Analytical paper summary (Persian)
├── data/                    # Data directory (add your datasets here)
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🏗️ Model Architecture

### Transformer Decoder

```
Input Text
    ↓
[Token Embedding + Positional Encoding]
    ↓
[Transformer Block 1]
  ├─ Causal Self-Attention (masked)
  ├─ Feed-Forward Network
  └─ Layer Normalization + Residual
    ↓
[Transformer Block 2]
    ↓
    ...
    ↓
[Transformer Block N]
    ↓
[Output Projection → Vocabulary]
    ↓
Next Token Prediction
```

### Key Components

1. **Causal Self-Attention**
   - Multi-head attention mechanism
   - Causal masking (lower triangular)
   - Prevents attending to future tokens

2. **Position Embeddings**
   - Learned positional encoding
   - Captures sequence order information

3. **Feed-Forward Network**
   - 2-layer MLP with GELU activation
   - Applied position-wise

4. **Layer Normalization**
   - Pre-LN architecture (GPT-2 style)
   - Stabilizes training

---

## 🔬 Experiments & Results

### Model Configurations

**Neural Model (Transformer):**
- Vocabulary size: ~5000 tokens (character-level)
- Hidden size: 256
- Number of layers: 4
- Number of heads: 4
- Feed-forward size: 1024
- Max sequence length: 256
- Total parameters: ~5M

**Baseline (BiGram):**
- Statistical n-gram model
- Add-k smoothing (k=1)
- No parameters

### Performance Metrics

| Model | Parameters | Val Perplexity | Improvement |
|-------|-----------|----------------|-------------|
| BiGram Baseline | 0 | ~45-60 | - |
| Transformer (Neural) | ~5M | ~20-30 | ~40-50% |

### Text Generation Examples

**Prompt:** "Machine learning"

**BiGram Output:**
```
Machine learning is important...
[Statistical, less coherent]
```

**Transformer Output:**
```
Machine learning is a subset of artificial intelligence that enables 
computers to learn from data and improve their performance...
[More coherent and contextually relevant]
```

---

## 📊 Visualizations

The demo notebook includes:
- Model comparison charts
- Attention pattern visualization
- Perplexity vs sequence length analysis
- Token distribution analysis
- Temperature effect on generation

All visualizations are saved in `demo/` directory.

---

## 🛠️ Usage Examples

### 1. Training Custom Model

```python
from model import CausalLanguageModel
from train import SimpleTokenizer, Trainer

# Load your data
train_texts = ["your", "training", "data"]
val_texts = ["validation", "data"]

# Build tokenizer
tokenizer = SimpleTokenizer(tokenizer_type='char')
tokenizer.build_vocab(train_texts)

# Create model
model = CausalLanguageModel(
    vocab_size=len(tokenizer.vocab),
    d_model=512,
    n_layers=6,
    n_heads=8
)

# Train
trainer = Trainer(...)
trainer.train(num_epochs=10)
```

### 2. Text Generation

```python
from inference import TextGenerator

generator = TextGenerator(
    model_path='models/best_model.pt',
    tokenizer_path='models/tokenizer.json'
)

# Generate text
generated = generator.generate(
    prompt="The future of AI",
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)
print(generated[0])
```

### 3. Model Evaluation

```python
from inference import calculate_perplexity

ppl = calculate_perplexity(
    model_path='models/best_model.pt',
    tokenizer_path='models/tokenizer.json',
    text_file='test_data.txt'
)
print(f"Perplexity: {ppl:.2f}")
```

---

## 📚 Data Sources

### English Datasets
- **WikiText-2/103**: [Link](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
- **Penn Treebank**: [Link](https://catalog.ldc.upenn.edu/LDC99T42)
- **OpenWebText**: [Link](https://huggingface.co/datasets/openwebtext)

### Persian Datasets
- **OSCAR Corpus**: [Link](https://huggingface.co/datasets/oscar)
- **Persian Wikipedia**: [Link](https://dumps.wikimedia.org/fawiki/)
- **Hamshahri Corpus**: [Link](http://ece.ut.ac.ir/dbrg/hamshahri/)

To use custom data:
1. Place text files in `data/` directory
2. Update data loading in `train.py`
3. Run training

---

## 🔧 Hyperparameter Tuning

Key hyperparameters to tune:

```python
config = {
    'd_model': 256,          # Hidden dimension (128, 256, 512, 768)
    'n_layers': 4,           # Number of layers (2, 4, 6, 8, 12)
    'n_heads': 4,            # Attention heads (4, 8, 12, 16)
    'd_ff': 1024,            # FFN dimension (2048, 4096)
    'dropout': 0.1,          # Dropout rate (0.0, 0.1, 0.2)
    'learning_rate': 3e-4,   # LR (1e-4, 3e-4, 5e-4)
    'batch_size': 32,        # Batch size (16, 32, 64)
    'max_seq_len': 256,      # Max sequence length (128, 256, 512)
}
```

---

## 🧪 Testing

```bash
# Run unit tests
cd tests
python -m pytest test_model.py
python -m pytest test_tokenizer.py
```

---

## 📈 Performance Tips

1. **Use GPU**: Training is much faster on GPU
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **Gradient Accumulation**: For larger batch sizes
   ```python
   accumulation_steps = 4
   ```

3. **Mixed Precision**: Use `torch.cuda.amp` for faster training

4. **Larger Datasets**: More data → better performance

5. **Longer Training**: Language models benefit from more epochs

---

## 🎯 Applications

- **Text Completion**: Auto-complete user input
- **Code Generation**: Generate code snippets
- **Chatbots**: Build conversational agents
- **Creative Writing**: Story generation
- **Domain Adaptation**: Fine-tune for specific domains (medical, legal)

---

## 🔬 Advanced Topics

### 1. Different Tokenization

Replace character-level with BPE or SentencePiece:

```python
# Use HuggingFace tokenizers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

### 2. Larger Models

Scale up the architecture:

```python
model = CausalLanguageModel(
    vocab_size=50000,
    d_model=768,      # GPT-2 small
    n_layers=12,
    n_heads=12,
    d_ff=3072
)
# ~117M parameters
```

### 3. Fine-tuning

Fine-tune on specific tasks:

```python
# Load pre-trained model
model.load_state_dict(checkpoint['state_dict'])

# Fine-tune on domain-specific data
trainer.train(domain_texts, num_epochs=3, learning_rate=1e-5)
```

---

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `batch_size`
   - Reduce `max_seq_len`
   - Use gradient accumulation

2. **Poor Generation Quality**
   - Train longer
   - Use more data
   - Increase model size
   - Adjust temperature

3. **High Perplexity**
   - Check data quality
   - Verify tokenizer
   - Increase model capacity
   - Train longer

---

## 📖 References

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original Transformer paper

2. **Language Models are Unsupervised Multitask Learners** (GPT-2, Radford et al., 2019)
   - Causal language modeling at scale

3. **Improving Language Understanding by Generative Pre-Training** (GPT, Radford et al., 2018)
   - Pre-training approach

4. **The Illustrated Transformer** (Jay Alammar)
   - Great visualization: http://jalammar.github.io/illustrated-transformer/

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is for educational purposes as part of a university course.

---


## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- HuggingFace for transformers inspiration
- Course instructor and TAs

---

<div dir="rtl">]

</div>

---

**Last Updated:** February 2026
