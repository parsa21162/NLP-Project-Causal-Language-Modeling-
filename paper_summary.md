# خلاصه تحلیلی: Causal Language Modeling

## مسئله اصلی و اهمیت آن

Causal Language Modeling یکی از اساسی‌ترین وظایف در پردازش زبان طبیعی است که هدف آن **پیش‌بینی کلمه بعدی در یک توالی متنی** بر اساس کلمات قبلی است. این مسئله از این جهت مهم است که:

- **پایه مدل‌های زبانی مدرن**: اساس معماری‌هایی مانند GPT، GPT-2، GPT-3، و LLaMA را تشکیل می‌دهد
- **تولید متن**: قابلیت تولید متن منسجم و معنادار را فراهم می‌کند
- **یادگیری بدون نظارت**: بدون نیاز به داده‌های برچسب‌دار، صرفاً از متن خام یاد می‌گیرد
- **انتقال یادگیری**: مدل آموزش‌دیده قابل fine-tune کردن برای وظایف مختلف است

## ورودی‌ها و خروجی‌های سیستم

### ورودی‌ها:
1. **متن خام (Raw Text)**: دنباله‌ای از کلمات یا توکن‌ها
   - مثال: "امروز هوا بسیار"
2. **Context Window**: طول پنجره زمینه (مثلاً 512 یا 1024 توکن)

### خروجی‌ها:
1. **توزیع احتمالی**: احتمال هر کلمه در vocabulary برای موقعیت بعدی
   - P(w_t | w_1, w_2, ..., w_{t-1})
2. **پیش‌بینی کلمه بعدی**: کلمه با بیشترین احتمال
3. **متن تولیدی**: در حالت inference، ادامه متن ورودی

## داده مورد استفاده

### نوع داده:
- متن خام (Raw Text) - داده‌های unlabeled
- زبان‌های مختلف: انگلیسی، فارسی، عربی، و...

### منابع رایج:
- **Wikipedia**: دانشنامه آزاد
- **Common Crawl**: کراول وب
- **BookCorpus**: مجموعه کتاب‌ها
- **OpenWebText**: نسخه باز Reddit
- **برای فارسی**: 
  - مجموعه‌های OSCAR، CC100
  - اخبار فارسی (Hamshahri, BBC Persian)
  - ویکی‌پدیا فارسی

### اندازه:
- مدل‌های کوچک: 10M-100M توکن
- مدل‌های متوسط: 1B-10B توکن  
- مدل‌های بزرگ (GPT-3): 300B+ توکن

## روش پیشنهادی

### معماری اصلی: Transformer Decoder

```
1. Tokenization:
   Text → Tokens (BPE, WordPiece, SentencePiece)

2. Embedding Layer:
   Tokens → Vector representations
   + Positional Encoding

3. Transformer Decoder Blocks (N layers):
   └─ Masked Self-Attention
      - فقط به توکن‌های قبلی توجه می‌کند (Causal Masking)
      - جلوی دسترسی به آینده را می‌گیرد
   └─ Feed-Forward Network
   └─ Layer Normalization + Residual Connections

4. Output Layer:
   Hidden States → Vocabulary Distribution
   Linear(d_model → vocab_size) + Softmax

5. Loss Function:
   Cross-Entropy Loss بین پیش‌بینی و کلمه واقعی
```

### شبه‌کد اصلی:

```python
# Training Loop
for batch in dataloader:
    # Forward pass
    tokens = batch['input_ids']           # [batch, seq_len]
    targets = tokens[:, 1:]               # shift for next-token prediction
    inputs = tokens[:, :-1]
    
    # Model prediction
    logits = model(inputs)                # [batch, seq_len-1, vocab_size]
    
    # Calculate loss
    loss = cross_entropy(logits.view(-1, vocab_size), 
                         targets.view(-1))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
# Inference (Text Generation)
def generate(prompt, max_length):
    tokens = tokenize(prompt)
    for _ in range(max_length):
        logits = model(tokens)            # [1, current_len, vocab_size]
        next_token = logits[:, -1, :].argmax()  # greedy decoding
        tokens = append(tokens, next_token)
        if next_token == EOS_TOKEN:
            break
    return detokenize(tokens)
```

### نکات کلیدی:

1. **Causal Masking**: 
   - از attention mask استفاده می‌شود تا مدل فقط به گذشته نگاه کند
   - شکل ماسک: مثلث پایینی (lower triangular)

2. **Autoregressive Generation**:
   - در زمان تولید، هر کلمه بر اساس کلمات قبلی پیش‌بینی می‌شود
   - روش‌های sampling: Greedy, Beam Search, Top-k, Top-p (nucleus)

3. **Optimization**:
   - Adam/AdamW optimizer
   - Learning rate scheduling (warmup + decay)
   - Gradient clipping

## نتایج اصلی

### معیارهای ارزیابی:
1. **Perplexity (PPL)**: 
   - اندازه‌گیری کیفیت مدل زبانی
   - PPL پایین‌تر = مدل بهتر
   - فرمول: exp(average cross-entropy loss)

2. **Accuracy**: 
   - دقت پیش‌بینی کلمه بعدی

3. **کیفیت متن تولیدی**:
   - Coherence (انسجام)
   - Fluency (روانی)
   - Diversity (تنوع)

### نتایج معمول:
- مدل‌های کوچک (125M params): PPL ≈ 20-30
- مدل‌های متوسط (350M-1.3B params): PPL ≈ 15-25  
- مدل‌های بزرگ (6.7B+ params): PPL ≈ 10-15

## محدودیت‌ها

1. **حافظه محدود**: فقط به context window محدود است (نمی‌تواند متن‌های خیلی طولانی را در نظر بگیرد)

2. **هزینه محاسباتی بالا**: 
   - آموزش مدل‌های بزرگ نیاز به GPU/TPU قوی دارد
   - Complexity: O(n²) برای self-attention

3. **عدم درک واقعی**: 
   - مدل صرفاً الگوهای آماری را یاد می‌گیرد
   - ممکن است واقعیت‌های دنیای واقعی را ندانود

4. **تعصدات (Bias)**: 
   - تعصدات موجود در داده‌های آموزشی به مدل منتقل می‌شود

5. **Hallucination**: 
   - احتمال تولید اطلاعات غلط یا ساختگی

## ایده‌های ادامه کار

1. **بهبود کارایی**:
   - استفاده از Sparse Attention
   - Model Compression (Pruning, Quantization)
   - Mixture of Experts (MoE)

2. **افزایش طول Context**:
   - Sliding Window Attention
   - Recurrent Memory
   - Retrieval-Augmented Generation (RAG)

3. **بهبود کیفیت تولید**:
   - Reinforcement Learning from Human Feedback (RLHF)
   - Constitutional AI
   - Multi-task Learning

4. **کاهش تعصدات**:
   - Data Filtering
   - Fairness-aware Training
   - Prompt Engineering

5. **توسعه مدل‌های چندزبانه**:
   - Cross-lingual Transfer Learning
   - مدل‌های مختص زبان فارسی با کیفیت بالا

6. **کاربردهای تخصصی**:
   - Code Generation (مدل‌های کد)
   - Medical/Legal Domain Adaptation
   - Conversational AI

---

**نویسنده**: [نام شما]  
**تاریخ**: بهمن 1403  
**موضوع**: Causal Language Modeling - پیاده‌سازی و تحلیل
