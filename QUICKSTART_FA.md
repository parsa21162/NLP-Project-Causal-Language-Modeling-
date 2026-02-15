# راهنمای سریع - Causal Language Modeling

## 🚀 شروع سریع (به فارسی)

### مرحله 1: نصب

```bash
# نصب وابستگی‌ها
pip install torch numpy matplotlib seaborn tqdm pandas jupyter python-docx

# یا استفاده از requirements.txt
pip install -r requirements.txt
```

### مرحله 2: تست نصب

```bash
# اجرای تست
python test_setup.py
```

اگر پیغام "All tests passed! ✓" را دیدید، نصب موفق بوده است.

---

## 📚 ساختار پروژه

```
causal-lm-project/
├── src/              # کدهای اصلی
│   ├── model.py      # مدل Transformer
│   ├── train.py      # آموزش مدل
│   ├── inference.py  # تولید متن
│   └── baseline.py   # مدل پایه (BiGram)
├── demo/             # نوت‌بوک آموزشی
├── docs/             # مستندات
├── models/           # مدل‌های ذخیره‌شده
├── data/             # داده‌ها
└── README.md         # راهنما
```

---

## 🎯 سه روش برای استفاده

### روش 1: استفاده از Jupyter Notebook (پیشنهادی!)

این روش بهترین گزینه برای یادگیری و آزمایش است:

```bash
# باز کردن نوت‌بوک
jupyter notebook demo/demo_notebook.ipynb
```

در این نوت‌بوک:
- ✅ آموزش کامل مدل
- ✅ مقایسه با Baseline
- ✅ تولید متن
- ✅ تجسم نتایج
- ✅ تحلیل‌های مختلف

همه چیز مرحله به مرحله توضیح داده شده است.

---

### روش 2: آموزش از طریق Command Line

اگر می‌خواهید فقط مدل را آموزش دهید:

```bash
cd src
python train.py
```

این کار:
- مدل را آموزش می‌دهد
- Checkpointها را ذخیره می‌کند
- Tokenizer را می‌سازد

مدل آموزش‌دیده در `models/best_model.pt` ذخیره می‌شود.

---

### روش 3: تولید متن (بعد از آموزش)

```bash
# حالت تعاملی (Interactive)
python src/inference.py --mode interactive

# تولید از یک prompt
python src/inference.py --mode generate --prompt "این یک تست"

# محاسبه Perplexity
python src/inference.py --mode perplexity --text_file data.txt
```

---

## 💡 مثال‌های عملی

### مثال 1: آموزش سریع

```python
from model import CausalLanguageModel
from train import SimpleTokenizer, TextDataset, Trainer
import torch

# داده‌های نمونه
texts = ["این یک متن نمونه است", "ما در حال آموزش مدل هستیم"] * 100

# ساخت Tokenizer
tokenizer = SimpleTokenizer(tokenizer_type='char')
tokenizer.build_vocab(texts)

# ساخت مدل
model = CausalLanguageModel(
    vocab_size=len(tokenizer.vocab),
    d_model=128,
    n_layers=2,
    n_heads=4
)

# آموزش...
```

### مثال 2: تولید متن

```python
from inference import TextGenerator

# بارگذاری مدل
generator = TextGenerator(
    model_path='models/best_model.pt',
    tokenizer_path='models/tokenizer.json'
)

# تولید متن
text = generator.generate(
    prompt="هوش مصنوعی",
    max_new_tokens=50,
    temperature=0.8
)

print(text[0])
```

---

## 🔧 تنظیمات پیشنهادی

### برای تست سریع:
```python
config = {
    'd_model': 128,
    'n_layers': 2,
    'n_heads': 4,
    'batch_size': 16,
    'num_epochs': 3
}
```

### برای کیفیت بهتر:
```python
config = {
    'd_model': 512,
    'n_layers': 6,
    'n_heads': 8,
    'batch_size': 32,
    'num_epochs': 20
}
```

---

## ❓ سوالات متداول

### 1. چطور داده خودم را استفاده کنم؟

داده‌های متنی خود را در فایل txt قرار دهید:

```python
# خواندن داده
with open('my_data.txt', 'r', encoding='utf-8') as f:
    texts = f.readlines()

# ادامه مراحل...
```

### 2. چطور مدل را بزرگ‌تر کنم؟

در `train.py` یا notebook، پارامترها را افزایش دهید:

```python
model = CausalLanguageModel(
    vocab_size=vocab_size,
    d_model=768,      # بزرگ‌تر
    n_layers=12,      # بیشتر
    n_heads=12,       # بیشتر
    d_ff=3072         # بزرگ‌تر
)
```

### 3. آیا GPU لازم است؟

- برای تست و یادگیری: خیر، CPU کافی است
- برای مدل‌های بزرگ: بله، GPU خیلی سریع‌تر است

کد به طور خودکار GPU را تشخیص می‌دهد:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 4. چطور کیفیت تولید متن را بهبود دهم؟

1. داده بیشتر استفاده کنید
2. مدل را بزرگ‌تر کنید
3. زمان آموزش را افزایش دهید
4. Temperature را تنظیم کنید (0.7-0.9 معمولاً خوب است)
5. از top_k و top_p استفاده کنید

### 5. Perplexity چیست؟

Perplexity معیاری برای سنجش کیفیت مدل زبانی است:
- **عدد کوچک‌تر = مدل بهتر**
- مثلاً: PPL=20 بهتر از PPL=50

---

## 🐛 رفع مشکلات رایج

### مشکل: Out of Memory

**راه‌حل:**
```python
# کاهش batch_size
batch_size = 8  # به جای 32

# کاهش max_seq_len
max_seq_len = 128  # به جای 512

# کاهش اندازه مدل
d_model = 256  # به جای 512
```

### مشکل: Training خیلی کند است

**راه‌حل:**
- از GPU استفاده کنید
- داده کمتری استفاده کنید (برای تست)
- مدل کوچک‌تر بسازید

### مشکل: متن تولیدی بی‌معنی است

**راه‌حل:**
- آموزش بیشتر (epochs بیشتر)
- داده بیشتر
- مدل بزرگ‌تر
- Temperature را کاهش دهید (مثلاً 0.7)

---

## 📊 نمونه خروجی

بعد از آموزش موفق، چیزی شبیه این خواهید دید:

```
Epoch 10/10
Train Loss: 1.8234 | Train PPL: 6.19
Val Loss: 2.0156 | Val PPL: 7.51

✓ Model saved to models/best_model.pt
```

و می‌توانید متن تولید کنید:

```
Prompt: "Machine learning"
Generated: "Machine learning is a subset of artificial 
intelligence that enables computers to learn from data..."
```

---

## 🎓 منابع یادگیری

### مقالات اصلی:
1. "Attention Is All You Need" - Transformer اصلی
2. "Language Models are Unsupervised Multitask Learners" - GPT-2
3. "Improving Language Understanding..." - GPT اولیه

### آموزش‌های توصیه‌شده:
- The Illustrated Transformer (Jay Alammar)
- CS224N - Stanford NLP
- Hugging Face Course

---

## ✅ چک‌لیست نهایی

قبل از شروع، مطمئن شوید:

- [x] Python نصب است (3.8+)
- [x] PyTorch نصب است
- [x] وابستگی‌ها نصب شده‌اند (`pip install -r requirements.txt`)
- [x] تست نصب موفق بود (`python test_setup.py`)
- [x] داده آماده است (یا از داده نمونه استفاده می‌کنید)

حالا آماده‌اید! 🚀

---

## 🆘 کمک بیشتر

اگر مشکلی داشتید:
1. ابتدا README.md اصلی را بخوانید
2. فایل `docs/experiments.md` را برای جزئیات بیشتر ببینید
3. نوت‌بوک `demo/demo_notebook.ipynb` را اجرا کنید
4. به Issues در GitHub مراجعه کنید

---

**موفق باشید!** 🎉

این پروژه را با ❤️ برای یادگیری و آموزش ساخته‌ایم.
