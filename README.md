# Machine_Translate_zh_vi_using_Transformer

# Neural Machine Translation for Chinese-Vietnamese (zh→vi)

## 📌 Overview

This repository contains the implementation of a Neural Machine Translation (NMT) system for the Chinese-to-Vietnamese (zh→vi) language pair. The project explores and compares three different approaches:

1. **Custom Transformer** – Built from scratch using PyTorch with 4 encoder/decoder layers (≈15.2M parameters)
2. **Hirashiba-mt-tiny-zh-vi** – Fine-tuned a small task-specific model (≈15.1M parameters)
3. **Helsinki-NLP/opus-mt-zh-vi** – Fine-tuned a large multilingual model (≈144.5M parameters)

## 📊 Dataset

- **CCMatrix** – 8,045,074 raw Chinese-Vietnamese sentence pairs with LASER similarity scores
- **VLSP 2022** – Official test set (1,000 sentences) for evaluation

## 🛠️ Preprocessing Pipeline

A comprehensive 10-step filtering pipeline was applied:
- Null filtering
- LASER score filtering (≥ 1.05)
- Length filtering (zh: 4–150 chars, vi: 3–100 tokens)
- Length ratio filtering (zh/vi: 0.18–1.2)
- Content filtering (URLs, spam, numeric-only)
- Noise character filtering (box-drawing, odd dashes)
- CJK ratio filtering (≥30% Han characters for zh)
- Latin ratio filtering (≥50% Latin characters for vi)
- Deduplication (keep highest LASER score)
- Text normalization (Traditional→Simplified, Unicode NFC, punctuation ASCII, lowercase)

Finally, **494,000 high-quality sentence pairs** (6.14% of original) were used for training.

## 🤖 Tokenization

- **SentencePiece BPE** trained separately for each language:
  - Chinese: 8,000 vocabulary, character_coverage=0.9999
  - Vietnamese: 16,000 vocabulary, character_coverage=1.0

## 🏆 Results (on VLSP 2022 test set)

| Model | Parameters | BLEU-13a | BLEU-char | chrF |
|-------|------------|----------|-----------|------|
| **Hirashiba** (fine-tuned) | 15.1M | **27.81** | **53.78** | **48.57** |
| Custom Transformer | 15.2M | 24.45 | 50.28 | 45.28 |
| Helsinki OPUS-MT (fine-tuned) | 144.5M | 23.26 | 49.17 | 44.36 |

## 💡 Key Findings

- A small (15.1M) but **task-specific** model outperforms a large (144.5M) **multilingual** model by 4.55 BLEU points
- Quality filtering is more important than quantity – only 6.14% of raw data was retained
- Proper normalization (Traditional→Simplified, Unicode NFC) significantly impacts performance

## 🧠 Technologies Used

- **Frameworks:** PyTorch, HuggingFace Transformers, SentencePiece
- **Evaluation:** SacreBLEU
- **Environment:** Kaggle Notebook, Google Colab
- **Hardware:** NVIDIA Tesla T4 GPU (16GB VRAM)

## 📁 Repository Structure
