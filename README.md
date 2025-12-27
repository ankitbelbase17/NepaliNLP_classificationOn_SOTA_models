
# Nepali Text Classification

Complete production-ready system for Nepali text classification using state-of-the-art transformer models.

## Features

- **NepBERT & CANINE Models**: State-of-the-art NLP models for Nepali text
- **Nepali-Specific**: Optimized for low-resource Nepali language
- **Dataset**: np20ng (Nepali 20 Newsgroups) from Hugging Face
- **WandB Integration**: Real-time training monitoring
- **Production Ready**: Complete training, inference, evaluation
- **Checkpointing**: Auto-save every checkpoint
- **Mixed Precision**: Faster training with AMP

## Installation

```bash
git clone <your-repo>
cd nepali_text_classification

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
wandb login

# Download dataset
bash scripts/download_data.sh
```

## Quick Start

### Train NepBERT (Recommended)
```bash
bash scripts/train.sh nepbert
```

### Train CANINE
```bash
bash scripts/train.sh canine
```

### Run Inference

Batch:
```bash
bash scripts/inference.sh nepbert ./experiments/checkpoints/nepbert_best.pth batch
```

Single Text:
```bash
bash scripts/inference.sh nepbert ./checkpoints/nepbert_best.pth single "यो नेपाली वाक्य हो"
```

### Evaluate Metrics
```bash
bash scripts/metrics.sh nepbert ./experiments/checkpoints/nepbert_best.pth
```

## Models

### 1. NepBERT (Recommended)
- BERT model specifically trained on Nepali corpus
- Pre-trained with Nepali language patterns and vocabulary
- Excellent performance on Nepali text classification tasks
- Better contextual understanding for Nepali language

### 2. CANINE
- Character-level model that works directly with Unicode characters
- No tokenization required - processes text at character level
- Handles language variations and spelling variations well
- Excellent multilingual capabilities with strong Nepali support

## Dataset: np20ng

Nepali 20 Newsgroups dataset with 20 categories:
- News categories in Nepali language
- Balanced distribution
- High-quality annotations

## Training Features

- ✅ Automatic checkpointing (every 250 steps)
- ✅ WandB logging with real-time metrics
- ✅ Resume training from latest checkpoint
- ✅ Mixed precision training (AMP)
- ✅ Xavier initialization
- ✅ Gradient clipping
- ✅ Learning rate scheduling
- ✅ Label smoothing

## Testing

```bash
python tests/test_model.py
python tests/test_train.py
python tests/test_inference.py
python tests/test_dataloader.py
```

## Results

PerformanceNepBERT | CANINE |
|--------|---------|--------|
| Accuracy | 0.98815 | 0.99071 |
| Precision (Macro) | 0.80663 | 0.84832 |
| Recall (Macro) | 0.83855 | 0.86455 |
| F1 Score (Macro) | 0.80451 | 0.83297 |
| Matthews Correlation Coefficient | 0.82426 | 0.85238 |
| Cohen's Kappa | 0.82422 | 0.85228 |
| ROC-AUC (Macro) | 0.98895 | 0.990710.97912 | 0.97625 |
| Cohen's Kappa | 0.97911 | 0.97622 |
| ROC-AUC (Macro) | 0.99936 | 0.99892 |

## Model Comparison

**NepBERT** delivers strong Nepali-specific performance with a dedicated pre-trained vocabulary and tokenizer. It's ideal for capturing nuanced Nepali language patterns and achieves excellent results on standard benchmarks.

**CANINE** provides superior overall performance (99.07% accuracy) by operating at the character level, eliminating tokenization bottlenecks. It's particularly robust to spelling variations and language variations common in Nepali text.

## Tips

1. **Start with CANINE** - Highest overall accuracy (99.07%)
2. **Use NepBERT** - Better Nepali language understanding
3. **Adjust batch size** based on GPU memory
4. **Use mixed precision** for faster training
5. **Monitor WandB** for real-time metrics

## Citation

```bibtex
@misc{nepali_text_classification_2024,
  title={Nepali Text Classification with Transformers},
  author={Ankit Belbase},
  year={2024}
}
```

## License

MIT License
