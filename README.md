
# Nepali Text Classification

Complete production-ready system for Nepali text classification using state-of-the-art transformer models.

## Features

- **9 Transformer Models**: BERT, BART, ELECTRA, Reformer, mBART, Canine, NepBERT, T5, Qwen2
- **Nepali-Specific**: Optimized for low-resource Nepali language
- **Dataset**: np20ng (Nepali 20 Newsgroups) from Hugging Face
- **WandB Integration**: Real-time training monitoring
- **Production Ready**: Complete training, inference, evaluation
- **Checkpointing**: Auto-save every 250 steps
- **Mixed Precision**: Faster training

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

### Train NepBERT (Recommended for Nepali)
```bash
bash scripts/train.sh nepbert
```

### Train Other Models
```bash
bash scripts/train.sh bert
bash scripts/train.sh t5
bash scripts/train.sh qwen2
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

### 1. BERT (Multilingual)
- Baseline multilingual model
- Good for Nepali text
- ~177M parameters

### 2. NepBERT (Recommended)
- Specifically trained on Nepali corpus
- Best performance for Nepali
- ~110M parameters

### 3. mBART
- Multilingual BART
- Strong for sequence tasks
- ~610M parameters

### 4. T5 (mT5)
- Text-to-text framework
- Versatile architecture
- ~580M parameters

### 5. Qwen2
- Latest multilingual LLM
- State-of-the-art performance
- ~500M parameters

### 6. ELECTRA
- Efficient pre-training
- Fast inference
- ~110M parameters

### 7. Canine
- Character-level model
- No tokenization needed
- ~125M parameters

### 8. Reformer
- Efficient for long sequences
- Lower memory
- ~150M parameters

### 9. BART
- Denoising autoencoder
- Strong representations
- ~400M parameters

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

Expected performance on np20ng test set:

| Model | Accuracy | F1 (macro) | Parameters |
|-------|----------|------------|------------|
| NepBERT | ~85% | ~0.84 | 110M |
| BERT | ~82% | ~0.81 | 177M |
| mBART | ~84% | ~0.83 | 610M |
| T5 | ~83% | ~0.82 | 580M |

## Tips

1. **Start with NepBERT** - Best for Nepali
2. **Adjust batch size** based on GPU memory
3. **Use mixed precision** for faster training
4. **Monitor WandB** for real-time metrics

## Citation

```bibtex
@misc{nepali_text_classification_2024,
  title={Nepali Text Classification with Transformers},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
