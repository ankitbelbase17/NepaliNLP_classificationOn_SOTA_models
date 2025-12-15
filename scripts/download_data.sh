# ==============================================================================
# scripts/download_data.sh - Download np20ng dataset
# ==============================================================================

#!/bin/bash

echo "========================================"
echo "Downloading Nepali 20 Newsgroups Dataset"
echo "========================================"

python -c "
from datasets import load_dataset

print('Downloading np20ng dataset from Hugging Face...')
try:
    dataset = load_dataset('Suyogyart/np20ng', trust_remote_code=True)
    print('\n✓ Dataset downloaded successfully!')
    print(f'  Train samples: {len(dataset[\"train\"])}')
    if 'validation' in dataset:
        print(f'  Validation samples: {len(dataset[\"validation\"])}')
    if 'test' in dataset:
        print(f'  Test samples: {len(dataset[\"test\"])}')
    print(f'  Features: {dataset[\"train\"].features}')
except Exception as e:
    print(f'Error downloading dataset: {e}')
    print('Please ensure you have internet connection and try again.')
"

echo "========================================"

