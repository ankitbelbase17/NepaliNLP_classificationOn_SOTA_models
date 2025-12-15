"""
model.py - All transformer models for Nepali text classification
Models: BERT, BART, ELECTRA, Reformer, mBART, Canine, NepBERT, T5, Qwen2
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    BertForSequenceClassification,
    BartForSequenceClassification,
    ElectraForSequenceClassification,
    ReformerForSequenceClassification,
    MBartForSequenceClassification,
    CanineForSequenceClassification,
    T5ForConditionalGeneration,
    T5EncoderModel,
    AutoModel
)
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class BERTClassifier(nn.Module):
    """Multilingual BERT for Nepali text classification"""
    def __init__(self, num_classes: int, model_name: str = "bert-base-multilingual-cased", dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Xavier initialization for classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'pooled_output': pooled_output
        }


class BARTClassifier(nn.Module):
    """BART model for text classification"""
    def __init__(self, num_classes: int, model_name: str = "facebook/mbart-large-cc25", dropout: float = 0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        
        # Load encoder only for classification
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.d_model, num_classes)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # Use last hidden state and pool
        hidden_state = outputs.last_hidden_state
        pooled = hidden_state[:, 0, :]  # Use first token (like CLS)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'pooled_output': pooled
        }


class ELECTRAClassifier(nn.Module):
    """ELECTRA model for text classification"""
    def __init__(self, num_classes: int, model_name: str = "google/electra-base-discriminator", dropout: float = 0.1):
        super().__init__()
        self.electra = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.electra.config.hidden_size, num_classes)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # ELECTRA doesn't have pooler_output, use first token
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'pooled_output': pooled_output
        }


class ReformerClassifier(nn.Module):
    """Reformer model for long text classification"""
    def __init__(self, num_classes: int, model_name: str = "google/reformer-enwik8", dropout: float = 0.1):
        super().__init__()
        self.reformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.reformer.config.hidden_size, num_classes)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.reformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # Pool over sequence
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits, {
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
            'hidden_states': outputs.hidden_states,
            'pooled_output': pooled_output
        }


class MBARTClassifier(nn.Module):
    """Multilingual BART for text classification"""
    def __init__(self, num_classes: int, model_name: str = "facebook/mbart-large-50", dropout: float = 0.1):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.model.config.d_model, num_classes)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'pooled_output': pooled
        }


class CanineClassifier(nn.Module):
    """Character-level Canine model"""
    def __init__(self, num_classes: int, model_name: str = "google/canine-s", dropout: float = 0.1):
        super().__init__()
        self.canine = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.canine.config.hidden_size, num_classes)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.canine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'pooled_output': pooled_output
        }


class NepBERTClassifier(nn.Module):
    """NepBERT - BERT specifically trained on Nepali corpus"""
    def __init__(self, num_classes: int, model_name: str = "Rajan/NepaliBERT", dropout: float = 0.1):
        super().__init__()
        try:
            self.bert = AutoModel.from_pretrained(model_name)
        except:
            # Fallback to multilingual BERT if NepBERT not available
            print(f"Warning: Could not load {model_name}, using multilingual BERT")
            self.bert = AutoModel.from_pretrained("bert-base-multilingual-cased")
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'pooled_output': pooled_output
        }


class T5Classifier(nn.Module):
    """T5 model adapted for classification"""
    def __init__(self, num_classes: int, model_name: str = "google/mt5-base", dropout: float = 0.1):
        super().__init__()
        # Use encoder-only version for efficiency
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.d_model, num_classes)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # Pool over sequence
        pooled = outputs.last_hidden_state.mean(dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'pooled_output': pooled
        }


class Qwen2Classifier(nn.Module):
    """Qwen2 model for text classification"""
    def __init__(self, num_classes: int, model_name: str = "Qwen/Qwen2-0.5B", dropout: float = 0.1):
        super().__init__()
        try:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            hidden_size = self.model.config.hidden_size
        except:
            # Fallback to smaller model
            print(f"Warning: Could not load {model_name}, using smaller model")
            self.model = AutoModel.from_pretrained("bert-base-multilingual-cased")
            hidden_size = self.model.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # Pool last hidden state
        pooled = outputs.last_hidden_state[:, 0, :] if outputs.last_hidden_state.size(1) > 0 else outputs.last_hidden_state.mean(dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits, {
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'pooled_output': pooled
        }


def get_model(model_name: str, num_classes: int, dropout: float = 0.1) -> nn.Module:
    """
    Factory function to get model by name
    
    Args:
        model_name: One of ['bert', 'bart', 'electra', 'reformer', 'mbart', 'canine', 'nepbert', 't5', 'qwen2']
        num_classes: Number of output classes
        dropout: Dropout rate
    """
    
    model_map = {
        'bert': lambda: BERTClassifier(num_classes, dropout=dropout),
        'bart': lambda: BARTClassifier(num_classes, dropout=dropout),
        'electra': lambda: ELECTRAClassifier(num_classes, dropout=dropout),
        'reformer': lambda: ReformerClassifier(num_classes, dropout=dropout),
        'mbart': lambda: MBARTClassifier(num_classes, dropout=dropout),
        'canine': lambda: CanineClassifier(num_classes, dropout=dropout),
        'nepbert': lambda: NepBERTClassifier(num_classes, dropout=dropout),
        't5': lambda: T5Classifier(num_classes, dropout=dropout),
        'qwen2': lambda: Qwen2Classifier(num_classes, dropout=dropout),
    }
    
    if model_name.lower() not in model_map:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(model_map.keys())}")
    
    model = model_map[model_name.lower()]()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Loaded {model_name} model")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test all models
    batch_size = 2
    seq_len = 128
    num_classes = 20
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    for model_name in ['bert', 'electra', 'nepbert', 't5']:
        print(f"\nTesting {model_name}...")
        try:
            model = get_model(model_name, num_classes)
            model.eval()
            
            with torch.no_grad():
                logits, aux = model(input_ids, attention_mask)
                print(f"  Input: {input_ids.shape}, Output: {logits.shape}")
                print(f"  Auxiliary outputs: {list(aux.keys())}")
        except Exception as e:
            print(f"  Error: {e}")