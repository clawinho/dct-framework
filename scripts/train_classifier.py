#!/usr/bin/env python3
"""
Train a distilBERT DCT classifier on synthetic data.

Usage:
    python scripts/train_classifier.py --data data/synthetic_train.jsonl --output models/dct-v0.1
"""

import argparse
import json
import sys
from pathlib import Path

print("DCT Classifier Training")
print("=" * 60)
print()
print("This script trains a distilBERT token classifier for DCT.")
print("Requirements: torch, transformers, datasets")
print()
print("NOTE: Training requires GPU for reasonable speed.")
print("On CPU, expect ~2-4 hours for 50K examples.")
print("On a T4 GPU, expect ~30-60 minutes.")
print()

def main():
    parser = argparse.ArgumentParser(description="Train DCT classifier")
    parser.add_argument("--data", type=str, required=True, help="Path to synthetic training data JSONL")
    parser.add_argument("--output", type=str, default="models/dct-v0.1", help="Output model directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256, help="Max token length (context window)")
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    # Lazy imports (heavy)
    import torch
    import numpy as np
    from transformers import (
        DistilBertForTokenClassification,
        DistilBertTokenizerFast,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print(f"Loading data from {args.data}...")
    examples = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples")

    # Prepare training data
    # Label: for each token in response_without_tools, is it at a divergence point?
    print("Preparing training data...")
    
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    texts = []
    labels_list = []
    
    for ex in examples:
        response = ex.get("response_without_tools", "")
        if not response:
            continue
        
        # Simple labeling: if the model should have used a tool but didn't,
        # mark tokens that are in critical patterns as positive
        used_tool = ex.get("used_tool", False)
        
        # Tokenize
        encoding = tokenizer(
            response,
            truncation=True,
            max_length=args.max_length,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        # Generate labels
        token_labels = [0] * len(encoding["input_ids"])
        
        if not used_tool:
            # Model didn't use tool -- check if it should have (based on logprobs)
            tokens_data = ex.get("tokens_without_tools", [])
            response_lower = response.lower()
            
            # Heuristic: mark low-confidence tokens in responses that should have used tools
            for i, td in enumerate(tokens_data):
                if i >= args.max_length:
                    break
                logprob = td.get("logprob", 0)
                token_text = td.get("token", "").lower()
                
                # Mark as critical if: low confidence + contains decision words
                decision_words = [
                    "not", "n't", "no", "never", "cannot", "can't",
                    "don't", "doesn't", "isn't", "aren't", "won't",
                    "admin", "owner", "user", "permission", "access",
                    "allowed", "denied", "verified", "confirmed",
                ]
                
                is_decision_word = any(w in token_text for w in decision_words)
                is_low_confidence = logprob < -0.5
                
                if is_decision_word and is_low_confidence:
                    # Find corresponding position in tokenizer output
                    # (approximate mapping)
                    if i < len(token_labels):
                        token_labels[i] = 1
        
        texts.append(response)
        labels_list.append(token_labels)
    
    print(f"Prepared {len(texts)} training examples")
    positive_count = sum(sum(l) for l in labels_list)
    total_tokens = sum(len(l) for l in labels_list)
    print(f"Positive labels: {positive_count}/{total_tokens} ({positive_count/total_tokens:.2%})")
    
    # Create dataset
    def tokenize_and_align(examples_batch):
        tokenized = tokenizer(
            examples_batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tokenized["labels"] = examples_batch["labels"]
        return tokenized
    
    # Split
    split_idx = int(len(texts) * (1 - args.val_split))
    
    train_dataset = Dataset.from_dict({
        "text": texts[:split_idx],
        "labels": labels_list[:split_idx],
    }).map(tokenize_and_align, batched=True, remove_columns=["text"])
    
    val_dataset = Dataset.from_dict({
        "text": texts[split_idx:],
        "labels": labels_list[split_idx:],
    }).map(tokenize_and_align, batched=True, remove_columns=["text"])
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Model
    model = DistilBertForTokenClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    ).to(device)
    
    # Training
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Flatten and ignore padding (-100)
        true = []
        pred = []
        for p_seq, l_seq in zip(predictions, labels):
            for p, l in zip(p_seq, l_seq):
                if l != -100:
                    true.append(l)
                    pred.append(p)
        
        true = np.array(true)
        pred = np.array(pred)
        
        tp = ((pred == 1) & (true == 1)).sum()
        fp = ((pred == 1) & (true == 0)).sum()
        fn = ((pred == 0) & (true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print(f"\nTraining for {args.epochs} epochs...")
    trainer.train()
    
    # Save
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    # Final eval
    eval_results = trainer.evaluate()
    print(f"\nFinal evaluation:")
    print(f"  Precision: {eval_results.get('eval_precision', 0):.3f}")
    print(f"  Recall:    {eval_results.get('eval_recall', 0):.3f}")
    print(f"  F1:        {eval_results.get('eval_f1', 0):.3f}")
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    main()
