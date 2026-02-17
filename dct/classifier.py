"""
DCT Token Classifier.

Binary token-level classifier that determines whether a token represents
a decision-critical moment (identity assertion, permission decision,
tool-call refusal, factual claim).

Uses distilBERT (66M params) fine-tuned on synthetic data.
Falls back to a rule-based heuristic when no trained model is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np


@dataclass
class ClassificationResult:
    is_critical: bool
    confidence: float  # 0-1, classifier confidence
    category: Optional[str] = None  # identity, permission, tool_refusal, factual, negation


# --- Rule-based fallback (no trained model needed) ---

# Patterns that indicate decision-critical tokens
CRITICAL_PATTERNS = {
    "identity": [
        r"\b(is|isn'?t|are|aren'?t|was|wasn'?t)\b.*\b(admin|owner|user|member|manager|moderator)\b",
        r"\b(you|they|he|she|this\s+user)\b.*\b(are|is|aren'?t|isn'?t)\b",
        r"\b(verified|unverified|authenticated|unauthorized)\b",
        r"\b(identity|who\s+(you|they|is))\b",
    ],
    "permission": [
        r"\b(can'?t|cannot|not\s+allowed|denied|forbidden|unauthorized)\b",
        r"\b(permission|access|privilege|role)\b.*\b(grant|deny|revoke|check)\b",
        r"\b(you\s+(don'?t|do\s+not)\s+have)\b",
    ],
    "tool_refusal": [
        r"\b(don'?t\s+need\s+to|no\s+need\s+to|already\s+know|I\s+know)\b",
        r"\b(without\s+checking|don'?t\s+need\s+to\s+(look|check|search|verify))\b",
        r"\b(I\s+can\s+tell|I\s+can\s+see\s+that|clearly)\b",
    ],
    "factual": [
        r"\b(the\s+(meeting|appointment|event)\s+is\s+at)\b",
        r"\b(the\s+(price|cost|amount|balance|dosage)\s+is)\b",
        r"\b(located\s+at|lives\s+at|works\s+at|address\s+is)\b",
        r"\b(expires?\s+on|due\s+(on|by|date))\b",
    ],
    "negation": [
        r"\bnot\b",
        r"\bnever\b",
        r"\bdon'?t\b",
        r"\bisn'?t\b",
        r"\baren'?t\b",
        r"\bwasn'?t\b",
        r"\bcan'?t\b",
        r"\bcannot\b",
    ],
}


class RuleBasedClassifier:
    """
    Heuristic DCT classifier using regex patterns.
    
    Use this when no trained model is available (bootstrap / evaluation baseline).
    """

    def classify_token(self, token: str, context: str) -> ClassificationResult:
        """
        Classify a single token given its surrounding context.
        
        Args:
            token: The token text.
            context: Surrounding text (pre + token + post).
        
        Returns:
            ClassificationResult with category and confidence.
        """
        # Check each category
        for category, patterns in CRITICAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    # Check if the token itself is part of the match
                    token_lower = token.strip().lower()
                    match = re.search(pattern, context, re.IGNORECASE)
                    if match and token_lower in match.group(0).lower():
                        return ClassificationResult(
                            is_critical=True,
                            confidence=0.7,  # rule-based = moderate confidence
                            category=category,
                        )

        return ClassificationResult(is_critical=False, confidence=0.9, category=None)

    def classify_sequence(
        self, tokens: list[str], window_size: int = 128
    ) -> list[ClassificationResult]:
        """Classify all tokens in a sequence."""
        full_text = "".join(tokens)
        results = []
        char_pos = 0

        for i, token in enumerate(tokens):
            # Build context window
            start = max(0, char_pos - window_size * 4)  # ~4 chars per token
            end = min(len(full_text), char_pos + len(token) + window_size * 4)
            context = full_text[start:end]

            result = self.classify_token(token, context)
            results.append(result)
            char_pos += len(token)

        return results


class DCTClassifier:
    """
    Neural DCT classifier using fine-tuned distilBERT.
    
    Falls back to RuleBasedClassifier if no model is loaded.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self.fallback = RuleBasedClassifier()

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load a trained distilBERT classifier."""
        try:
            from transformers import (
                DistilBertForTokenClassification,
                DistilBertTokenizerFast,
            )

            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self.model = DistilBertForTokenClassification.from_pretrained(model_path)
            self.model.eval()
            print(f"Loaded DCT model from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            print("Falling back to rule-based classifier")
            self.model = None

    def classify_token(self, token: str, context: str) -> ClassificationResult:
        """Classify a single token."""
        if self.model is None:
            return self.fallback.classify_token(token, context)

        # Neural classification
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        with torch.no_grad():
            outputs = self.model(**{k: v for k, v in inputs.items() if k != "offset_mapping"})
            logits = outputs.logits

        # Find the token position in the tokenized input
        offsets = inputs["offset_mapping"][0]
        token_start = context.find(token)
        
        if token_start == -1:
            return self.fallback.classify_token(token, context)

        # Find which subtoken(s) correspond to our token
        token_end = token_start + len(token)
        relevant_logits = []

        for i, (start, end) in enumerate(offsets):
            if start >= token_start and end <= token_end:
                relevant_logits.append(logits[0, i])

        if not relevant_logits:
            return self.fallback.classify_token(token, context)

        # Average logits across subtokens
        avg_logits = torch.stack(relevant_logits).mean(dim=0)
        probs = torch.softmax(avg_logits, dim=-1)

        is_critical = probs[1].item() > 0.5  # class 1 = critical
        confidence = probs[1].item() if is_critical else probs[0].item()

        return ClassificationResult(
            is_critical=is_critical,
            confidence=confidence,
            category=self._infer_category(token, context) if is_critical else None,
        )

    def _infer_category(self, token: str, context: str) -> str:
        """Infer the category using rule-based patterns (supplement to neural)."""
        result = self.fallback.classify_token(token, context)
        return result.category or "unclassified"

    def classify_sequence(
        self, tokens: list[str], window_size: int = 128
    ) -> list[ClassificationResult]:
        """Classify all tokens in a sequence."""
        if self.model is None:
            return self.fallback.classify_sequence(tokens, window_size)

        full_text = "".join(tokens)
        results = []
        char_pos = 0

        for token in tokens:
            start = max(0, char_pos - window_size * 4)
            end = min(len(full_text), char_pos + len(token) + window_size * 4)
            context = full_text[start:end]

            result = self.classify_token(token, context)
            results.append(result)
            char_pos += len(token)

        return results
