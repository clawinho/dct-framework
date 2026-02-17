"""
Epistemic vs. Stylistic Uncertainty Filters.

Differentiates genuine epistemic uncertainty (model doesn't know)
from non-epistemic low logprobs (stylistic variation, syntax branching,
multi-token entities).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FilterResult:
    is_epistemic: bool
    reason: Optional[str] = None
    adjusted_confidence: Optional[float] = None


# Token groups that are semantically equivalent
SEMANTIC_EQUIVALENTS = [
    {"isn't", "is not", "is n't"},
    {"aren't", "are not", "are n't"},
    {"wasn't", "was not", "was n't"},
    {"can't", "cannot", "can not", "can n't"},
    {"don't", "do not", "do n't"},
    {"doesn't", "does not", "does n't"},
    {"won't", "will not", "will n't"},
    {"wouldn't", "would not", "would n't"},
    {"shouldn't", "should not", "should n't"},
    {"couldn't", "could not", "could n't"},
    {"haven't", "have not", "have n't"},
    {"hasn't", "has not", "has n't"},
    {"hadn't", "had not", "had n't"},
    {"I'm", "I am"},
    {"you're", "you are"},
    {"they're", "they are"},
    {"we're", "we are"},
    {"it's", "it is"},
    {"that's", "that is"},
]

# Build lookup: token -> set of equivalents
_EQUIV_LOOKUP: dict[str, set[str]] = {}
for group in SEMANTIC_EQUIVALENTS:
    for token in group:
        _EQUIV_LOOKUP[token.lower()] = group


class EpistemicFilter:
    """
    Multi-signal filter to distinguish epistemic uncertainty from stylistic variation.
    
    Implements four filtering approaches:
    1. Semantic equivalence check
    2. Entropy margin analysis
    3. Context-category gating
    4. Multi-token entity detection
    """

    def __init__(
        self,
        margin_threshold: float = 0.3,
        high_stakes_categories: Optional[set[str]] = None,
    ):
        self.margin_threshold = margin_threshold
        self.high_stakes_categories = high_stakes_categories or {
            "identity", "permission"
        }

    def filter(
        self,
        token_text: str,
        logprob: float,
        top_alternatives: list[dict],
        category: Optional[str] = None,
        context: str = "",
    ) -> FilterResult:
        """
        Determine if a low-logprob decision-critical token represents
        genuine epistemic uncertainty.
        
        Args:
            token_text: The token text.
            logprob: The token's log probability.
            top_alternatives: List of {"token": str, "logprob": float}.
            category: DCT category (identity, permission, etc.)
            context: Surrounding text for entity detection.
        
        Returns:
            FilterResult indicating whether uncertainty is epistemic.
        """
        # Filter 1: Semantic equivalence
        equiv_result = self._check_semantic_equivalence(token_text, top_alternatives)
        if not equiv_result.is_epistemic:
            return equiv_result

        # Filter 2: Entropy margin
        margin_result = self._check_margin(logprob, top_alternatives)
        
        # Filter 3: Context-category gating
        # High-stakes categories have lower bar for triggering
        if category in self.high_stakes_categories:
            # For identity/permission, even moderate uncertainty should trigger
            return FilterResult(
                is_epistemic=True,
                reason=f"high-stakes category '{category}' with uncertainty",
                adjusted_confidence=None,
            )

        # For non-high-stakes, require stronger uncertainty signal
        if not margin_result.is_epistemic:
            return margin_result

        # Filter 4: Multi-token entity detection
        entity_result = self._check_entity_boundary(token_text, context)
        if not entity_result.is_epistemic:
            return entity_result

        return FilterResult(
            is_epistemic=True,
            reason="passed all filters: genuine epistemic uncertainty",
        )

    def _check_semantic_equivalence(
        self, token_text: str, top_alternatives: list[dict]
    ) -> FilterResult:
        """
        Check if alternatives are semantically equivalent.
        
        "isn't" vs "is not" = stylistic, not epistemic.
        "isn't" vs "is" = genuine semantic difference.
        """
        token_lower = token_text.strip().lower()
        equiv_group = _EQUIV_LOOKUP.get(token_lower)

        if equiv_group and top_alternatives:
            alt_token = top_alternatives[0].get("token", "").strip().lower()
            if alt_token in equiv_group:
                return FilterResult(
                    is_epistemic=False,
                    reason=f"stylistic variation: '{token_text}' vs '{alt_token}' are semantic equivalents",
                )

        return FilterResult(is_epistemic=True)

    def _check_margin(
        self, logprob: float, top_alternatives: list[dict]
    ) -> FilterResult:
        """
        Check the margin between top-1 and top-2 logprobs.
        
        Large margin + low logprob = model confident in meaning, low surface probability.
        Small margin + low logprob = genuine indecision.
        """
        if not top_alternatives:
            return FilterResult(is_epistemic=True, reason="no alternatives available")

        alt_logprob = top_alternatives[0].get("logprob", float("-inf"))
        margin = logprob - alt_logprob

        if margin > self.margin_threshold:
            return FilterResult(
                is_epistemic=False,
                reason=f"large margin ({margin:+.3f}): model confident despite low absolute logprob",
            )

        return FilterResult(
            is_epistemic=True,
            reason=f"small margin ({margin:+.3f}): genuine indecision between alternatives",
        )

    def _check_entity_boundary(
        self, token_text: str, context: str
    ) -> FilterResult:
        """
        Check if the token is part of a multi-token entity.
        
        Proper nouns, URLs, technical terms have naturally low per-token
        probabilities due to composition, not uncertainty.
        """
        token_stripped = token_text.strip()

        # Simple heuristic: capitalized mid-sentence likely entity
        if token_stripped and token_stripped[0].isupper():
            # Find position in context
            pos = context.find(token_text)
            if pos > 0:
                preceding = context[:pos].rstrip()
                # If preceded by another capitalized word (not sentence start),
                # likely multi-token entity
                if preceding and not preceding.endswith((".", "!", "?", "\n")):
                    words = preceding.split()
                    if words and words[-1][0].isupper():
                        return FilterResult(
                            is_epistemic=False,
                            reason=f"multi-token entity boundary: '{token_text}' follows capitalized word",
                        )

        # URLs and technical strings
        if any(c in token_stripped for c in ["@", "/", ":", ".", "_"]) and len(token_stripped) > 1:
            return FilterResult(
                is_epistemic=False,
                reason=f"technical token/entity: '{token_text}'",
            )

        return FilterResult(is_epistemic=True)
