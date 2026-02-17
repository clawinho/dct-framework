"""
DCT Intervention Engine.

Produces complete recovery plans when the classifier + logprob evaluator
flag a decision-critical token with low confidence.

Three action types:
- pass: continue streaming
- pause_and_retrieve: inject context, resume
- abort_and_redo: kill generation, force tool call, regenerate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievalPlan:
    category: str
    query: str
    sources: list[str] = field(default_factory=list)


@dataclass
class Instruction:
    type: str  # "context_inject", "windowed_rewrite", "full_regenerate"
    template: str = ""
    force_tool_calls: list[str] = field(default_factory=list)
    position: str = "prepend"  # where to inject context


@dataclass
class InterventionPlan:
    action: str  # "pass", "pause_and_retrieve", "abort_and_redo"
    retrieval: Optional[RetrievalPlan] = None
    instruction: Optional[Instruction] = None
    reason: str = ""


# Category -> default retrieval sources
CATEGORY_SOURCES = {
    "identity": ["identity", "contacts", "users"],
    "permission": ["permissions", "policies", "roles"],
    "factual": ["knowledge_base", "documents"],
    "tool_refusal": ["knowledge_base"],
    "negation": ["knowledge_base"],
}

# Category -> risk level determines intervention severity
CATEGORY_RISK = {
    "identity": "critical",
    "permission": "critical",
    "tool_refusal": "high",
    "factual": "high",
    "negation": "medium",
}


class InterventionEngine:
    """
    Produces intervention plans based on DCT classification + confidence signals.
    
    Implements tiered strategy:
    - Critical categories + low confidence -> abort_and_redo
    - High categories + low confidence -> pause_and_retrieve
    - Medium categories + low confidence -> pause_and_retrieve (context inject only)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        critical_threshold: float = 0.4,
    ):
        """
        Args:
            confidence_threshold: Below this, trigger pause_and_retrieve.
            critical_threshold: Below this for critical categories, trigger abort_and_redo.
        """
        self.confidence_threshold = confidence_threshold
        self.critical_threshold = critical_threshold

    def decide(
        self,
        is_critical: bool,
        category: Optional[str],
        logprob: float,
        is_epistemic: bool,
        context_snippet: str = "",
    ) -> InterventionPlan:
        """
        Produce an intervention plan.
        
        Args:
            is_critical: Whether the token is decision-critical.
            category: The DCT category.
            logprob: Token log probability.
            is_epistemic: Whether the uncertainty is epistemic (not stylistic).
            context_snippet: Text around the flagged token for query generation.
        
        Returns:
            InterventionPlan with action and recovery instructions.
        """
        import math
        confidence = math.exp(logprob)

        # Not critical -> pass
        if not is_critical:
            return InterventionPlan(action="pass", reason="token not decision-critical")

        # Critical but not epistemic uncertainty -> pass
        if not is_epistemic:
            return InterventionPlan(
                action="pass",
                reason="low logprob is non-epistemic (stylistic/entity)",
            )

        risk = CATEGORY_RISK.get(category, "medium")
        sources = CATEGORY_SOURCES.get(category, ["knowledge_base"])

        # Generate retrieval query from context
        query = self._generate_query(context_snippet, category)

        # Critical risk + very low confidence -> abort and redo
        if risk == "critical" and confidence < self.critical_threshold:
            return InterventionPlan(
                action="abort_and_redo",
                retrieval=RetrievalPlan(
                    category=category or "unknown",
                    query=query,
                    sources=sources,
                ),
                instruction=Instruction(
                    type="full_regenerate",
                    template="MANDATORY: Verify {category} claim before responding. Retrieved context: {{retrieved_facts}}",
                    force_tool_calls=["memory_search", "lookup_user"]
                    if category == "identity"
                    else ["knowledge_search"],
                ),
                reason=f"critical category '{category}' with confidence {confidence:.1%}",
            )

        # Below threshold -> pause and retrieve
        if confidence < self.confidence_threshold:
            return InterventionPlan(
                action="pause_and_retrieve",
                retrieval=RetrievalPlan(
                    category=category or "unknown",
                    query=query,
                    sources=sources,
                ),
                instruction=Instruction(
                    type="context_inject",
                    template="VERIFIED CONTEXT: {{retrieved_facts}}\n\nRe-evaluate your previous assertion.",
                    position="prepend",
                ),
                reason=f"category '{category}' with confidence {confidence:.1%} below threshold {self.confidence_threshold:.1%}",
            )

        # Above threshold -> pass
        return InterventionPlan(
            action="pass",
            reason=f"confidence {confidence:.1%} above threshold",
        )

    def _generate_query(self, context: str, category: Optional[str]) -> str:
        """Generate a retrieval query from context snippet."""
        # Simple extraction: take key nouns/entities from context
        # In production, this could use a lightweight NER or keyword extractor
        words = context.split()
        
        # Filter to likely meaningful words (length > 3, not stopwords)
        stopwords = {
            "this", "that", "with", "from", "they", "them", "their",
            "have", "been", "were", "will", "would", "could", "should",
            "about", "into", "your", "does", "also", "just", "than",
            "more", "some", "such", "when", "what", "which", "there",
            "very", "after", "before",
        }
        
        keywords = [
            w.strip(".,!?;:'\"()[]{}") for w in words
            if len(w) > 3 and w.lower() not in stopwords
        ][:8]

        prefix = f"{category} " if category else ""
        return prefix + " ".join(keywords)
