"""
Per-token logprob extraction from OpenAI-compatible APIs.

This is the core building block for DCT evaluation. It streams tokens
with their log probabilities, top alternatives, and metadata needed
for DCT classification and intervention decisions.

Works with: OpenAI, Azure OpenAI, Together AI, any OpenAI-compatible endpoint.
Does NOT work with: Anthropic (no logprobs exposed).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional

from openai import OpenAI


@dataclass
class TokenInfo:
    """A single token with its logprob and DCT metadata."""

    text: str
    logprob: float
    # Top alternative tokens and their logprobs
    top_alternatives: list[dict] = field(default_factory=list)
    # Position in the response
    index: int = 0
    # Timestamp for latency measurement
    timestamp_ms: float = 0.0
    # DCT classification (set by classifier)
    is_critical: bool = False
    critical_category: Optional[str] = None
    # Intervention decision (set by intervention engine)
    intervention: Optional[str] = None

    @property
    def top_alternative(self) -> Optional[str]:
        """The most likely alternative token."""
        if self.top_alternatives:
            return self.top_alternatives[0].get("token", None)
        return None

    @property
    def confidence(self) -> float:
        """Convert logprob to probability (0-1)."""
        import math
        return math.exp(self.logprob)

    @property
    def margin(self) -> float:
        """Margin between top-1 and top-2 logprobs.
        
        Large margin = model strongly prefers this token.
        Small margin = model is genuinely undecided.
        """
        if not self.top_alternatives:
            return float("inf")
        alt_logprob = self.top_alternatives[0].get("logprob", float("-inf"))
        return self.logprob - alt_logprob

    def to_dict(self) -> dict:
        return asdict(self)


class LogprobStream:
    """
    Streams tokens with per-token logprobs from an OpenAI-compatible API.

    Usage:
        stream = LogprobStream(model="gpt-4o", api_key="sk-...")
        
        for token in stream.generate(messages, tools):
            print(f"{token.text:15s} logprob={token.logprob:+.3f} "
                  f"confidence={token.confidence:.1%} "
                  f"margin={token.margin:+.3f} "
                  f"alt={token.top_alternative}")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        top_logprobs: int = 5,
    ):
        self.model = model
        self.top_logprobs = top_logprobs

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def generate(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[TokenInfo]:
        """
        Stream a completion and yield TokenInfo for each token.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Tool definitions (optional).
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Yields:
            TokenInfo with logprob, alternatives, and metadata.
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }

        if tools:
            kwargs["tools"] = tools

        start_time = time.time()
        stream = self.client.chat.completions.create(**kwargs)

        token_index = 0
        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta
            if not delta or not delta.content:
                continue

            # Extract logprob info
            logprobs_data = choice.logprobs
            if logprobs_data and logprobs_data.content:
                for token_logprob in logprobs_data.content:
                    alternatives = []
                    if token_logprob.top_logprobs:
                        for alt in token_logprob.top_logprobs:
                            if alt.token != token_logprob.token:
                                alternatives.append({
                                    "token": alt.token,
                                    "logprob": alt.logprob,
                                })

                    yield TokenInfo(
                        text=token_logprob.token,
                        logprob=token_logprob.logprob,
                        top_alternatives=alternatives[:4],
                        index=token_index,
                        timestamp_ms=(time.time() - start_time) * 1000,
                    )
                    token_index += 1

    def generate_full(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> list[TokenInfo]:
        """Non-streaming version that returns all tokens at once."""
        return list(self.generate(messages, tools, temperature, max_tokens))

    def generate_with_text(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> tuple[str, list[TokenInfo]]:
        """Returns (full_text, [TokenInfo, ...])."""
        tokens = self.generate_full(messages, tools, temperature, max_tokens)
        text = "".join(t.text for t in tokens)
        return text, tokens


def extract_logprobs_batch(
    messages_list: list[list[dict]],
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    tools: Optional[list[dict]] = None,
    show_progress: bool = True,
) -> list[list[TokenInfo]]:
    """
    Extract logprobs for a batch of conversations.
    
    Useful for benchmark runs. Returns list of token lists.
    """
    from tqdm import tqdm

    stream = LogprobStream(model=model, api_key=api_key)
    results = []

    iterator = tqdm(messages_list, desc="Extracting logprobs") if show_progress else messages_list

    for messages in iterator:
        tokens = stream.generate_full(messages, tools=tools)
        results.append(tokens)

    return results


# --- CLI usage ---
if __name__ == "__main__":
    import sys

    print("DCT Logprob Extractor - Demo")
    print("=" * 60)

    stream = LogprobStream(model="gpt-4o")

    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to a user database. You can look up user information using the lookup_user tool."},
        {"role": "user", "content": "Is the user with handle @xK7mQ9 an admin?"},
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_user",
                "description": "Look up a user's role and permissions by their handle.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "handle": {
                            "type": "string",
                            "description": "The user's handle/username",
                        }
                    },
                    "required": ["handle"],
                },
            },
        }
    ]

    print(f"\nPrompt: {messages[-1]['content']}")
    print(f"\nModel: {stream.model}")
    print("-" * 60)
    print(f"{'Token':15s} {'LogProb':>8s} {'Conf%':>7s} {'Margin':>8s} {'Top Alt':15s}")
    print("-" * 60)

    full_text = ""
    low_confidence_tokens = []

    for token in stream.generate(messages, tools=tools):
        full_text += token.text
        flag = ""
        if token.logprob < -0.5:
            flag = " <-- LOW CONFIDENCE"
            low_confidence_tokens.append(token)

        print(
            f"{repr(token.text):15s} {token.logprob:>+8.3f} "
            f"{token.confidence:>6.1%} {token.margin:>+8.3f} "
            f"{repr(token.top_alternative or ''):15s}{flag}"
        )

    print("-" * 60)
    print(f"\nFull response: {full_text}")
    print(f"\nLow confidence tokens ({len(low_confidence_tokens)}):")
    for t in low_confidence_tokens:
        print(f"  [{t.index}] {repr(t.text)} logprob={t.logprob:+.3f} alt={t.top_alternative}")
