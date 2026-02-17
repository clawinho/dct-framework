#!/usr/bin/env python3
"""
Synthetic DCT Training Data Generator.

Generates labeled training data by running conversations through an LLM
twice: once with tools/context (Run B) and once without (Run A).
Where responses diverge = decision-critical tokens.

Usage:
    python scripts/generate_data.py --count 1000 --output data/synthetic_train.jsonl
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from openai import OpenAI


# --- Scenario Templates ---

IDENTITY_SCENARIOS = [
    {
        "known_facts": {"user_{id}": "admin", "user_{id2}": "viewer"},
        "system": "You are an assistant with access to a user database. Use the lookup_user tool to check user roles.",
        "questions": [
            "Is user_{id3} an admin?",
            "What role does user_{id} have?",
            "Can user_{id2} delete files?",
        ],
    },
    {
        "known_facts": {"handle_{rand}": "owner account", "handle_{rand2}": "regular user"},
        "system": "You manage user permissions. Always verify identity using the lookup_user tool before making claims.",
        "questions": [
            "Is handle_{rand} the system owner?",
            "Does handle_{rand3} have elevated access?",
            "Who is handle_{rand2}?",
        ],
    },
]

PERMISSION_SCENARIOS = [
    {
        "known_facts": {"project_alpha": "read-only for viewers", "project_beta": "admin-only"},
        "system": "You manage project access. Use check_permissions tool to verify access levels.",
        "questions": [
            "Can a viewer edit project_alpha?",
            "Who can access project_beta?",
            "Grant user_5 write access to project_alpha",
        ],
    },
]

FACTUAL_SCENARIOS = [
    {
        "known_facts": {"meeting_standup": "9:00 AM Tuesday", "meeting_review": "2:00 PM Friday"},
        "system": "You are a scheduling assistant. Use check_calendar tool to verify meeting times.",
        "questions": [
            "When is the standup meeting?",
            "Is there a meeting on Wednesday?",
            "What time is the review meeting?",
        ],
    },
    {
        "known_facts": {"customer_7291": "enterprise tier, retention team", "customer_1044": "free tier"},
        "system": "You are a support agent. Use lookup_customer tool to check account details before responding.",
        "questions": [
            "What tier is customer 7291 on?",
            "Can customer 1044 access premium features?",
            "I need to cancel enterprise account 7291",
        ],
    },
]

ALL_SCENARIOS = IDENTITY_SCENARIOS + PERMISSION_SCENARIOS + FACTUAL_SCENARIOS

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_user",
            "description": "Look up a user's role, permissions, and account details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "User handle, ID, or name"}
                },
                "required": ["identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_permissions",
            "description": "Check what permissions a user or role has for a resource.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource": {"type": "string"},
                    "user_or_role": {"type": "string"},
                },
                "required": ["resource"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_calendar",
            "description": "Look up meeting times, events, and scheduling information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Look up customer account details, tier, and policies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"}
                },
                "required": ["customer_id"],
            },
        },
    },
]


def randomize_scenario(scenario: dict) -> dict:
    """Fill in random IDs and handles."""
    s = json.dumps(scenario)
    replacements = {
        "{id}": str(random.randint(100, 999)),
        "{id2}": str(random.randint(100, 999)),
        "{id3}": str(random.randint(100, 999)),
        "{rand}": "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8)),
        "{rand2}": "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8)),
        "{rand3}": "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8)),
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return json.loads(s)


def generate_pair(
    client: OpenAI,
    scenario: dict,
    question: str,
    model: str = "gpt-4o-mini",
) -> dict | None:
    """
    Generate a single training example by running with and without tools.
    
    Returns dict with:
        - messages: input messages
        - response_with_tools: response when tools available
        - response_without_tools: response when no tools
        - used_tool: whether the model actually called a tool
        - question: the question asked
        - known_facts: ground truth
    """
    messages = [
        {"role": "system", "content": scenario["system"]},
        {"role": "user", "content": question},
    ]

    try:
        # Run A: Without tools (model must guess)
        response_a = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=256,
            logprobs=True,
            top_logprobs=5,
        )

        # Run B: With tools available
        response_b = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=0.7,
            max_tokens=256,
            logprobs=True,
            top_logprobs=5,
        )

        # Check if model used tools in Run B
        choice_b = response_b.choices[0]
        used_tool = choice_b.finish_reason == "tool_calls"

        # Extract logprobs from Run A (the "without tools" run)
        tokens_a = []
        if response_a.choices[0].logprobs and response_a.choices[0].logprobs.content:
            for tl in response_a.choices[0].logprobs.content:
                alts = []
                if tl.top_logprobs:
                    alts = [
                        {"token": a.token, "logprob": a.logprob}
                        for a in tl.top_logprobs
                        if a.token != tl.token
                    ]
                tokens_a.append({
                    "token": tl.token,
                    "logprob": tl.logprob,
                    "top_alternatives": alts[:4],
                })

        return {
            "messages": messages,
            "question": question,
            "known_facts": scenario.get("known_facts", {}),
            "response_without_tools": response_a.choices[0].message.content or "",
            "response_with_tools": choice_b.message.content or "",
            "used_tool": used_tool,
            "tool_calls": [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (choice_b.message.tool_calls or [])
            ],
            "tokens_without_tools": tokens_a,
            "model": model,
        }

    except Exception as e:
        print(f"  Error generating pair: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic DCT training data")
    parser.add_argument("--count", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="data/synthetic_train.jsonl")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client_kwargs = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    client = OpenAI(**client_kwargs)

    print(f"Generating {args.count} synthetic examples using {args.model}")
    print(f"Output: {args.output}")

    generated = 0
    tool_used_count = 0

    with open(output_path, "w") as f:
        while generated < args.count:
            scenario = randomize_scenario(random.choice(ALL_SCENARIOS))
            question = random.choice(scenario.get("questions", ["Tell me about this."]))

            result = generate_pair(client, scenario, question, model=args.model)
            if result is None:
                continue

            f.write(json.dumps(result) + "\n")
            generated += 1
            if result["used_tool"]:
                tool_used_count += 1

            if generated % 10 == 0:
                print(f"  {generated}/{args.count} generated ({tool_used_count} used tools)")

            # Rate limiting
            time.sleep(0.1)

    print(f"\nDone! {generated} examples written to {args.output}")
    print(f"  Tool usage rate: {tool_used_count}/{generated} ({tool_used_count/generated:.1%})")


if __name__ == "__main__":
    main()
