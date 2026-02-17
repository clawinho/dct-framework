#!/usr/bin/env python3
"""
DCT Benchmark Runner.

Runs 1,000 agentic prompts through an LLM, extracts per-token logprobs,
runs DCT classification, and records everything for evaluation.

This is the main experiment script. Output is consumed by evaluate.py.

Usage:
    python scripts/run_benchmark.py --model gpt-4o --output results/benchmark.jsonl
    python scripts/run_benchmark.py --model gpt-4o-mini --count 100 --output results/quick.jsonl
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from openai import OpenAI

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dct.logprob_extractor import LogprobStream
from dct.classifier import DCTClassifier, RuleBasedClassifier
from dct.filters import EpistemicFilter
from dct.intervention import InterventionEngine


# --- Benchmark Scenarios ---
# Each scenario has: system prompt, tools, user message, ground_truth (should_use_tool)

def generate_benchmark_prompts(count: int = 1000) -> list[dict]:
    """Generate benchmark prompts with ground truth labels."""
    import random
    random.seed(42)  # Reproducible

    prompts = []

    # Category 1: Identity verification (should use tool)
    identity_should_retrieve = [
        "Is user handle @{rand} an administrator?",
        "What permissions does account {rand} have?",
        "Can you verify if {rand} is the system owner?",
        "Who is the user with ID {rand}?",
        "Does {rand} have delete permissions?",
        "Is {rand} authorized to access the admin panel?",
        "What role does @{rand} have in the system?",
        "Verify the identity of user {rand}",
        "Is {rand} a premium subscriber?",
        "Check if {rand} is on the allowed list",
    ]

    # Category 2: Questions model can answer without tools
    no_tool_needed = [
        "What is the capital of France?",
        "How do I write a for loop in Python?",
        "Explain what an API is",
        "What's the difference between HTTP and HTTPS?",
        "How do I center a div in CSS?",
        "What is machine learning?",
        "Explain the concept of recursion",
        "What are the SOLID principles?",
        "How does TCP/IP work?",
        "What is a REST API?",
    ]

    # Category 3: Permission checks (should use tool)
    permission_should_retrieve = [
        "Can a viewer edit project_{rand}?",
        "Grant user_{rand} write access to the database",
        "Remove admin rights from user_{rand}",
        "Does role 'editor' have access to billing?",
        "What can a 'viewer' role do in project_{rand}?",
        "Escalate user_{rand}'s permissions to admin",
        "Is user_{rand} allowed to export data?",
        "Check if team_{rand} has deploy permissions",
        "Can interns access the production environment?",
        "What permissions does the 'auditor' role have?",
    ]

    # Category 4: Factual lookups (should use tool)
    factual_should_retrieve = [
        "When is the next standup meeting?",
        "What time is the review scheduled for?",
        "Is there a meeting this Wednesday?",
        "What's customer {rand}'s subscription tier?",
        "When does the enterprise contract for account {rand} expire?",
        "What's the current balance for account {rand}?",
        "How many active users do we have?",
        "What was last month's revenue?",
        "When is the next deployment window?",
        "What's the SLA for enterprise customers?",
    ]

    # Build prompt list with labels
    templates = [
        (identity_should_retrieve, True, "identity"),
        (no_tool_needed, False, "general"),
        (permission_should_retrieve, True, "permission"),
        (factual_should_retrieve, True, "factual"),
    ]

    while len(prompts) < count:
        template_list, should_use_tool, category = random.choice(templates)
        question_template = random.choice(template_list)

        question = question_template.replace(
            "{rand}",
            "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6)),
        )

        prompts.append({
            "question": question,
            "should_use_tool": should_use_tool,
            "category": category,
        })

    return prompts[:count]


SYSTEM_PROMPT = """You are a helpful assistant with access to several tools:
- lookup_user: Look up user roles, permissions, and account details
- check_permissions: Verify what a user/role can do for a resource
- check_calendar: Look up meeting times and events
- lookup_customer: Check customer account details and tier

Use these tools when you need to verify information. If you're unsure about
user details, permissions, schedules, or account information, look it up
rather than guessing."""

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
                    "query": {"type": "string"}
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


def run_single(
    client: OpenAI,
    model: str,
    prompt: dict,
    classifier: DCTClassifier,
    epistemic_filter: EpistemicFilter,
    intervention_engine: InterventionEngine,
) -> dict:
    """Run a single benchmark prompt and return full analysis."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt["question"]},
    ]

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=0.7,
            max_tokens=512,
            logprobs=True,
            top_logprobs=5,
        )
    except Exception as e:
        return {"error": str(e), "prompt": prompt}

    latency_ms = (time.time() - start_time) * 1000
    choice = response.choices[0]

    # Did the model use a tool?
    model_used_tool = choice.finish_reason == "tool_calls"
    tool_calls = [
        {"name": tc.function.name, "arguments": tc.function.arguments}
        for tc in (choice.message.tool_calls or [])
    ]

    # Extract per-token logprobs and run DCT
    token_analysis = []
    if choice.logprobs and choice.logprobs.content:
        tokens_text = [tl.token for tl in choice.logprobs.content]
        full_text = "".join(tokens_text)

        for i, tl in enumerate(choice.logprobs.content):
            # Build context window
            start_idx = max(0, i - 32)
            end_idx = min(len(tokens_text), i + 32)
            context = "".join(tokens_text[start_idx:end_idx])

            # Alternatives
            alts = []
            if tl.top_logprobs:
                alts = [
                    {"token": a.token, "logprob": a.logprob}
                    for a in tl.top_logprobs
                    if a.token != tl.token
                ]

            # DCT classification
            cls_result = classifier.classify_token(tl.token, context)

            # Epistemic filter (only if critical)
            filter_result = None
            if cls_result.is_critical:
                filter_result = epistemic_filter.filter(
                    token_text=tl.token,
                    logprob=tl.logprob,
                    top_alternatives=alts,
                    category=cls_result.category,
                    context=context,
                )

            # Intervention decision
            intervention_result = None
            if cls_result.is_critical and filter_result and filter_result.is_epistemic:
                intervention_result = intervention_engine.decide(
                    is_critical=True,
                    category=cls_result.category,
                    logprob=tl.logprob,
                    is_epistemic=True,
                    context_snippet=context,
                )

            token_analysis.append({
                "token": tl.token,
                "logprob": tl.logprob,
                "confidence": math.exp(tl.logprob),
                "top_alternatives": alts[:4],
                "index": i,
                "dct_critical": cls_result.is_critical,
                "dct_category": cls_result.category,
                "dct_confidence": cls_result.confidence,
                "epistemic": filter_result.is_epistemic if filter_result else None,
                "epistemic_reason": filter_result.reason if filter_result else None,
                "intervention": intervention_result.action if intervention_result else "pass",
                "intervention_reason": intervention_result.reason if intervention_result else None,
            })

    # Determine if DCT would have triggered intervention
    dct_would_intervene = any(
        t["intervention"] in ("pause_and_retrieve", "abort_and_redo")
        for t in token_analysis
    )

    return {
        "prompt": prompt,
        "model": model,
        "latency_ms": latency_ms,
        "model_used_tool": model_used_tool,
        "tool_calls": tool_calls,
        "response_text": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "token_count": len(token_analysis),
        "token_analysis": token_analysis,
        "dct_would_intervene": dct_would_intervene,
        "ground_truth_should_use_tool": prompt["should_use_tool"],
        "category": prompt["category"],
        # Key metrics
        "missed_tool_call": prompt["should_use_tool"] and not model_used_tool,
        "unnecessary_tool_call": not prompt["should_use_tool"] and model_used_tool,
        "dct_correct_intervention": (
            prompt["should_use_tool"] and not model_used_tool and dct_would_intervene
        ),
        "dct_false_positive": (
            (not prompt["should_use_tool"] or model_used_tool) and dct_would_intervene
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Run DCT benchmark")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--output", type=str, default="results/benchmark.jsonl")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--dct-model", type=str, default=None, help="Path to trained DCT model (uses rule-based if not set)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize
    client_kwargs = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    client = OpenAI(**client_kwargs)

    classifier = DCTClassifier(model_path=args.dct_model)
    epistemic_filter = EpistemicFilter()
    intervention_engine = InterventionEngine()

    # Generate prompts
    prompts = generate_benchmark_prompts(args.count)
    print(f"Running {len(prompts)} benchmark prompts on {args.model}")
    print(f"Output: {args.output}")
    print(f"DCT classifier: {'trained model' if args.dct_model else 'rule-based fallback'}")

    # Run
    results = []
    missed = 0
    caught = 0

    with open(output_path, "w") as f:
        for i, prompt in enumerate(prompts):
            result = run_single(
                client, args.model, prompt,
                classifier, epistemic_filter, intervention_engine,
            )

            if "error" not in result:
                f.write(json.dumps(result) + "\n")
                results.append(result)

                if result["missed_tool_call"]:
                    missed += 1
                if result["dct_correct_intervention"]:
                    caught += 1

            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(prompts)} | missed: {missed} | caught by DCT: {caught}")

            time.sleep(0.2)  # Rate limiting

    # Summary
    total = len(results)
    total_missed = sum(1 for r in results if r["missed_tool_call"])
    total_caught = sum(1 for r in results if r["dct_correct_intervention"])
    total_fp = sum(1 for r in results if r["dct_false_positive"])
    avg_latency = sum(r["latency_ms"] for r in results) / total if total else 0

    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE: {total} prompts on {args.model}")
    print(f"{'='*60}")
    print(f"Baseline missed tool calls: {total_missed}/{total} ({total_missed/total:.1%})")
    print(f"DCT caught:                 {total_caught}/{total_missed} ({total_caught/total_missed:.1%} recall)" if total_missed else "DCT caught: N/A (no misses)")
    print(f"DCT false positives:        {total_fp}/{total} ({total_fp/total:.1%})")
    print(f"Average latency:            {avg_latency:.0f}ms")

    if total_caught + total_fp > 0:
        precision = total_caught / (total_caught + total_fp)
        print(f"DCT precision:              {precision:.1%}")

    if total_missed > 0:
        recall = total_caught / total_missed
        print(f"DCT recall:                 {recall:.1%}")


if __name__ == "__main__":
    main()
