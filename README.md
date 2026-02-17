# DCT Framework

**Decision-Critical Token Classification for Confidence-Aware Tool Calling in LLMs**

A middleware framework that monitors LLM output streams in real-time, identifies tokens where the model makes high-stakes decisions with low confidence, and triggers automated retrieval and re-generation.

## Status

Pre-experimental. This repo contains:
- Synthetic data generation pipeline
- DCT classifier training code
- Evaluation benchmark (1,000 prompts)
- Per-token logprob extraction for OpenAI-compatible APIs
- Ablation study runner

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY=sk-...

# Generate synthetic training data (1K examples)
python scripts/generate_data.py --count 1000 --output data/synthetic_train.jsonl

# Train the DCT classifier
python scripts/train_classifier.py --data data/synthetic_train.jsonl --output models/dct-v0.1

# Run the benchmark (1K prompts, extracts per-token logprobs)
python scripts/run_benchmark.py --model gpt-4o --output results/benchmark.jsonl

# Evaluate: precision, recall, ROC, ablations
python scripts/evaluate.py --results results/benchmark.jsonl --output results/report/
```

## Architecture

See `docs/whitepaper-v0.3.pdf` for full details.

```
Token Stream + Logprobs
        |
   [DCT Classifier] -- Is this token decision-critical?
        |
   [Logprob Check]  -- Is the model uncertain here?
        |
   [Intervention]   -- Retrieve context, force tool call, regenerate
```

## Per-Token Logprob Extraction

The key building block. `dct/logprob_extractor.py` wraps any OpenAI-compatible API and returns per-token logprobs alongside DCT classification:

```python
from dct import LogprobStream

stream = LogprobStream(
    model="gpt-4o",
    api_key="sk-...",
)

for token in stream.generate(messages, tools):
    print(f"{token.text:15s} logprob={token.logprob:+.3f} "
          f"critical={token.is_critical} top_alt={token.top_alternative}")
```

## Repo Structure

```
dct-framework/
  dct/                    # Core library
    __init__.py
    logprob_extractor.py  # Per-token logprob streaming
    classifier.py         # DCT token classifier
    intervention.py       # Intervention engine
    filters.py            # Epistemic vs stylistic filters
  scripts/                # CLI tools
    generate_data.py      # Synthetic data pipeline
    train_classifier.py   # Train distilBERT classifier
    run_benchmark.py      # Run 1K prompt benchmark
    evaluate.py           # Generate evaluation report
  data/                   # Training data (generated)
  models/                 # Trained models
  results/                # Benchmark results
  docs/                   # Whitepaper
  tests/                  # Unit tests
```

## Authors

- **Clawinho** — architecture, implementation
- **Erwin AI** — review, evaluation design

## License

MIT
