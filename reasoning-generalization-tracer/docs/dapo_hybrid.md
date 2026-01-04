# DAPO Hybrid Training (GEPA + GRN)

This repository integrates the `gepa-dapo-grn` package to train Reasoning/Coding models
with GEPA feedback vectors and DAPO-style policy optimization. The integration is
adapter-based: rg-tracer provides lightweight wrappers that translate local scoring output
into `GEPAFeedback` objects and wire them into the DAPO training loop.

## What `gepa-dapo-grn` provides

The external package supplies:

- DAPO trainer and config objects (`DAPOTrainer`, `DAPOConfig`).
- GRN configuration and safety/curriculum controllers.
- `GEPAFeedback` interface consumed by the trainer.

The rg-tracer integration **imports** these interfaces and does not copy code from the
upstream package.

## How rg-tracer uses it

Adapters live in `src/rg_tracer/dapo/`:

- `hf_policy_adapter.py` wraps a HuggingFace causal LM into the `Policy` interface.
- `feedback_adapter.py` maps local scoring metrics into `GEPAFeedback`.
- `dapo_hybrid_trainer.py` orchestrates sampling, scoring, feedback mapping, and training.
- `logging.py` emits JSONL logs with full GEPA vectors and RL metrics.

## Integration contract

Your pipeline should satisfy the following contract:

1. **Scorer** returns `local_metrics` plus metadata.
2. **Adapter** maps `local_metrics` into `GEPAFeedback` via `FeedbackMappingConfig`.
3. **DAPOTrainer** consumes `GEPAFeedback` and returns RL metrics for logging.

> **Note:** The DAPO trainer expects `GEPAFeedback` objects, so any custom scorer only needs
> to return a dict of floats plus the relevant IDs for task/prompt tracking.

## Custom scorers

Implement a scorer with a simple interface:

```python
class MyScorer:
    def score(self, prompts, completions):
        return [
            {
                "correctness": 1.0,
                "reasoning_quality": 0.8,
                "format_penalty": 0.0,
            }
            for _ in completions
        ]
```

Then map metrics to GEPA keys using `FeedbackMappingConfig`:

```python
from rg_tracer.dapo import FeedbackMappingConfig

cfg = FeedbackMappingConfig(
    reward_keys={
        "correctness": "correctness",
        "reasoning_quality": "reasoning_quality",
    },
    tag_keys={"format_penalty": "format_penalty"},
)
```

## Reward mixer configuration

Provide reward mixer weights in YAML/JSON (passed to `RewardMixerConfig`):

```yaml
correctness: 1.0
reasoning_quality: 0.5
format_penalty: -0.25
```

## Running hybrid training

Use the CLI entrypoint:

```bash
python scripts/train_dapo_hybrid.py \
  --model gpt2 \
  --dataset datasets/toy_math/addition_small.jsonl \
  --output-dir runs/dapo_hybrid \
  --batch-size 4 \
  --group-size 2 \
  --learning-rate 1e-5 \
  --clip-ratio 0.2 \
  --kl-target 0.1 \
  --kl-coef 0.1 \
  --reward-mixer configs/reward_mixer.yaml \
  --eval-every 100 \
  --seed 42
```

## Minimal CPU example

Run a mocked example that finishes in seconds:

```bash
python examples/dapo_hybrid_minimal.py
```

It writes JSONL logs to `examples/dapo_hybrid_minimal.jsonl`.

## Grok/EGGROLL analysis

The JSONL logs contain full GEPA vectors and core RL metrics. Use them as fitness signals
or analysis traces for grokking experiments:

- `gepa[].rewards` and `gepa[].tags` capture the full reward/tag vectors per sample.
- `rl` records loss, policy loss, KL, clip ratio, LR, and grad norm.
- `generation` captures prompt/completion IDs and lengths.

These records can be ingested to correlate circuit-level shifts with specific reward axes.
