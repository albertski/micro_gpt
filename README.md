# MicroGPT

A minimal GPT (transformer language model) in pure Ruby, ported from [Andrej Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) Python implementation (~243 lines).

This is the *full* algorithmic content of a GPT — autograd engine, transformer architecture, Adam optimizer, training loop, and autoregressive inference — implemented from scratch with **zero external dependencies** (only Ruby stdlib).

Educational / demonstrative purposes only — extremely inefficient by design.

## In Simple Terms

It reads a text file of names, learns the character patterns in those names (which letters tend to follow which), and then generates new, made-up names one character at a time based on what it learned. Same architecture as ChatGPT, just with ~4,000 parameters instead of hundreds of billions.

## What's Inside

| Component | Description |
|---|---|
| `Value` | Scalar-valued autograd engine with reverse-mode differentiation |
| `NN` | Neural network primitives: `linear`, `softmax`, `rmsnorm` |
| `Tokenizer` | Character-level tokenizer (unique chars → token ids) |
| `Model` | GPT-2-style transformer (RMSNorm, multi-head attention, ReLU MLP) |
| `AdamOptimizer` | Adam with bias correction and linear LR decay |
| `Trainer` | Training loop with cross-entropy loss |
| `Sampler` | Temperature-controlled autoregressive text generation |
| `Config` | All hyperparameters in one immutable struct |

The default configuration produces a model with **4,192 parameters** — compare that to GPT-4's hundreds of billions. Same architecture, just much smaller.

## Requirements

- Ruby 3.2+
- Bundler (for running tests)

## Quick Start

Install test dependencies:

```sh
bundle install
```

Train the model and generate samples:

```sh
ruby bin/train
```

This will:
1. Load the names dataset from `input.txt` (32k names)
2. Train for 1,000 steps (~minutes on a modern machine)
3. Generate 20 hallucinated names

Use `--steps 50` when running the model for testing to keep it fast.
- Example: `ruby bin/train train --steps 50`

You can also pass a custom dataset file:

```sh
ruby bin/train train path/to/your/data.txt
```

The dataset should be a text file with one document (e.g. name, word, short sentence) per line.

## Running Tests

```sh
bundle exec rspec
```

97 examples covering every class: autograd correctness, gradient propagation, softmax numerical stability, encode/decode roundtrips, optimizer updates, training loss decrease, and sampling determinism.

## Project Structure

```
├── bin/train                    # Runner script
├── input.txt                    # Names dataset (one name per line)
├── lib/
│   ├── micro_gpt.rb             # Top-level module
│   └── micro_gpt/
│       ├── value.rb             # Autograd engine
│       ├── nn.rb                # linear, softmax, rmsnorm
│       ├── random.rb            # Gaussian RNG, weighted sampling
│       ├── config.rb            # Hyperparameters
│       ├── tokenizer.rb         # Character-level tokenizer
│       ├── dataset.rb           # Local file dataset loader
│       ├── model.rb             # GPT model + KV cache
│       ├── optimizer.rb         # Adam optimizer
│       ├── trainer.rb           # Training loop
│       └── sampler.rb           # Text generation
└── spec/                        # RSpec tests for everything
```

## How It Works

The model learns character-level patterns from the dataset. During training, each name is wrapped in BOS (Beginning of Sequence) tokens, fed through the transformer one character at a time, and the model learns to predict the next character. At inference time, it generates new text by sampling from the predicted distribution.

The entire forward and backward pass operates on `Value` objects — scalar floats that track their computation graph. Calling `loss.backward` walks the graph in reverse topological order, applying the chain rule to compute gradients for every parameter. This is the same algorithm (backpropagation) used by PyTorch and TensorFlow, just on individual scalars instead of tensors.

## Credits

Original Python implementation by [Andrej Karpathy](https://karpathy.ai/microgpt.html) — part of a six-year compression arc from micrograd (2020) to microGPT (2026), stripping away every layer of abstraction to reveal the core algorithm.
