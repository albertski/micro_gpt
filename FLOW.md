# MicroGPT Flow

A GPT-2-style transformer trained from scratch in pure Ruby, with no ML libraries. Every operation — including backpropagation — is implemented by hand. The default use case is training on a list of names (one per line in `input.txt`) and generating new made-up names.

---

## High-Level Flow

```
bin/train → CLI → build_config → load_dataset → Tokenizer → Model → run_training → generate_samples
```

---

## Step 1: Entry Point (`bin/train`)

Calls `MicroGPT::CLI.start(ARGV)`. That's it.

---

## Step 2: CLI (`cli.rb`)

Parses options (steps, temperature, etc.) and orchestrates everything:

1. `build_config` — assembles hyperparameters into a `Config` struct
2. `load_dataset` — reads `input.txt`, shuffles documents
3. `Tokenizer.new` — builds character vocabulary
4. `Model.new` — initializes weights
5. `run_training` — trains the model
6. `generate_samples` — runs inference and prints results

---

## Step 3: Dataset & Tokenizer

**Dataset** reads `input.txt`, splits by line → array of documents (e.g. `["alice", "bob", ...]`).

**Tokenizer** scans all characters, assigns each a unique integer ID. Adds one special `BOS` (beginning-of-sequence) token at the end. If your vocab has 26 letters, token IDs are `0-25` for letters, `26` for BOS.

---

## Step 4: Model Initialization (`model.rb`)

The model is a collection of weight matrices filled with small random values (`Random.gauss`):

- `wte` — token embedding table: maps each token ID → a 16-dim vector
- `wpe` — position embedding table: maps each position (0, 1, 2...) → a 16-dim vector
- Per-layer attention weights (`Wq`, `Wk`, `Wv`, `Wo`) and MLP weights
- `lm_head` — final projection back to vocab size (logits)

At initialization these numbers are random garbage. Training will fix them.

Every single number is a `Value` object (the autograd engine), not a plain float. That's what makes gradients possible later.

---

## Step 5: The Training Loop (`trainer.rb`)

For each step, `train_step` does the following:

1. Pick a document (cycling through dataset), wrap it with BOS tokens:
   ```
   "alice" → [BOS, a, l, i, c, e, BOS] → [26, 0, 11, 8, 2, 4, 26]
   ```
2. Run `compute_loss` — forward pass for each position, measure how wrong the predictions are
3. Call `loss.backward` — backpropagation, computes gradients
4. Call `optimizer.step` — update weights using gradients
5. Call `optimizer.zero_grad` — reset gradients for next step

---

## Step 6: `compute_loss` in Detail

The game: at each position, look at the current token and predict what comes next.

```
pos 0: see BOS(26) → predict a(0)
pos 1: see a(0)    → predict l(11)
pos 2: see l(11)   → predict i(8)
pos 3: see i(8)    → predict c(2)
pos 4: see c(2)    → predict e(4)
pos 5: see e(4)    → predict BOS(26)  ← "I'm done"
```

For each position:

**1. Forward pass → logits**

```ruby
logits = @model.forward(token_id, pos_id, kv_cache)
```

Returns one raw score per vocab token (27 numbers). Random garbage early in training, meaningful later.

**2. Softmax → probabilities**

```ruby
probs = NN.softmax(logits)
```

Converts raw scores into a probability distribution that sums to 1.0.

**3. Cross-entropy loss**

```ruby
losses << -probs[target_id].log
```

Look at only the probability assigned to the correct next token and take its negative log:

- `prob = 1.0` (perfect) → `-log(1.0) = 0` — zero loss
- `prob = 0.5` → `-log(0.5) = 0.69`
- `prob = 0.01` (very wrong) → `-log(0.01) = 4.6` — high loss

The loss gets bigger the more wrong the model is.

**4. Average over all positions**

```ruby
(1.0 / n) * losses.inject(:+)
```

A loss around `log(vocab_size) ≈ 3.3` at the start means the model is guessing randomly. As loss drops, the model assigns higher probability to correct next characters.

---

## Step 7: The Forward Pass (`model.rb#forward`)

Traces one token at one position through the transformer.

**7a. Embedding lookup**

```ruby
tok_emb = wte[token_id]    # "what token am I?"
pos_emb = wpe[pos_id]      # "where am I in the sequence?"
x = tok_emb + pos_emb      # combined 16-dim vector
```

**7b. RMSNorm**

Rescales `x` so values don't explode or vanish. A stabilization step.

**7c. Attention block** (repeated per layer)

```ruby
q = linear(x, Wq)   # "what am I looking for?"
k = linear(x, Wk)   # "what do I offer?"
v = linear(x, Wv)   # "what's my content?"
```

The KV cache stores `k` and `v` for every past position. For each attention head:

```
score = dot(my_query, each_past_key) / sqrt(head_dim)
weights = softmax(scores)
output = sum(weight[t] * value[t] for each past position t)
```

Attention output is a blend of past information, weighted by relevance. A residual connection adds the original `x` back in so the signal isn't lost.

**7d. MLP block** (repeated per layer)

```ruby
x = linear(x, mlp_fc1)   # expand: 16 → 64
x = x.map(&:relu)         # zero out negatives
x = linear(x, mlp_fc2)   # compress: 64 → 16
x = x + x_residual        # residual
```

Attention decides *where* to look. MLP decides *what to do* with what it found.

**7e. Project to logits**

```ruby
logits = linear(x, lm_head)   # 16 → vocab_size
```

One score per token in the vocabulary. Higher = model thinks that token is more likely next.

---

## Step 8: The Autograd Engine (`value.rb`)

Every number in the model is a `Value` object. Operations like `+`, `*`, `exp`, `log` don't just compute a result — they record the computation graph:

```ruby
a = Value.new(2.0)
b = Value.new(3.0)
c = a * b   # c.data = 6.0
            # c.children = [a, b]
            # c.local_grads = [3.0, 2.0]  ← d(a*b)/da=b, d(a*b)/db=a
```

After the entire forward pass, the loss `Value` sits at the top of a massive computation graph — every node connected back to every weight in the model.

`loss.backward` walks that graph in reverse, applying the chain rule at every node:

```ruby
@grad = 1.0   # loss affects itself 100%

topo.reverse_each do |v|
  v.children.zip(v.local_grads).each do |child, local_grad|
    child.grad += local_grad * v.grad
  end
end
```

After `backward`, every weight has a `.grad` — how much the loss changes if you nudge that weight.

---

## Step 9: The Optimizer (`optimizer.rb`)

Adam uses gradients to update each weight:

```ruby
# Smoothed gradient (momentum)
@m[i] = 0.85 * @m[i] + 0.15 * param.grad

# Smoothed squared gradient (adaptive scaling)
@v[i] = 0.99 * @v[i] + 0.01 * param.grad**2

# Bias-corrected versions
m_hat = @m[i] / (1 - 0.85 ** (step + 1))
v_hat = @v[i] / (1 - 0.99 ** (step + 1))

# Update the weight
param.data -= lr * m_hat / (sqrt(v_hat) + 1e-8)
```

`param.data` is directly mutated — only the float changes, not the computation graph.

The `v` term in the denominator makes updates smaller for weights with large gradients (already moving fast) and larger for weights that haven't moved much. This helps all parameters learn at a useful pace.

`zero_grad` resets every `.grad` to 0.0 after each step so gradients don't accumulate across steps.

---

## Step 10: Sampling / Inference (`sampler.rb`)

After training, generate new text one character at a time:

```ruby
token_id = bos_id   # start signal
collected = []

16.times do |pos_id|
  logits   = model.forward(token_id, pos_id, kv_cache)
  probs    = softmax(logits / temperature)
  token_id = weighted_random_choice(probs)

  break if tokenizer.bos?(token_id)   # model signaled "I'm done"
  collected << token_id
end

tokenizer.decode(collected)   # [0, 11, 8, 2] → "alic"
```

**Temperature** controls randomness:
- Low (0.1) — almost always picks the highest-probability token → safe but repetitive
- High (1.5) — spreads probability more evenly → creative but potentially nonsense
- Default (0.5) — in between

The model feeds its own output back as the next input, generating one character at a time until it emits BOS again (meaning "end of name").

---

## The Full Cycle Per Training Step

```
compute_loss   → builds computation graph, returns loss Value
backward       → fills .grad on every weight (chain rule)
optimizer.step → nudges .data on every weight using .grad
zero_grad      → clears .grad so next step starts clean
```

After 1000 steps, each weight has been nudged thousands of times, each time slightly reducing the loss, until the model has learned the character patterns in the training data.

---

## One-Sentence Summary

The model is a pile of numbers (weights). Training adjusts those numbers so the model gets better at predicting the next character. Autograd (`Value`) makes it possible to compute exactly how each weight should change. After training, we run the model forwards-only to generate new text one character at a time.
