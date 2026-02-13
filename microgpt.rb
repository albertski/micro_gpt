#!/usr/bin/env ruby
# frozen_string_literal: true

# microGPT.rb - A minimal GPT (transformer language model) in pure Ruby
# Ported from Andrej Karpathy's microGPT Python implementation (~243 lines)
# Educational / demonstrative purposes only — extremely inefficient
#
# The most atomic way to train and inference a GPT in pure, dependency-free Ruby.
# This file is the complete algorithm. Everything else is just efficiency.
#
# Translation notes:
#   - Ruby's `coerce` replaces Python's __radd__, __rmul__, __rsub__, __rtruediv__
#   - Box-Muller transform replaces Python's random.gauss (same distribution, different RNG)
#   - `inject(:+)` replaces Python's sum() on Value objects
#   - `def` methods don't close over locals in Ruby, so state_dict is passed explicitly

$stdout.sync = true       # flush output immediately so we see training progress
require 'open-uri'        # stdlib: equivalent to Python's urllib.request

srand(42) # Let there be order among chaos

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Box-Muller transform for Gaussian random numbers (replaces random.gauss)
def gauss(mean = 0.0, stddev = 1.0)
  u1 = rand
  u2 = rand
  z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
  mean + stddev * z
end

# Weighted random sampling by index (replaces random.choices with weights)
def weighted_choice(weights)
  r = rand * weights.sum
  cumulative = 0.0
  weights.each_with_index do |w, i|
    cumulative += w
    return i if r <= cumulative
  end
  weights.size - 1 # floating-point edge-case fallback
end

# -----------------------------------------------------------------------------
# Autograd Engine
# -----------------------------------------------------------------------------

# Let there be Autograd, to recursively apply the chain rule through a computation graph
class Value
  attr_accessor :data, :grad
  attr_reader :children, :local_grads

  def initialize(data, children = [], local_grads = [])
    @data = data.to_f       # scalar value of this node calculated during forward pass
    @grad = 0.0             # derivative of the loss w.r.t. this node, calculated in backward pass
    @children = children    # children of this node in the computation graph
    @local_grads = local_grads # local derivative of this node w.r.t. its children
  end

  def +(other)
    other = other.is_a?(Value) ? other : Value.new(other)
    Value.new(@data + other.data, [self, other], [1.0, 1.0])
  end

  def *(other)
    other = other.is_a?(Value) ? other : Value.new(other)
    Value.new(@data * other.data, [self, other], [other.data, @data])
  end

  def **(other) # other is always a plain number, not a Value
    Value.new(@data**other, [self], [other * @data**(other - 1)])
  end

  def log
    Value.new(Math.log(@data), [self], [1.0 / @data])
  end

  def exp
    e = Math.exp(@data)
    Value.new(e, [self], [e])
  end

  def relu
    Value.new(@data > 0 ? @data : 0.0, [self], [@data > 0 ? 1.0 : 0.0])
  end

  def -@
    self * -1
  end

  def -(other)
    self + (-other)
  end

  def /(other)
    self * (other**-1)
  end

  # Enables `number <op> value` (e.g. `5 + value`, `2.0 * value`).
  # Ruby calls value.coerce(number) -> [Value(number), value],
  # then evaluates Value(number) <op> value. This single method replaces
  # Python's __radd__, __rmul__, __rsub__, __rtruediv__.
  def coerce(other)
    [Value.new(other), self]
  end

  def backward
    # Topological sort of the computation graph
    topo = []
    visited = {}

    build_topo = lambda do |v|
      unless visited.key?(v)
        visited[v] = true
        v.children.each { |child| build_topo.call(child) }
        topo << v
      end
    end

    build_topo.call(self)
    @grad = 1.0

    # Propagate gradients in reverse topological order
    topo.reverse_each do |v|
      v.children.zip(v.local_grads).each do |child, local_grad|
        child.grad += local_grad * v.grad
      end
    end
  end
end

# -----------------------------------------------------------------------------
# Model primitives
# -----------------------------------------------------------------------------

# Follow GPT-2, blessed among the GPTs, with minor differences:
#   layernorm -> rmsnorm, no biases, GeLU -> ReLU

def linear(x, w)
  w.map { |wo| wo.zip(x).map { |wi, xi| wi * xi }.inject(:+) }
end

def softmax(logits)
  max_val = logits.map(&:data).max
  exps = logits.map { |val| (val - max_val).exp }
  total = exps.inject(:+)
  exps.map { |e| e / total }
end

def rmsnorm(x)
  ms = x.map { |xi| xi * xi }.inject(:+) / x.size
  scale = (ms + 1e-5)**-0.5
  x.map { |xi| xi * scale }
end

# The GPT forward pass: maps a single token + position to logits over vocab.
# keys/values are mutated in-place (KV cache for causal attention).
# sd (state_dict) is passed explicitly because Ruby `def` doesn't close over locals.
def gpt(token_id, pos_id, keys, values, sd, n_layer, n_head, head_dim)
  tok_emb = sd['wte'][token_id] # token embedding
  pos_emb = sd['wpe'][pos_id]   # position embedding
  x = tok_emb.zip(pos_emb).map { |t, p| t + p } # joint token and position embedding

  x = rmsnorm(x)

  n_layer.times do |li|
    # 1) Multi-head attention block
    x_residual = x
    x = rmsnorm(x)

    q = linear(x, sd["layer#{li}.attn_wq"])
    k = linear(x, sd["layer#{li}.attn_wk"])
    v = linear(x, sd["layer#{li}.attn_wv"])

    keys[li] << k
    values[li] << v

    x_attn = []

    n_head.times do |h|
      hs = h * head_dim
      q_h = q[hs, head_dim]
      k_h = keys[li].map { |ki| ki[hs, head_dim] }
      v_h = values[li].map { |vi| vi[hs, head_dim] }

      attn_logits = (0...k_h.size).map { |t|
        (0...head_dim).map { |j| q_h[j] * k_h[t][j] }.inject(:+) / head_dim**0.5
      }
      attn_weights = softmax(attn_logits)

      head_out = (0...head_dim).map { |j|
        (0...v_h.size).map { |t| attn_weights[t] * v_h[t][j] }.inject(:+)
      }

      x_attn.concat(head_out)
    end

    x = linear(x_attn, sd["layer#{li}.attn_wo"])
    x = x.zip(x_residual).map { |a, b| a + b }

    # 2) MLP block
    x_residual = x
    x = rmsnorm(x)

    x = linear(x, sd["layer#{li}.mlp_fc1"])
    x = x.map(&:relu)
    x = linear(x, sd["layer#{li}.mlp_fc2"])
    x = x.zip(x_residual).map { |a, b| a + b }
  end

  linear(x, sd['lm_head'])
end

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

# Let there be an input dataset `docs`: array of strings (e.g. a dataset of names)
unless File.exist?('input.txt')
  names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
  File.write('input.txt', URI.open(names_url).read)
end

docs = File.read('input.txt').strip.split("\n").map(&:strip).reject(&:empty?)
docs.shuffle!
puts "num docs: #{docs.size}"

# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------

# Let there be a Tokenizer to translate strings to discrete symbols and back
uchars = docs.join.chars.uniq.sort # unique characters in the dataset become token ids 0..n-1
bos = uchars.size                  # token id for the special Beginning of Sequence (BOS) token
vocab_size = uchars.size + 1       # total number of unique tokens, +1 is for BOS
puts "vocab size: #{vocab_size}"

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

# Initialize the parameters, to store the knowledge of the model.
n_embd     = 16                  # embedding dimension
n_head     = 4                   # number of attention heads
n_layer    = 1                   # number of layers
block_size = 16                  # maximum sequence length
head_dim   = n_embd / n_head     # dimension of each head

matrix = ->(nout, nin, std = 0.08) {
  Array.new(nout) { Array.new(nin) { Value.new(gauss(0.0, std)) } }
}

sd = {
  'wte'     => matrix.(vocab_size, n_embd),
  'wpe'     => matrix.(block_size, n_embd),
  'lm_head' => matrix.(vocab_size, n_embd),
}

n_layer.times do |i|
  sd["layer#{i}.attn_wq"] = matrix.(n_embd, n_embd)
  sd["layer#{i}.attn_wk"] = matrix.(n_embd, n_embd)
  sd["layer#{i}.attn_wv"] = matrix.(n_embd, n_embd)
  sd["layer#{i}.attn_wo"] = matrix.(n_embd, n_embd)
  sd["layer#{i}.mlp_fc1"] = matrix.(4 * n_embd, n_embd)
  sd["layer#{i}.mlp_fc2"] = matrix.(n_embd, 4 * n_embd)
end

# Flatten params into a single array of Value objects
params = sd.values.flat_map(&:flatten)
puts "num params: #{params.size}"

# -----------------------------------------------------------------------------
# Adam Optimizer
# -----------------------------------------------------------------------------

# Let there be Adam, the blessed optimizer and its buffers
learning_rate = 0.01
beta1         = 0.85
beta2         = 0.99
eps_adam       = 1e-8

m_buf = Array.new(params.size, 0.0) # first moment buffer
v_buf = Array.new(params.size, 0.0) # second moment buffer

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

num_steps = 1000 # number of training steps

num_steps.times do |step|
  # Take single document, tokenize it, surround it with BOS special token on both sides
  doc = docs[step % docs.size]
  tokens = [bos] + doc.chars.map { |ch| uchars.index(ch) } + [bos]
  n = [block_size, tokens.size - 1].min

  # Forward the token sequence through the model, building up the computation graph
  # all the way to the loss.
  keys   = Array.new(n_layer) { [] }
  values = Array.new(n_layer) { [] }
  losses = []

  n.times do |pos_id|
    token_id  = tokens[pos_id]
    target_id = tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys, values, sd, n_layer, n_head, head_dim)
    probs  = softmax(logits)
    loss_t = -probs[target_id].log
    losses << loss_t
  end

  loss = (1.0 / n) * losses.inject(:+) # final average loss. May yours be low.

  # Backward the loss, calculating the gradients with respect to all model parameters.
  loss.backward

  # Adam optimizer update: update the model parameters based on the corresponding gradients.
  lr_t = learning_rate * (1.0 - step.to_f / num_steps) # linear learning rate decay

  params.each_with_index do |p, i|
    m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
    v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad**2

    m_hat = m_buf[i] / (1.0 - beta1**(step + 1))
    v_hat = v_buf[i] / (1.0 - beta2**(step + 1))

    p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
    p.grad = 0.0
  end

  printf("step %4d / %4d | loss %.4f\n", step + 1, num_steps, loss.data)
end

# -----------------------------------------------------------------------------
# Inference: may the model babble back to us
# -----------------------------------------------------------------------------

temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high

puts "\n--- inference (new, hallucinated names) ---"

20.times do |sample_idx|
  keys   = Array.new(n_layer) { [] }
  values = Array.new(n_layer) { [] }
  token_id = bos
  sample = []

  block_size.times do |pos_id|
    logits = gpt(token_id, pos_id, keys, values, sd, n_layer, n_head, head_dim)
    probs  = softmax(logits.map { |l| l / temperature })
    token_id = weighted_choice(probs.map(&:data))

    break if token_id == bos

    sample << uchars[token_id]
  end

  printf("sample %2d: %s\n", sample_idx + 1, sample.join)
end
