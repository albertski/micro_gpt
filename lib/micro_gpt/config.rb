# frozen_string_literal: true

module MicroGPT
  # Immutable hyperparameter configuration for the micro GPT model.
  # All fields have sensible defaults matching Karpathy's original.
  Config = Data.define(
    :n_embd,
    :n_head,
    :n_layer,
    :block_size,
    :learning_rate,
    :beta1,
    :beta2,
    :eps,
    :num_steps,
    :temperature,
    :init_std,
    :num_samples
  ) do
    def initialize(
      n_embd: 16,
      n_head: 4,
      n_layer: 1,
      block_size: 16,
      learning_rate: 0.01,
      beta1: 0.85,
      beta2: 0.99,
      eps: 1e-8,
      num_steps: 1000,
      temperature: 0.5,
      init_std: 0.08,
      num_samples: 20
    )
      super
    end

    def head_dim
      n_embd / n_head
    end
  end
end
