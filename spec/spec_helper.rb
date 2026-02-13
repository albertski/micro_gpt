# frozen_string_literal: true

require_relative "../lib/micro_gpt"

RSpec.configure do |config|
  config.before(:suite) { srand(42) }
  config.order = :defined
end

def tiny_config(**overrides)
  MicroGPT::Config.new(
    n_embd: 4, n_head: 2, n_layer: 1, block_size: 4,
    num_steps: 5, num_samples: 2, **overrides
  )
end

def tiny_docs
  %w[abc bca cab]
end
