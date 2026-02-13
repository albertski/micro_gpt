# frozen_string_literal: true

module MicroGPT
  # Generates text samples autoregressively from a trained model.
  class Sampler
    def initialize(model:, tokenizer:, config:)
      @model = model
      @tokenizer = tokenizer
      @config = config
    end

    # Generates a single sample string.
    def generate(temperature: nil)
      temp = temperature || @config.temperature
      kv_cache = @model.new_kv_cache
      token_id = @tokenizer.bos_id
      collected = []

      @config.block_size.times do |pos_id|
        logits = @model.forward(token_id, pos_id, kv_cache)
        probs  = NN.softmax(logits.map { |l| l / temp })
        token_id = Random.weighted_choice(probs.map(&:data))

        break if @tokenizer.bos?(token_id)

        collected << token_id
      end

      @tokenizer.decode(collected)
    end

    # Generates n samples. Defaults to config.num_samples.
    def generate_batch(n: nil, temperature: nil)
      count = n || @config.num_samples
      Array.new(count) { generate(temperature: temperature) }
    end
  end
end
