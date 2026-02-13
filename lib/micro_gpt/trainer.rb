# frozen_string_literal: true

module MicroGPT
  # Orchestrates the training loop: tokenizing documents, running the forward
  # pass, computing cross-entropy loss, backpropagation, and optimizer updates.
  class Trainer
    def initialize(model:, optimizer:, tokenizer:, dataset:, config:)
      @model = model
      @optimizer = optimizer
      @tokenizer = tokenizer
      @dataset = dataset
      @config = config
    end

    # Runs the full training loop for config.num_steps steps.
    # Yields (step_number, loss_float) after each step if a block is given.
    def train
      @config.num_steps.times do |step|
        loss_val = train_step(step)
        yield(step + 1, loss_val) if block_given?
      end
    end

    # Runs a single training step on one document. Returns the loss as a Float.
    def train_step(step)
      tokens = tokenize_with_bos(@dataset[step])
      n = [@config.block_size, tokens.size - 1].min

      loss = compute_loss(tokens, n)
      loss.backward
      @optimizer.step(step)
      @optimizer.zero_grad

      loss.data
    end

    private

    # Wraps a document in BOS tokens: [BOS, ...char_ids, BOS]
    def tokenize_with_bos(doc)
      [@tokenizer.bos_id] + @tokenizer.encode(doc) + [@tokenizer.bos_id]
    end

    # Computes average cross-entropy loss over a token sequence.
    def compute_loss(tokens, n)
      kv_cache = @model.new_kv_cache
      losses = []

      n.times do |pos_id|
        token_id  = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        logits = @model.forward(token_id, pos_id, kv_cache)
        probs  = NN.softmax(logits)
        losses << -probs[target_id].log
      end

      (1.0 / n) * losses.inject(:+)
    end
  end
end
