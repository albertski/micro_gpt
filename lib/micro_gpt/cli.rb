# frozen_string_literal: true

require "thor"

module MicroGPT
  # Command-line interface for training and running microGPT.
  # Uses Thor for option parsing, help text, and subcommand support.
  class CLI < Thor
    def self.exit_on_failure?
      true
    end

    desc "train [INPUT_FILE]", "Train a micro GPT model and generate samples"
    long_desc <<~DESC
      Trains a tiny GPT-2-style transformer on a character-level dataset,
      then generates new samples via autoregressive sampling.

      INPUT_FILE defaults to input.txt in the project root.
      The file should contain one document (e.g. name) per line.
    DESC

    method_option :steps,       type: :numeric, default: 1000,  desc: "Number of training steps"
    method_option :temperature, type: :numeric, default: 0.5,   desc: "Sampling temperature (0, 1]"
    method_option :samples,     type: :numeric, default: 20,    desc: "Number of samples to generate"
    method_option :seed,        type: :numeric, default: 42,    desc: "Random seed"
    method_option :n_embd,      type: :numeric, default: 16,    desc: "Embedding dimension"
    method_option :n_head,      type: :numeric, default: 4,     desc: "Number of attention heads"
    method_option :n_layer,     type: :numeric, default: 1,     desc: "Number of transformer layers"
    method_option :block_size,  type: :numeric, default: 16,    desc: "Maximum sequence length"
    method_option :lr,          type: :numeric, default: 0.01,  desc: "Learning rate"

    def train(input_file = nil)
      $stdout.sync = true
      srand(options[:seed])

      config    = build_config
      dataset   = load_dataset(input_file || default_input_path)
      tokenizer = Tokenizer.new(dataset.documents)

      print_summary(dataset, tokenizer)

      model = Model.new(config: config, vocab_size: tokenizer.vocab_size)
      say "num params: #{model.params.size}"

      run_training(model: model, tokenizer: tokenizer, dataset: dataset, config: config)
      generate_samples(model: model, tokenizer: tokenizer, config: config)
    end

    default_command :train

    private

    def build_config
      Config.new(
        n_embd:        options[:n_embd],
        n_head:        options[:n_head],
        n_layer:       options[:n_layer],
        block_size:    options[:block_size],
        learning_rate: options[:lr],
        num_steps:     options[:steps],
        temperature:   options[:temperature],
        num_samples:   options[:samples]
      )
    end

    def load_dataset(path)
      dataset = Dataset.new(path)
      dataset.shuffle!
      dataset
    end

    def print_summary(dataset, tokenizer)
      say "num docs: #{dataset.size}"
      say "vocab size: #{tokenizer.vocab_size}"
    end

    def run_training(model:, tokenizer:, dataset:, config:)
      optimizer = AdamOptimizer.new(params: model.params, config: config)
      trainer   = Trainer.new(
        model: model, optimizer: optimizer,
        tokenizer: tokenizer, dataset: dataset, config: config
      )

      trainer.train do |step, loss|
        say format("step %4d / %4d | loss %.4f", step, config.num_steps, loss)
      end
    end

    def generate_samples(model:, tokenizer:, config:)
      say "\n--- inference (new, hallucinated names) ---"

      sampler = Sampler.new(model: model, tokenizer: tokenizer, config: config)

      sampler.generate_batch.each_with_index do |sample, i|
        say format("sample %2d: %s", i + 1, sample)
      end
    end

    def default_input_path
      File.expand_path("../../input.txt", __dir__)
    end
  end
end
