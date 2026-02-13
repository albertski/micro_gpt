# frozen_string_literal: true

require "tempfile"

RSpec.describe MicroGPT::Trainer do
  let(:config) { tiny_config(num_steps: 10) }
  let(:tmpfile) do
    file = Tempfile.new("train_data")
    file.write("abc\nbca\ncab\n")
    file.close
    file
  end
  let(:dataset) { MicroGPT::Dataset.new(tmpfile.path) }
  let(:tokenizer) { MicroGPT::Tokenizer.new(dataset.documents) }
  let(:model) { MicroGPT::Model.new(config: config, vocab_size: tokenizer.vocab_size) }
  let(:optimizer) { MicroGPT::AdamOptimizer.new(params: model.params, config: config) }
  subject(:trainer) do
    described_class.new(
      model: model, optimizer: optimizer,
      tokenizer: tokenizer, dataset: dataset, config: config
    )
  end

  before { srand(42) }
  after { tmpfile.unlink }

  describe "#train_step" do
    it "returns a positive float loss" do
      loss = trainer.train_step(0)
      expect(loss).to be_a(Float)
      expect(loss).to be > 0
    end

    it "resets gradients after the step" do
      trainer.train_step(0)
      model.params.each { |p| expect(p.grad).to eq(0.0) }
    end
  end

  describe "#train" do
    it "yields step number and loss for each step" do
      steps = []
      trainer.train { |step, loss| steps << [step, loss] }

      expect(steps.size).to eq(10)
      expect(steps.first[0]).to eq(1) # 1-based step
      expect(steps.last[0]).to eq(10)
      steps.each { |_, loss| expect(loss).to be > 0 }
    end

    it "decreases loss over training" do
      losses = []
      trainer.train { |_, loss| losses << loss }

      expect(losses.last).to be < losses.first
    end

    it "runs without a block" do
      expect { trainer.train }.not_to raise_error
    end
  end
end
