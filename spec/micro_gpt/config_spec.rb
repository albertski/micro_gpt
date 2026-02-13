# frozen_string_literal: true

RSpec.describe MicroGPT::Config do
  describe "defaults" do
    subject(:config) { described_class.new }

    it "has correct default hyperparameters" do
      expect(config.n_embd).to eq(16)
      expect(config.n_head).to eq(4)
      expect(config.n_layer).to eq(1)
      expect(config.block_size).to eq(16)
      expect(config.learning_rate).to eq(0.01)
      expect(config.beta1).to eq(0.85)
      expect(config.beta2).to eq(0.99)
      expect(config.eps).to eq(1e-8)
      expect(config.num_steps).to eq(1000)
      expect(config.temperature).to eq(0.5)
      expect(config.init_std).to eq(0.08)
      expect(config.num_samples).to eq(20)
    end
  end

  describe "#head_dim" do
    it "computes embedding dimension per head" do
      config = described_class.new(n_embd: 16, n_head: 4)
      expect(config.head_dim).to eq(4)
    end

    it "computes correctly for custom values" do
      config = described_class.new(n_embd: 8, n_head: 2)
      expect(config.head_dim).to eq(4)
    end
  end

  describe "custom overrides" do
    it "accepts custom values" do
      config = described_class.new(n_embd: 32, n_layer: 2)
      expect(config.n_embd).to eq(32)
      expect(config.n_layer).to eq(2)
    end

    it "is immutable" do
      config = described_class.new
      expect(config).to be_frozen
    end
  end
end
