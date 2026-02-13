# frozen_string_literal: true

RSpec.describe MicroGPT::Model do
  let(:config) { tiny_config }
  let(:vocab_size) { 4 } # a, b, c + BOS
  subject(:model) { described_class.new(config: config, vocab_size: vocab_size) }

  before { srand(42) }

  describe "#initialize" do
    it "creates the correct number of parameters" do
      # For tiny_config (n_embd=4, n_head=2, n_layer=1, block_size=4, vocab_size=4):
      # wte: 4*4=16, wpe: 4*4=16, lm_head: 4*4=16 = 48
      # Per layer (1 layer):
      #   attn_wq: 4*4=16, attn_wk: 16, attn_wv: 16, attn_wo: 16
      #   mlp_fc1: 16*4=64, mlp_fc2: 4*16=64
      #   Layer total: 16*4 + 64*2 = 192
      # Grand total: 48 + 192 = 240
      expect(model.params.size).to eq(240)
    end

    it "stores the config" do
      expect(model.config).to eq(config)
    end

    it "initializes all params as Value objects" do
      expect(model.params).to all(be_a(MicroGPT::Value))
    end
  end

  describe "#new_kv_cache" do
    it "returns a KVCache with correct number of layers" do
      kv = model.new_kv_cache
      expect(kv.keys.size).to eq(config.n_layer)
      expect(kv.values.size).to eq(config.n_layer)
    end

    it "starts with empty caches" do
      kv = model.new_kv_cache
      kv.keys.each { |layer| expect(layer).to be_empty }
      kv.values.each { |layer| expect(layer).to be_empty }
    end
  end

  describe "#forward" do
    it "returns logits of size vocab_size" do
      kv = model.new_kv_cache
      logits = model.forward(0, 0, kv)
      expect(logits.size).to eq(vocab_size)
    end

    it "returns Value objects" do
      kv = model.new_kv_cache
      logits = model.forward(0, 0, kv)
      expect(logits).to all(be_a(MicroGPT::Value))
    end

    it "appends to KV cache on each call" do
      kv = model.new_kv_cache
      model.forward(0, 0, kv)
      expect(kv.keys[0].size).to eq(1)

      model.forward(1, 1, kv)
      expect(kv.keys[0].size).to eq(2)
    end

    it "produces differentiable outputs" do
      kv = model.new_kv_cache
      logits = model.forward(0, 0, kv)
      probs = MicroGPT::NN.softmax(logits)
      loss = -probs[0].log

      expect { loss.backward }.not_to raise_error
      # At least some params should have nonzero gradients
      nonzero_grads = model.params.count { |p| p.grad != 0.0 }
      expect(nonzero_grads).to be > 0
    end
  end
end
