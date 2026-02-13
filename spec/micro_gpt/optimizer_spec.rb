# frozen_string_literal: true

RSpec.describe MicroGPT::AdamOptimizer do
  let(:config) { tiny_config }

  before { srand(42) }

  describe "#step" do
    it "updates parameter data based on gradients" do
      param = MicroGPT::Value.new(1.0)
      param.grad = 2.0
      optimizer = described_class.new(params: [param], config: config)

      original_data = param.data
      optimizer.step(0)

      expect(param.data).not_to eq(original_data)
    end

    it "moves parameter in negative gradient direction" do
      param = MicroGPT::Value.new(0.0)
      param.grad = 1.0 # positive gradient
      optimizer = described_class.new(params: [param], config: config)

      optimizer.step(0)

      expect(param.data).to be < 0.0 # should decrease
    end

    it "applies bias correction" do
      param = MicroGPT::Value.new(0.0)
      param.grad = 1.0
      optimizer = described_class.new(params: [param], config: config)

      # Step 0 has large bias correction
      optimizer.step(0)
      step0_update = param.data

      # Reset
      param.data = 0.0
      param.grad = 1.0
      optimizer2 = described_class.new(params: [param], config: config)

      # Warm up moment buffers first
      optimizer2.step(0)
      param.grad = 1.0
      data_before = param.data
      optimizer2.step(1)
      step1_update = param.data - data_before

      # The updates should differ due to bias correction
      expect(step0_update).not_to eq(step1_update)
    end
  end

  describe "#zero_grad" do
    it "resets all gradients to zero" do
      params = [MicroGPT::Value.new(1.0), MicroGPT::Value.new(2.0)]
      params.each { |p| p.grad = 5.0 }
      optimizer = described_class.new(params: params, config: config)

      optimizer.zero_grad

      params.each { |p| expect(p.grad).to eq(0.0) }
    end
  end
end
