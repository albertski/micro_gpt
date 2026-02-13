# frozen_string_literal: true

RSpec.describe MicroGPT::NN do
  let(:v) { ->(x) { MicroGPT::Value.new(x) } }

  describe ".linear" do
    it "computes matrix-vector product" do
      # Identity-like 2x2 matrix
      w = [[v.(1), v.(0)], [v.(0), v.(1)]]
      x = [v.(3), v.(5)]

      result = described_class.linear(x, w)
      expect(result.map(&:data)).to eq([3.0, 5.0])
    end

    it "returns correct dimensions" do
      w = Array.new(3) { Array.new(2) { v.(1) } }
      x = [v.(1), v.(2)]

      result = described_class.linear(x, w)
      expect(result.size).to eq(3)
    end

    it "computes weighted sums correctly" do
      w = [[v.(2), v.(3)]]
      x = [v.(4), v.(5)]

      result = described_class.linear(x, w)
      expect(result[0].data).to eq(23.0) # 2*4 + 3*5
    end
  end

  describe ".softmax" do
    it "returns probabilities summing to 1" do
      logits = [v.(1), v.(2), v.(3)]
      probs = described_class.softmax(logits)

      total = probs.map(&:data).sum
      expect(total).to be_within(1e-10).of(1.0)
    end

    it "returns uniform distribution for equal inputs" do
      logits = [v.(5), v.(5), v.(5)]
      probs = described_class.softmax(logits)

      probs.each do |p|
        expect(p.data).to be_within(1e-10).of(1.0 / 3.0)
      end
    end

    it "assigns highest probability to largest logit" do
      logits = [v.(1), v.(10), v.(2)]
      probs = described_class.softmax(logits)

      expect(probs[1].data).to be > probs[0].data
      expect(probs[1].data).to be > probs[2].data
    end

    it "handles large values without overflow" do
      logits = [v.(1000), v.(1001), v.(1002)]
      probs = described_class.softmax(logits)

      total = probs.map(&:data).sum
      expect(total).to be_within(1e-6).of(1.0)
    end

    it "produces differentiable outputs" do
      logits = [v.(1), v.(2)]
      probs = described_class.softmax(logits)

      loss = -probs[0].log
      expect { loss.backward }.not_to raise_error
      expect(logits[0].grad).not_to eq(0.0)
    end
  end

  describe ".rmsnorm" do
    it "normalizes to approximately unit RMS" do
      x = [v.(3), v.(4), v.(5)]
      normed = described_class.rmsnorm(x)

      rms = Math.sqrt(normed.map { |xi| xi.data**2 }.sum / normed.size)
      expect(rms).to be_within(0.01).of(1.0)
    end

    it "preserves relative magnitudes" do
      x = [v.(2), v.(4)]
      normed = described_class.rmsnorm(x)

      ratio = normed[1].data / normed[0].data
      expect(ratio).to be_within(1e-6).of(2.0)
    end

    it "returns same number of elements" do
      x = [v.(1), v.(2), v.(3), v.(4)]
      expect(described_class.rmsnorm(x).size).to eq(4)
    end
  end
end
