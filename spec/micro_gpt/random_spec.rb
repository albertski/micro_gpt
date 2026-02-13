# frozen_string_literal: true

RSpec.describe MicroGPT::Random do
  describe ".gauss" do
    it "returns a Float" do
      expect(described_class.gauss).to be_a(Float)
    end

    it "is deterministic with srand" do
      srand(123)
      a = described_class.gauss
      srand(123)
      b = described_class.gauss
      expect(a).to eq(b)
    end

    it "produces values near the specified mean" do
      srand(42)
      samples = Array.new(10_000) { described_class.gauss(5.0, 1.0) }
      mean = samples.sum / samples.size
      expect(mean).to be_within(0.1).of(5.0)
    end

    it "produces values with approximately the specified stddev" do
      srand(42)
      samples = Array.new(10_000) { described_class.gauss(0.0, 2.0) }
      mean = samples.sum / samples.size
      variance = samples.map { |s| (s - mean)**2 }.sum / samples.size
      stddev = Math.sqrt(variance)
      expect(stddev).to be_within(0.15).of(2.0)
    end
  end

  describe ".weighted_choice" do
    it "returns an index within range" do
      idx = described_class.weighted_choice([0.1, 0.2, 0.7])
      expect(idx).to be_between(0, 2)
    end

    it "always picks the only nonzero weight" do
      10.times do
        expect(described_class.weighted_choice([0, 0, 1])).to eq(2)
      end
    end

    it "is deterministic with srand" do
      srand(99)
      a = described_class.weighted_choice([0.3, 0.3, 0.4])
      srand(99)
      b = described_class.weighted_choice([0.3, 0.3, 0.4])
      expect(a).to eq(b)
    end

    it "samples proportionally over many trials" do
      srand(42)
      counts = [0, 0, 0]
      n = 10_000
      n.times { counts[described_class.weighted_choice([0.2, 0.3, 0.5])] += 1 }

      expect(counts[0].to_f / n).to be_within(0.03).of(0.2)
      expect(counts[1].to_f / n).to be_within(0.03).of(0.3)
      expect(counts[2].to_f / n).to be_within(0.03).of(0.5)
    end
  end
end
