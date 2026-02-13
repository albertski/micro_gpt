# frozen_string_literal: true

require "tempfile"

RSpec.describe MicroGPT::Sampler do
  let(:config) { tiny_config(num_samples: 3) }
  let(:tmpfile) do
    file = Tempfile.new("sample_data")
    file.write("abc\nbca\ncab\n")
    file.close
    file
  end
  let(:dataset) { MicroGPT::Dataset.new(tmpfile.path) }
  let(:tokenizer) { MicroGPT::Tokenizer.new(dataset.documents) }
  let(:model) { MicroGPT::Model.new(config: config, vocab_size: tokenizer.vocab_size) }
  subject(:sampler) do
    described_class.new(model: model, tokenizer: tokenizer, config: config)
  end

  before { srand(42) }
  after { tmpfile.unlink }

  describe "#generate" do
    it "returns a String" do
      expect(sampler.generate).to be_a(String)
    end

    it "is deterministic with srand" do
      srand(42)
      m1 = MicroGPT::Model.new(config: config, vocab_size: tokenizer.vocab_size)
      s1 = described_class.new(model: m1, tokenizer: tokenizer, config: config)
      result1 = s1.generate

      srand(42)
      m2 = MicroGPT::Model.new(config: config, vocab_size: tokenizer.vocab_size)
      s2 = described_class.new(model: m2, tokenizer: tokenizer, config: config)
      result2 = s2.generate

      expect(result1).to eq(result2)
    end

    it "accepts a custom temperature" do
      expect { sampler.generate(temperature: 0.8) }.not_to raise_error
    end
  end

  describe "#generate_batch" do
    it "returns the configured number of samples" do
      samples = sampler.generate_batch
      expect(samples.size).to eq(3)
    end

    it "returns strings" do
      sampler.generate_batch.each do |s|
        expect(s).to be_a(String)
      end
    end

    it "accepts a custom count" do
      expect(sampler.generate_batch(n: 2).size).to eq(2)
    end
  end
end
