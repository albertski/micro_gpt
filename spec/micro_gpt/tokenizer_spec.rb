# frozen_string_literal: true

RSpec.describe MicroGPT::Tokenizer do
  subject(:tokenizer) { described_class.new(tiny_docs) }

  describe "#initialize" do
    it "builds vocabulary from documents" do
      # tiny_docs = ["abc", "bca", "cab"] -> unique chars: a, b, c
      expect(tokenizer.vocab_size).to eq(4) # a, b, c + BOS
    end
  end

  describe "#bos_id" do
    it "is the last token id (after all characters)" do
      expect(tokenizer.bos_id).to eq(3) # a=0, b=1, c=2, BOS=3
    end
  end

  describe "#encode" do
    it "encodes a string to token ids" do
      expect(tokenizer.encode("abc")).to eq([0, 1, 2])
    end

    it "encodes repeated characters" do
      expect(tokenizer.encode("aab")).to eq([0, 0, 1])
    end

    it "returns an empty array for empty string" do
      expect(tokenizer.encode("")).to eq([])
    end

    it "raises on unknown characters" do
      expect { tokenizer.encode("xyz") }.to raise_error(KeyError)
    end
  end

  describe "#decode" do
    it "decodes token ids to a string" do
      expect(tokenizer.decode([0, 1, 2])).to eq("abc")
    end

    it "skips BOS tokens" do
      expect(tokenizer.decode([3, 0, 1, 3])).to eq("ab")
    end

    it "returns empty string for empty array" do
      expect(tokenizer.decode([])).to eq("")
    end
  end

  describe "#bos?" do
    it "returns true for BOS token" do
      expect(tokenizer.bos?(3)).to be true
    end

    it "returns false for non-BOS token" do
      expect(tokenizer.bos?(0)).to be false
    end
  end

  describe "encode/decode roundtrip" do
    it "roundtrips correctly" do
      original = "cab"
      expect(tokenizer.decode(tokenizer.encode(original))).to eq(original)
    end
  end
end
