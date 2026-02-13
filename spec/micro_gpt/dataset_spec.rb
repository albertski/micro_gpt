# frozen_string_literal: true

require "tempfile"

RSpec.describe MicroGPT::Dataset do
  let(:tmpfile) do
    file = Tempfile.new("dataset")
    file.write("alice\nbob\n\ncharlie\n  \ndiana\n")
    file.close
    file
  end

  after { tmpfile.unlink }

  subject(:dataset) { described_class.new(tmpfile.path) }

  describe "#initialize" do
    it "loads documents from file" do
      expect(dataset.documents).to eq(%w[alice bob charlie diana])
    end

    it "strips whitespace and skips empty lines" do
      expect(dataset.documents).not_to include("")
      expect(dataset.documents).not_to include("  ")
    end

    it "raises on missing file" do
      expect { described_class.new("/nonexistent/file.txt") }
        .to raise_error(ArgumentError, /File not found/)
    end
  end

  describe "#size" do
    it "returns number of documents" do
      expect(dataset.size).to eq(4)
    end
  end

  describe "#[]" do
    it "returns document at index" do
      expect(dataset[0]).to eq("alice")
      expect(dataset[2]).to eq("charlie")
    end

    it "wraps around with modulo" do
      expect(dataset[4]).to eq("alice")
      expect(dataset[5]).to eq("bob")
    end
  end

  describe "#shuffle!" do
    it "returns self for chaining" do
      expect(dataset.shuffle!).to be(dataset)
    end

    it "shuffles documents in place" do
      srand(42)
      original = dataset.documents.dup
      dataset.shuffle!
      # With 4 elements and a fixed seed, at least one should move
      expect(dataset.documents).not_to eq(original)
    end
  end
end
