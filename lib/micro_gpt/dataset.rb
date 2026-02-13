# frozen_string_literal: true

module MicroGPT
  # Loads and manages a text dataset from a local file.
  # Each non-empty line in the file is treated as one document.
  class Dataset
    attr_reader :documents

    def initialize(path)
      raise ArgumentError, "File not found: #{path}" unless File.exist?(path)

      @documents = File.foreach(path).map(&:strip).reject(&:empty?)
    end

    # Returns the document at the given index, wrapping around with modulo.
    def [](index)
      @documents[index % @documents.size]
    end

    # Number of documents in the dataset.
    def size
      @documents.size
    end

    # Shuffles documents in place.
    def shuffle!
      @documents.shuffle!
      self
    end
  end
end
