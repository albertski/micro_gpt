# frozen_string_literal: true

module MicroGPT
  # Character-level tokenizer. Builds a vocabulary from a corpus of documents,
  # mapping each unique character to an integer token id. A special BOS
  # (Beginning of Sequence) token is appended after all character ids.
  class Tokenizer
    attr_reader :vocab_size, :bos_id

    # Builds vocabulary from an array of document strings.
    def initialize(documents)
      @chars = documents.join.chars.uniq.sort
      @char_to_id = @chars.each_with_index.to_h
      @bos_id = @chars.size
      @vocab_size = @chars.size + 1
    end

    # Encodes a string into an array of integer token ids.
    def encode(string)
      string.chars.map { |ch| @char_to_id.fetch(ch) }
    end

    # Decodes an array of integer token ids back into a string.
    # BOS tokens are silently skipped.
    def decode(token_ids)
      token_ids.filter_map { |id| @chars[id] unless id == @bos_id }.join
    end

    # Returns true if the given token id is the BOS token.
    def bos?(token_id)
      token_id == @bos_id
    end
  end
end
