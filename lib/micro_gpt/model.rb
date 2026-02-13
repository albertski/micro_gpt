# frozen_string_literal: true

module MicroGPT
  # A minimal GPT-2-style transformer language model.
  # Owns the weight parameters and implements the forward pass.
  #
  # Architecture (following GPT-2 with minor differences):
  #   - RMSNorm instead of LayerNorm
  #   - No biases
  #   - ReLU instead of GeLU
  class Model
    include NN

    attr_reader :config, :params

    # Simple container for per-layer key/value attention cache.
    class KVCache
      attr_reader :keys, :values

      def initialize(n_layer)
        @keys   = Array.new(n_layer) { [] }
        @values = Array.new(n_layer) { [] }
      end
    end

    def initialize(config:, vocab_size:)
      @config = config
      @vocab_size = vocab_size
      @state_dict = init_parameters
      @params = @state_dict.values.flat_map(&:flatten)
    end

    # Creates a fresh KV cache for a new sequence.
    def new_kv_cache
      KVCache.new(@config.n_layer)
    end

    # Forward pass: maps a single token at a given position to logits over vocab.
    # The kv_cache is mutated in-place (keys/values appended for causal attention).
    # Returns an array of Value objects of size vocab_size.
    def forward(token_id, pos_id, kv_cache)
      tok_emb = @state_dict["wte"][token_id]
      pos_emb = @state_dict["wpe"][pos_id]
      x = tok_emb.zip(pos_emb).map { |t, p| t + p }

      x = rmsnorm(x)

      @config.n_layer.times do |li|
        # 1) Multi-head attention block
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, @state_dict["layer#{li}.attn_wq"])
        k = linear(x, @state_dict["layer#{li}.attn_wk"])
        v = linear(x, @state_dict["layer#{li}.attn_wv"])

        kv_cache.keys[li] << k
        kv_cache.values[li] << v

        x_attn = []

        @config.n_head.times do |h|
          hs = h * @config.head_dim
          q_h = q[hs, @config.head_dim]
          k_h = kv_cache.keys[li].map { |ki| ki[hs, @config.head_dim] }
          v_h = kv_cache.values[li].map { |vi| vi[hs, @config.head_dim] }

          attn_logits = (0...k_h.size).map { |t|
            (0...@config.head_dim).map { |j| q_h[j] * k_h[t][j] }.inject(:+) / @config.head_dim**0.5
          }
          attn_weights = softmax(attn_logits)

          head_out = (0...@config.head_dim).map { |j|
            (0...v_h.size).map { |t| attn_weights[t] * v_h[t][j] }.inject(:+)
          }

          x_attn.concat(head_out)
        end

        x = linear(x_attn, @state_dict["layer#{li}.attn_wo"])
        x = x.zip(x_residual).map { |a, b| a + b }

        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)

        x = linear(x, @state_dict["layer#{li}.mlp_fc1"])
        x = x.map(&:relu)
        x = linear(x, @state_dict["layer#{li}.mlp_fc2"])
        x = x.zip(x_residual).map { |a, b| a + b }
      end

      linear(x, @state_dict["lm_head"])
    end

    private

    def init_parameters
      sd = {
        "wte"     => make_matrix(@vocab_size, @config.n_embd),
        "wpe"     => make_matrix(@config.block_size, @config.n_embd),
        "lm_head" => make_matrix(@vocab_size, @config.n_embd),
      }

      @config.n_layer.times do |i|
        sd["layer#{i}.attn_wq"] = make_matrix(@config.n_embd, @config.n_embd)
        sd["layer#{i}.attn_wk"] = make_matrix(@config.n_embd, @config.n_embd)
        sd["layer#{i}.attn_wv"] = make_matrix(@config.n_embd, @config.n_embd)
        sd["layer#{i}.attn_wo"] = make_matrix(@config.n_embd, @config.n_embd)
        sd["layer#{i}.mlp_fc1"] = make_matrix(4 * @config.n_embd, @config.n_embd)
        sd["layer#{i}.mlp_fc2"] = make_matrix(@config.n_embd, 4 * @config.n_embd)
      end

      sd
    end

    def make_matrix(nout, nin)
      Array.new(nout) { Array.new(nin) { Value.new(Random.gauss(0.0, @config.init_std)) } }
    end
  end
end
