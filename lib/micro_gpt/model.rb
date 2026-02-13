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
        x = attention_block(x, li, kv_cache)
        x = mlp_block(x, li)
      end

      linear(x, @state_dict["lm_head"])
    end

    private

    def attention_block(x, layer_idx, kv_cache)
      x_residual = x
      x = rmsnorm(x)

      q = linear(x, @state_dict["layer#{layer_idx}.attn_wq"])
      k = linear(x, @state_dict["layer#{layer_idx}.attn_wk"])
      v = linear(x, @state_dict["layer#{layer_idx}.attn_wv"])

      kv_cache.keys[layer_idx] << k
      kv_cache.values[layer_idx] << v

      x_attn = multi_head_attention(q, kv_cache.keys[layer_idx], kv_cache.values[layer_idx])

      x = linear(x_attn, @state_dict["layer#{layer_idx}.attn_wo"])
      add_residual(x, x_residual)
    end

    def multi_head_attention(q, cached_keys, cached_values)
      scale = Math.sqrt(@config.head_dim)

      @config.n_head.times.flat_map do |h|
        hs = h * @config.head_dim
        q_h = q[hs, @config.head_dim]
        k_h = cached_keys.map { |ki| ki[hs, @config.head_dim] }
        v_h = cached_values.map { |vi| vi[hs, @config.head_dim] }

        attn_logits = k_h.map { |kt|
          q_h.zip(kt).map { |qi, ki| qi * ki }.inject(:+) / scale
        }
        attn_weights = softmax(attn_logits)

        (0...@config.head_dim).map { |j|
          attn_weights.each_with_index.map { |w, t| w * v_h[t][j] }.inject(:+)
        }
      end
    end

    def mlp_block(x, layer_idx)
      x_residual = x
      x = rmsnorm(x)

      x = linear(x, @state_dict["layer#{layer_idx}.mlp_fc1"])
      x = x.map(&:relu)
      x = linear(x, @state_dict["layer#{layer_idx}.mlp_fc2"])
      add_residual(x, x_residual)
    end

    def add_residual(x, residual)
      x.zip(residual).map { |a, b| a + b }
    end

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
