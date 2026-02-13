# frozen_string_literal: true

module MicroGPT
  # Neural network primitive operations. These are pure functions (no state)
  # operating on arrays of Value objects.
  module NN
    module_function

    # Matrix-vector product. w is [nout][nin] of Value, x is [nin] of Value.
    # Returns [nout] of Value.
    def linear(x, w)
      w.map { |wo| wo.zip(x).map { |wi, xi| wi * xi }.inject(:+) }
    end

    # Numerically-stable softmax over an array of Value objects.
    # Subtracts max before exponentiating to prevent overflow.
    # Returns array of Value objects summing to 1.
    def softmax(logits)
      max_val = logits.map(&:data).max
      exps = logits.map { |val| (val - max_val).exp }
      total = exps.inject(:+)
      exps.map { |e| e / total }
    end

    # RMS (Root Mean Square) normalization over an array of Value objects.
    # Returns array of Value objects with approximately unit RMS.
    def rmsnorm(x)
      ms = x.map { |xi| xi * xi }.inject(:+) / x.size
      scale = (ms + 1e-5)**-0.5
      x.map { |xi| xi * scale }
    end
  end
end
