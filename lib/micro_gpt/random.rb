# frozen_string_literal: true

module MicroGPT
  # Random number generation utilities used for parameter initialization
  # and inference sampling.
  module Random
    module_function

    # Gaussian random number via the Box-Muller transform.
    # Replaces Python's random.gauss — same distribution, different RNG.
    def gauss(mean = 0.0, stddev = 1.0)
      u1 = rand
      u2 = rand
      z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
      mean + stddev * z
    end

    # Weighted random sampling by index.
    # Returns the index chosen according to the given weight distribution.
    # Replaces Python's random.choices with weights.
    def weighted_choice(weights)
      r = rand * weights.sum
      cumulative = 0.0
      weights.each_with_index do |w, i|
        cumulative += w
        return i if r <= cumulative
      end
      weights.size - 1
    end
  end
end
