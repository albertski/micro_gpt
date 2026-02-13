# frozen_string_literal: true

module MicroGPT
  # Adam optimizer with bias correction and linear learning rate decay.
  # Maintains first and second moment buffers for each parameter.
  class AdamOptimizer
    def initialize(params:, config:)
      @params = params
      @config = config
      @m = Array.new(params.size, 0.0) # first moment
      @v = Array.new(params.size, 0.0) # second moment
    end

    # Performs one Adam update step with linear LR decay.
    # step is 0-based.
    def step(step)
      lr_t = @config.learning_rate * (1.0 - step.to_f / @config.num_steps)

      @params.each_with_index do |p, i|
        @m[i] = @config.beta1 * @m[i] + (1 - @config.beta1) * p.grad
        @v[i] = @config.beta2 * @v[i] + (1 - @config.beta2) * p.grad**2

        m_hat = @m[i] / (1.0 - @config.beta1**(step + 1))
        v_hat = @v[i] / (1.0 - @config.beta2**(step + 1))

        p.data -= lr_t * m_hat / (v_hat**0.5 + @config.eps)
      end
    end

    # Resets all parameter gradients to zero.
    def zero_grad
      @params.each { |p| p.grad = 0.0 }
    end
  end
end
