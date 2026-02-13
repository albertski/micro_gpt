# frozen_string_literal: true

module MicroGPT
  # Adam optimizer with bias correction and linear learning rate decay.
  # Maintains first and second moment buffers for each parameter.
  class AdamOptimizer
    def initialize(params:, config:)
      @params = params
      @config = config
      @m = Array.new(params.size, 0.0)
      @v = Array.new(params.size, 0.0)
    end

    # Performs one Adam update with linear LR decay.
    # step_index is 0-based.
    def step(step_index)
      lr = decayed_lr(step_index)

      @params.each_with_index do |param, i|
        @m[i] = @config.beta1 * @m[i] + (1 - @config.beta1) * param.grad
        @v[i] = @config.beta2 * @v[i] + (1 - @config.beta2) * param.grad**2

        m_hat = @m[i] / (1.0 - @config.beta1**(step_index + 1))
        v_hat = @v[i] / (1.0 - @config.beta2**(step_index + 1))

        param.data -= lr * m_hat / (Math.sqrt(v_hat) + @config.eps)
      end
    end

    def zero_grad
      @params.each { |p| p.grad = 0.0 }
    end

    private

    def decayed_lr(step_index)
      @config.learning_rate * (1.0 - step_index.to_f / @config.num_steps)
    end
  end
end
