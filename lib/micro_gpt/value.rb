# frozen_string_literal: true

module MicroGPT
  # Scalar-valued autograd engine. Each Value wraps a float and tracks the
  # computation graph so that gradients can be computed via reverse-mode
  # automatic differentiation (backpropagation).
  class Value
    attr_accessor :data, :grad
    attr_reader :children, :local_grads

    def initialize(data, children = [], local_grads = [])
      @data = data.to_f
      @grad = 0.0
      @children = children.freeze
      @local_grads = local_grads.freeze
    end

    # --- Arithmetic operations ---

    def +(other)
      other = wrap(other)
      Value.new(@data + other.data, [self, other], [1.0, 1.0])
    end

    def *(other)
      other = wrap(other)
      Value.new(@data * other.data, [self, other], [other.data, @data])
    end

    def **(exponent)
      Value.new(@data**exponent, [self], [exponent * @data**(exponent - 1)])
    end

    def -@
      self * -1
    end

    def -(other)
      self + (-other)
    end

    def /(other)
      self * (other**-1)
    end

    # --- Activation / math functions ---

    def log
      Value.new(Math.log(@data), [self], [1.0 / @data])
    end

    def exp
      e = Math.exp(@data)
      Value.new(e, [self], [e])
    end

    def relu
      active = @data.positive?
      Value.new(active ? @data : 0.0, [self], [active ? 1.0 : 0.0])
    end

    # Enables `number <op> value` (e.g. `5 + value`, `2.0 * value`).
    # Ruby calls value.coerce(number) -> [Value(number), value],
    # then evaluates Value(number) <op> value.
    def coerce(other)
      [Value.new(other), self]
    end

    # Reverse-mode automatic differentiation. Computes gradients of this
    # node (the loss) with respect to all upstream Value nodes.
    def backward
      topo = topological_sort
      @grad = 1.0

      topo.reverse_each do |v|
        v.children.zip(v.local_grads).each do |child, local_grad|
          child.grad += local_grad * v.grad
        end
      end
    end

    private

    def wrap(other)
      other.is_a?(Value) ? other : Value.new(other)
    end

    def topological_sort
      topo = []
      visited = {}.compare_by_identity
      build_topo(self, topo, visited)
      topo
    end

    def build_topo(node, topo, visited)
      return if visited.key?(node)

      visited[node] = true
      node.children.each { |child| build_topo(child, topo, visited) }
      topo << node
    end
  end
end
