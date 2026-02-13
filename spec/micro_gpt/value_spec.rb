# frozen_string_literal: true

RSpec.describe MicroGPT::Value do
  describe "#initialize" do
    it "stores data as a float" do
      v = described_class.new(3)
      expect(v.data).to eq(3.0)
      expect(v.data).to be_a(Float)
    end

    it "initializes grad to 0.0" do
      expect(described_class.new(5).grad).to eq(0.0)
    end

    it "defaults to empty children and local_grads" do
      v = described_class.new(1)
      expect(v.children).to eq([])
      expect(v.local_grads).to eq([])
    end
  end

  describe "forward pass (operators)" do
    let(:a) { described_class.new(3.0) }
    let(:b) { described_class.new(4.0) }

    it "adds two Values" do
      expect((a + b).data).to eq(7.0)
    end

    it "adds a Value and a number" do
      expect((a + 2).data).to eq(5.0)
    end

    it "multiplies two Values" do
      expect((a * b).data).to eq(12.0)
    end

    it "multiplies a Value and a number" do
      expect((a * 3).data).to eq(9.0)
    end

    it "raises to a power" do
      expect((a ** 2).data).to eq(9.0)
    end

    it "subtracts two Values" do
      expect((a - b).data).to eq(-1.0)
    end

    it "divides two Values" do
      expect((a / b).data).to be_within(1e-10).of(0.75)
    end

    it "negates a Value" do
      expect((-a).data).to eq(-3.0)
    end
  end

  describe "unary operations" do
    it "#log computes natural log" do
      v = described_class.new(Math::E)
      expect(v.log.data).to be_within(1e-10).of(1.0)
    end

    it "#exp computes exponential" do
      v = described_class.new(0.0)
      expect(v.exp.data).to eq(1.0)
    end

    it "#relu returns data when positive" do
      expect(described_class.new(3.0).relu.data).to eq(3.0)
    end

    it "#relu returns 0 when negative" do
      expect(described_class.new(-2.0).relu.data).to eq(0.0)
    end

    it "#relu returns 0 when zero" do
      expect(described_class.new(0.0).relu.data).to eq(0.0)
    end
  end

  describe "#coerce" do
    let(:v) { described_class.new(2.0) }

    it "enables number + Value" do
      result = 5 + v
      expect(result.data).to eq(7.0)
    end

    it "enables number * Value" do
      result = 3.0 * v
      expect(result.data).to eq(6.0)
    end

    it "enables number - Value" do
      result = 10 - v
      expect(result.data).to eq(8.0)
    end

    it "enables number / Value" do
      result = 6.0 / v
      expect(result.data).to be_within(1e-10).of(3.0)
    end
  end

  describe "#backward" do
    it "computes gradients for a simple product" do
      a = described_class.new(3.0)
      b = described_class.new(4.0)
      c = a * b
      c.backward

      expect(a.grad).to eq(4.0) # dc/da = b
      expect(b.grad).to eq(3.0) # dc/db = a
    end

    it "computes gradients for addition" do
      a = described_class.new(3.0)
      b = described_class.new(4.0)
      c = a + b
      c.backward

      expect(a.grad).to eq(1.0)
      expect(b.grad).to eq(1.0)
    end

    it "computes gradients for a compound expression" do
      # f = (a * b) + c
      a = described_class.new(2.0)
      b = described_class.new(3.0)
      c = described_class.new(5.0)
      f = a * b + c
      f.backward

      expect(a.grad).to eq(3.0) # df/da = b
      expect(b.grad).to eq(2.0) # df/db = a
      expect(c.grad).to eq(1.0) # df/dc = 1
    end

    it "accumulates gradients when a node is used multiple times" do
      a = described_class.new(3.0)
      f = a + a # df/da = 2
      f.backward

      expect(a.grad).to eq(2.0)
    end

    it "computes gradients through log" do
      a = described_class.new(2.0)
      f = a.log
      f.backward

      expect(a.grad).to be_within(1e-10).of(0.5) # d(log(a))/da = 1/a
    end

    it "computes gradients through exp" do
      a = described_class.new(1.0)
      f = a.exp
      f.backward

      expect(a.grad).to be_within(1e-10).of(Math::E) # d(exp(a))/da = exp(a)
    end

    it "computes gradients through relu" do
      a = described_class.new(3.0)
      f = a.relu
      f.backward
      expect(a.grad).to eq(1.0)

      b = described_class.new(-2.0)
      g = b.relu
      g.backward
      expect(b.grad).to eq(0.0)
    end

    it "computes gradients through power" do
      a = described_class.new(3.0)
      f = a ** 2
      f.backward

      expect(a.grad).to be_within(1e-10).of(6.0) # d(a^2)/da = 2a
    end

    it "computes gradients through division" do
      a = described_class.new(6.0)
      b = described_class.new(3.0)
      f = a / b
      f.backward

      expect(a.grad).to be_within(1e-10).of(1.0 / 3.0)  # df/da = 1/b
      expect(b.grad).to be_within(1e-10).of(-6.0 / 9.0)  # df/db = -a/b^2
    end
  end
end
