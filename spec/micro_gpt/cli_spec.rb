# frozen_string_literal: true

require "tempfile"

RSpec.describe MicroGPT::CLI do
  let(:tmpfile) do
    file = Tempfile.new("cli_data")
    file.write("abc\nbca\ncab\n")
    file.close
    file
  end

  after { tmpfile.unlink }

  def capture_cli(*args)
    output = StringIO.new
    original_stdout = $stdout
    $stdout = output
    begin
      described_class.start(args)
    rescue SystemExit
      # Thor may call exit on errors
    ensure
      $stdout = original_stdout
    end
    output.string
  end

  def tiny_cli_args
    [
      "train", tmpfile.path,
      "--steps", "3",
      "--samples", "2",
      "--seed", "42",
      "--n-embd", "4",
      "--n-head", "2",
      "--block-size", "4"
    ]
  end

  describe "train command" do
    it "runs training and produces output" do
      output = capture_cli(*tiny_cli_args)
      expect(output).to include("num docs:")
      expect(output).to include("vocab size:")
      expect(output).to include("num params:")
    end

    it "prints loss at each step" do
      output = capture_cli(*tiny_cli_args)
      expect(output).to include("step    1 /    3 | loss")
      expect(output).to include("step    3 /    3 | loss")
    end

    it "generates samples after training" do
      output = capture_cli(*tiny_cli_args)
      expect(output).to include("--- inference (new, hallucinated names) ---")
      expect(output).to include("sample  1:")
      expect(output).to include("sample  2:")
    end

    it "does not generate more samples than requested" do
      output = capture_cli(*tiny_cli_args)
      expect(output).not_to include("sample  3:")
    end

    it "uses the provided seed for determinism" do
      output1 = capture_cli(*tiny_cli_args)
      output2 = capture_cli(*tiny_cli_args)
      expect(output1).to eq(output2)
    end
  end

  describe "option defaults" do
    it "exposes all expected options on the train command" do
      command = described_class.all_commands["train"]
      option_names = command.options.keys

      %i[steps temperature samples seed n_embd n_head n_layer block_size lr].each do |opt|
        expect(option_names).to include(opt), "expected option '#{opt}' to be defined"
      end
    end
  end

  describe "default_command" do
    it "is set to train" do
      expect(described_class.default_command).to eq("train")
    end
  end

  describe "exit_on_failure?" do
    it "returns true" do
      expect(described_class.exit_on_failure?).to be true
    end
  end
end
