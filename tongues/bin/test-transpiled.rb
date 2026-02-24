#!/usr/bin/env ruby
# frozen_string_literal: true

# Native Ruby test harness for transpiled Tongues binaries.
# Loads the transpiled file once, then runs all .tests cases in-process.

require "stringio"
require "json"
require "tempfile"
require "fileutils"

TONGUES_DIR = File.expand_path("..", __dir__)
TESTS_DIR = File.join(TONGUES_DIR, "tests")

# Phase → test config: [dir, runner, is_taytsh, cli_args, expect_json]
# Runners: :cli, :phase, :lowering, :codegen, :emit, :app, :ordering, :taytsh_app
TESTS = {
  "cli" => {
    "cli" => { dir: "02_cli", run: :cli },
  },
  "frontend" => {
    "parse"     => { dir: "03_parse",      run: :phase, taytsh: false, args: ["--stop-at", "parse"],      json: true  },
    "subset"    => { dir: "04_subset",      run: :phase, taytsh: false, args: ["--stop-at", "subset"],     json: false },
    "names"     => { dir: "05_names",       run: :phase, taytsh: false, args: ["--stop-at", "names"],      json: true  },
    "sigs"      => { dir: "06_signatures",  run: :phase, taytsh: false, args: ["--stop-at", "signatures"], json: true  },
    "fields"    => { dir: "07_fields",      run: :phase, taytsh: false, args: ["--stop-at", "fields"],     json: true  },
    "hierarchy" => { dir: "08_hierarchy",   run: :phase, taytsh: false, args: ["--stop-at", "hierarchy"],  json: true  },
    "inference" => { dir: "09_inference",   run: :phase, taytsh: false, args: ["--stop-at", "inference"],  json: true  },
    "lowering"  => { dir: "10_lowering",    run: :lowering },
  },
  "middleend" => {
    "type_checking" => { dir: "13_type_checking", run: :phase, taytsh: true, args: ["--stop-at", "check"],     json: false },
    "scope"         => { dir: "14_scope",          run: :phase, taytsh: true, args: ["--stop-at", "scope"],     json: true  },
    "returns"       => { dir: "15_returns",        run: :phase, taytsh: true, args: ["--stop-at", "returns"],   json: true  },
    "liveness"      => { dir: "16_liveness",       run: :phase, taytsh: true, args: ["--stop-at", "liveness"],  json: true  },
    "strings"       => { dir: "17_strings",        run: :phase, taytsh: true, args: ["--stop-at", "strings"],   json: true  },
    "hoisting"      => { dir: "18_hoisting",       run: :phase, taytsh: true, args: ["--stop-at", "hoisting"],  json: true  },
    "ownership"     => { dir: "19_ownership",      run: :phase, taytsh: true, args: ["--stop-at", "ownership"], json: true  },
    "callgraph"     => { dir: "20_callgraph",      run: :phase, taytsh: true, args: ["--stop-at", "callgraph"], json: true  },
  },
  "backend" => {
    "codegen"  => { dir: "21_codegen",  run: :codegen },
    "emit"     => { dir: "25_emit",     run: :emit },
    "app"      => { dir: "22_app",      run: :app },
    "ordering" => { dir: "24_ordering", run: :ordering },
  },
  "taytsh" => {
    "taytsh_parse" => { dir: "11_taytsh_parse", run: :phase, taytsh: true, args: ["--stop-at", "parse"], json: true  },
    "taytsh_check" => { dir: "12_taytsh_check", run: :phase, taytsh: true, args: ["--stop-at", "check"], json: false },
    "taytsh_app"   => { dir: "23_taytsh_app",   run: :taytsh_app },
  },
}

EMITTER_LANGS = %w[python perl ruby]
RUNTIMES = {
  "python" => ["python3"],
  "perl"   => ["perl"],
  "ruby"   => ["ruby"],
}

# ---------------------------------------------------------------------------
# In-process execution
# ---------------------------------------------------------------------------

def run_inprocess(argv, stdin_data: "")
  old_argv = ARGV.dup
  old_stdout = $stdout
  old_stderr = $stderr
  old_stdin = $stdin
  out = StringIO.new
  err = StringIO.new
  code = 0
  begin
    ARGV.replace(argv)
    $stdout = out
    $stderr = err
    $stdin = StringIO.new(stdin_data)
    main
  rescue SystemExit => e
    code = e.status
  rescue => e
    err.write("#{e}\n")
    code = 1
  ensure
    ARGV.replace(old_argv)
    $stdout = old_stdout
    $stderr = old_stderr
    $stdin = old_stdin
  end
  { stdout: out.string, stderr: err.string, exit: code }
end

def run_transpiled_phase(source, cli_args, is_taytsh:, expect_json: true)
  suffix = is_taytsh ? ".ty" : ".py"
  tmp = Tempfile.new(["test", suffix])
  begin
    tmp.write(source)
    tmp.flush
    argv = if is_taytsh
             ["taytsh", *cli_args, tmp.path]
           else
             [*cli_args, tmp.path]
           end
    result = run_inprocess(argv)
  ensure
    tmp.close
    tmp.unlink
  end
  stderr_text = result[:stderr].strip
  if result[:exit] != 0
    errors = stderr_text.split("\n").reject(&:empty?)
    return { errors: errors, warnings: [], data: nil, reveals: [] }
  end
  warnings = stderr_text.empty? ? [] : stderr_text.split("\n").reject(&:empty?)
  unless expect_json
    return { errors: [], warnings: warnings, data: nil, reveals: [] }
  end
  stdout_text = result[:stdout].strip
  if stdout_text.empty?
    return { errors: [], warnings: warnings, data: nil, reveals: [] }
  end
  begin
    data = JSON.parse(stdout_text)
  rescue JSON::ParserError
    return { errors: ["Invalid JSON output: #{stdout_text[0, 200]}"], warnings: [], data: nil, reveals: [] }
  end
  { errors: [], warnings: warnings, data: data, reveals: [] }
end

# ---------------------------------------------------------------------------
# .tests file parsing
# ---------------------------------------------------------------------------

def parse_spec_file(path)
  lines = File.read(path).split("\n")
  result = []
  i = 0
  while i < lines.length
    if lines[i].start_with?("=== ")
      test_name = lines[i][4..].strip
      i += 1
      input_lines = []
      while i < lines.length && !lines[i].start_with?("---")
        input_lines << lines[i]
        i += 1
      end
      i += 1 if i < lines.length && lines[i] == "---"
      expected_lines = []
      while i < lines.length && !lines[i].start_with?("---")
        expected_lines << lines[i]
        i += 1
      end
      i += 1 if i < lines.length && lines[i] == "---"
      result << [test_name, input_lines.join("\n"), expected_lines.join("\n").strip]
    else
      i += 1
    end
  end
  result
end

def parse_cli_test_file(path)
  lines = File.read(path).split("\n")
  result = []
  i = 0
  while i < lines.length
    if lines[i].start_with?("=== ")
      test_name = lines[i][4..].strip
      i += 1
      input_lines = []
      while i < lines.length && !lines[i].start_with?("---")
        input_lines << lines[i]
        i += 1
      end
      i += 1 if i < lines.length && lines[i] == "---"
      expected_lines = []
      while i < lines.length && !lines[i].start_with?("---")
        expected_lines << lines[i]
        i += 1
      end
      i += 1 if i < lines.length && lines[i] == "---"
      spec = parse_cli_spec(input_lines, expected_lines)
      result << [test_name, spec]
    else
      i += 1
    end
  end
  result
end

def parse_cli_spec(input_lines, expected_lines)
  spec = { args: [], stdin: nil, stdin_bytes: nil, assertions: [] }
  body_start = 0
  if !input_lines.empty? && input_lines[0].start_with?("args:")
    args_str = input_lines[0][5..].strip
    spec[:args] = args_str.empty? ? [] : args_str.split
    body_start = 1
  end
  remaining = input_lines[body_start..]
  if !remaining.empty? && remaining[0].start_with?("stdin-bytes:")
    hex_str = remaining[0]["stdin-bytes:".length..].strip
    spec[:stdin_bytes] = [hex_str].pack("H*")
  else
    spec[:stdin] = remaining.join("\n")
  end
  expected_lines.each do |line|
    line = line.strip
    next if line.empty?
    if line.start_with?("exit:")
      spec[:assertions] << [:exit, line[5..].strip.to_i]
    elsif line.start_with?("exit-not:")
      spec[:assertions] << [:"exit-not", line[9..].strip.to_i]
    elsif line.start_with?("stderr:")
      spec[:assertions] << [:stderr, line[7..].strip]
    elsif line.start_with?("stderr-contains:")
      spec[:assertions] << [:"stderr-contains", line[16..].strip]
    elsif line.start_with?("stderr-empty:")
      spec[:assertions] << [:"stderr-empty", nil]
    elsif line.start_with?("stdout-contains:")
      spec[:assertions] << [:"stdout-contains", line[16..].strip]
    elsif line.start_with?("stdout-empty:")
      spec[:assertions] << [:"stdout-empty", nil]
    end
  end
  spec
end

def parse_simple_tests(path)
  lines = File.read(path).split("\n")
  result = []
  i = 0
  while i < lines.length
    if lines[i].start_with?("=== ")
      name = lines[i][4..].strip
      i += 1
      content_lines = []
      while i < lines.length && !lines[i].start_with?("=== ")
        content_lines << lines[i]
        i += 1
      end
      result << [name, content_lines.join("\n").strip]
    else
      i += 1
    end
  end
  result
end

# ---------------------------------------------------------------------------
# Assertion checking
# ---------------------------------------------------------------------------

def resolve_dotpath(obj, path)
  parts = path.split(".")
  current = obj
  i = 0
  while i < parts.length
    part = parts[i]
    if part == "length"
      return current.length
    end
    if current.is_a?(Array)
      current = current[part.to_i]
      i += 1
    elsif current.is_a?(Hash)
      if current.key?(part)
        current = current[part]
        i += 1
      else
        found = false
        (i + 1...parts.length).each do |j|
          composite = parts[i..j].join(".")
          if current.key?(composite)
            current = current[composite]
            i = j + 1
            found = true
            break
          end
        end
        raise KeyError, "key not found: #{part}" unless found
      end
    else
      raise KeyError, "cannot traverse #{current.class} with key '#{part}'"
    end
  end
  current
end

def to_comparable(value)
  return "null" if value.nil?
  return (value ? "true" : "false") if value == true || value == false
  return value.to_s if value.is_a?(Integer)
  return value if value.is_a?(String)
  value.to_s
end

def check_expected(expected, result, phase, lenient_errors: false)
  reveal_assertions = []
  verdict_lines = []
  expected.split("\n").each do |line|
    stripped = line.strip
    if stripped.start_with?("reveal:")
      rest = stripped[7..]
      eq_pos = rest.index("=")
      lineno = rest[0...eq_pos].strip.to_i
      expected_type = rest[(eq_pos + 1)..].strip
      reveal_assertions << [lineno, expected_type]
    else
      verdict_lines << line
    end
  end
  expected = verdict_lines.join("\n").strip
  expected = "ok" if expected.empty?
  if expected == "ok"
    unless result[:errors].empty?
      return "Expected ok, got error: #{result[:errors][0]}"
    end
    err = check_reveals(reveal_assertions, result[:reveals])
    return err if err
    return nil
  end
  if expected.start_with?("error:")
    expected_msg = expected[6..].strip
    if result[:errors].empty?
      return "Expected error containing '#{expected_msg}', got ok"
    end
    if !lenient_errors && !expected_msg.empty?
      found = result[:errors].any? { |e| e.downcase.include?(expected_msg.downcase) }
      return "Expected error containing '#{expected_msg}', got: #{result[:errors]}" unless found
    end
    return nil
  end
  if expected.start_with?("warning:")
    expected_msg = expected[8..].strip
    if result[:warnings].empty?
      return "Expected warning containing '#{expected_msg}', got none"
    end
    found = result[:warnings].any? { |w| w.downcase.include?(expected_msg.downcase) }
    return "Expected warning containing '#{expected_msg}', got: #{result[:warnings]}" unless found
    return nil
  end
  unless result[:errors].empty?
    return "#{phase} failed: #{result[:errors][0]}"
  end
  if result[:data].nil?
    return "No data returned from #{phase}"
  end
  expected.split("\n").each do |line|
    line = line.strip
    next if line.empty?
    unless line.include?("=")
      return "Bad assertion (no '='): #{line}"
    end
    path, expected_val = line.split("=", 2)
    path = path.strip
    expected_val = expected_val.strip
    begin
      actual = resolve_dotpath(result[:data], path)
    rescue KeyError, IndexError, TypeError => e
      return "Path '#{path}' not found in result: #{e}"
    end
    actual_str = to_comparable(actual)
    if expected_val.include?(".") && !expected_val.include?(" ")
      begin
        ref_val = resolve_dotpath(result[:data], expected_val)
        expected_val = to_comparable(ref_val)
      rescue KeyError, IndexError, TypeError
        # treat as literal
      end
    end
    if actual_str != expected_val
      return "Assertion failed: #{path}\n  expected: #{expected_val.inspect}\n  actual:   #{actual_str.inspect}"
    end
  end
  nil
end

def check_reveals(assertions, actuals)
  assertions.each do |lineno, expected_type|
    found = false
    actuals.each do |actual_line, actual_type|
      if actual_line == lineno
        if actual_type != expected_type
          return "reveal_type at line #{lineno}: expected '#{expected_type}', got '#{actual_type}'"
        end
        found = true
        break
      end
    end
    return "No reveal_type found at line #{lineno}" unless found
  end
  nil
end

def contains_normalized(haystack, needle)
  needle_lines = needle.strip.split("\n").map(&:strip).reject(&:empty?)
  haystack_lines = haystack.split("\n").map(&:strip).reject(&:empty?)
  return true if needle_lines.empty?
  haystack_lines.each_index do |i|
    if haystack_lines[i].include?(needle_lines[0])
      match = true
      (1...needle_lines.length).each do |j|
        if i + j >= haystack_lines.length || !haystack_lines[i + j].include?(needle_lines[j])
          match = false
          break
        end
      end
      return true if match
    end
  end
  false
end

# ---------------------------------------------------------------------------
# CLI test helpers
# ---------------------------------------------------------------------------

def cli_needs_backend(spec)
  args = spec[:args]
  return false if args.include?("--stop-at")
  expects_success = spec[:assertions].any? { |k, v| k == :exit && v == 0 }
  return false unless expects_success
  return false unless args.include?("--target")
  target = args[args.index("--target") + 1]
  !EMITTER_LANGS.include?(target)
end

def check_cli_assertions(result, assertions)
  assertions.each do |kind, value|
    case kind
    when :exit
      if result[:exit] != value
        return "expected exit #{value}, got #{result[:exit]}\nstderr: #{result[:stderr]}"
      end
    when :"exit-not"
      if result[:exit] == value
        return "expected exit != #{value}, got #{result[:exit]}"
      end
    when :stderr
      actual = result[:stderr].rstrip
      if actual != value
        return "expected stderr #{value.inspect}, got #{actual.inspect}"
      end
    when :"stderr-contains"
      unless result[:stderr].include?(value)
        return "expected stderr to contain #{value.inspect}, got #{result[:stderr].inspect}"
      end
    when :"stderr-empty"
      unless result[:stderr].empty?
        return "expected empty stderr, got #{result[:stderr].inspect}"
      end
    when :"stdout-contains"
      unless result[:stdout].include?(value)
        return "expected stdout to contain #{value.inspect}, got #{result[:stdout].inspect}"
      end
    when :"stdout-empty"
      unless result[:stdout].empty?
        return "expected empty stdout, got #{result[:stdout][0, 200].inspect}"
      end
    end
  end
  nil
end

# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def run_cli_tests(test_dir)
  results = []
  Dir.glob(File.join(test_dir, "*.tests")).sort.each do |f|
    stem = File.basename(f, ".tests")
    parse_cli_test_file(f).each do |name, spec|
      test_id = "#{stem}/#{name}"
      if cli_needs_backend(spec)
        results << [:skip, test_id, nil]
        next
      end
      stdin_data = if spec[:stdin_bytes]
                     spec[:stdin_bytes]
                   elsif spec[:stdin]
                     spec[:stdin]
                   else
                     ""
                   end
      result = run_inprocess(spec[:args], stdin_data: stdin_data)
      err = check_cli_assertions(result, spec[:assertions])
      results << (err ? [:fail, test_id, err] : [:pass, test_id, nil])
    end
  end
  results
end

def run_phase_tests(test_dir, phase_name, cfg)
  results = []
  Dir.glob(File.join(test_dir, "*.tests")).sort.each do |f|
    stem = File.basename(f, ".tests")
    parse_spec_file(f).each do |name, input, expected|
      test_id = "#{stem}/#{name}"
      lenient = %w[parse inference taytsh_parse taytsh_check].include?(phase_name)
      phase_result = run_transpiled_phase(
        input, cfg[:args],
        is_taytsh: cfg[:taytsh],
        expect_json: cfg[:json]
      )
      if phase_name == "inference" && phase_result[:errors].empty? && phase_result[:data]
        if phase_result[:data].is_a?(Hash) && phase_result[:data].key?("reveals")
          reveals = phase_result[:data]["reveals"].map { |r| [r["line"], r["type"]] }
          phase_result[:reveals] = reveals
        end
      end
      err = check_expected(expected, phase_result, phase_name, lenient_errors: lenient)
      results << (err ? [:fail, test_id, err] : [:pass, test_id, nil])
    end
  end
  results
end

def run_lowering_tests(test_dir)
  results = []
  Dir.glob(File.join(test_dir, "*.tests")).sort.each do |f|
    stem = File.basename(f, ".tests")
    parse_spec_file(f).each do |name, input, expected|
      test_id = "#{stem}/#{name}"
      tmp = Tempfile.new(["test", ".py"])
      begin
        tmp.write(input)
        tmp.flush
        result = run_inprocess(["--stop-at", "lowering-text", tmp.path])
      ensure
        tmp.close
        tmp.unlink
      end
      if expected.start_with?("error:")
        expected_msg = expected[6..].strip
        if result[:exit] == 0
          results << [:fail, test_id, "Expected error containing '#{expected_msg}', got success"]
          next
        end
        stderr = result[:stderr].strip.split("\n")[0] || ""
        if !expected_msg.empty? && !stderr.downcase.include?(expected_msg.downcase)
          results << [:fail, test_id, "Expected error containing '#{expected_msg}', got: #{stderr}"]
          next
        end
        results << [:pass, test_id, nil]
        next
      end
      if result[:exit] != 0
        err_msg = result[:stderr].strip.split("\n")[0] || "lowering failed"
        results << [:fail, test_id, "Lowering error: #{err_msg}"]
        next
      end
      output = result[:stdout]
      unless contains_normalized(output, expected)
        results << [:fail, test_id, "Expected not found in output:\n--- expected ---\n#{expected}\n--- got ---\n#{output}"]
        next
      end
      results << [:pass, test_id, nil]
    end
  end
  results
end

def run_codegen_tests(test_dir)
  results = []
  base_dir = File.join(test_dir, "base")
  return results unless File.directory?(base_dir)
  lang_dirs = Dir.children(test_dir)
    .select { |d| d != "base" && File.directory?(File.join(test_dir, d)) }
    .select { |d| EMITTER_LANGS.include?(d) }
    .sort
  lang_dirs.each do |lang|
    lang_dir = File.join(test_dir, lang)
    Dir.glob(File.join(base_dir, "*.tests")).sort.each do |base_file|
      basename = File.basename(base_file)
      stem = File.basename(base_file, ".tests")
      lang_file = File.join(lang_dir, basename)
      base_tests = parse_simple_tests(base_file)
      next if base_tests.empty?
      unless File.exist?(lang_file)
        base_tests.each do |name, _|
          results << [:fail, "#{stem}/#{name}[#{lang}]", "#{lang}/#{basename} missing"]
        end
        next
      end
      lang_tests = parse_simple_tests(lang_file)
      base_names = base_tests.map(&:first)
      lang_names = lang_tests.map(&:first)
      if base_names != lang_names
        base_tests.each do |name, _|
          results << [:fail, "#{stem}/#{name}[#{lang}]", "base/lang name mismatch"]
        end
        next
      end
      lang_by_name = lang_tests.to_h
      base_tests.each do |name, source|
        test_id = "#{stem}/#{name}[#{lang}]"
        expected = lang_by_name[name]
        # Transpile via taytsh --emit
        tmp = Tempfile.new(["test", ".ty"])
        begin
          tmp.write(source)
          tmp.flush
          result = run_inprocess(["taytsh", "--emit", lang, tmp.path])
        ensure
          tmp.close
          tmp.unlink
        end
        if result[:exit] != 0
          stderr = result[:stderr].strip.split("\n")[0] || "transpile failed"
          results << [:fail, test_id, "Transpile error: #{stderr}"]
          next
        end
        output = result[:stdout]
        unless contains_normalized(output, expected)
          results << [:fail, test_id, "Expected not found in output:\n--- expected ---\n#{expected}\n--- got ---\n#{output}"]
          next
        end
        results << [:pass, test_id, nil]
      end
    end
  end
  results
end

def run_emit_tests(test_dir)
  results = []
  base_dir = File.join(test_dir, "base")
  return results unless File.directory?(base_dir)
  lang_dirs = Dir.children(test_dir)
    .select { |d| d != "base" && File.directory?(File.join(test_dir, d)) }
    .select { |d| EMITTER_LANGS.include?(d) }
    .sort
  lang_dirs.each do |lang|
    lang_dir = File.join(test_dir, lang)
    Dir.glob(File.join(base_dir, "*.tests")).sort.each do |base_file|
      basename = File.basename(base_file)
      stem = File.basename(base_file, ".tests")
      lang_file = File.join(lang_dir, basename)
      base_tests = parse_simple_tests(base_file)
      next if base_tests.empty?
      unless File.exist?(lang_file)
        next
      end
      lang_tests = parse_simple_tests(lang_file)
      lang_by_name = lang_tests.to_h
      base_tests.each do |name, source|
        next unless lang_by_name.key?(name)
        test_id = "#{stem}/#{name}[#{lang}]"
        expected = lang_by_name[name]
        tmp = Tempfile.new(["test", ".py"])
        begin
          tmp.write(source)
          tmp.flush
          result = run_inprocess(["--target", lang, tmp.path])
        ensure
          tmp.close
          tmp.unlink
        end
        if result[:exit] != 0
          stderr = result[:stderr].strip.split("\n")[0] || "emit failed"
          results << [:fail, test_id, "Emit error: #{stderr}"]
          next
        end
        output = result[:stdout]
        unless contains_normalized(output, expected)
          results << [:fail, test_id, "Expected not found in output:\n--- expected ---\n#{expected}\n--- got ---\n#{output}"]
          next
        end
        results << [:pass, test_id, nil]
      end
    end
  end
  results
end

def run_app_tests(test_dir)
  results = []
  available = RUNTIMES.select { |_, cmd| system("which", cmd[0], out: File::NULL, err: File::NULL) }.keys
  Dir.glob(File.join(test_dir, "apptest_*.py")).sort.each do |test_file|
    stem = File.basename(test_file, ".py")
    source = File.read(test_file)
    available.each do |target|
      test_id = "#{stem}[#{target}]"
      tmp = Tempfile.new(["test", ".py"])
      begin
        tmp.write(source)
        tmp.flush
        result = run_inprocess(["--target", target, tmp.path])
      ensure
        tmp.close
        tmp.unlink
      end
      if result[:exit] != 0
        stderr = result[:stderr].strip.split("\n")[0] || "transpile failed"
        results << [:fail, test_id, "Transpile error (#{target}): #{stderr}"]
        next
      end
      transpiled_code = result[:stdout]
      runtime = RUNTIMES[target]
      io = IO.popen([*runtime], "r+", err: [:child, :out])
      io.write(transpiled_code)
      io.close_write
      output = io.read
      io.close
      exit_code = $?.exitstatus
      if exit_code != 0
        results << [:fail, test_id, "App test failed with exit #{exit_code}\n#{output}"]
        next
      end
      results << [:pass, test_id, nil]
    end
  end
  results
end

def run_taytsh_app_tests(test_dir)
  results = []
  Dir.glob(File.join(test_dir, "*.ty")).sort.each do |test_file|
    stem = File.basename(test_file, ".ty")
    test_id = stem
    result = run_inprocess(["taytsh", test_file])
    if result[:exit] != 0
      output = (result[:stdout] + result[:stderr]).strip
      results << [:fail, test_id, "Exit code #{result[:exit]}:\n#{output}"]
      next
    end
    results << [:pass, test_id, nil]
  end
  results
end

def run_ordering_tests(test_dir)
  results = []
  available = RUNTIMES.select { |_, cmd| system("which", cmd[0], out: File::NULL, err: File::NULL) }.keys
  Dir.glob(File.join(test_dir, "*.ty")).sort.each do |test_file|
    stem = File.basename(test_file, ".ty")
    available.each do |target|
      test_id = "#{stem}[#{target}]"
      result = run_inprocess(["taytsh", "--emit", target, test_file])
      if result[:exit] != 0
        stderr = result[:stderr].strip.split("\n")[0] || "transpile failed"
        results << [:fail, test_id, "Transpile error (#{target}): #{stderr}"]
        next
      end
      transpiled_code = result[:stdout]
      runtime = RUNTIMES[target]
      io = IO.popen([*runtime], "r+", err: [:child, :out])
      io.write(transpiled_code)
      io.close_write
      output = io.read
      io.close
      exit_code = $?.exitstatus
      if exit_code != 0
        results << [:fail, test_id, "Ordering test failed with exit #{exit_code}\n#{output}"]
        next
      end
      results << [:pass, test_id, nil]
    end
  end
  results
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if ARGV.length < 1
  $stderr.puts "Usage: ruby test-transpiled.rb <transpiled.rb>"
  exit 1
end

transpiled_path = File.expand_path(ARGV[0], TONGUES_DIR)
unless File.exist?(transpiled_path)
  $stderr.puts "Transpiled file not found: #{transpiled_path}"
  exit 1
end

puts "Loading transpiled binary: #{transpiled_path}"
t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
begin
  load transpiled_path
rescue SyntaxError => e
  $stderr.puts "Failed to load transpiled binary: syntax error"
  $stderr.puts e.message.split("\n").first(5).join("\n")
  exit 1
end
t1 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
puts "Loaded in #{"%.1f" % (t1 - t0)}s"
puts

total_pass = 0
total_fail = 0
total_skip = 0
failures = []

TESTS.each do |section_name, phases|
  phases.each do |phase_name, cfg|
    test_dir = File.join(TESTS_DIR, cfg[:dir])
    next unless File.directory?(test_dir)
    phase_results = case cfg[:run]
                    when :cli
                      run_cli_tests(test_dir)
                    when :phase
                      run_phase_tests(test_dir, phase_name, cfg)
                    when :lowering
                      run_lowering_tests(test_dir)
                    when :codegen
                      run_codegen_tests(test_dir)
                    when :emit
                      run_emit_tests(test_dir)
                    when :app
                      run_app_tests(test_dir)
                    when :taytsh_app
                      run_taytsh_app_tests(test_dir)
                    when :ordering
                      run_ordering_tests(test_dir)
                    else
                      []
                    end
    pass = phase_results.count { |s, _, _| s == :pass }
    fail_count = phase_results.count { |s, _, _| s == :fail }
    skip = phase_results.count { |s, _, _| s == :skip }
    total_pass += pass
    total_fail += fail_count
    total_skip += skip
    status = fail_count > 0 ? "FAIL" : "ok"
    counts = "#{pass} passed"
    counts += ", #{fail_count} failed" if fail_count > 0
    counts += ", #{skip} skipped" if skip > 0
    puts "#{phase_name}: #{status} (#{counts})"
    phase_results.each do |s, tid, err|
      if s == :fail
        failures << [phase_name, tid, err]
        puts "  FAIL #{tid}"
      end
    end
  end
end

puts
if !failures.empty?
  puts "=" * 60
  puts "FAILURES"
  puts "=" * 60
  failures.each do |phase, tid, err|
    puts
    puts "#{phase} :: #{tid}"
    puts err
  end
  puts
end

puts "=" * 60
total = total_pass + total_fail + total_skip
puts "#{total} tests: #{total_pass} passed, #{total_fail} failed, #{total_skip} skipped"
puts "=" * 60

exit(total_fail > 0 ? 1 : 0)
