#!/usr/bin/env ruby
# frozen_string_literal: true

# Native Ruby test harness for transpiled Tongues binaries.
# Loads the transpiled file once, then runs all .tests cases in-process.
# Shared parsing/assertion logic is transpiled from tests/shared/test_harness.py.

require "stringio"
require "tempfile"
require "fileutils"

TONGUES_DIR = File.expand_path("..", __dir__)
TESTS_DIR = File.join(TONGUES_DIR, "tests")
LIB_DIR = File.join(TONGUES_DIR, "src", "lib")

# Phase → test config: [dir, runner, is_taytsh, cli_args, expect_json]
# Runners: :cli, :linker, :phase, :lowering, :codegen, :emit, :app, :ordering, :ty_app
TESTS = {
  "cli" => {
    "cli" => { dir: "frontend/cli", run: :cli },
  },
  "linker" => {
    "linker" => { dir: "frontend/linker", run: :linker },
  },
  "frontend" => {
    "parse"     => { dir: "frontend/parse",      run: :phase, taytsh: false, args: ["--stop-at", "parse"],      json: true  },
    "subset"    => { dir: "frontend/subset",      run: :phase, taytsh: false, args: ["--stop-at", "subset"],     json: false },
    "names"     => { dir: "frontend/names",       run: :phase, taytsh: false, args: ["--stop-at", "names"],      json: true  },
    "sigs"      => { dir: "frontend/signatures",  run: :phase, taytsh: false, args: ["--stop-at", "signatures"], json: true  },
    "fields"    => { dir: "frontend/fields",      run: :phase, taytsh: false, args: ["--stop-at", "fields"],     json: true  },
    "hierarchy" => { dir: "frontend/hierarchy",   run: :phase, taytsh: false, args: ["--stop-at", "hierarchy"],  json: true  },
    "pycheck"   => { dir: "frontend/pycheck",     run: :phase, taytsh: false, args: ["--stop-at", "pycheck"],    json: true  },
    "lowering"  => { dir: "frontend/lowering",    run: :lowering },
  },
  "middleend" => {
    "scope"         => { dir: "middleend/scope",          run: :phase, taytsh: true, args: ["--stop-at", "scope"],     json: true  },
    "returns"       => { dir: "middleend/returns",        run: :phase, taytsh: true, args: ["--stop-at", "returns"],   json: true  },
    "liveness"      => { dir: "middleend/liveness",       run: :phase, taytsh: true, args: ["--stop-at", "liveness"],  json: true  },
    "strings"       => { dir: "middleend/strings",        run: :phase, taytsh: true, args: ["--stop-at", "strings"],   json: true  },
    "hoisting"      => { dir: "middleend/hoisting",       run: :phase, taytsh: true, args: ["--stop-at", "hoisting"],  json: true  },
    "ownership"     => { dir: "middleend/ownership",      run: :phase, taytsh: true, args: ["--stop-at", "ownership"], json: true  },
    "callgraph"     => { dir: "middleend/callgraph",      run: :phase, taytsh: true, args: ["--stop-at", "callgraph"], json: true  },
  },
  "backend" => {
    "codegen"  => { dir: "backend/codegen",  run: :codegen },
    "emit"     => { dir: "backend/emit",     run: :emit },
    "app"      => { dir: "backend/app",      run: :app },
    "ordering" => { dir: "backend/ordering", run: :ordering },
  },
  "taytsh" => {
    "typarse" => { dir: "taytsh/typarse",  run: :phase, taytsh: true, args: ["--stop-at", "parse"], json: true  },
    "tycheck" => { dir: "taytsh/tycheck",  run: :phase, taytsh: true, args: ["--stop-at", "check"], json: true  },
    "ty_app"  => { dir: "taytsh/app",      run: :ty_app },
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
    data = json_parse(stdout_text)
  rescue => e
    return { errors: ["Invalid JSON output: #{stdout_text[0, 200]}"], warnings: [], data: nil, reveals: [] }
  end
  { errors: [], warnings: warnings, data: data, reveals: [] }
end

# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def run_cli_tests(test_dir)
  results = []
  Dir.glob(File.join(test_dir, "*.tests")).sort.each do |f|
    stem = File.basename(f, ".tests")
    parse_cli_test_file(File.read(f)).each do |name, spec|
      test_id = "#{stem}/#{name}"
      if cli_needs_backend(spec.args, spec.assertions, EMITTER_LANGS)
        results << [:skip, test_id, nil]
        next
      end
      stdin_data = if !spec.stdin_hex.empty?
                     [spec.stdin_hex].pack("H*")
                   else
                     spec.stdin
                   end
      result = run_inprocess(spec.args, stdin_data: stdin_data)
      err = check_cli_assertions(result[:exit], result[:stdout], result[:stderr], spec.assertions)
      results << (err.empty? ? [:pass, test_id, nil] : [:fail, test_id, err])
    end
  end
  results
end

def run_linker_tests(test_dir)
  results = []
  Dir.glob(File.join(test_dir, "*.tests")).sort.each do |f|
    stem = File.basename(f, ".tests")
    parse_linker_test_file(File.read(f)).each do |name, spec|
      test_id = "#{stem}/#{name}"
      parts = spec.files.flat_map { |lf| [lf.path, lf.source] }
      stdin_data = parts.join("\0")
      args = spec.args
      if args.include?("--target")
        target = args[args.index("--target") + 1]
        unless EMITTER_LANGS.include?(target)
          results << [:skip, test_id, nil]
          next
        end
      end
      result = run_inprocess(args, stdin_data: stdin_data)
      err = check_cli_assertions(result[:exit], result[:stdout], result[:stderr], spec.assertions)
      results << (err.empty? ? [:pass, test_id, nil] : [:fail, test_id, err])
    end
  end
  results
end

def run_phase_tests(test_dir, phase_name, cfg)
  results = []
  pattern = cfg[:glob] || "*.tests"
  Dir.glob(File.join(test_dir, pattern)).sort.each do |f|
    stem = File.basename(f, ".tests")
    parse_spec_file(File.read(f)).each do |entry|
      test_id = "#{stem}/#{entry.name}"
      lenient = %w[parse pycheck typarse tycheck].include?(phase_name)
      phase_result = run_transpiled_phase(
        entry.input, cfg[:args],
        is_taytsh: cfg[:taytsh],
        expect_json: cfg[:json]
      )
      reveals = phase_result[:reveals]
      if %w[pycheck tycheck].include?(phase_name) && phase_result[:errors].empty? && phase_result[:data]
        if phase_result[:data].is_a?(JsonObject)
          begin
            reveals_arr = get_items(get_field(phase_result[:data], "reveals"))
            reveals = reveals_arr.map { |r| [get_number(get_field(r, "line")).to_i, get_string(get_field(r, "type"))] }
          rescue
          end
        end
      end
      err = check_expected(entry.expected, phase_result[:errors], phase_result[:warnings],
                           phase_result[:data], reveals, phase_name, lenient)
      results << (err.empty? ? [:pass, test_id, nil] : [:fail, test_id, err])
    end
  end
  results
end

def run_lowering_tests(test_dir)
  results = []
  Dir.glob(File.join(test_dir, "*.tests")).sort.each do |f|
    stem = File.basename(f, ".tests")
    parse_spec_file(File.read(f)).each do |entry|
      test_id = "#{stem}/#{entry.name}"
      tmp = Tempfile.new(["test", ".py"])
      begin
        tmp.write(entry.input)
        tmp.flush
        result = run_inprocess(["--stop-at", "lowering-text", tmp.path])
      ensure
        tmp.close
        tmp.unlink
      end
      if entry.expected.start_with?("error:")
        expected_msg = entry.expected[6..].strip
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
      unless contains_normalized(output, entry.expected)
        results << [:fail, test_id, "Expected not found in output:\n--- expected ---\n#{entry.expected}\n--- got ---\n#{output}"]
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
      base_tests = parse_simple_tests(File.read(base_file))
      next if base_tests.empty?
      unless File.exist?(lang_file)
        base_tests.each do |entry|
          results << [:fail, "#{stem}/#{entry.name}[#{lang}]", "#{lang}/#{basename} missing"]
        end
        next
      end
      lang_tests = parse_simple_tests(File.read(lang_file))
      base_names = base_tests.map(&:name)
      lang_names = lang_tests.map(&:name)
      if base_names != lang_names
        base_tests.each do |entry|
          results << [:fail, "#{stem}/#{entry.name}[#{lang}]", "base/lang name mismatch"]
        end
        next
      end
      lang_by_name = lang_tests.to_h { |e| [e.name, e.content] }
      base_tests.each do |entry|
        test_id = "#{stem}/#{entry.name}[#{lang}]"
        expected = lang_by_name[entry.name]
        tmp = Tempfile.new(["test", ".ty"])
        begin
          tmp.write(entry.content)
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
      base_tests = parse_simple_tests(File.read(base_file))
      next if base_tests.empty?
      unless File.exist?(lang_file)
        next
      end
      lang_tests = parse_simple_tests(File.read(lang_file))
      lang_by_name = lang_tests.to_h { |e| [e.name, e.content] }
      base_tests.each do |entry|
        next unless lang_by_name.key?(entry.name)
        test_id = "#{stem}/#{entry.name}[#{lang}]"
        expected = lang_by_name[entry.name]
        tmp = Tempfile.new(["test", ".py"])
        begin
          tmp.write(entry.content)
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
    lib_names = find_lib_imports(source)
    available.each do |target|
      test_id = "#{stem}[#{target}]"
      if lib_names.empty?
        tmp = Tempfile.new(["test", ".py"])
        begin
          tmp.write(source)
          tmp.flush
          result = run_inprocess(["--target", target, tmp.path])
        ensure
          tmp.close
          tmp.unlink
        end
      else
        lib_sources = lib_names.map do |name|
          lib_path = File.join(LIB_DIR, "#{name}.py")
          [lib_path, name]
        end
        parts = []
        lib_sources.each do |lib_path, name|
          parts << ["lib/#{name}.py", File.read(lib_path)]
        end
        stdin_data = build_project_input("apptest.py", source, parts)
        result = run_inprocess(["--project", "--target", target], stdin_data: stdin_data)
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

def run_ty_app_tests(test_dir)
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

harness_path = File.join(TONGUES_DIR, ".out", "test_harness.rb")
unless File.exist?(harness_path)
  $stderr.puts "Transpiled harness not found: #{harness_path}"
  exit 1
end
load harness_path
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
                    when :linker
                      run_linker_tests(test_dir)
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
                    when :ty_app
                      run_ty_app_tests(test_dir)
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
