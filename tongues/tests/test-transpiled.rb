#!/usr/bin/env ruby
# frozen_string_literal: true

# Native Ruby test harness for transpiled Tongues binaries.
# Loads the transpiled file once, then runs all .tests cases in-process.
# Shared parsing/assertion logic is transpiled from tests/shared/test_harness.py.
#
# Supports parallel execution with -n <num> or -n auto (like pytest-xdist).

require "stringio"
require "tempfile"
require "fileutils"
require "parallel"
require "timeout"

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

EMITTER_LANGS = %w[ruby]
RUNTIMES = {
  "ruby" => ["ruby"],
}

# ---------------------------------------------------------------------------
# VM mode: parse + compile .ty once, invoke per test
# ---------------------------------------------------------------------------

$vm_compiled = nil

def load_vm_module(ty_path)
  source = File.read(ty_path, encoding: "BINARY")
  mod = taytsh_taytsh_parse(source)
  $vm_compiled = vm_prepare(mod)
  puts "VM module compiled"
end

def run_vm_inprocess(argv, stdin_data: "")
  old_stdin = $stdin
  begin
    $stdin = StringIO.new(stdin_data)
    builtins = XBuiltinDispatch.allocate
    builtins.instance_variable_set(:@vm, nil)
    builtins.instance_variable_set(:@_table, {})
    instance = VM.new(module_: $vm_compiled, builtins: builtins)
    builtins.vm = instance
    result = instance.invoke(stdin_data, ["tongues"] + argv)
    { stdout: result.stdout.to_s, stderr: result.stderr.to_s, exit: result.exit_code }
  ensure
    $stdin = old_stdin
  end
end

# ---------------------------------------------------------------------------
# In-process execution
# ---------------------------------------------------------------------------

$use_vm = false

def run_inprocess(argv, stdin_data: "")
  return run_vm_inprocess(argv, stdin_data: stdin_data) if $use_vm
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
    TONGUES_MAIN.call
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
      annotations = {}
      if %w[pycheck tycheck].include?(phase_name) && phase_result[:errors].empty? && phase_result[:data]
        if phase_result[:data].is_a?(JsonObject)
          begin
            reveals_arr = json_get_items(json_get_field(phase_result[:data], "reveals"))
            reveals = reveals_arr.map { |r| [json_get_number(json_get_field(r, "line")).to_i, json_get_string(json_get_field(r, "type"))] }
          rescue
          end
          begin
            anns_obj = json_get_field(phase_result[:data], "annotations")
            if anns_obj.is_a?(JsonObject)
              anns_obj.entries.each do |line_str, line_anns|
                line_dict = {}
                if line_anns.is_a?(JsonObject)
                  line_anns.entries.each do |k, v|
                    line_dict[k] = v.value if v.is_a?(JsonString)
                  end
                end
                annotations[line_str.to_i] = line_dict
              end
            end
          rescue
          end
        end
      end
      err = check_expected(entry.expected, phase_result[:errors], phase_result[:warnings],
                           phase_result[:data], reveals, annotations, phase_name, lenient)
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

# Set of "stem|target" combos expected to fail (see known-failures.txt)
def load_known_failures(test_dir)
  path = File.join(test_dir, "known-failures.txt")
  known = Set.new
  return known unless File.exist?(path)
  File.read(path).each_line do |raw|
    line = raw.strip
    next if line.empty? || line.start_with?("#")
    tokens = line.split
    known << "#{tokens[0]}|#{tokens[1]}"
  end
  known
end

def run_app_tests(test_dir)
  results = []
  available = RUNTIMES.select { |_, cmd| system("which", cmd[0], out: File::NULL, err: File::NULL) }.keys
  known_failures = load_known_failures(test_dir)
  Dir.glob(File.join(test_dir, "apptest_*.py")).sort.each do |test_file|
    stem = File.basename(test_file, ".py")
    source = File.read(test_file)
    lib_names = find_lib_imports(source)
    # Transitively resolve cross-lib imports
    seen = lib_names.to_a.dup
    queue = lib_names.to_a.dup
    until queue.empty?
      name = queue.shift
      lib_path = File.join(LIB_DIR, "#{name}.py")
      next unless File.exist?(lib_path)
      find_lib_imports(File.read(lib_path)).each do |dep|
        unless seen.include?(dep)
          seen << dep
          queue << dep
        end
      end
    end
    lib_names = seen
    available.each do |target|
      test_id = "#{stem}[#{target}]"
      if known_failures.include?("#{stem}|#{target}")
        results << [:skip, test_id, nil]
        next
      end
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
# Parallel execution support
# ---------------------------------------------------------------------------

def get_cpu_count
  Parallel.processor_count
end

def collect_tests
  collected = []
  TESTS.each do |section_name, phases|
    phases.each do |phase_name, cfg|
      test_dir = File.join(TESTS_DIR, cfg[:dir])
      next unless File.directory?(test_dir)
      case cfg[:run]
      when :cli
        Dir.glob(File.join(test_dir, "*.tests")).sort.each do |f|
          stem = File.basename(f, ".tests")
          parse_cli_test_file(File.read(f)).each do |name, spec|
            test_id = "#{stem}/#{name}"
            if cli_needs_backend(spec.args, spec.assertions, EMITTER_LANGS)
              collected << [phase_name, test_id, :skip, nil]
            else
              collected << [phase_name, test_id, :cli, spec]
            end
          end
        end
      when :phase
        Dir.glob(File.join(test_dir, "*.tests")).sort.each do |f|
          stem = File.basename(f, ".tests")
          parse_spec_file(File.read(f)).each do |entry|
            test_id = "#{stem}/#{entry.name}"
            collected << [phase_name, test_id, :phase, [entry, cfg]]
          end
        end
      when :lowering
        Dir.glob(File.join(test_dir, "*.tests")).sort.each do |f|
          stem = File.basename(f, ".tests")
          parse_spec_file(File.read(f)).each do |entry|
            test_id = "#{stem}/#{entry.name}"
            collected << [phase_name, test_id, :lowering, entry]
          end
        end
      when :linker
        Dir.glob(File.join(test_dir, "*.tests")).sort.each do |f|
          stem = File.basename(f, ".tests")
          parse_linker_test_file(File.read(f)).each do |name, spec|
            test_id = "#{stem}/#{name}"
            if spec.args.include?("--target")
              target = spec.args[spec.args.index("--target") + 1]
              unless EMITTER_LANGS.include?(target)
                collected << [phase_name, test_id, :skip, nil]
                next
              end
            end
            collected << [phase_name, test_id, :linker, spec]
          end
        end
      when :codegen
        base_dir = File.join(test_dir, "base")
        next unless File.directory?(base_dir)
        lang_dirs = Dir.children(test_dir)
          .select { |d| d != "base" && File.directory?(File.join(test_dir, d)) && EMITTER_LANGS.include?(d) }
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
                collected << [phase_name, "#{stem}/#{entry.name}[#{lang}]", :prefail, "#{lang}/#{basename} missing"]
              end
              next
            end
            lang_tests = parse_simple_tests(File.read(lang_file))
            base_names = base_tests.map(&:name)
            lang_names = lang_tests.map(&:name)
            if base_names != lang_names
              base_tests.each do |entry|
                collected << [phase_name, "#{stem}/#{entry.name}[#{lang}]", :prefail, "base/lang name mismatch"]
              end
              next
            end
            lang_by_name = lang_tests.to_h { |e| [e.name, e.content] }
            base_tests.each do |entry|
              test_id = "#{stem}/#{entry.name}[#{lang}]"
              collected << [phase_name, test_id, :codegen, [lang, entry.content, lang_by_name[entry.name]]]
            end
          end
        end
      when :emit
        base_dir = File.join(test_dir, "base")
        next unless File.directory?(base_dir)
        lang_dirs = Dir.children(test_dir)
          .select { |d| d != "base" && File.directory?(File.join(test_dir, d)) && EMITTER_LANGS.include?(d) }
          .sort
        lang_dirs.each do |lang|
          lang_dir = File.join(test_dir, lang)
          Dir.glob(File.join(base_dir, "*.tests")).sort.each do |base_file|
            basename = File.basename(base_file)
            stem = File.basename(base_file, ".tests")
            lang_file = File.join(lang_dir, basename)
            base_tests = parse_simple_tests(File.read(base_file))
            next if base_tests.empty?
            next unless File.exist?(lang_file)
            lang_tests = parse_simple_tests(File.read(lang_file))
            lang_by_name = lang_tests.to_h { |e| [e.name, e.content] }
            base_tests.each do |entry|
              next unless lang_by_name.key?(entry.name)
              test_id = "#{stem}/#{entry.name}[#{lang}]"
              collected << [phase_name, test_id, :emit, [lang, entry.content, lang_by_name[entry.name]]]
            end
          end
        end
      when :app
        available = RUNTIMES.select { |_, cmd| system("which", cmd[0], out: File::NULL, err: File::NULL) }.keys
        known_failures = load_known_failures(test_dir)
        Dir.glob(File.join(test_dir, "apptest_*.py")).sort.each do |test_file|
          stem = File.basename(test_file, ".py")
          source = File.read(test_file)
          lib_names = find_lib_imports(source)
          seen = lib_names.to_a.dup
          queue = lib_names.to_a.dup
          until queue.empty?
            name = queue.shift
            lib_path = File.join(LIB_DIR, "#{name}.py")
            next unless File.exist?(lib_path)
            find_lib_imports(File.read(lib_path)).each do |dep|
              unless seen.include?(dep)
                seen << dep
                queue << dep
              end
            end
          end
          lib_names = seen
          lib_parts = lib_names.map do |name|
            lib_path = File.join(LIB_DIR, "#{name}.py")
            ["lib/#{name}.py", File.read(lib_path)]
          end
          available.each do |target|
            test_id = "#{stem}[#{target}]"
            if known_failures.include?("#{stem}|#{target}")
              collected << [phase_name, test_id, :skip, nil]
              next
            end
            collected << [phase_name, test_id, :app, [target, source, lib_parts]]
          end
        end
      when :ordering
        available = RUNTIMES.select { |_, cmd| system("which", cmd[0], out: File::NULL, err: File::NULL) }.keys
        Dir.glob(File.join(test_dir, "*.ty")).sort.each do |test_file|
          stem = File.basename(test_file, ".ty")
          available.each do |target|
            test_id = "#{stem}[#{target}]"
            collected << [phase_name, test_id, :ordering, [target, test_file]]
          end
        end
      when :ty_app
        Dir.glob(File.join(test_dir, "*.ty")).sort.each do |test_file|
          stem = File.basename(test_file, ".ty")
          collected << [phase_name, stem, :ty_app, test_file]
        end
      end
    end
  end
  collected
end

def run_single_test(phase_name, test_id, test_type, test_data)
  case test_type
  when :skip
    [phase_name, test_id, :skip, nil]
  when :prefail
    [phase_name, test_id, :fail, test_data]
  when :cli
    spec = test_data
    stdin_data = !spec.stdin_hex.empty? ? [spec.stdin_hex].pack("H*") : spec.stdin
    result = run_inprocess(spec.args, stdin_data: stdin_data)
    err = check_cli_assertions(result[:exit], result[:stdout], result[:stderr], spec.assertions)
    err.empty? ? [phase_name, test_id, :pass, nil] : [phase_name, test_id, :fail, err]
  when :phase
    entry, cfg = test_data
    lenient = %w[parse pycheck typarse tycheck].include?(phase_name)
    phase_result = run_transpiled_phase(entry.input, cfg[:args], is_taytsh: cfg[:taytsh], expect_json: cfg[:json])
    reveals = phase_result[:reveals]
    annotations = {}
    if %w[pycheck tycheck].include?(phase_name) && phase_result[:errors].empty? && phase_result[:data]
      if phase_result[:data].is_a?(JsonObject)
        begin
          reveals_arr = json_get_items(json_get_field(phase_result[:data], "reveals"))
          reveals = reveals_arr.map { |r| [json_get_number(json_get_field(r, "line")).to_i, json_get_string(json_get_field(r, "type"))] }
        rescue
        end
        begin
          anns_obj = json_get_field(phase_result[:data], "annotations")
          if anns_obj.is_a?(JsonObject)
            anns_obj.entries.each do |line_str, line_anns|
              line_dict = {}
              if line_anns.is_a?(JsonObject)
                line_anns.entries.each do |k, v|
                  line_dict[k] = v.value if v.is_a?(JsonString)
                end
              end
              annotations[line_str.to_i] = line_dict
            end
          end
        rescue
        end
      end
    end
    err = check_expected(entry.expected, phase_result[:errors], phase_result[:warnings], phase_result[:data], reveals, annotations, phase_name, lenient)
    err.empty? ? [phase_name, test_id, :pass, nil] : [phase_name, test_id, :fail, err]
  when :lowering
    entry = test_data
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
        return [phase_name, test_id, :fail, "Expected error containing '#{expected_msg}', got success"]
      end
      stderr = result[:stderr].strip.split("\n")[0] || ""
      if !expected_msg.empty? && !stderr.downcase.include?(expected_msg.downcase)
        return [phase_name, test_id, :fail, "Expected error containing '#{expected_msg}', got: #{stderr}"]
      end
      return [phase_name, test_id, :pass, nil]
    end
    if result[:exit] != 0
      err_msg = result[:stderr].strip.split("\n")[0] || "lowering failed"
      return [phase_name, test_id, :fail, "Lowering error: #{err_msg}"]
    end
    output = result[:stdout]
    unless contains_normalized(output, entry.expected)
      return [phase_name, test_id, :fail, "Expected not found in output:\n--- expected ---\n#{entry.expected}\n--- got ---\n#{output}"]
    end
    [phase_name, test_id, :pass, nil]
  when :linker
    spec = test_data
    parts = spec.files.flat_map { |lf| [lf.path, lf.source] }
    stdin_data = parts.join("\0")
    result = run_inprocess(spec.args, stdin_data: stdin_data)
    err = check_cli_assertions(result[:exit], result[:stdout], result[:stderr], spec.assertions)
    err.empty? ? [phase_name, test_id, :pass, nil] : [phase_name, test_id, :fail, err]
  when :codegen
    lang, input_content, expected = test_data
    tmp = Tempfile.new(["test", ".ty"])
    begin
      tmp.write(input_content)
      tmp.flush
      result = run_inprocess(["taytsh", "--emit", lang, tmp.path])
    ensure
      tmp.close
      tmp.unlink
    end
    if result[:exit] != 0
      stderr = result[:stderr].strip.split("\n")[0] || "transpile failed"
      return [phase_name, test_id, :fail, "Transpile error: #{stderr}"]
    end
    output = result[:stdout]
    unless contains_normalized(output, expected)
      return [phase_name, test_id, :fail, "Expected not found in output:\n--- expected ---\n#{expected}\n--- got ---\n#{output}"]
    end
    [phase_name, test_id, :pass, nil]
  when :emit
    lang, input_content, expected = test_data
    tmp = Tempfile.new(["test", ".py"])
    begin
      tmp.write(input_content)
      tmp.flush
      result = run_inprocess(["--target", lang, tmp.path])
    ensure
      tmp.close
      tmp.unlink
    end
    if result[:exit] != 0
      stderr = result[:stderr].strip.split("\n")[0] || "emit failed"
      return [phase_name, test_id, :fail, "Emit error: #{stderr}"]
    end
    output = result[:stdout]
    unless contains_normalized(output, expected)
      return [phase_name, test_id, :fail, "Expected not found in output:\n--- expected ---\n#{expected}\n--- got ---\n#{output}"]
    end
    [phase_name, test_id, :pass, nil]
  when :app
    target, source, lib_parts = test_data
    if lib_parts.empty?
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
      stdin_data = build_project_input("apptest.py", source, lib_parts)
      result = run_inprocess(["--project", "--target", target], stdin_data: stdin_data)
    end
    if result[:exit] != 0
      stderr = result[:stderr].strip.split("\n")[0] || "transpile failed"
      return [phase_name, test_id, :fail, "Transpile error (#{target}): #{stderr}"]
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
      return [phase_name, test_id, :fail, "App test failed with exit #{exit_code}\n#{output}"]
    end
    [phase_name, test_id, :pass, nil]
  when :ordering
    target, test_file = test_data
    result = run_inprocess(["taytsh", "--emit", target, test_file])
    if result[:exit] != 0
      stderr = result[:stderr].strip.split("\n")[0] || "transpile failed"
      return [phase_name, test_id, :fail, "Transpile error (#{target}): #{stderr}"]
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
      return [phase_name, test_id, :fail, "Ordering test failed with exit #{exit_code}\n#{output}"]
    end
    [phase_name, test_id, :pass, nil]
  when :ty_app
    test_file = test_data
    result = run_inprocess(["taytsh", test_file])
    if result[:exit] != 0
      output = (result[:stdout] + result[:stderr]).strip
      return [phase_name, test_id, :fail, "Exit code #{result[:exit]}:\n#{output}"]
    end
    [phase_name, test_id, :pass, nil]
  else
    [phase_name, test_id, :fail, "Unknown test type: #{test_type}"]
  end
rescue => e
  [phase_name, test_id, :fail, "Exception: #{e}\n#{e.backtrace.first(5).join("\n")}"]
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if ARGV.length < 1
  $stderr.puts "Usage: ruby test-transpiled.rb <transpiled.rb> [options]"
  $stderr.puts "Options:"
  $stderr.puts "  --via-vm <tongues.ty>  Run tests through the VM"
  $stderr.puts "  --target <name>        Set target name for reporting"
  $stderr.puts "  -n <num|auto>          Number of parallel workers (default: auto)"
  exit 1
end

via_vm_path = nil
$target_name = nil
$num_workers = get_cpu_count
filtered_argv = []
i = 0
while i < ARGV.length
  if ARGV[i] == "--via-vm"
    if i + 1 >= ARGV.length
      $stderr.puts "--via-vm requires a path to a .ty file"
      exit 1
    end
    via_vm_path = File.expand_path(ARGV[i + 1], TONGUES_DIR)
    i += 2
  elsif ARGV[i] == "--target"
    if i + 1 >= ARGV.length
      $stderr.puts "--target requires a name"
      exit 1
    end
    $target_name = ARGV[i + 1]
    i += 2
  elsif ARGV[i] == "-n"
    if i + 1 >= ARGV.length
      $stderr.puts "-n requires a number or 'auto'"
      exit 1
    end
    val = ARGV[i + 1]
    $num_workers = val == "auto" ? get_cpu_count : val.to_i
    if $num_workers < 1
      $stderr.puts "Worker count must be positive"
      exit 1
    end
    i += 2
  else
    filtered_argv << ARGV[i]
    i += 1
  end
end

transpiled_path = File.expand_path(filtered_argv[0], TONGUES_DIR)
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
# The transpiled harness also defines main (its self-test entrypoint) and
# overwrites the compiler's on load; keep a reference to the compiler's.
TONGUES_MAIN = method(:main)
puts "Loaded in #{"%.1f" % (t1 - t0)}s"

if via_vm_path
  unless File.exist?(via_vm_path)
    $stderr.puts "VM module not found: #{via_vm_path}"
    exit 1
  end
  puts "Loading VM module: #{via_vm_path}"
  vm_t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  load_vm_module(via_vm_path)
  puts "VM compiled in #{"%.1f" % (Process.clock_gettime(Process::CLOCK_MONOTONIC) - vm_t0)}s"
  $use_vm = true
end

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

# Collect all tests
collected = collect_tests
puts "Collected #{collected.length} tests"
puts "Running with #{$num_workers} workers"

vm_tag = $use_vm ? "[vm] " : ""
t_start = Process.clock_gettime(Process::CLOCK_MONOTONIC)

# Run tests in parallel with timeout (longer for VM mode)
test_timeout = $use_vm ? 60 : 10
results = Parallel.map(collected, in_processes: $num_workers) do |test|
  phase_name, test_id, test_type, test_data = test
  begin
    Timeout.timeout(test_timeout) do
      run_single_test(phase_name, test_id, test_type, test_data)
    end
  rescue Timeout::Error
    [phase_name, test_id, :fail, "Test timed out after #{test_timeout}s"]
  rescue => e
    [phase_name, test_id, :fail, "Worker error: #{e}\n#{e.backtrace.first(3).join("\n")}"]
  end
end

# Print results and count
results.each do |phase_name, test_id, status, err|
  case status
  when :pass
    puts "PASS #{vm_tag}#{phase_name}::#{test_id}"
    total_pass += 1
  when :skip
    puts "SKIP #{vm_tag}#{phase_name}::#{test_id}"
    total_skip += 1
  else
    puts "FAIL #{vm_tag}#{phase_name}::#{test_id}"
    if err
      err.split("\n").each { |line| puts "  #{line}" }
    end
    total_fail += 1
    failures << [phase_name, test_id, err]
  end
end

t_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - t_start
puts "Completed in #{"%.1f" % t_elapsed}s"

puts
if !failures.empty?
  puts "=" * 60
  puts $target_name ? "FAILURES [#{$target_name}]" : "FAILURES"
  puts "=" * 60
  failures.each do |phase, tid, err|
    puts "  #{phase}::#{tid}"
  end
  puts
end

puts "=" * 60
total = total_pass + total_fail + total_skip
prefix = $target_name ? "[#{$target_name}] " : ""
summary_line = "#{prefix}#{total} tests: #{total_pass} passed, #{total_fail} failed, #{total_skip} skipped"
puts summary_line
puts "=" * 60

# GitHub Actions notice annotation
if total_fail == 0
  puts "::notice::#{summary_line}"
end

# GitHub Actions job summary
summary_file = ENV["GITHUB_STEP_SUMMARY"]
if summary_file
  status_emoji = total_fail == 0 ? "✅" : "❌"
  File.open(summary_file, "a") do |f|
    f.puts "## #{status_emoji} #{$target_name || 'Test Results'}\n"
    f.puts "| Passed | Failed | Skipped | Total |"
    f.puts "|--------|--------|---------|-------|"
    f.puts "| #{total_pass} | #{total_fail} | #{total_skip} | #{total} |\n"
    if !failures.empty?
      f.puts "### Failures\n"
      failures.each do |phase, tid, err|
        f.puts "<details><summary><code>#{phase} :: #{tid}</code></summary>\n"
        f.puts "```\n#{err}\n```\n"
        f.puts "</details>\n"
      end
    end
  end
end

exit(total_fail > 0 ? 1 : 0)
