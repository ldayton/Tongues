#!/usr/bin/env perl
# Native Perl test harness for transpiled Tongues binaries.
# Loads the transpiled file once, then runs all .tests cases in-process.
# Shared parsing/assertion logic is transpiled from tests/shared/test_harness.py.

use v5.36;
use utf8;
use File::Basename;
use File::Spec;
use File::Temp qw(tempfile);
use Time::HiRes qw(time);
use IPC::Open3;
use POSIX ();
use Scalar::Util qw(blessed);
use Symbol qw(gensym);

my $TONGUES_DIR = File::Spec->rel2abs(File::Spec->catdir(dirname(__FILE__), ".."));
my $TESTS_DIR = File::Spec->catdir($TONGUES_DIR, "tests");
my $LIB_DIR = File::Spec->catdir($TONGUES_DIR, "src", "lib");

# Phase -> test config
# Runners: cli, linker, phase, lowering, codegen, emit, app, ordering, ty_app
my @TESTS = (
    ["cli", [
        ["cli", { dir => "frontend/cli", run => "cli" }],
    ]],
    ["linker", [
        ["linker", { dir => "frontend/linker", run => "linker" }],
    ]],
    ["frontend", [
        ["parse",     { dir => "frontend/parse",      run => "phase", taytsh => 0, args => ["--stop-at", "parse"],      json => 1  }],
        ["subset",    { dir => "frontend/subset",      run => "phase", taytsh => 0, args => ["--stop-at", "subset"],     json => 0  }],
        ["names",     { dir => "frontend/names",       run => "phase", taytsh => 0, args => ["--stop-at", "names"],      json => 1  }],
        ["sigs",      { dir => "frontend/signatures",  run => "phase", taytsh => 0, args => ["--stop-at", "signatures"], json => 1  }],
        ["fields",    { dir => "frontend/fields",      run => "phase", taytsh => 0, args => ["--stop-at", "fields"],     json => 1  }],
        ["hierarchy", { dir => "frontend/hierarchy",   run => "phase", taytsh => 0, args => ["--stop-at", "hierarchy"],  json => 1  }],
        ["pycheck",   { dir => "frontend/pycheck",     run => "phase", taytsh => 0, args => ["--stop-at", "pycheck"],    json => 1  }],
        ["lowering",  { dir => "frontend/lowering",    run => "lowering" }],
    ]],
    ["middleend", [
        ["scope",         { dir => "middleend/scope",          run => "phase", taytsh => 1, args => ["--stop-at", "scope"],     json => 1  }],
        ["returns",       { dir => "middleend/returns",        run => "phase", taytsh => 1, args => ["--stop-at", "returns"],   json => 1  }],
        ["liveness",      { dir => "middleend/liveness",       run => "phase", taytsh => 1, args => ["--stop-at", "liveness"],  json => 1  }],
        ["strings",       { dir => "middleend/strings",        run => "phase", taytsh => 1, args => ["--stop-at", "strings"],   json => 1  }],
        ["hoisting",      { dir => "middleend/hoisting",       run => "phase", taytsh => 1, args => ["--stop-at", "hoisting"],  json => 1  }],
        ["ownership",     { dir => "middleend/ownership",      run => "phase", taytsh => 1, args => ["--stop-at", "ownership"], json => 1  }],
        ["callgraph",     { dir => "middleend/callgraph",      run => "phase", taytsh => 1, args => ["--stop-at", "callgraph"], json => 1  }],
    ]],
    ["backend", [
        ["codegen",  { dir => "backend/codegen",  run => "codegen" }],
        ["emit",     { dir => "backend/emit",     run => "emit" }],
        ["app",      { dir => "backend/app",      run => "app" }],
        ["ordering", { dir => "backend/ordering", run => "ordering" }],
    ]],
    ["taytsh", [
        ["typarse", { dir => "taytsh/typarse",  run => "phase", taytsh => 1, args => ["--stop-at", "parse"], json => 1  }],
        ["tycheck", { dir => "taytsh/tycheck",  run => "phase", taytsh => 1, args => ["--stop-at", "check"], json => 1  }],
        ["ty_app",  { dir => "taytsh/app",      run => "ty_app" }],
    ]],
);

my @EMITTER_LANGS = ("perl");
my %RUNTIMES = (
    perl => ["perl"],
);

# ---------------------------------------------------------------------------
# Exit capture
# ---------------------------------------------------------------------------

our $FORK_MODE = 0;
BEGIN {
    *CORE::GLOBAL::exit = sub {
        if ($FORK_MODE) {
            close STDOUT; close STDERR;
            POSIX::_exit($_[0] // 0);
        }
        CORE::exit($_[0] // 0);
    };
}

# ---------------------------------------------------------------------------
# VM mode: parse + compile .ty once, invoke per test
# ---------------------------------------------------------------------------

my $_vm_compiled = undef;

sub load_vm_module ($ty_path) {
    my $source = _read_file($ty_path);
    my $module = taytsh_taytsh_parse($source);
    $_vm_compiled = vm_prepare($module);
    say "VM module compiled";
}

sub run_vm_inprocess ($argv, $stdin_data = "") {
    my $placeholder = bless {}, "VM";
    my $builtins = _BuiltinDispatch->new($placeholder, {});
    my $instance = VM->new($_vm_compiled, [], [], [], [], [], [], "", 0, [], {}, $builtins, undef, {});
    $builtins->{vm} = $instance;
    my $result = $instance->invoke($stdin_data, $argv);
    return {
        stdout => $result->{stdout} // "",
        stderr => $result->{stderr} // "",
        exit   => $result->{exit_code} // 0,
    };
}

# ---------------------------------------------------------------------------
# In-process execution (fork per test, 3s timeout)
# ---------------------------------------------------------------------------

my $run_inprocess_fn = \&_run_inprocess_fork;

sub run_inprocess ($argv, $stdin_data = "") {
    return $run_inprocess_fn->($argv, $stdin_data);
}

sub _run_inprocess_fork ($argv, $stdin_data = "") {
    my ($out_fh, $out_file) = tempfile("out_XXXXXX", TMPDIR => 1);
    my ($err_fh, $err_file) = tempfile("err_XXXXXX", TMPDIR => 1);
    my ($in_fh, $in_file) = tempfile("in_XXXXXX", TMPDIR => 1);
    close $out_fh;
    close $err_fh;
    print $in_fh $stdin_data;
    close $in_fh;
    my $pid = fork();
    die "fork failed: $!" unless defined $pid;
    if ($pid == 0) {
        open(STDOUT, ">", $out_file) or POSIX::_exit(99);
        open(STDERR, ">", $err_file) or POSIX::_exit(99);
        STDOUT->autoflush(1);
        STDERR->autoflush(1);
        open(STDIN, "<", $in_file) or POSIX::_exit(99);
        @ARGV = @$argv;
        $FORK_MODE = 1;
        eval { main() };
        if ($@) {
            my $err = $@;
            if (ref($err) && ref($err) eq 'HASH' || blessed($err)) {
                $err = $err->{msg} // $err->{message} // "$err";
            }
            print STDERR $err;
            close STDOUT; close STDERR;
            POSIX::_exit(1);
        }
        close STDOUT; close STDERR;
        POSIX::_exit(0);
    }
    my $timed_out = 0;
    eval {
        local $SIG{ALRM} = sub { die "TIMEOUT\n" };
        alarm(3);
        waitpid($pid, 0);
        alarm(0);
    };
    if ($@ && $@ eq "TIMEOUT\n") {
        $timed_out = 1;
        kill 9, $pid;
        waitpid($pid, 0);
    }
    my $exit_raw = $?;
    my $code;
    if ($timed_out) {
        $code = 1;
    } elsif ($exit_raw & 127) {
        $code = 1;
    } else {
        $code = $exit_raw >> 8;
    }
    my $out = do { local $/; open my $f, "<", $out_file; $f ? <$f> : "" };
    my $err_out = do { local $/; open my $f, "<", $err_file; $f ? <$f> : "" };
    $err_out .= "TIMEOUT after 3s\n" if $timed_out;
    unlink $out_file, $err_file, $in_file;
    return { stdout => $out // "", stderr => $err_out // "", exit => $code };
}

sub run_transpiled_phase ($source, $cli_args, %opts) {
    my $is_taytsh = $opts{is_taytsh};
    my $expect_json = $opts{expect_json} // 1;
    my $suffix = $is_taytsh ? ".ty" : ".py";
    my ($fh, $tmpfile) = tempfile("testXXXXXX", SUFFIX => $suffix, TMPDIR => 1);
    print $fh $source;
    close $fh;
    my @argv;
    if ($is_taytsh) {
        @argv = ("taytsh", @$cli_args, $tmpfile);
    } else {
        @argv = (@$cli_args, $tmpfile);
    }
    my $result = run_inprocess(\@argv);
    unlink $tmpfile;
    my $stderr_text = $result->{stderr};
    $stderr_text =~ s/^\s+|\s+$//g;
    if ($result->{exit} != 0) {
        my @errors = grep { $_ ne "" } split(/\n/, $stderr_text);
        return { errors => \@errors, warnings => [], data => undef, reveals => [] };
    }
    my @warnings = $stderr_text eq "" ? () : grep { $_ ne "" } split(/\n/, $stderr_text);
    unless ($expect_json) {
        return { errors => [], warnings => \@warnings, data => undef, reveals => [] };
    }
    my $stdout_text = $result->{stdout};
    $stdout_text =~ s/^\s+|\s+$//g;
    if ($stdout_text eq "") {
        return { errors => [], warnings => \@warnings, data => undef, reveals => [] };
    }
    my $data;
    eval { $data = json_parse($stdout_text); };
    if ($@) {
        my $preview = substr($stdout_text, 0, 200);
        return { errors => ["Invalid JSON output: $preview"], warnings => [], data => undef, reveals => [] };
    }
    return { errors => [], warnings => \@warnings, data => $data, reveals => [] };
}

# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

sub _read_file ($path) {
    open(my $fh, "<", $path) or die "Cannot open $path: $!";
    my $content = do { local $/; <$fh> };
    close $fh;
    return $content;
}

sub run_cli_tests ($test_dir) {
    my @results;
    for my $f (sort glob("$test_dir/*.tests")) {
        my $stem = basename($f, ".tests");
        my $tests = parse_cli_test_file(_read_file($f));
        for my $t (@$tests) {
            my ($name, $spec) = @$t;
            my $test_id = "$stem/$name";
            if (cli_needs_backend($spec->{args}, $spec->{assertions}, \@EMITTER_LANGS)) {
                push @results, ["skip", $test_id, undef];
                next;
            }
            my $stdin_data;
            if ($spec->{stdin_hex} ne "") {
                $stdin_data = pack("H*", $spec->{stdin_hex});
            } else {
                $stdin_data = $spec->{stdin};
            }
            my $result = run_inprocess($spec->{args}, $stdin_data);
            my $err = check_cli_assertions($result->{exit}, $result->{stdout}, $result->{stderr}, $spec->{assertions});
            push @results, [$err eq "" ? "pass" : "fail", $test_id, $err eq "" ? undef : $err];
        }
    }
    return \@results;
}

sub run_linker_tests ($test_dir) {
    my @results;
    for my $f (sort glob("$test_dir/*.tests")) {
        my $stem = basename($f, ".tests");
        my $tests = parse_linker_test_file(_read_file($f));
        for my $t (@$tests) {
            my ($name, $spec) = @$t;
            my $test_id = "$stem/$name";
            my @parts;
            for my $lf (@{$spec->{files}}) {
                push @parts, $lf->{path}, $lf->{source};
            }
            my $stdin_data = join("\0", @parts);
            my @args = @{$spec->{args}};
            if (grep { $_ eq "--target" } @args) {
                my $target_idx;
                for my $i (0 .. $#args) {
                    if ($args[$i] eq "--target") {
                        $target_idx = $i;
                        last;
                    }
                }
                if (defined $target_idx) {
                    my $target = $args[$target_idx + 1];
                    unless (grep { $_ eq $target } @EMITTER_LANGS) {
                        push @results, ["skip", $test_id, undef];
                        next;
                    }
                }
            }
            my $result = run_inprocess($spec->{args}, $stdin_data);
            my $err = check_cli_assertions($result->{exit}, $result->{stdout}, $result->{stderr}, $spec->{assertions});
            push @results, [$err eq "" ? "pass" : "fail", $test_id, $err eq "" ? undef : $err];
        }
    }
    return \@results;
}

sub run_phase_tests ($test_dir, $phase_name, $cfg) {
    my @results;
    my $pattern = $cfg->{glob} // "*.tests";
    for my $f (sort glob("$test_dir/$pattern")) {
        my $stem = basename($f, ".tests");
        my $tests = parse_spec_file(_read_file($f));
        for my $entry (@$tests) {
            my $test_id = "$stem/$entry->{name}";
            my $lenient = ($phase_name =~ /^(parse|pycheck|typarse|tycheck)$/);
            my $phase_result = run_transpiled_phase(
                $entry->{input}, $cfg->{args},
                is_taytsh => $cfg->{taytsh},
                expect_json => $cfg->{json},
            );
            my $reveals = $phase_result->{reveals};
            if ($phase_name =~ /^(pycheck|tycheck)$/ && !@{$phase_result->{errors}} && defined $phase_result->{data}) {
                if (blessed($phase_result->{data}) && $phase_result->{data}->isa("JsonObject")) {
                    eval {
                        my $reveals_arr = json_get_items(json_get_field($phase_result->{data}, "reveals"));
                        $reveals = [map { [int(json_get_number(json_get_field($_, "line"))), json_get_string(json_get_field($_, "type"))] } @$reveals_arr];
                    };
                }
            }
            my $err = check_expected($entry->{expected}, $phase_result->{errors}, $phase_result->{warnings},
                                     $phase_result->{data}, $reveals, $phase_name, $lenient ? 1 : 0);
            push @results, [$err eq "" ? "pass" : "fail", $test_id, $err eq "" ? undef : $err];
        }
    }
    return \@results;
}

sub run_lowering_tests ($test_dir) {
    my @results;
    for my $f (sort glob("$test_dir/*.tests")) {
        my $stem = basename($f, ".tests");
        my $tests = parse_spec_file(_read_file($f));
        for my $entry (@$tests) {
            my $test_id = "$stem/$entry->{name}";
            my ($fh, $tmpfile) = tempfile("testXXXXXX", SUFFIX => ".py", TMPDIR => 1);
            print $fh $entry->{input};
            close $fh;
            my $result = run_inprocess(["--stop-at", "lowering-text", $tmpfile]);
            unlink $tmpfile;
            if ($entry->{expected} =~ /^error:(.*)/) {
                my $expected_msg = $1;
                $expected_msg =~ s/^\s+|\s+$//g;
                if ($result->{exit} == 0) {
                    push @results, ["fail", $test_id, "Expected error containing '$expected_msg', got success"];
                    next;
                }
                my $stderr = $result->{stderr};
                $stderr =~ s/^\s+|\s+$//g;
                my $first_line = (split(/\n/, $stderr))[0] // "";
                if ($expected_msg ne "" && index(lc($first_line), lc($expected_msg)) < 0) {
                    push @results, ["fail", $test_id, "Expected error containing '$expected_msg', got: $first_line"];
                    next;
                }
                push @results, ["pass", $test_id, undef];
                next;
            }
            if ($result->{exit} != 0) {
                my $stderr = $result->{stderr};
                $stderr =~ s/^\s+|\s+$//g;
                my $err_msg = (split(/\n/, $stderr))[0] // "lowering failed";
                push @results, ["fail", $test_id, "Lowering error: $err_msg"];
                next;
            }
            my $output = $result->{stdout};
            unless (contains_normalized($output, $entry->{expected})) {
                push @results, ["fail", $test_id, "Expected not found in output:\n--- expected ---\n$entry->{expected}\n--- got ---\n$output"];
                next;
            }
            push @results, ["pass", $test_id, undef];
        }
    }
    return \@results;
}

sub run_codegen_tests ($test_dir) {
    my @results;
    my $base_dir = File::Spec->catdir($test_dir, "base");
    return \@results unless -d $base_dir;
    my @lang_dirs;
    opendir(my $dh, $test_dir) or return \@results;
    for my $d (sort readdir($dh)) {
        next if $d eq "base" || $d eq "." || $d eq "..";
        next unless -d File::Spec->catdir($test_dir, $d);
        next unless grep { $_ eq $d } @EMITTER_LANGS;
        push @lang_dirs, $d;
    }
    closedir $dh;
    for my $lang (@lang_dirs) {
        my $lang_dir = File::Spec->catdir($test_dir, $lang);
        for my $base_file (sort glob("$base_dir/*.tests")) {
            my $basename_file = basename($base_file);
            my $stem = basename($base_file, ".tests");
            my $lang_file = File::Spec->catfile($lang_dir, $basename_file);
            my $base_tests = parse_simple_tests(_read_file($base_file));
            next unless @$base_tests;
            unless (-f $lang_file) {
                for my $entry (@$base_tests) {
                    push @results, ["fail", "$stem/$entry->{name}[$lang]", "$lang/$basename_file missing"];
                }
                next;
            }
            my $lang_tests = parse_simple_tests(_read_file($lang_file));
            my @base_names = map { $_->{name} } @$base_tests;
            my @lang_names = map { $_->{name} } @$lang_tests;
            if ("@base_names" ne "@lang_names") {
                for my $entry (@$base_tests) {
                    push @results, ["fail", "$stem/$entry->{name}[$lang]", "base/lang name mismatch"];
                }
                next;
            }
            my %lang_by_name;
            for my $lt (@$lang_tests) {
                $lang_by_name{$lt->{name}} = $lt->{content};
            }
            for my $entry (@$base_tests) {
                my $test_id = "$stem/$entry->{name}" . "[$lang]";
                my $expected = $lang_by_name{$entry->{name}};
                my ($fh, $tmpfile) = tempfile("testXXXXXX", SUFFIX => ".ty", TMPDIR => 1);
                print $fh $entry->{content};
                close $fh;
                my $result = run_inprocess(["taytsh", "--emit", $lang, $tmpfile]);
                unlink $tmpfile;
                if ($result->{exit} != 0) {
                    my $stderr = $result->{stderr};
                    $stderr =~ s/^\s+|\s+$//g;
                    my $err = (split(/\n/, $stderr))[0] // "transpile failed";
                    push @results, ["fail", $test_id, "Transpile error: $err"];
                    next;
                }
                my $output = $result->{stdout};
                unless (contains_normalized($output, $expected)) {
                    push @results, ["fail", $test_id, "Expected not found in output:\n--- expected ---\n$expected\n--- got ---\n$output"];
                    next;
                }
                push @results, ["pass", $test_id, undef];
            }
        }
    }
    return \@results;
}

sub run_emit_tests ($test_dir) {
    my @results;
    my $base_dir = File::Spec->catdir($test_dir, "base");
    return \@results unless -d $base_dir;
    my @lang_dirs;
    opendir(my $dh, $test_dir) or return \@results;
    for my $d (sort readdir($dh)) {
        next if $d eq "base" || $d eq "." || $d eq "..";
        next unless -d File::Spec->catdir($test_dir, $d);
        next unless grep { $_ eq $d } @EMITTER_LANGS;
        push @lang_dirs, $d;
    }
    closedir $dh;
    for my $lang (@lang_dirs) {
        my $lang_dir = File::Spec->catdir($test_dir, $lang);
        for my $base_file (sort glob("$base_dir/*.tests")) {
            my $basename_file = basename($base_file);
            my $stem = basename($base_file, ".tests");
            my $lang_file = File::Spec->catfile($lang_dir, $basename_file);
            my $base_tests = parse_simple_tests(_read_file($base_file));
            next unless @$base_tests;
            next unless -f $lang_file;
            my $lang_tests = parse_simple_tests(_read_file($lang_file));
            my %lang_by_name;
            for my $lt (@$lang_tests) {
                $lang_by_name{$lt->{name}} = $lt->{content};
            }
            for my $entry (@$base_tests) {
                next unless exists $lang_by_name{$entry->{name}};
                my $test_id = "$stem/$entry->{name}" . "[$lang]";
                my $expected = $lang_by_name{$entry->{name}};
                my ($fh, $tmpfile) = tempfile("testXXXXXX", SUFFIX => ".py", TMPDIR => 1);
                print $fh $entry->{content};
                close $fh;
                my $result = run_inprocess(["--target", $lang, $tmpfile]);
                unlink $tmpfile;
                if ($result->{exit} != 0) {
                    my $stderr = $result->{stderr};
                    $stderr =~ s/^\s+|\s+$//g;
                    my $err = (split(/\n/, $stderr))[0] // "emit failed";
                    push @results, ["fail", $test_id, "Emit error: $err"];
                    next;
                }
                my $output = $result->{stdout};
                unless (contains_normalized($output, $expected)) {
                    push @results, ["fail", $test_id, "Expected not found in output:\n--- expected ---\n$expected\n--- got ---\n$output"];
                    next;
                }
                push @results, ["pass", $test_id, undef];
            }
        }
    }
    return \@results;
}

sub run_app_tests ($test_dir) {
    my @results;
    my @available;
    for my $lang (keys %RUNTIMES) {
        my $cmd = $RUNTIMES{$lang}[0];
        if (system("which $cmd >/dev/null 2>&1") == 0) {
            push @available, $lang;
        }
    }
    @available = sort @available;
    for my $test_file (sort glob("$test_dir/apptest_*.py")) {
        my $stem = basename($test_file, ".py");
        my $source = _read_file($test_file);
        my $lib_names = find_lib_imports($source);
        # Transitively resolve cross-lib imports
        my %seen = map { $_ => 1 } @$lib_names;
        my @queue = @$lib_names;
        while (@queue) {
            my $name = shift @queue;
            my $lib_path = File::Spec->catfile($LIB_DIR, "$name.py");
            next unless -f $lib_path;
            my $deps = find_lib_imports(_read_file($lib_path));
            for my $dep (@$deps) {
                unless ($seen{$dep}) {
                    $seen{$dep} = 1;
                    push @$lib_names, $dep;
                    push @queue, $dep;
                }
            }
        }
        for my $target (@available) {
            my $test_id = "$stem" . "[$target]";
            my $result;
            if (@$lib_names == 0) {
                my ($tfh, $tmpfile) = tempfile("testXXXXXX", SUFFIX => ".py", TMPDIR => 1);
                print $tfh $source;
                close $tfh;
                $result = run_inprocess(["--target", $target, $tmpfile]);
                unlink $tmpfile;
            } else {
                my @lib_sources;
                for my $name (@$lib_names) {
                    my $lib_path = File::Spec->catfile($LIB_DIR, "$name.py");
                    my $lib_src = _read_file($lib_path);
                    push @lib_sources, ["lib/$name.py", $lib_src];
                }
                my $stdin_data = build_project_input("apptest.py", $source, \@lib_sources);
                $result = run_inprocess(["--project", "--target", $target], $stdin_data);
            }
            if ($result->{exit} != 0) {
                my $stderr = $result->{stderr};
                $stderr =~ s/^\s+|\s+$//g;
                my $err = (split(/\n/, $stderr))[0] // "transpile failed";
                push @results, ["fail", $test_id, "Transpile error ($target): $err"];
                next;
            }
            my $transpiled_code = $result->{stdout};
            my @runtime = @{$RUNTIMES{$target}};
            my $err_fh = gensym;
            my $pid = open3(my $child_in, my $child_out, $err_fh, @runtime);
            print $child_in $transpiled_code;
            close $child_in;
            my $output = do { local $/; <$child_out> };
            my $child_err = do { local $/; <$err_fh> };
            waitpid($pid, 0);
            my $exit_code = $? >> 8;
            if ($exit_code != 0) {
                push @results, ["fail", $test_id, "App test failed with exit $exit_code\n$output$child_err"];
                next;
            }
            push @results, ["pass", $test_id, undef];
        }
    }
    return \@results;
}

sub run_ty_app_tests ($test_dir) {
    my @results;
    for my $test_file (sort glob("$test_dir/*.ty")) {
        my $stem = basename($test_file, ".ty");
        my $test_id = $stem;
        my $result = run_inprocess(["taytsh", $test_file]);
        if ($result->{exit} != 0) {
            my $output = $result->{stdout} . $result->{stderr};
            $output =~ s/^\s+|\s+$//g;
            push @results, ["fail", $test_id, "Exit code $result->{exit}:\n$output"];
            next;
        }
        push @results, ["pass", $test_id, undef];
    }
    return \@results;
}

sub run_ordering_tests ($test_dir) {
    my @results;
    my @available;
    for my $lang (keys %RUNTIMES) {
        my $cmd = $RUNTIMES{$lang}[0];
        if (system("which $cmd >/dev/null 2>&1") == 0) {
            push @available, $lang;
        }
    }
    @available = sort @available;
    for my $test_file (sort glob("$test_dir/*.ty")) {
        my $stem = basename($test_file, ".ty");
        for my $target (@available) {
            my $test_id = "$stem" . "[$target]";
            my $result = run_inprocess(["taytsh", "--emit", $target, $test_file]);
            if ($result->{exit} != 0) {
                my $stderr = $result->{stderr};
                $stderr =~ s/^\s+|\s+$//g;
                my $err = (split(/\n/, $stderr))[0] // "transpile failed";
                push @results, ["fail", $test_id, "Transpile error ($target): $err"];
                next;
            }
            my $transpiled_code = $result->{stdout};
            my @runtime = @{$RUNTIMES{$target}};
            my $err_fh = gensym;
            my $pid = open3(my $child_in, my $child_out, $err_fh, @runtime);
            print $child_in $transpiled_code;
            close $child_in;
            my $output = do { local $/; <$child_out> };
            my $child_err = do { local $/; <$err_fh> };
            waitpid($pid, 0);
            my $exit_code = $? >> 8;
            if ($exit_code != 0) {
                push @results, ["fail", $test_id, "Ordering test failed with exit $exit_code\n$output$child_err"];
                next;
            }
            push @results, ["pass", $test_id, undef];
        }
    }
    return \@results;
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if (@ARGV < 1) {
    print STDERR "Usage: perl test-transpiled.pl <transpiled.pl> [--via-vm <tongues.ty>] [--target <name>]\n";
    exit 1;
}

my $via_vm_path = undef;
my $target_name = undef;
my @filtered_argv;
for (my $i = 0; $i < @ARGV; $i++) {
    if ($ARGV[$i] eq "--via-vm") {
        if ($i + 1 >= @ARGV) {
            print STDERR "--via-vm requires a path to a .ty file\n";
            exit 1;
        }
        $via_vm_path = File::Spec->rel2abs($ARGV[$i + 1], $TONGUES_DIR);
        $i++;
    } elsif ($ARGV[$i] eq "--target") {
        if ($i + 1 >= @ARGV) {
            print STDERR "--target requires a name\n";
            exit 1;
        }
        $target_name = $ARGV[$i + 1];
        $i++;
    } else {
        push @filtered_argv, $ARGV[$i];
    }
}

my $transpiled_path = File::Spec->rel2abs($filtered_argv[0], $TONGUES_DIR);
unless (-f $transpiled_path) {
    print STDERR "Transpiled file not found: $transpiled_path\n";
    exit 1;
}

say "Loading transpiled binary: $transpiled_path";
my $t0 = time();
eval {
    do $transpiled_path;
    die $@ if $@;
};
if ($@) {
    print STDERR "Failed to load transpiled binary:\n";
    my @lines = split(/\n/, "$@");
    print STDERR join("\n", @lines[0 .. ($#lines < 4 ? $#lines : 4)]) . "\n";
    exit 1;
}
my $t1 = time();
printf("Loaded in %.1fs\n", $t1 - $t0);

if (defined $via_vm_path) {
    unless (-f $via_vm_path) {
        print STDERR "VM module not found: $via_vm_path\n";
        exit 1;
    }
    say "Loading VM module: $via_vm_path";
    my $vm_t0 = time();
    load_vm_module($via_vm_path);
    printf("VM compiled in %.1fs\n", time() - $vm_t0);
    $run_inprocess_fn = \&run_vm_inprocess;
}

my $harness_path = File::Spec->catfile($TONGUES_DIR, ".out", "test_harness.pl");
unless (-f $harness_path) {
    print STDERR "Transpiled harness not found: $harness_path\n";
    exit 1;
}
eval {
    do $harness_path;
    die $@ if $@;
};
if ($@) {
    print STDERR "Failed to load transpiled harness:\n$@\n";
    exit 1;
}
say "";

my $total_pass = 0;
my $total_fail = 0;
my $total_skip = 0;
my @failures;

for my $section (@TESTS) {
    my ($section_name, $phases) = @$section;
    for my $phase_entry (@$phases) {
        my ($phase_name, $cfg) = @$phase_entry;
        my $test_dir = File::Spec->catdir($TESTS_DIR, $cfg->{dir});
        next unless -d $test_dir;
        my $phase_results;
        my $runner = $cfg->{run};
        if ($runner eq "cli") {
            $phase_results = run_cli_tests($test_dir);
        } elsif ($runner eq "linker") {
            $phase_results = run_linker_tests($test_dir);
        } elsif ($runner eq "phase") {
            $phase_results = run_phase_tests($test_dir, $phase_name, $cfg);
        } elsif ($runner eq "lowering") {
            $phase_results = run_lowering_tests($test_dir);
        } elsif ($runner eq "codegen") {
            $phase_results = run_codegen_tests($test_dir);
        } elsif ($runner eq "emit") {
            $phase_results = run_emit_tests($test_dir);
        } elsif ($runner eq "app") {
            $phase_results = run_app_tests($test_dir);
        } elsif ($runner eq "ty_app") {
            $phase_results = run_ty_app_tests($test_dir);
        } elsif ($runner eq "ordering") {
            $phase_results = run_ordering_tests($test_dir);
        } else {
            $phase_results = [];
        }
        my $pass = scalar grep { $_->[0] eq "pass" } @$phase_results;
        my $fail_count = scalar grep { $_->[0] eq "fail" } @$phase_results;
        my $skip = scalar grep { $_->[0] eq "skip" } @$phase_results;
        $total_pass += $pass;
        $total_fail += $fail_count;
        $total_skip += $skip;
        my $status = $fail_count > 0 ? "FAIL" : "ok";
        my $counts = "$pass passed";
        $counts .= ", $fail_count failed" if $fail_count > 0;
        $counts .= ", $skip skipped" if $skip > 0;
        say "$phase_name: $status ($counts)";
        for my $r (@$phase_results) {
            if ($r->[0] eq "fail") {
                push @failures, [$phase_name, $r->[1], $r->[2]];
                say "  FAIL $r->[1]";
            }
        }
    }
}

say "";
if (@failures) {
    say "=" x 60;
    say $target_name ? "FAILURES [$target_name]" : "FAILURES";
    say "=" x 60;
    for my $f (@failures) {
        my ($phase, $tid, $err) = @$f;
        say "";
        say "$phase :: $tid";
        say $err;
    }
    say "";
}

say "=" x 60;
my $total = $total_pass + $total_fail + $total_skip;
my $prefix = $target_name ? "[$target_name] " : "";
say "${prefix}$total tests: $total_pass passed, $total_fail failed, $total_skip skipped";
say "=" x 60;

exit($total_fail > 0 ? 1 : 0);
