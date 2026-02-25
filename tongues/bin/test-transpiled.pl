#!/usr/bin/env perl
# Native Perl test harness for transpiled Tongues binaries.
# Loads the transpiled file once, then runs all .tests cases in-process.

use v5.36;
use utf8;
use File::Basename;
use File::Spec;
use File::Temp qw(tempfile);
use JSON::PP ();
use Time::HiRes qw(time);
use IPC::Open3;
use POSIX ();
use Symbol qw(gensym);

my $TONGUES_DIR = File::Spec->rel2abs(File::Spec->catdir(dirname(__FILE__), ".."));
my $TESTS_DIR = File::Spec->catdir($TONGUES_DIR, "tests");

# Phase -> test config
# Runners: cli, phase, lowering, codegen, emit, app, ordering, taytsh_app
my @TESTS = (
    ["cli", [
        ["cli", { dir => "02_cli", run => "cli" }],
    ]],
    ["frontend", [
        ["parse",     { dir => "03_parse",      run => "phase", taytsh => 0, args => ["--stop-at", "parse"],      json => 1  }],
        ["subset",    { dir => "04_subset",      run => "phase", taytsh => 0, args => ["--stop-at", "subset"],     json => 0  }],
        ["names",     { dir => "05_names",       run => "phase", taytsh => 0, args => ["--stop-at", "names"],      json => 1  }],
        ["sigs",      { dir => "06_signatures",  run => "phase", taytsh => 0, args => ["--stop-at", "signatures"], json => 1  }],
        ["fields",    { dir => "07_fields",      run => "phase", taytsh => 0, args => ["--stop-at", "fields"],     json => 1  }],
        ["hierarchy", { dir => "08_hierarchy",   run => "phase", taytsh => 0, args => ["--stop-at", "hierarchy"],  json => 1  }],
        ["inference", { dir => "09_inference",   run => "phase", taytsh => 0, args => ["--stop-at", "inference"],  json => 1  }],
        ["lowering",  { dir => "10_lowering",    run => "lowering" }],
    ]],
    ["middleend", [
        ["type_checking", { dir => "13_type_checking", run => "phase", taytsh => 1, args => ["--stop-at", "check"],     json => 0  }],
        ["scope",         { dir => "14_scope",          run => "phase", taytsh => 1, args => ["--stop-at", "scope"],     json => 1  }],
        ["returns",       { dir => "15_returns",        run => "phase", taytsh => 1, args => ["--stop-at", "returns"],   json => 1  }],
        ["liveness",      { dir => "16_liveness",       run => "phase", taytsh => 1, args => ["--stop-at", "liveness"],  json => 1  }],
        ["strings",       { dir => "17_strings",        run => "phase", taytsh => 1, args => ["--stop-at", "strings"],   json => 1  }],
        ["hoisting",      { dir => "18_hoisting",       run => "phase", taytsh => 1, args => ["--stop-at", "hoisting"],  json => 1  }],
        ["ownership",     { dir => "19_ownership",      run => "phase", taytsh => 1, args => ["--stop-at", "ownership"], json => 1  }],
        ["callgraph",     { dir => "20_callgraph",      run => "phase", taytsh => 1, args => ["--stop-at", "callgraph"], json => 1  }],
    ]],
    ["backend", [
        ["codegen",  { dir => "21_codegen",  run => "codegen" }],
        ["emit",     { dir => "25_emit",     run => "emit" }],
        ["app",      { dir => "22_app",      run => "app" }],
        ["ordering", { dir => "24_ordering", run => "ordering" }],
    ]],
    ["taytsh", [
        ["taytsh_parse", { dir => "11_taytsh_parse", run => "phase", taytsh => 1, args => ["--stop-at", "parse"], json => 1  }],
        ["taytsh_check", { dir => "12_taytsh_check", run => "phase", taytsh => 1, args => ["--stop-at", "check"], json => 0  }],
        ["taytsh_app",   { dir => "23_taytsh_app",   run => "taytsh_app" }],
    ]],
);

my @EMITTER_LANGS = ("python", "perl", "ruby");
my %RUNTIMES = (
    python => ["python3"],
    perl   => ["perl"],
    ruby   => ["ruby"],
);

# ---------------------------------------------------------------------------
# Exit capture: a blessed object so we can distinguish exit() from die()
# Must be installed before loading the transpiled binary so that exit()
# resolves to the override at compile time.
# ---------------------------------------------------------------------------

our $FORK_MODE = 0;
BEGIN {
    *CORE::GLOBAL::exit = sub {
        if ($FORK_MODE) {
            # In forked child: flush and _exit immediately (bypasses eval catches)
            close STDOUT; close STDERR;
            POSIX::_exit($_[0] // 0);
        }
        CORE::exit($_[0] // 0);
    };
}

# ---------------------------------------------------------------------------
# In-process execution (fork per test, 3s timeout)
# ---------------------------------------------------------------------------

sub run_inprocess ($argv, $stdin_data = "") {
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
            print STDERR $@;
            close STDOUT; close STDERR;
            POSIX::_exit(1);
        }
        close STDOUT; close STDERR;
        POSIX::_exit(0);
    }
    # Parent: wait with timeout
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
        $code = 1;  # killed by signal
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
    eval { $data = JSON::PP->new->decode($stdout_text); };
    if ($@) {
        my $preview = substr($stdout_text, 0, 200);
        return { errors => ["Invalid JSON output: $preview"], warnings => [], data => undef, reveals => [] };
    }
    return { errors => [], warnings => \@warnings, data => $data, reveals => [] };
}

# ---------------------------------------------------------------------------
# .tests file parsing
# ---------------------------------------------------------------------------

sub parse_spec_file ($path) {
    open(my $fh, "<", $path) or die "Cannot open $path: $!";
    my @lines = <$fh>;
    close $fh;
    chomp @lines;
    my @result;
    my $i = 0;
    while ($i < scalar @lines) {
        if ($lines[$i] =~ /^=== (.*)/) {
            my $test_name = $1;
            $test_name =~ s/\s+$//;
            $i++;
            my @input_lines;
            while ($i < scalar @lines && $lines[$i] !~ /^---/) {
                push @input_lines, $lines[$i];
                $i++;
            }
            $i++ if $i < scalar @lines && $lines[$i] eq "---";
            my @expected_lines;
            while ($i < scalar @lines && $lines[$i] !~ /^---/) {
                push @expected_lines, $lines[$i];
                $i++;
            }
            $i++ if $i < scalar @lines && $lines[$i] eq "---";
            my $input = join("\n", @input_lines);
            my $expected = join("\n", @expected_lines);
            $expected =~ s/\s+$//;
            push @result, [$test_name, $input, $expected];
        } else {
            $i++;
        }
    }
    return \@result;
}

sub parse_cli_test_file ($path) {
    open(my $fh, "<", $path) or die "Cannot open $path: $!";
    my @lines = <$fh>;
    close $fh;
    chomp @lines;
    my @result;
    my $i = 0;
    while ($i < scalar @lines) {
        if ($lines[$i] =~ /^=== (.*)/) {
            my $test_name = $1;
            $test_name =~ s/\s+$//;
            $i++;
            my @input_lines;
            while ($i < scalar @lines && $lines[$i] !~ /^---/) {
                push @input_lines, $lines[$i];
                $i++;
            }
            $i++ if $i < scalar @lines && $lines[$i] eq "---";
            my @expected_lines;
            while ($i < scalar @lines && $lines[$i] !~ /^---/) {
                push @expected_lines, $lines[$i];
                $i++;
            }
            $i++ if $i < scalar @lines && $lines[$i] eq "---";
            my $spec = parse_cli_spec(\@input_lines, \@expected_lines);
            push @result, [$test_name, $spec];
        } else {
            $i++;
        }
    }
    return \@result;
}

sub parse_cli_spec ($input_lines, $expected_lines) {
    my %spec = (args => [], stdin => undef, stdin_bytes => undef, assertions => []);
    my $body_start = 0;
    if (@$input_lines && $input_lines->[0] =~ /^args:(.*)/) {
        my $args_str = $1;
        $args_str =~ s/^\s+|\s+$//g;
        $spec{args} = $args_str eq "" ? [] : [split(/\s+/, $args_str)];
        $body_start = 1;
    }
    my @remaining = @$input_lines[$body_start .. $#$input_lines];
    if (@remaining && $remaining[0] =~ /^stdin-bytes:(.*)/) {
        my $hex_str = $1;
        $hex_str =~ s/^\s+|\s+$//g;
        $spec{stdin_bytes} = pack("H*", $hex_str);
    } else {
        $spec{stdin} = join("\n", @remaining);
    }
    for my $line (@$expected_lines) {
        my $stripped = $line;
        $stripped =~ s/^\s+|\s+$//g;
        next if $stripped eq "";
        if ($stripped =~ /^exit:(.*)/) {
            push @{$spec{assertions}}, ["exit", int($1 =~ s/^\s+|\s+$//gr)];
        } elsif ($stripped =~ /^exit-not:(.*)/) {
            push @{$spec{assertions}}, ["exit-not", int($1 =~ s/^\s+|\s+$//gr)];
        } elsif ($stripped =~ /^stderr:(.*)/) {
            push @{$spec{assertions}}, ["stderr", $1 =~ s/^\s+|\s+$//gr];
        } elsif ($stripped =~ /^stderr-contains:(.*)/) {
            push @{$spec{assertions}}, ["stderr-contains", $1 =~ s/^\s+|\s+$//gr];
        } elsif ($stripped =~ /^stderr-empty:/) {
            push @{$spec{assertions}}, ["stderr-empty", undef];
        } elsif ($stripped =~ /^stdout-contains:(.*)/) {
            push @{$spec{assertions}}, ["stdout-contains", $1 =~ s/^\s+|\s+$//gr];
        } elsif ($stripped =~ /^stdout-empty:/) {
            push @{$spec{assertions}}, ["stdout-empty", undef];
        }
    }
    return \%spec;
}

sub parse_simple_tests ($path) {
    open(my $fh, "<", $path) or die "Cannot open $path: $!";
    my @lines = <$fh>;
    close $fh;
    chomp @lines;
    my @result;
    my $i = 0;
    while ($i < scalar @lines) {
        if ($lines[$i] =~ /^=== (.*)/) {
            my $name = $1;
            $name =~ s/\s+$//;
            $i++;
            my @content_lines;
            while ($i < scalar @lines && $lines[$i] !~ /^=== /) {
                push @content_lines, $lines[$i];
                $i++;
            }
            my $content = join("\n", @content_lines);
            $content =~ s/\s+$//;
            push @result, [$name, $content];
        } else {
            $i++;
        }
    }
    return \@result;
}

# ---------------------------------------------------------------------------
# Assertion checking
# ---------------------------------------------------------------------------

sub resolve_dotpath ($obj, $dotpath) {
    my @parts = split(/\./, $dotpath);
    my $current = $obj;
    my $i = 0;
    while ($i < scalar @parts) {
        my $part = $parts[$i];
        if ($part eq "length") {
            if (ref($current) eq "ARRAY") {
                return scalar @$current;
            }
            return length($current);
        }
        if (ref($current) eq "ARRAY") {
            $current = $current->[$part];
            $i++;
        } elsif (ref($current) eq "HASH") {
            if (exists $current->{$part}) {
                $current = $current->{$part};
                $i++;
            } else {
                my $found = 0;
                for my $j ($i + 1 .. $#parts) {
                    my $composite = join(".", @parts[$i .. $j]);
                    if (exists $current->{$composite}) {
                        $current = $current->{$composite};
                        $i = $j + 1;
                        $found = 1;
                        last;
                    }
                }
                die "key not found: $part" unless $found;
            }
        } else {
            die "cannot traverse " . ref($current) . " with key '$part'";
        }
    }
    return $current;
}

sub to_comparable ($value) {
    return "null" unless defined $value;
    if (JSON::PP::is_bool($value)) {
        return $value ? "true" : "false";
    }
    if (ref($value) eq "" && _is_numeric($value) && $value =~ /^-?\d+$/) {
        return "$value";
    }
    return "$value";
}

sub _is_numeric ($v) {
    return $v =~ /^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
}

sub check_expected ($expected, $result, $phase, %opts) {
    my $lenient_errors = $opts{lenient_errors} // 0;
    my @reveal_assertions;
    my @verdict_lines;
    for my $line (split(/\n/, $expected)) {
        my $stripped = $line;
        $stripped =~ s/^\s+|\s+$//g;
        if ($stripped =~ /^reveal:(.*)/) {
            my $rest = $1;
            my $eq_pos = index($rest, "=");
            my $lineno = int(substr($rest, 0, $eq_pos) =~ s/^\s+|\s+$//gr);
            my $expected_type = substr($rest, $eq_pos + 1);
            $expected_type =~ s/^\s+|\s+$//g;
            push @reveal_assertions, [$lineno, $expected_type];
        } else {
            push @verdict_lines, $line;
        }
    }
    $expected = join("\n", @verdict_lines);
    $expected =~ s/\s+$//;
    $expected = "ok" if $expected eq "";
    if ($expected eq "ok") {
        if (@{$result->{errors}}) {
            return "Expected ok, got error: $result->{errors}[0]";
        }
        my $err = check_reveals(\@reveal_assertions, $result->{reveals});
        return $err if $err;
        return undef;
    }
    if ($expected =~ /^error:(.*)/) {
        my $expected_msg = $1;
        $expected_msg =~ s/^\s+|\s+$//g;
        if (!@{$result->{errors}}) {
            return "Expected error containing '$expected_msg', got ok";
        }
        if (!$lenient_errors && $expected_msg ne "") {
            my $found = 0;
            for my $e (@{$result->{errors}}) {
                if (index(lc($e), lc($expected_msg)) >= 0) {
                    $found = 1;
                    last;
                }
            }
            return "Expected error containing '$expected_msg', got: @{$result->{errors}}" unless $found;
        }
        return undef;
    }
    if ($expected =~ /^warning:(.*)/) {
        my $expected_msg = $1;
        $expected_msg =~ s/^\s+|\s+$//g;
        if (!@{$result->{warnings}}) {
            return "Expected warning containing '$expected_msg', got none";
        }
        my $found = 0;
        for my $w (@{$result->{warnings}}) {
            if (index(lc($w), lc($expected_msg)) >= 0) {
                $found = 1;
                last;
            }
        }
        return "Expected warning containing '$expected_msg', got: @{$result->{warnings}}" unless $found;
        return undef;
    }
    if (@{$result->{errors}}) {
        return "$phase failed: $result->{errors}[0]";
    }
    if (!defined $result->{data}) {
        return "No data returned from $phase";
    }
    for my $line (split(/\n/, $expected)) {
        $line =~ s/^\s+|\s+$//g;
        next if $line eq "";
        unless ($line =~ /=/) {
            return "Bad assertion (no '='): $line";
        }
        my ($path, $expected_val) = split(/=/, $line, 2);
        $path =~ s/^\s+|\s+$//g;
        $expected_val =~ s/^\s+|\s+$//g;
        my $actual;
        eval { $actual = resolve_dotpath($result->{data}, $path); };
        if ($@) {
            return "Path '$path' not found in result: $@";
        }
        my $actual_str = to_comparable($actual);
        if ($expected_val =~ /\./ && $expected_val !~ / /) {
            eval {
                my $ref_val = resolve_dotpath($result->{data}, $expected_val);
                $expected_val = to_comparable($ref_val);
            };
            # on error, treat as literal
        }
        if ($actual_str ne $expected_val) {
            return "Assertion failed: $path\n  expected: \"$expected_val\"\n  actual:   \"$actual_str\"";
        }
    }
    return undef;
}

sub check_reveals ($assertions, $actuals) {
    for my $a (@$assertions) {
        my ($lineno, $expected_type) = @$a;
        my $found = 0;
        for my $act (@$actuals) {
            my ($actual_line, $actual_type) = @$act;
            if ($actual_line == $lineno) {
                if ($actual_type ne $expected_type) {
                    return "reveal_type at line $lineno: expected '$expected_type', got '$actual_type'";
                }
                $found = 1;
                last;
            }
        }
        return "No reveal_type found at line $lineno" unless $found;
    }
    return undef;
}

sub contains_normalized ($haystack, $needle) {
    my @needle_lines = grep { $_ ne "" } map { s/^\s+|\s+$//gr } split(/\n/, $needle);
    my @haystack_lines = grep { $_ ne "" } map { s/^\s+|\s+$//gr } split(/\n/, $haystack);
    return 1 if !@needle_lines;
    for my $i (0 .. $#haystack_lines) {
        if (index($haystack_lines[$i], $needle_lines[0]) >= 0) {
            my $match = 1;
            for my $j (1 .. $#needle_lines) {
                if ($i + $j > $#haystack_lines || index($haystack_lines[$i + $j], $needle_lines[$j]) < 0) {
                    $match = 0;
                    last;
                }
            }
            return 1 if $match;
        }
    }
    return 0;
}

# ---------------------------------------------------------------------------
# CLI test helpers
# ---------------------------------------------------------------------------

sub cli_needs_backend ($spec) {
    my @args = @{$spec->{args}};
    return 0 if grep { $_ eq "--stop-at" } @args;
    my $expects_success = grep { $_->[0] eq "exit" && $_->[1] == 0 } @{$spec->{assertions}};
    return 0 unless $expects_success;
    my $target_idx;
    for my $i (0 .. $#args) {
        if ($args[$i] eq "--target") {
            $target_idx = $i;
            last;
        }
    }
    return 0 unless defined $target_idx;
    my $target = $args[$target_idx + 1];
    return !grep { $_ eq $target } @EMITTER_LANGS;
}

sub check_cli_assertions ($result, $assertions) {
    for my $a (@$assertions) {
        my ($kind, $value) = @$a;
        if ($kind eq "exit") {
            if ($result->{exit} != $value) {
                return "expected exit $value, got $result->{exit}\nstderr: $result->{stderr}";
            }
        } elsif ($kind eq "exit-not") {
            if ($result->{exit} == $value) {
                return "expected exit != $value, got $result->{exit}";
            }
        } elsif ($kind eq "stderr") {
            my $actual = $result->{stderr};
            $actual =~ s/\s+$//;
            if ($actual ne $value) {
                return "expected stderr \"$value\", got \"$actual\"";
            }
        } elsif ($kind eq "stderr-contains") {
            unless (index($result->{stderr}, $value) >= 0) {
                return "expected stderr to contain \"$value\", got \"$result->{stderr}\"";
            }
        } elsif ($kind eq "stderr-empty") {
            unless ($result->{stderr} eq "") {
                return "expected empty stderr, got \"$result->{stderr}\"";
            }
        } elsif ($kind eq "stdout-contains") {
            unless (index($result->{stdout}, $value) >= 0) {
                return "expected stdout to contain \"$value\", got \"$result->{stdout}\"";
            }
        } elsif ($kind eq "stdout-empty") {
            unless ($result->{stdout} eq "") {
                return "expected empty stdout, got \"" . substr($result->{stdout}, 0, 200) . "\"";
            }
        }
    }
    return undef;
}

# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

sub run_cli_tests ($test_dir) {
    my @results;
    for my $f (sort glob("$test_dir/*.tests")) {
        my $stem = basename($f, ".tests");
        my $tests = parse_cli_test_file($f);
        for my $t (@$tests) {
            my ($name, $spec) = @$t;
            my $test_id = "$stem/$name";
            if (cli_needs_backend($spec)) {
                push @results, ["skip", $test_id, undef];
                next;
            }
            my $stdin_data;
            if (defined $spec->{stdin_bytes}) {
                $stdin_data = $spec->{stdin_bytes};
            } elsif (defined $spec->{stdin}) {
                $stdin_data = $spec->{stdin};
            } else {
                $stdin_data = "";
            }
            my $result = run_inprocess($spec->{args}, $stdin_data);
            my $err = check_cli_assertions($result, $spec->{assertions});
            push @results, [$err ? "fail" : "pass", $test_id, $err];
        }
    }
    return \@results;
}

sub run_phase_tests ($test_dir, $phase_name, $cfg) {
    my @results;
    for my $f (sort glob("$test_dir/*.tests")) {
        my $stem = basename($f, ".tests");
        my $tests = parse_spec_file($f);
        for my $t (@$tests) {
            my ($name, $input, $expected) = @$t;
            my $test_id = "$stem/$name";
            my $lenient = ($phase_name =~ /^(parse|inference|taytsh_parse|taytsh_check)$/);
            my $phase_result = run_transpiled_phase(
                $input, $cfg->{args},
                is_taytsh => $cfg->{taytsh},
                expect_json => $cfg->{json},
            );
            if ($phase_name eq "inference" && !@{$phase_result->{errors}} && defined $phase_result->{data}) {
                if (ref($phase_result->{data}) eq "HASH" && exists $phase_result->{data}{reveals}) {
                    my @reveals = map { [$_->{line}, $_->{type}] } @{$phase_result->{data}{reveals}};
                    $phase_result->{reveals} = \@reveals;
                }
            }
            my $err = check_expected($expected, $phase_result, $phase_name, lenient_errors => $lenient);
            push @results, [$err ? "fail" : "pass", $test_id, $err];
        }
    }
    return \@results;
}

sub run_lowering_tests ($test_dir) {
    my @results;
    for my $f (sort glob("$test_dir/*.tests")) {
        my $stem = basename($f, ".tests");
        my $tests = parse_spec_file($f);
        for my $t (@$tests) {
            my ($name, $input, $expected) = @$t;
            my $test_id = "$stem/$name";
            my ($fh, $tmpfile) = tempfile("testXXXXXX", SUFFIX => ".py", TMPDIR => 1);
            print $fh $input;
            close $fh;
            my $result = run_inprocess(["--stop-at", "lowering-text", $tmpfile]);
            unlink $tmpfile;
            if ($expected =~ /^error:(.*)/) {
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
            unless (contains_normalized($output, $expected)) {
                push @results, ["fail", $test_id, "Expected not found in output:\n--- expected ---\n$expected\n--- got ---\n$output"];
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
            my $base_tests = parse_simple_tests($base_file);
            next unless @$base_tests;
            unless (-f $lang_file) {
                for my $bt (@$base_tests) {
                    push @results, ["fail", "$stem/$bt->[0][$lang]", "$lang/$basename_file missing"];
                }
                next;
            }
            my $lang_tests = parse_simple_tests($lang_file);
            my @base_names = map { $_->[0] } @$base_tests;
            my @lang_names = map { $_->[0] } @$lang_tests;
            if ("@base_names" ne "@lang_names") {
                for my $bt (@$base_tests) {
                    push @results, ["fail", "$stem/$bt->[0][$lang]", "base/lang name mismatch"];
                }
                next;
            }
            my %lang_by_name;
            for my $lt (@$lang_tests) {
                $lang_by_name{$lt->[0]} = $lt->[1];
            }
            for my $bt (@$base_tests) {
                my ($name, $source) = @$bt;
                my $test_id = "$stem/$name" . "[$lang]";
                my $expected = $lang_by_name{$name};
                my ($fh, $tmpfile) = tempfile("testXXXXXX", SUFFIX => ".ty", TMPDIR => 1);
                print $fh $source;
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
            my $base_tests = parse_simple_tests($base_file);
            next unless @$base_tests;
            next unless -f $lang_file;
            my $lang_tests = parse_simple_tests($lang_file);
            my %lang_by_name;
            for my $lt (@$lang_tests) {
                $lang_by_name{$lt->[0]} = $lt->[1];
            }
            for my $bt (@$base_tests) {
                my ($name, $source) = @$bt;
                next unless exists $lang_by_name{$name};
                my $test_id = "$stem/$name" . "[$lang]";
                my $expected = $lang_by_name{$name};
                my ($fh, $tmpfile) = tempfile("testXXXXXX", SUFFIX => ".py", TMPDIR => 1);
                print $fh $source;
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
        open(my $fh, "<", $test_file) or next;
        my $source = do { local $/; <$fh> };
        close $fh;
        for my $target (@available) {
            my $test_id = "$stem" . "[$target]";
            my ($tfh, $tmpfile) = tempfile("testXXXXXX", SUFFIX => ".py", TMPDIR => 1);
            print $tfh $source;
            close $tfh;
            my $result = run_inprocess(["--target", $target, $tmpfile]);
            unlink $tmpfile;
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

sub run_taytsh_app_tests ($test_dir) {
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
    print STDERR "Usage: perl test-transpiled.pl <transpiled.pl>\n";
    exit 1;
}

my $transpiled_path = File::Spec->rel2abs($ARGV[0], $TONGUES_DIR);
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
printf("Loaded in %.1fs\n\n", $t1 - $t0);

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
        } elsif ($runner eq "taytsh_app") {
            $phase_results = run_taytsh_app_tests($test_dir);
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
    say "FAILURES";
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
say "$total tests: $total_pass passed, $total_fail failed, $total_skip skipped";
say "=" x 60;

exit($total_fail > 0 ? 1 : 0);
