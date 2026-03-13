#!/usr/bin/env perl
# Benchmark individual app tests through the Perl VM.
#
# Usage:
#   cd tongues
#   perl tests/bench_vm_apptests.pl apptest_bits apptest_bools apptest_ints
#
# Loads tongues.pl and compiles the VM once (not timed), then for each
# requested test measures the VM execution time (tongues.ty interpreting
# the app test via tongues.pl's bytecode interpreter).

use v5.36;
no warnings 'redefine';
use utf8;
use File::Basename;
use File::Spec;
use File::Temp qw(tempfile);
use Time::HiRes qw(time);
use IPC::Open3;
use Symbol qw(gensym);
use POSIX ();
STDOUT->autoflush(1);

my $TONGUES_DIR = File::Spec->rel2abs(File::Spec->catdir(dirname(__FILE__), ".."));
my $LIB_DIR     = File::Spec->catdir($TONGUES_DIR, "src", "lib");
my $APP_DIR     = File::Spec->catdir($TONGUES_DIR, "tests", "backend", "app");

die "Usage: $0 <test_name> [test_name ...]\n  e.g. $0 apptest_bits apptest_bools\n"
    unless @ARGV;

my @test_names = @ARGV;

# Validate test files exist before doing expensive setup.
my @test_files;
for my $name (@test_names) {
    $name = "apptest_$name" unless $name =~ /^apptest_/;
    my $path = File::Spec->catfile($APP_DIR, "$name.py");
    die "Test not found: $path\n" unless -f $path;
    push @test_files, { name => $name, path => $path };
}

sub _read_file ($path) {
    open my $fh, "<", $path or die "$path: $!";
    local $/;
    return <$fh>;
}

sub find_lib_imports ($source) {
    my @names;
    for my $line (split /\n/, $source) {
        $line =~ s/^\s+|\s+$//g;
        if ($line =~ /^import lib\.(\w+)/) {
            push @names, $1 unless grep { $_ eq $1 } @names;
        }
    }
    return \@names;
}

# ── Setup (not timed) ────────────────────────────────────────────────

say "--- Setup (not timed) ---";

my $transpiled_path = File::Spec->catfile($TONGUES_DIR, ".out", "tongues.pl");
die "tongues.pl not found: $transpiled_path\n" unless -f $transpiled_path;

my $s0 = time();
say "Loading tongues.pl ...";
{
    local $SIG{__WARN__} = sub { warn $_[0] unless $_[0] =~ /redefined/ };
    eval { do $transpiled_path; die $@ if $@ };
    if ($@) {
        my @lines = split(/\n/, "$@");
        print STDERR "Failed to load tongues.pl: $lines[0]\n";
        exit 1;
    }
    printf("  tongues.pl loaded in %.1fs\n", time() - $s0);

    my $harness_path = File::Spec->catfile($TONGUES_DIR, ".out", "test_harness.pl");
    die "test_harness.pl not found: $harness_path\n" unless -f $harness_path;
    eval { do $harness_path; die $@ if $@ };
    die "Failed to load test_harness.pl: $@" if $@;
}
say "  test_harness.pl loaded";

my $vm_path = File::Spec->catfile($TONGUES_DIR, ".out", "tongues.ty");
die "tongues.ty not found: $vm_path\n" unless -f $vm_path;

my $s1 = time();
say "Compiling VM module ...";
my $vm_source = _read_file($vm_path);
my $vm_module = taytsh_taytsh_parse($vm_source);
my $vm_compiled = vm_prepare($vm_module);
printf("  VM compiled in %.1fs\n", time() - $s1);
say "";

# ── Helpers ───────────────────────────────────────────────────────────

sub vm_invoke ($argv, $stdin_data = "") {
    my $builtins = _BuiltinDispatch->new(bless({}, "VM"), {});
    my $instance = VM->new(
        $vm_compiled, [], [], [], [], [], [], "", 0, [], {}, $builtins, undef, {}
    );
    $builtins->{vm} = $instance;
    return $instance->invoke($stdin_data, ["tongues", @$argv]);
}

sub verify_output ($code) {
    my $err_fh = gensym;
    my $pid = open3(my $in, my $out, $err_fh, "perl");
    print $in $code;
    close $in;
    my $stderr = do { local $/; <$err_fh> };
    waitpid($pid, 0);
    return $? >> 8;
}

# ── Run each test ─────────────────────────────────────────────────────

my @results;
my $timeout = 120;

for my $tf (@test_files) {
    say "=== $tf->{name} ===";

    my $source    = _read_file($tf->{path});
    my $lib_names = find_lib_imports($source);

    # Resolve transitive lib deps
    my %seen = map { $_ => 1 } @$lib_names;
    my @queue = @$lib_names;
    while (@queue) {
        my $name = shift @queue;
        my $lp   = File::Spec->catfile($LIB_DIR, "$name.py");
        next unless -f $lp;
        for my $dep (@{ find_lib_imports(_read_file($lp)) }) {
            unless ($seen{$dep}) {
                $seen{$dep} = 1;
                push @$lib_names, $dep;
                push @queue, $dep;
            }
        }
    }

    # Prepare input (same as harness)
    my ($argv, $stdin_data, $tmpfile);
    if (@$lib_names == 0) {
        my $fh;
        ($fh, $tmpfile) = tempfile("testXXXXXX", SUFFIX => ".py", TMPDIR => 1);
        print $fh $source;
        close $fh;
        $argv = ["--target", "perl", $tmpfile];
        $stdin_data = "";
    } else {
        my @parts = ("apptest.py", $source);
        for my $name (@$lib_names) {
            my $lp = File::Spec->catfile($LIB_DIR, "$name.py");
            push @parts, "lib/$name.py", _read_file($lp);
        }
        $stdin_data = join("\x{0}", @parts);
        $argv = ["--project", "--target", "perl"];
        $tmpfile = undef;
    }

    # Time the VM execution
    my ($ok, $output_code, $elapsed) = (0, "", 0);
    my $t0 = time();
    eval {
        local $SIG{ALRM} = sub { die "TIMEOUT\n" };
        alarm($timeout);
        my $result = vm_invoke($argv, $stdin_data);
        alarm(0);
        $elapsed = time() - $t0;
        if (($result->{exit_code} // 0) != 0) {
            my $err = $result->{stderr} // "unknown error";
            die "VM failed (exit $result->{exit_code}): $err\n";
        }
        $output_code = $result->{stdout} // "";
        $ok = 1;
    };
    if ($@) {
        if ($@ eq "TIMEOUT\n") {
            $elapsed = time() - $t0;
            printf("  TIMEOUT (>%ds)\n", $timeout);
        } else {
            $elapsed = time() - $t0;
            my $err = $@;
            if (ref($err)) {
                $err = $err->{message} // $err->{msg} // "$err";
            }
            chomp($err);
            say "  FAILED - $err";
        }
        push @results, { name => $tf->{name}, status => "FAIL", time => $elapsed };
        say "";
        next;
    }
    unlink $tmpfile if defined $tmpfile;

    # Verify the output is correct (not timed)
    my $exit = verify_output($output_code);
    my $status = $exit == 0 ? "PASS" : "FAIL";
    printf("  %6.1fs  [%s]\n", $elapsed, $status);
    push @results, { name => $tf->{name}, status => $status, time => $elapsed };
    say "";
}

# ── Summary ───────────────────────────────────────────────────────────

say "--- Summary ---";
printf("%-30s %10s %s\n", "Test", "VM time", "Status");
printf("%-30s %10s %s\n", "-" x 30, "-" x 10, "-" x 6);
for my $r (@results) {
    if (defined $r->{time}) {
        printf("%-30s %9.1fs %s\n", $r->{name}, $r->{time}, $r->{status});
    } else {
        printf("%-30s %10s %s\n", $r->{name}, "-", $r->{status});
    }
}
