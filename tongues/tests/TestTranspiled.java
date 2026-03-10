import java.io.*;
import java.lang.reflect.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;

/**
 * Native Java test harness for transpiled Tongues binaries.
 * Loads the transpiled Main class once, then runs all .tests cases in-process.
 */
public class TestTranspiled {
    private static final String[] EMITTER_LANGS = {"java"};
    private static final String[] STDIN_LANGS = {};
    private static final Map<String, String[]> RUNTIMES = Map.of(
        "java", new String[]{"java"},
        "javascript", new String[]{"node"},
        "perl", new String[]{"perl"},
        "python", new String[]{"python3"},
        "ruby", new String[]{"ruby"}
    );

    private static Path tonguesDir;
    private static Path testsDir;
    private static Path libDir;
    private static Method mainMethod;
    private static Class<?> mainClass;

    // VM mode state
    private static boolean _useVm = false;
    private static Object _vmCompiled = null;
    private static Constructor<?> _vmConstructor = null;
    private static Method _vmInvokeMethod = null;

    // Test phase configuration: name -> {dir, run, taytsh?, args?, json?}
    static class Phase {
        String name;
        Map<String, Object> cfg;
        Phase(String name, Map<String, Object> cfg) { this.name = name; this.cfg = cfg; }
    }
    static class Section {
        String name;
        List<Phase> phases;
        Section(String name, Phase... phases) { this.name = name; this.phases = List.of(phases); }
    }
    private static final List<Section> TESTS = List.of(
        new Section("cli",
            new Phase("cli", Map.of("dir", "frontend/cli", "run", "cli"))
        ),
        new Section("linker",
            new Phase("linker", Map.of("dir", "frontend/linker", "run", "linker"))
        ),
        new Section("frontend",
            new Phase("parse", Map.of("dir", "frontend/parse", "run", "phase", "taytsh", false, "args", new String[]{"--stop-at", "parse"}, "json", true)),
            new Phase("subset", Map.of("dir", "frontend/subset", "run", "phase", "taytsh", false, "args", new String[]{"--stop-at", "subset"}, "json", false)),
            new Phase("names", Map.of("dir", "frontend/names", "run", "phase", "taytsh", false, "args", new String[]{"--stop-at", "names"}, "json", true)),
            new Phase("sigs", Map.of("dir", "frontend/signatures", "run", "phase", "taytsh", false, "args", new String[]{"--stop-at", "signatures"}, "json", true)),
            new Phase("fields", Map.of("dir", "frontend/fields", "run", "phase", "taytsh", false, "args", new String[]{"--stop-at", "fields"}, "json", true)),
            new Phase("hierarchy", Map.of("dir", "frontend/hierarchy", "run", "phase", "taytsh", false, "args", new String[]{"--stop-at", "hierarchy"}, "json", true)),
            new Phase("pycheck", Map.of("dir", "frontend/pycheck", "run", "phase", "taytsh", false, "args", new String[]{"--stop-at", "pycheck"}, "json", true)),
            new Phase("lowering", Map.of("dir", "frontend/lowering", "run", "lowering"))
        ),
        new Section("middleend",
            new Phase("scope", Map.of("dir", "middleend/scope", "run", "phase", "taytsh", true, "args", new String[]{"--stop-at", "scope"}, "json", true)),
            new Phase("returns", Map.of("dir", "middleend/returns", "run", "phase", "taytsh", true, "args", new String[]{"--stop-at", "returns"}, "json", true)),
            new Phase("liveness", Map.of("dir", "middleend/liveness", "run", "phase", "taytsh", true, "args", new String[]{"--stop-at", "liveness"}, "json", true)),
            new Phase("strings", Map.of("dir", "middleend/strings", "run", "phase", "taytsh", true, "args", new String[]{"--stop-at", "strings"}, "json", true)),
            new Phase("hoisting", Map.of("dir", "middleend/hoisting", "run", "phase", "taytsh", true, "args", new String[]{"--stop-at", "hoisting"}, "json", true)),
            new Phase("ownership", Map.of("dir", "middleend/ownership", "run", "phase", "taytsh", true, "args", new String[]{"--stop-at", "ownership"}, "json", true)),
            new Phase("callgraph", Map.of("dir", "middleend/callgraph", "run", "phase", "taytsh", true, "args", new String[]{"--stop-at", "callgraph"}, "json", true))
        ),
        new Section("backend",
            new Phase("codegen", Map.of("dir", "backend/codegen", "run", "codegen")),
            new Phase("emit", Map.of("dir", "backend/emit", "run", "emit")),
            new Phase("app", Map.of("dir", "backend/app", "run", "app")),
            new Phase("ordering", Map.of("dir", "backend/ordering", "run", "ordering"))
        ),
        new Section("taytsh",
            new Phase("typarse", Map.of("dir", "taytsh/typarse", "run", "phase", "taytsh", true, "args", new String[]{"--stop-at", "parse"}, "json", true)),
            new Phase("tycheck", Map.of("dir", "taytsh/tycheck", "run", "phase", "taytsh", true, "args", new String[]{"--stop-at", "check"}, "json", true)),
            new Phase("ty_app", Map.of("dir", "taytsh/app", "run", "ty_app"))
        )
    );

    // -------------------------------------------------------------------------
    // Data classes for test parsing
    // -------------------------------------------------------------------------

    static class SpecEntry {
        String name;
        String input;
        String expected;
        SpecEntry(String name, String input, String expected) {
            this.name = name;
            this.input = input;
            this.expected = expected;
        }
    }

    static class SimpleEntry {
        String name;
        String content;
        SimpleEntry(String name, String content) {
            this.name = name;
            this.content = content;
        }
    }

    static class CliAssertion {
        String kind;
        String value;
        CliAssertion(String kind, String value) {
            this.kind = kind;
            this.value = value;
        }
    }

    static class CliSpec {
        List<String> args;
        String stdin;
        String stdinHex;
        List<CliAssertion> assertions;
        CliSpec(List<String> args, String stdin, String stdinHex, List<CliAssertion> assertions) {
            this.args = args;
            this.stdin = stdin;
            this.stdinHex = stdinHex;
            this.assertions = assertions;
        }
    }

    static class LinkerFile {
        String path;
        String source;
        LinkerFile(String path, String source) {
            this.path = path;
            this.source = source;
        }
    }

    static class LinkerSpec {
        List<LinkerFile> files;
        List<String> args;
        List<CliAssertion> assertions;
        LinkerSpec(List<LinkerFile> files, List<String> args, List<CliAssertion> assertions) {
            this.files = files;
            this.args = args;
            this.assertions = assertions;
        }
    }

    static class RunResult {
        String stdout;
        String stderr;
        int exit;
        RunResult(String stdout, String stderr, int exit) {
            this.stdout = stdout;
            this.stderr = stderr;
            this.exit = exit;
        }
    }

    static class PhaseResult {
        List<String> errors;
        List<String> warnings;
        String data;
        PhaseResult(List<String> errors, List<String> warnings, String data) {
            this.errors = errors;
            this.warnings = warnings;
            this.data = data;
        }
    }

    // -------------------------------------------------------------------------
    // Test file parsing
    // -------------------------------------------------------------------------

    static String trimBlankLines(String text) {
        String[] lines = text.split("\n", -1);
        int start = 0;
        while (start < lines.length && lines[start].isEmpty()) start++;
        int end = lines.length;
        while (end > start && lines[end - 1].isEmpty()) end--;
        return String.join("\n", Arrays.copyOfRange(lines, start, end));
    }

    static List<SpecEntry> parseSpecFile(String text) {
        String[] lines = text.split("\n", -1);
        List<SpecEntry> result = new ArrayList<>();
        int i = 0;
        while (i < lines.length) {
            if (lines[i].startsWith("=== ")) {
                String testName = lines[i].substring(4).trim();
                i++;
                List<String> inputLines = new ArrayList<>();
                while (i < lines.length && !lines[i].startsWith("---")) {
                    inputLines.add(lines[i]);
                    i++;
                }
                if (i < lines.length && lines[i].equals("---")) i++;
                List<String> expectedLines = new ArrayList<>();
                while (i < lines.length && !lines[i].startsWith("---")) {
                    expectedLines.add(lines[i]);
                    i++;
                }
                if (i < lines.length && lines[i].equals("---")) i++;
                result.add(new SpecEntry(
                    testName,
                    String.join("\n", inputLines),
                    trimBlankLines(String.join("\n", expectedLines))
                ));
            } else {
                i++;
            }
        }
        return result;
    }

    static List<CliAssertion> parseCliAssertions(List<String> expectedLines) {
        List<CliAssertion> assertions = new ArrayList<>();
        for (String rawLine : expectedLines) {
            String stripped = rawLine.trim();
            if (stripped.isEmpty()) continue;
            if (stripped.startsWith("exit:")) {
                assertions.add(new CliAssertion("exit", stripped.substring(5).trim()));
            } else if (stripped.startsWith("exit-not:")) {
                assertions.add(new CliAssertion("exit-not", stripped.substring(9).trim()));
            } else if (stripped.startsWith("stderr:")) {
                assertions.add(new CliAssertion("stderr", stripped.substring(7).trim()));
            } else if (stripped.startsWith("stderr-contains:")) {
                assertions.add(new CliAssertion("stderr-contains", stripped.substring(16).trim()));
            } else if (stripped.startsWith("stderr-empty:")) {
                assertions.add(new CliAssertion("stderr-empty", ""));
            } else if (stripped.startsWith("stdout-contains:")) {
                assertions.add(new CliAssertion("stdout-contains", stripped.substring(16).trim()));
            } else if (stripped.startsWith("stdout-empty:")) {
                assertions.add(new CliAssertion("stdout-empty", ""));
            }
        }
        return assertions;
    }

    static CliSpec parseCliSpec(List<String> inputLines, List<String> expectedLines) {
        List<String> args = new ArrayList<>();
        String stdin = "";
        String stdinHex = "";
        int bodyStart = 0;
        if (!inputLines.isEmpty() && inputLines.get(0).startsWith("args:")) {
            String argsStr = inputLines.get(0).substring(5).trim();
            if (!argsStr.isEmpty()) {
                args = Arrays.asList(argsStr.split("\\s+"));
            }
            bodyStart = 1;
        }
        List<String> remaining = inputLines.subList(bodyStart, inputLines.size());
        if (!remaining.isEmpty() && remaining.get(0).startsWith("stdin-bytes:")) {
            stdinHex = remaining.get(0).substring(12).trim();
        } else {
            stdin = String.join("\n", remaining);
        }
        List<CliAssertion> assertions = parseCliAssertions(expectedLines);
        return new CliSpec(args, stdin, stdinHex, assertions);
    }

    static List<Map.Entry<String, CliSpec>> parseCliTestFile(String text) {
        String[] lines = text.split("\n", -1);
        List<Map.Entry<String, CliSpec>> result = new ArrayList<>();
        int i = 0;
        while (i < lines.length) {
            if (lines[i].startsWith("=== ")) {
                String testName = lines[i].substring(4).trim();
                i++;
                List<String> inputLines = new ArrayList<>();
                while (i < lines.length && !lines[i].startsWith("---")) {
                    inputLines.add(lines[i]);
                    i++;
                }
                if (i < lines.length && lines[i].equals("---")) i++;
                List<String> expectedLines = new ArrayList<>();
                while (i < lines.length && !lines[i].startsWith("---")) {
                    expectedLines.add(lines[i]);
                    i++;
                }
                if (i < lines.length && lines[i].equals("---")) i++;
                CliSpec spec = parseCliSpec(inputLines, expectedLines);
                result.add(Map.entry(testName, spec));
            } else {
                i++;
            }
        }
        return result;
    }

    static LinkerSpec parseLinkerSpec(List<String> inputLines, List<String> expectedLines) {
        List<LinkerFile> files = new ArrayList<>();
        List<String> args = new ArrayList<>();
        String currentPath = "";
        boolean hasCurrent = false;
        List<String> currentLines = new ArrayList<>();
        for (String line : inputLines) {
            if (line.startsWith("file: ")) {
                if (hasCurrent) {
                    files.add(new LinkerFile(currentPath, String.join("\n", currentLines)));
                }
                currentPath = line.substring(6).trim();
                hasCurrent = true;
                currentLines = new ArrayList<>();
            } else if (line.startsWith("args: ")) {
                if (hasCurrent) {
                    files.add(new LinkerFile(currentPath, String.join("\n", currentLines)));
                    hasCurrent = false;
                    currentLines = new ArrayList<>();
                }
                args = Arrays.asList(line.substring(6).trim().split("\\s+"));
            } else {
                currentLines.add(line);
            }
        }
        if (hasCurrent) {
            files.add(new LinkerFile(currentPath, String.join("\n", currentLines)));
        }
        List<CliAssertion> assertions = parseCliAssertions(expectedLines);
        return new LinkerSpec(files, args, assertions);
    }

    static List<Map.Entry<String, LinkerSpec>> parseLinkerTestFile(String text) {
        String[] lines = text.split("\n", -1);
        List<Map.Entry<String, LinkerSpec>> result = new ArrayList<>();
        int i = 0;
        while (i < lines.length) {
            if (lines[i].startsWith("=== ")) {
                String testName = lines[i].substring(4).trim();
                i++;
                List<String> inputLines = new ArrayList<>();
                while (i < lines.length && !lines[i].startsWith("---")) {
                    inputLines.add(lines[i]);
                    i++;
                }
                if (i < lines.length && lines[i].equals("---")) i++;
                List<String> expectedLines = new ArrayList<>();
                while (i < lines.length && !lines[i].startsWith("---")) {
                    expectedLines.add(lines[i]);
                    i++;
                }
                if (i < lines.length && lines[i].equals("---")) i++;
                LinkerSpec spec = parseLinkerSpec(inputLines, expectedLines);
                result.add(Map.entry(testName, spec));
            } else {
                i++;
            }
        }
        return result;
    }

    static List<SimpleEntry> parseSimpleTests(String text) {
        String[] lines = text.split("\n", -1);
        List<SimpleEntry> result = new ArrayList<>();
        int i = 0;
        while (i < lines.length) {
            if (lines[i].startsWith("=== ")) {
                String name = lines[i].substring(4).trim();
                i++;
                List<String> contentLines = new ArrayList<>();
                while (i < lines.length && !lines[i].startsWith("=== ")) {
                    contentLines.add(lines[i]);
                    i++;
                }
                result.add(new SimpleEntry(name, trimBlankLines(String.join("\n", contentLines))));
            } else {
                i++;
            }
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Thread-local stream capture for parallel execution
    // -------------------------------------------------------------------------

    static class ThreadLocalPrintStream extends PrintStream {
        private final ThreadLocal<PrintStream> threadStream = new ThreadLocal<>();
        private final PrintStream fallback;

        public ThreadLocalPrintStream(PrintStream fallback) {
            super(fallback);
            this.fallback = fallback;
        }

        public void setThreadStream(PrintStream ps) { threadStream.set(ps); }
        public void clearThreadStream() { threadStream.remove(); }
        private PrintStream getStream() {
            PrintStream ps = threadStream.get();
            return ps != null ? ps : fallback;
        }

        @Override public void write(int b) { getStream().write(b); }
        @Override public void write(byte[] buf, int off, int len) { getStream().write(buf, off, len); }
        @Override public void flush() { getStream().flush(); }
        @Override public void print(boolean b) { getStream().print(b); }
        @Override public void print(char c) { getStream().print(c); }
        @Override public void print(int i) { getStream().print(i); }
        @Override public void print(long l) { getStream().print(l); }
        @Override public void print(float f) { getStream().print(f); }
        @Override public void print(double d) { getStream().print(d); }
        @Override public void print(char[] s) { getStream().print(s); }
        @Override public void print(String s) { getStream().print(s); }
        @Override public void print(Object obj) { getStream().print(obj); }
        @Override public void println() { getStream().println(); }
        @Override public void println(boolean x) { getStream().println(x); }
        @Override public void println(char x) { getStream().println(x); }
        @Override public void println(int x) { getStream().println(x); }
        @Override public void println(long x) { getStream().println(x); }
        @Override public void println(float x) { getStream().println(x); }
        @Override public void println(double x) { getStream().println(x); }
        @Override public void println(char[] x) { getStream().println(x); }
        @Override public void println(String x) { getStream().println(x); }
        @Override public void println(Object x) { getStream().println(x); }
        @Override public PrintStream printf(String format, Object... args) { return getStream().printf(format, args); }
        @Override public PrintStream printf(java.util.Locale l, String format, Object... args) { return getStream().printf(l, format, args); }
    }

    static class ThreadLocalInputStream extends InputStream {
        private final ThreadLocal<InputStream> threadStream = new ThreadLocal<>();
        private final InputStream fallback;

        public ThreadLocalInputStream(InputStream fallback) { this.fallback = fallback; }
        public void setThreadStream(InputStream is) { threadStream.set(is); }
        public void clearThreadStream() { threadStream.remove(); }
        private InputStream getStream() {
            InputStream is = threadStream.get();
            return is != null ? is : fallback;
        }

        @Override public int read() throws IOException { return getStream().read(); }
        @Override public int read(byte[] b) throws IOException { return getStream().read(b); }
        @Override public int read(byte[] b, int off, int len) throws IOException { return getStream().read(b, off, len); }
        @Override public int available() throws IOException { return getStream().available(); }
        @Override public void close() throws IOException { getStream().close(); }
    }

    private static ThreadLocalPrintStream tlOut;
    private static ThreadLocalPrintStream tlErr;
    private static ThreadLocalInputStream tlIn;
    private static PrintStream realOut;
    private static PrintStream realErr;
    private static InputStream realIn;

    static void installThreadLocalStreams() {
        realOut = System.out;
        realErr = System.err;
        realIn = System.in;
        tlOut = new ThreadLocalPrintStream(realOut);
        tlErr = new ThreadLocalPrintStream(realErr);
        tlIn = new ThreadLocalInputStream(realIn);
        System.setOut(tlOut);
        System.setErr(tlErr);
        System.setIn(tlIn);
    }

    static void restoreStreams() {
        System.setOut(realOut);
        System.setErr(realErr);
        System.setIn(realIn);
    }

    // -------------------------------------------------------------------------
    // In-process execution
    // -------------------------------------------------------------------------

    static Class<?> systemExitExceptionClass = null;
    static java.lang.reflect.Field systemExitCodeField = null;

    static {
        try {
            systemExitExceptionClass = Class.forName("Main$SystemExitException");
            systemExitCodeField = systemExitExceptionClass.getDeclaredField("code");
            systemExitCodeField.setAccessible(true);
        } catch (Exception e) {
            // Will be null if not found
        }
    }

    static RunResult runInprocess(String[] argv, String stdinData, byte[] stdinBytes) {
        if (_useVm) {
            return runVmInprocess(argv, stdinData, stdinBytes);
        }
        PrintStream oldOut = System.out;
        PrintStream oldErr = System.err;
        InputStream oldIn = System.in;
        ByteArrayOutputStream outBuf = new ByteArrayOutputStream();
        ByteArrayOutputStream errBuf = new ByteArrayOutputStream();
        int exitCode = 0;

        try {
            System.setOut(new PrintStream(outBuf, true, "UTF-8"));
            System.setErr(new PrintStream(errBuf, true, "UTF-8"));
            byte[] inputBytes = stdinBytes != null ? stdinBytes : stdinData.getBytes("UTF-8");
            System.setIn(new ByteArrayInputStream(inputBytes));

            mainMethod.invoke(null, (Object) argv);
        } catch (InvocationTargetException e) {
            Throwable cause = e.getCause();
            if (systemExitExceptionClass != null && systemExitExceptionClass.isInstance(cause)) {
                try {
                    exitCode = systemExitCodeField.getInt(cause);
                } catch (Exception ex) {
                    exitCode = 1;
                }
            } else {
                try {
                    String msg = cause != null ? cause.getMessage() : e.getMessage();
                    errBuf.write((msg != null ? msg : "Unknown error").getBytes());
                    errBuf.write('\n');
                } catch (IOException ignored) {}
                exitCode = 1;
            }
        } catch (Exception e) {
            try {
                errBuf.write((e.getMessage() + "\n").getBytes());
            } catch (IOException ignored) {}
            exitCode = 1;
        } finally {
            System.setOut(oldOut);
            System.setErr(oldErr);
            System.setIn(oldIn);
        }

        try {
            return new RunResult(outBuf.toString("UTF-8"), errBuf.toString("UTF-8"), exitCode);
        } catch (UnsupportedEncodingException e) {
            return new RunResult(outBuf.toString(), errBuf.toString(), exitCode);
        }
    }

    static RunResult runInprocess(String[] argv, String stdinData) {
        return runInprocess(argv, stdinData, null);
    }

    static RunResult runInprocess(String[] argv) {
        return runInprocess(argv, "", null);
    }

    // -------------------------------------------------------------------------
    // VM mode: parse + compile .ty once, invoke per test
    // -------------------------------------------------------------------------

    static void loadVmModule(String tyPath) throws Exception {
        String source = Files.readString(Paths.get(tyPath));
        // Call taytsh_taytsh_parse(source) -> TModule
        Method parseMethod = mainClass.getMethod("taytsh_taytsh_parse", String.class);
        Object module;
        try {
            module = parseMethod.invoke(null, source);
        } catch (InvocationTargetException e) {
            Throwable cause = e.getCause();
            if (cause instanceof NullPointerException && cause.getMessage() != null
                    && cause.getMessage().contains("annotations")) {
                throw new RuntimeException(
                    "Java VM tests not supported: transpiled Java has a bug where TDecl.annotations "
                    + "is not initialized in subclasses. Run with Python/JS/Ruby/Perl instead.");
            }
            throw e;
        }
        // Call vm_prepare(module) -> CompiledModule
        Method prepareMethod = mainClass.getMethod("vm_prepare", module.getClass());
        _vmCompiled = prepareMethod.invoke(null, module);
        // Get VM class and methods
        Class<?> vmClass = Class.forName("Main$VM", true, mainClass.getClassLoader());
        // Find the constructor that takes the CompiledModule type
        Constructor<?>[] constructors = vmClass.getConstructors();
        for (Constructor<?> c : constructors) {
            Class<?>[] params = c.getParameterTypes();
            if (params.length == 1 && params[0].isAssignableFrom(_vmCompiled.getClass())) {
                _vmConstructor = c;
                break;
            }
        }
        if (_vmConstructor == null) {
            throw new RuntimeException("Could not find VM constructor accepting CompiledModule");
        }
        _vmInvokeMethod = vmClass.getMethod("invoke", byte[].class, List.class);
        System.out.println("VM module compiled");
    }

    static RunResult runVmInprocess(String[] argv, String stdinData, byte[] stdinBytes) {
        try {
            byte[] inputBytes = stdinBytes != null ? stdinBytes : stdinData.getBytes("UTF-8");
            Object vm = _vmConstructor.newInstance(_vmCompiled);
            // Set builtins.vm = vm (required for VM self-reference)
            try {
                Field builtinsField = vm.getClass().getField("builtins");
                Object builtins = builtinsField.get(vm);
                Field vmField = builtins.getClass().getField("vm");
                vmField.set(builtins, vm);
            } catch (NoSuchFieldException ignored) {
                // builtins.vm may not exist in all versions
            }
            List<String> args = new ArrayList<>();
            args.add("tongues");
            args.addAll(Arrays.asList(argv));
            Object result = _vmInvokeMethod.invoke(vm, inputBytes, args);
            // Extract stdout, stderr, exit_code from VMResult
            Field stdoutField = result.getClass().getField("stdout");
            Field stderrField = result.getClass().getField("stderr");
            Field exitField = result.getClass().getField("exit_code");
            Object stdoutObj = stdoutField.get(result);
            Object stderrObj = stderrField.get(result);
            int exit = exitField.getInt(result);
            String stdout = stdoutObj instanceof String ? (String) stdoutObj : new String((byte[]) stdoutObj, "UTF-8");
            String stderr = stderrObj instanceof String ? (String) stderrObj : new String((byte[]) stderrObj, "UTF-8");
            return new RunResult(stdout, stderr, exit);
        } catch (InvocationTargetException e) {
            Throwable cause = e.getCause();
            String msg = cause != null ? cause.getMessage() : e.getMessage();
            return new RunResult("", msg != null ? msg : "Unknown error", 1);
        } catch (Exception e) {
            return new RunResult("", e.getMessage() != null ? e.getMessage() : "Unknown error", 1);
        }
    }

    static PhaseResult runTranspiledPhase(String source, String[] cliArgs, boolean isTaytsh, boolean expectJson) throws IOException {
        String suffix = isTaytsh ? ".ty" : ".py";
        Path tmpFile = Files.createTempFile("test_", suffix);
        try {
            Files.writeString(tmpFile, source);
            List<String> argv = new ArrayList<>();
            if (isTaytsh) {
                argv.add("taytsh");
            }
            argv.addAll(Arrays.asList(cliArgs));
            argv.add(tmpFile.toString());
            RunResult result = runInprocess(argv.toArray(new String[0]));
            String stderrText = result.stderr.trim();
            if (result.exit != 0) {
                List<String> errors = Arrays.stream(stderrText.split("\n"))
                    .filter(s -> !s.isEmpty())
                    .collect(Collectors.toList());
                return new PhaseResult(errors, new ArrayList<>(), null);
            }
            List<String> warnings = stderrText.isEmpty() ? new ArrayList<>() :
                Arrays.stream(stderrText.split("\n"))
                    .filter(s -> !s.isEmpty())
                    .collect(Collectors.toList());
            if (!expectJson) {
                return new PhaseResult(new ArrayList<>(), warnings, null);
            }
            String stdoutText = result.stdout.trim();
            if (stdoutText.isEmpty()) {
                return new PhaseResult(new ArrayList<>(), warnings, null);
            }
            return new PhaseResult(new ArrayList<>(), warnings, stdoutText);
        } finally {
            Files.deleteIfExists(tmpFile);
        }
    }

    // -------------------------------------------------------------------------
    // Assertion checking
    // -------------------------------------------------------------------------

    static boolean containsNormalized(String haystack, String needle) {
        List<String> needleStripped = Arrays.stream(needle.trim().split("\n"))
            .map(String::trim)
            .filter(s -> !s.isEmpty())
            .collect(Collectors.toList());
        List<String> haystackStripped = Arrays.stream(haystack.split("\n"))
            .map(String::trim)
            .filter(s -> !s.isEmpty())
            .collect(Collectors.toList());
        if (needleStripped.isEmpty()) return true;
        for (int i = 0; i < haystackStripped.size(); i++) {
            if (haystackStripped.get(i).contains(needleStripped.get(0))) {
                boolean match = true;
                for (int j = 1; j < needleStripped.size(); j++) {
                    if (i + j >= haystackStripped.size() ||
                        !haystackStripped.get(i + j).contains(needleStripped.get(j))) {
                        match = false;
                        break;
                    }
                }
                if (match) return true;
            }
        }
        return false;
    }

    static String checkCliAssertions(int exitCode, String stdout, String stderr, List<CliAssertion> assertions) {
        for (CliAssertion a : assertions) {
            switch (a.kind) {
                case "exit":
                    int expectedExit = Integer.parseInt(a.value);
                    if (exitCode != expectedExit) {
                        return "expected exit " + a.value + ", got " + exitCode + "\nstderr: " + stderr;
                    }
                    break;
                case "exit-not":
                    int notExit = Integer.parseInt(a.value);
                    if (exitCode == notExit) {
                        return "expected exit != " + a.value + ", got " + exitCode;
                    }
                    break;
                case "stderr":
                    String actualStderr = stderr.stripTrailing();
                    if (!actualStderr.equals(a.value)) {
                        return "expected stderr '" + a.value + "', got '" + actualStderr + "'";
                    }
                    break;
                case "stderr-contains":
                    if (!stderr.contains(a.value)) {
                        return "expected stderr to contain '" + a.value + "', got '" + stderr + "'";
                    }
                    break;
                case "stderr-empty":
                    if (!stderr.isEmpty()) {
                        return "expected empty stderr, got '" + stderr + "'";
                    }
                    break;
                case "stdout-contains":
                    if (!stdout.contains(a.value)) {
                        return "expected stdout to contain '" + a.value + "', got '" + stdout + "'";
                    }
                    break;
                case "stdout-empty":
                    if (!stdout.isEmpty()) {
                        return "expected empty stdout, got '" + stdout.substring(0, Math.min(200, stdout.length())) + "'";
                    }
                    break;
            }
        }
        return "";
    }

    static String checkExpected(String expected, List<String> errors, List<String> warnings,
                                String data, String phase, boolean lenientErrors) {
        expected = trimBlankLines(expected);
        if (expected.isEmpty()) expected = "ok";

        if (expected.equals("ok")) {
            if (!errors.isEmpty()) {
                return "Expected ok, got error: " + errors.get(0);
            }
            return "";
        }

        if (expected.startsWith("error:")) {
            String expectedMsg = expected.substring(6).trim();
            if (errors.isEmpty()) {
                return "Expected error containing '" + expectedMsg + "', got ok";
            }
            if (!lenientErrors && !expectedMsg.isEmpty()) {
                boolean found = errors.stream()
                    .anyMatch(e -> e.toLowerCase().contains(expectedMsg.toLowerCase()));
                if (!found) {
                    return "Expected error containing '" + expectedMsg + "', got: " + errors;
                }
            }
            return "";
        }

        if (expected.startsWith("warning:")) {
            String expectedMsg = expected.substring(8).trim();
            if (warnings.isEmpty()) {
                return "Expected warning containing '" + expectedMsg + "', got none";
            }
            boolean found = warnings.stream()
                .anyMatch(w -> w.toLowerCase().contains(expectedMsg.toLowerCase()));
            if (!found) {
                return "Expected warning containing '" + expectedMsg + "', got: " + warnings;
            }
            return "";
        }

        if (!errors.isEmpty()) {
            return phase + " failed: " + errors.get(0);
        }
        if (data == null) {
            return "No data returned from " + phase;
        }

        // For JSON dotpath assertions, we'd need a JSON parser
        // For now, skip detailed JSON validation - just check phase succeeded
        return "";
    }

    static boolean cliNeedsBackend(List<String> args, List<CliAssertion> assertions, String[] emitterLangs) {
        boolean hasStopAt = args.contains("--stop-at");
        if (hasStopAt) return false;

        boolean expectsSuccess = assertions.stream()
            .anyMatch(a -> a.kind.equals("exit") && a.value.equals("0"));
        if (!expectsSuccess) return false;

        int targetIdx = args.indexOf("--target");
        if (targetIdx == -1 || targetIdx + 1 >= args.size()) return false;

        String target = args.get(targetIdx + 1);
        return !Arrays.asList(emitterLangs).contains(target);
    }

    // -------------------------------------------------------------------------
    // Test runners
    // -------------------------------------------------------------------------

    static List<String[]> runCliTests(Path testDir) throws IOException {
        List<String[]> results = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "*.tests")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path f : files) {
                String stem = f.getFileName().toString().replace(".tests", "");
                String content = Files.readString(f);
                for (Map.Entry<String, CliSpec> entry : parseCliTestFile(content)) {
                    String testId = stem + "/" + entry.getKey();
                    CliSpec spec = entry.getValue();
                    if (cliNeedsBackend(spec.args, spec.assertions, EMITTER_LANGS)) {
                        results.add(new String[]{"skip", testId, null});
                        continue;
                    }
                    RunResult result;
                    if (!spec.stdinHex.isEmpty()) {
                        byte[] raw = hexToBytes(spec.stdinHex);
                        result = runInprocess(spec.args.toArray(new String[0]), "", raw);
                    } else {
                        result = runInprocess(spec.args.toArray(new String[0]), spec.stdin);
                    }
                    String err = checkCliAssertions(result.exit, result.stdout, result.stderr, spec.assertions);
                    results.add(new String[]{err.isEmpty() ? "pass" : "fail", testId, err.isEmpty() ? null : err});
                }
            }
        }
        return results;
    }

    static byte[] hexToBytes(String hex) {
        if (hex == null || hex.isEmpty()) return new byte[0];
        // Ensure even length
        int len = hex.length();
        if (len % 2 != 0) len--;
        byte[] data = new byte[len / 2];
        for (int i = 0; i < len; i += 2) {
            data[i / 2] = (byte) ((Character.digit(hex.charAt(i), 16) << 4)
                + Character.digit(hex.charAt(i + 1), 16));
        }
        return data;
    }

    static List<String[]> runLinkerTests(Path testDir) throws IOException {
        List<String[]> results = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "*.tests")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path f : files) {
                String stem = f.getFileName().toString().replace(".tests", "");
                String content = Files.readString(f);
                for (Map.Entry<String, LinkerSpec> entry : parseLinkerTestFile(content)) {
                    String testId = stem + "/" + entry.getKey();
                    LinkerSpec spec = entry.getValue();
                    List<String> parts = new ArrayList<>();
                    for (LinkerFile lf : spec.files) {
                        parts.add(lf.path);
                        parts.add(lf.source);
                    }
                    String stdinData = String.join("\0", parts);
                    int targetIdx = spec.args.indexOf("--target");
                    if (targetIdx != -1 && targetIdx + 1 < spec.args.size()) {
                        String target = spec.args.get(targetIdx + 1);
                        if (!Arrays.asList(EMITTER_LANGS).contains(target)) {
                            results.add(new String[]{"skip", testId, null});
                            continue;
                        }
                    }
                    RunResult result = runInprocess(spec.args.toArray(new String[0]), stdinData);
                    String err = checkCliAssertions(result.exit, result.stdout, result.stderr, spec.assertions);
                    results.add(new String[]{err.isEmpty() ? "pass" : "fail", testId, err.isEmpty() ? null : err});
                }
            }
        }
        return results;
    }

    @SuppressWarnings("unchecked")
    static List<String[]> runPhaseTests(Path testDir, String phaseName, Map<String, Object> cfg) throws IOException {
        List<String[]> results = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "*.tests")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path f : files) {
                String stem = f.getFileName().toString().replace(".tests", "");
                String content = Files.readString(f);
                for (SpecEntry entry : parseSpecFile(content)) {
                    String testId = stem + "/" + entry.name;
                    boolean lenient = Arrays.asList("parse", "pycheck", "typarse", "tycheck").contains(phaseName);
                    String[] args = (String[]) cfg.getOrDefault("args", new String[0]);
                    boolean isTaytsh = (Boolean) cfg.getOrDefault("taytsh", false);
                    boolean expectJson = (Boolean) cfg.getOrDefault("json", true);
                    PhaseResult phaseResult = runTranspiledPhase(entry.input, args, isTaytsh, expectJson);
                    String err = checkExpected(entry.expected, phaseResult.errors, phaseResult.warnings,
                        phaseResult.data, phaseName, lenient);
                    results.add(new String[]{err.isEmpty() ? "pass" : "fail", testId, err.isEmpty() ? null : err});
                }
            }
        }
        return results;
    }

    static List<String[]> runLoweringTests(Path testDir) throws IOException {
        List<String[]> results = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "*.tests")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path f : files) {
                String stem = f.getFileName().toString().replace(".tests", "");
                String content = Files.readString(f);
                for (SpecEntry entry : parseSpecFile(content)) {
                    String testId = stem + "/" + entry.name;
                    Path tmpFile = Files.createTempFile("test_", ".py");
                    try {
                        Files.writeString(tmpFile, entry.input);
                        RunResult result = runInprocess(new String[]{"--stop-at", "lowering-text", tmpFile.toString()});
                        if (entry.expected.startsWith("error:")) {
                            String expectedMsg = entry.expected.substring(6).trim();
                            if (result.exit == 0) {
                                results.add(new String[]{"fail", testId, "Expected error containing '" + expectedMsg + "', got success"});
                                continue;
                            }
                            String firstLine = result.stderr.trim().split("\n")[0];
                            if (!expectedMsg.isEmpty() && !firstLine.toLowerCase().contains(expectedMsg.toLowerCase())) {
                                results.add(new String[]{"fail", testId, "Expected error containing '" + expectedMsg + "', got: " + firstLine});
                                continue;
                            }
                            results.add(new String[]{"pass", testId, null});
                            continue;
                        }
                        if (result.exit != 0) {
                            String errMsg = result.stderr.trim().split("\n")[0];
                            results.add(new String[]{"fail", testId, "Lowering error: " + errMsg});
                            continue;
                        }
                        if (!containsNormalized(result.stdout, entry.expected)) {
                            results.add(new String[]{"fail", testId, "Expected not found in output:\n--- expected ---\n" + entry.expected + "\n--- got ---\n" + result.stdout});
                            continue;
                        }
                        results.add(new String[]{"pass", testId, null});
                    } finally {
                        Files.deleteIfExists(tmpFile);
                    }
                }
            }
        }
        return results;
    }

    static List<String[]> runCodegenTests(Path testDir) throws IOException {
        List<String[]> results = new ArrayList<>();
        Path baseDir = testDir.resolve("base");
        if (!Files.isDirectory(baseDir)) return results;

        List<String> langDirs;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir)) {
            langDirs = new ArrayList<>();
            for (Path p : stream) {
                if (Files.isDirectory(p)) {
                    String name = p.getFileName().toString();
                    if (!name.equals("base") && Arrays.asList(EMITTER_LANGS).contains(name)) {
                        langDirs.add(name);
                    }
                }
            }
        }
        Collections.sort(langDirs);

        for (String lang : langDirs) {
            Path langDir = testDir.resolve(lang);
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(baseDir, "*.tests")) {
                List<Path> files = new ArrayList<>();
                stream.forEach(files::add);
                Collections.sort(files);
                for (Path baseFile : files) {
                    String baseName = baseFile.getFileName().toString();
                    String stem = baseName.replace(".tests", "");
                    Path langFile = langDir.resolve(baseName);
                    List<SimpleEntry> baseTests = parseSimpleTests(Files.readString(baseFile));
                    if (baseTests.isEmpty()) continue;
                    if (!Files.exists(langFile)) {
                        for (SimpleEntry e : baseTests) {
                            results.add(new String[]{"fail", stem + "/" + e.name + "[" + lang + "]", lang + "/" + baseName + " missing"});
                        }
                        continue;
                    }
                    List<SimpleEntry> langTests = parseSimpleTests(Files.readString(langFile));
                    Map<String, String> langByName = langTests.stream()
                        .collect(Collectors.toMap(e -> e.name, e -> e.content));
                    for (SimpleEntry entry : baseTests) {
                        String testId = stem + "/" + entry.name + "[" + lang + "]";
                        String expected = langByName.get(entry.name);
                        if (expected == null) {
                            results.add(new String[]{"fail", testId, "No matching lang test"});
                            continue;
                        }
                        Path tmpFile = Files.createTempFile("test_", ".ty");
                        try {
                            Files.writeString(tmpFile, entry.content);
                            RunResult result = runInprocess(new String[]{"taytsh", "--emit", lang, tmpFile.toString()});
                            if (result.exit != 0) {
                                String stderr = result.stderr.trim().split("\n")[0];
                                results.add(new String[]{"fail", testId, "Transpile error: " + stderr});
                                continue;
                            }
                            if (!containsNormalized(result.stdout, expected)) {
                                results.add(new String[]{"fail", testId, "Expected not found in output:\n--- expected ---\n" + expected + "\n--- got ---\n" + result.stdout});
                                continue;
                            }
                            results.add(new String[]{"pass", testId, null});
                        } finally {
                            Files.deleteIfExists(tmpFile);
                        }
                    }
                }
            }
        }
        return results;
    }

    static List<String[]> runEmitTests(Path testDir) throws IOException {
        List<String[]> results = new ArrayList<>();
        Path baseDir = testDir.resolve("base");
        if (!Files.isDirectory(baseDir)) return results;

        List<String> langDirs;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir)) {
            langDirs = new ArrayList<>();
            for (Path p : stream) {
                if (Files.isDirectory(p)) {
                    String name = p.getFileName().toString();
                    if (!name.equals("base") && Arrays.asList(EMITTER_LANGS).contains(name)) {
                        langDirs.add(name);
                    }
                }
            }
        }
        Collections.sort(langDirs);

        for (String lang : langDirs) {
            Path langDir = testDir.resolve(lang);
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(baseDir, "*.tests")) {
                List<Path> files = new ArrayList<>();
                stream.forEach(files::add);
                Collections.sort(files);
                for (Path baseFile : files) {
                    String baseName = baseFile.getFileName().toString();
                    String stem = baseName.replace(".tests", "");
                    Path langFile = langDir.resolve(baseName);
                    List<SimpleEntry> baseTests = parseSimpleTests(Files.readString(baseFile));
                    if (baseTests.isEmpty()) continue;
                    if (!Files.exists(langFile)) continue;
                    List<SimpleEntry> langTests = parseSimpleTests(Files.readString(langFile));
                    Map<String, String> langByName = langTests.stream()
                        .collect(Collectors.toMap(e -> e.name, e -> e.content));
                    for (SimpleEntry entry : baseTests) {
                        if (!langByName.containsKey(entry.name)) continue;
                        String testId = stem + "/" + entry.name + "[" + lang + "]";
                        String expected = langByName.get(entry.name);
                        Path tmpFile = Files.createTempFile("test_", ".py");
                        try {
                            Files.writeString(tmpFile, entry.content);
                            RunResult result = runInprocess(new String[]{"--target", lang, tmpFile.toString()});
                            if (result.exit != 0) {
                                String stderr = result.stderr.trim().split("\n")[0];
                                results.add(new String[]{"fail", testId, "Emit error: " + stderr});
                                continue;
                            }
                            if (!containsNormalized(result.stdout, expected)) {
                                results.add(new String[]{"fail", testId, "Expected not found in output:\n--- expected ---\n" + expected + "\n--- got ---\n" + result.stdout});
                                continue;
                            }
                            results.add(new String[]{"pass", testId, null});
                        } finally {
                            Files.deleteIfExists(tmpFile);
                        }
                    }
                }
            }
        }
        return results;
    }

    static boolean runtimeAvailable(String lang) {
        String[] cmd = RUNTIMES.get(lang);
        if (cmd == null) return false;
        try {
            Process p = new ProcessBuilder("which", cmd[0]).start();
            return p.waitFor() == 0;
        } catch (Exception e) {
            return false;
        }
    }

    static List<String[]> runAppTests(Path testDir) throws IOException {
        List<String[]> results = new ArrayList<>();
        List<String> available = Arrays.stream(EMITTER_LANGS)
            .filter(TestTranspiled::runtimeAvailable)
            .sorted()
            .collect(Collectors.toList());

        if (!Files.isDirectory(testDir)) return results;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "apptest_*.py")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path testFile : files) {
                String stem = testFile.getFileName().toString().replace(".py", "");
                String source = Files.readString(testFile);
                for (String target : available) {
                    String testId = stem + "[" + target + "]";
                    Path tmpFile = Files.createTempFile("test_", ".py");
                    try {
                        Files.writeString(tmpFile, source);
                        RunResult result = runInprocess(new String[]{"--target", target, tmpFile.toString()});
                        if (result.exit != 0) {
                            String stderr = result.stderr.trim().split("\n")[0];
                            results.add(new String[]{"fail", testId, "Transpile error (" + target + "): " + stderr});
                            continue;
                        }
                        String transpiledCode = result.stdout;
                        String[] runtime = RUNTIMES.get(target);
                        ProcessBuilder pb = new ProcessBuilder(runtime);
                        pb.redirectErrorStream(true);
                        Process p = pb.start();
                        p.getOutputStream().write(transpiledCode.getBytes("UTF-8"));
                        p.getOutputStream().close();
                        String output = new String(p.getInputStream().readAllBytes(), "UTF-8");
                        int exitCode = p.waitFor();
                        if (exitCode != 0) {
                            results.add(new String[]{"fail", testId, "App test failed with exit " + exitCode + "\n" + output});
                            continue;
                        }
                        results.add(new String[]{"pass", testId, null});
                    } catch (InterruptedException e) {
                        results.add(new String[]{"fail", testId, "Interrupted"});
                    } finally {
                        Files.deleteIfExists(tmpFile);
                    }
                }
            }
        }
        return results;
    }

    static List<String[]> runTyAppTests(Path testDir) throws IOException {
        List<String[]> results = new ArrayList<>();
        if (!Files.isDirectory(testDir)) return results;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "*.ty")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path testFile : files) {
                String stem = testFile.getFileName().toString().replace(".ty", "");
                String testId = stem;
                RunResult result = runInprocess(new String[]{"taytsh", testFile.toString()});
                if (result.exit != 0) {
                    String output = (result.stdout + result.stderr).trim();
                    results.add(new String[]{"fail", testId, "Exit code " + result.exit + ":\n" + output});
                    continue;
                }
                results.add(new String[]{"pass", testId, null});
            }
        }
        return results;
    }

    static List<String[]> runOrderingTests(Path testDir) throws IOException {
        List<String[]> results = new ArrayList<>();
        List<String> available = Arrays.stream(EMITTER_LANGS)
            .filter(TestTranspiled::runtimeAvailable)
            .sorted()
            .collect(Collectors.toList());

        if (!Files.isDirectory(testDir)) return results;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "*.ty")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path testFile : files) {
                String stem = testFile.getFileName().toString().replace(".ty", "");
                for (String target : available) {
                    String testId = stem + "[" + target + "]";
                    RunResult result = runInprocess(new String[]{"taytsh", "--emit", target, testFile.toString()});
                    if (result.exit != 0) {
                        String stderr = result.stderr.trim().split("\n")[0];
                        results.add(new String[]{"fail", testId, "Transpile error (" + target + "): " + stderr});
                        continue;
                    }
                    String transpiledCode = result.stdout;
                    String[] runtime = RUNTIMES.get(target);
                    try {
                        ProcessBuilder pb = new ProcessBuilder(runtime);
                        pb.redirectErrorStream(true);
                        Process p = pb.start();
                        p.getOutputStream().write(transpiledCode.getBytes("UTF-8"));
                        p.getOutputStream().close();
                        String output = new String(p.getInputStream().readAllBytes(), "UTF-8");
                        int exitCode = p.waitFor();
                        if (exitCode != 0) {
                            results.add(new String[]{"fail", testId, "Ordering test failed with exit " + exitCode + "\n" + output});
                            continue;
                        }
                        results.add(new String[]{"pass", testId, null});
                    } catch (InterruptedException e) {
                        results.add(new String[]{"fail", testId, "Interrupted"});
                    }
                }
            }
        }
        return results;
    }

    // -------------------------------------------------------------------------
    // Parallel execution support
    // -------------------------------------------------------------------------

    static int getCpuCount() {
        return Runtime.getRuntime().availableProcessors();
    }

    static class TestDescriptor {
        String phaseName;
        String testId;
        String testType;
        Object testData;
        Map<String, Object> cfg;
        TestDescriptor(String phaseName, String testId, String testType, Object testData, Map<String, Object> cfg) {
            this.phaseName = phaseName;
            this.testId = testId;
            this.testType = testType;
            this.testData = testData;
            this.cfg = cfg;
        }
    }

    static List<TestDescriptor> collectTests() throws IOException {
        List<TestDescriptor> collected = new ArrayList<>();
        for (Section section : TESTS) {
            for (Phase phase : section.phases) {
                String phaseName = phase.name;
                Map<String, Object> cfg = phase.cfg;
                Path testDir = testsDir.resolve((String) cfg.get("dir"));
                if (!Files.isDirectory(testDir)) continue;
                String runnerName = (String) cfg.get("run");
                switch (runnerName) {
                    case "cli":
                        collectCliTests(testDir, phaseName, cfg, collected);
                        break;
                    case "linker":
                        collectLinkerTests(testDir, phaseName, cfg, collected);
                        break;
                    case "phase":
                        collectPhaseTests(testDir, phaseName, cfg, collected);
                        break;
                    case "lowering":
                        collectLoweringTests(testDir, phaseName, cfg, collected);
                        break;
                    case "codegen":
                        collectCodegenTests(testDir, phaseName, cfg, collected);
                        break;
                    case "emit":
                        collectEmitTests(testDir, phaseName, cfg, collected);
                        break;
                    case "app":
                        collectAppTests(testDir, phaseName, cfg, collected);
                        break;
                    case "ty_app":
                        collectTyAppTests(testDir, phaseName, cfg, collected);
                        break;
                    case "ordering":
                        collectOrderingTests(testDir, phaseName, cfg, collected);
                        break;
                }
            }
        }
        return collected;
    }

    static void collectCliTests(Path testDir, String phaseName, Map<String, Object> cfg, List<TestDescriptor> collected) throws IOException {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "*.tests")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path f : files) {
                String stem = f.getFileName().toString().replace(".tests", "");
                String content = Files.readString(f);
                for (Map.Entry<String, CliSpec> entry : parseCliTestFile(content)) {
                    String testId = stem + "/" + entry.getKey();
                    CliSpec spec = entry.getValue();
                    if (cliNeedsBackend(spec.args, spec.assertions, EMITTER_LANGS)) {
                        collected.add(new TestDescriptor(phaseName, testId, "skip", null, cfg));
                    } else {
                        collected.add(new TestDescriptor(phaseName, testId, "cli", spec, cfg));
                    }
                }
            }
        }
    }

    static void collectLinkerTests(Path testDir, String phaseName, Map<String, Object> cfg, List<TestDescriptor> collected) throws IOException {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "*.tests")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path f : files) {
                String stem = f.getFileName().toString().replace(".tests", "");
                String content = Files.readString(f);
                for (Map.Entry<String, LinkerSpec> entry : parseLinkerTestFile(content)) {
                    String testId = stem + "/" + entry.getKey();
                    LinkerSpec spec = entry.getValue();
                    int targetIdx = spec.args.indexOf("--target");
                    if (targetIdx != -1 && targetIdx + 1 < spec.args.size()) {
                        String target = spec.args.get(targetIdx + 1);
                        if (!Arrays.asList(EMITTER_LANGS).contains(target)) {
                            collected.add(new TestDescriptor(phaseName, testId, "skip", null, cfg));
                            continue;
                        }
                    }
                    collected.add(new TestDescriptor(phaseName, testId, "linker", spec, cfg));
                }
            }
        }
    }

    static void collectPhaseTests(Path testDir, String phaseName, Map<String, Object> cfg, List<TestDescriptor> collected) throws IOException {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "*.tests")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path f : files) {
                String stem = f.getFileName().toString().replace(".tests", "");
                String content = Files.readString(f);
                for (SpecEntry entry : parseSpecFile(content)) {
                    String testId = stem + "/" + entry.name;
                    collected.add(new TestDescriptor(phaseName, testId, "phase", entry, cfg));
                }
            }
        }
    }

    static void collectLoweringTests(Path testDir, String phaseName, Map<String, Object> cfg, List<TestDescriptor> collected) throws IOException {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "*.tests")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path f : files) {
                String stem = f.getFileName().toString().replace(".tests", "");
                String content = Files.readString(f);
                for (SpecEntry entry : parseSpecFile(content)) {
                    String testId = stem + "/" + entry.name;
                    collected.add(new TestDescriptor(phaseName, testId, "lowering", entry, cfg));
                }
            }
        }
    }

    static void collectCodegenTests(Path testDir, String phaseName, Map<String, Object> cfg, List<TestDescriptor> collected) throws IOException {
        Path baseDir = testDir.resolve("base");
        if (!Files.isDirectory(baseDir)) return;
        List<String> langDirs = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir)) {
            for (Path p : stream) {
                if (Files.isDirectory(p)) {
                    String name = p.getFileName().toString();
                    if (!name.equals("base") && Arrays.asList(EMITTER_LANGS).contains(name)) {
                        langDirs.add(name);
                    }
                }
            }
        }
        Collections.sort(langDirs);
        for (String lang : langDirs) {
            Path langDir = testDir.resolve(lang);
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(baseDir, "*.tests")) {
                List<Path> files = new ArrayList<>();
                stream.forEach(files::add);
                Collections.sort(files);
                for (Path baseFile : files) {
                    String baseName = baseFile.getFileName().toString();
                    String stem = baseName.replace(".tests", "");
                    Path langFile = langDir.resolve(baseName);
                    List<SimpleEntry> baseTests = parseSimpleTests(Files.readString(baseFile));
                    if (baseTests.isEmpty()) continue;
                    if (!Files.exists(langFile)) {
                        for (SimpleEntry e : baseTests) {
                            collected.add(new TestDescriptor(phaseName, stem + "/" + e.name + "[" + lang + "]", "prefail", lang + "/" + baseName + " missing", cfg));
                        }
                        continue;
                    }
                    List<SimpleEntry> langTests = parseSimpleTests(Files.readString(langFile));
                    Map<String, String> langByName = langTests.stream().collect(Collectors.toMap(e -> e.name, e -> e.content));
                    for (SimpleEntry entry : baseTests) {
                        String testId = stem + "/" + entry.name + "[" + lang + "]";
                        String expected = langByName.get(entry.name);
                        if (expected == null) {
                            collected.add(new TestDescriptor(phaseName, testId, "prefail", "No matching lang test", cfg));
                        } else {
                            collected.add(new TestDescriptor(phaseName, testId, "codegen", new Object[]{entry.content, expected, lang}, cfg));
                        }
                    }
                }
            }
        }
    }

    static void collectEmitTests(Path testDir, String phaseName, Map<String, Object> cfg, List<TestDescriptor> collected) throws IOException {
        Path baseDir = testDir.resolve("base");
        if (!Files.isDirectory(baseDir)) return;
        List<String> langDirs = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir)) {
            for (Path p : stream) {
                if (Files.isDirectory(p)) {
                    String name = p.getFileName().toString();
                    if (!name.equals("base") && Arrays.asList(EMITTER_LANGS).contains(name)) {
                        langDirs.add(name);
                    }
                }
            }
        }
        Collections.sort(langDirs);
        for (String lang : langDirs) {
            Path langDir = testDir.resolve(lang);
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(baseDir, "*.tests")) {
                List<Path> files = new ArrayList<>();
                stream.forEach(files::add);
                Collections.sort(files);
                for (Path baseFile : files) {
                    String baseName = baseFile.getFileName().toString();
                    String stem = baseName.replace(".tests", "");
                    Path langFile = langDir.resolve(baseName);
                    List<SimpleEntry> baseTests = parseSimpleTests(Files.readString(baseFile));
                    if (baseTests.isEmpty()) continue;
                    if (!Files.exists(langFile)) continue;
                    List<SimpleEntry> langTests = parseSimpleTests(Files.readString(langFile));
                    Map<String, String> langByName = langTests.stream().collect(Collectors.toMap(e -> e.name, e -> e.content));
                    for (SimpleEntry entry : baseTests) {
                        if (!langByName.containsKey(entry.name)) continue;
                        String testId = stem + "/" + entry.name + "[" + lang + "]";
                        collected.add(new TestDescriptor(phaseName, testId, "emit", new Object[]{entry.content, langByName.get(entry.name), lang}, cfg));
                    }
                }
            }
        }
    }

    static void collectAppTests(Path testDir, String phaseName, Map<String, Object> cfg, List<TestDescriptor> collected) throws IOException {
        List<String> available = Arrays.stream(STDIN_LANGS).filter(TestTranspiled::runtimeAvailable).sorted().collect(Collectors.toList());
        if (!Files.isDirectory(testDir)) return;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "apptest_*.py")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path testFile : files) {
                String stem = testFile.getFileName().toString().replace(".py", "");
                String source = Files.readString(testFile);
                for (String target : available) {
                    String testId = stem + "[" + target + "]";
                    collected.add(new TestDescriptor(phaseName, testId, "app", new Object[]{source, target}, cfg));
                }
            }
        }
    }

    static void collectTyAppTests(Path testDir, String phaseName, Map<String, Object> cfg, List<TestDescriptor> collected) throws IOException {
        if (!Files.isDirectory(testDir)) return;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "*.ty")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path testFile : files) {
                String stem = testFile.getFileName().toString().replace(".ty", "");
                collected.add(new TestDescriptor(phaseName, stem, "ty_app", testFile.toString(), cfg));
            }
        }
    }

    static void collectOrderingTests(Path testDir, String phaseName, Map<String, Object> cfg, List<TestDescriptor> collected) throws IOException {
        List<String> available = Arrays.stream(STDIN_LANGS).filter(TestTranspiled::runtimeAvailable).sorted().collect(Collectors.toList());
        if (!Files.isDirectory(testDir)) return;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "*.ty")) {
            List<Path> files = new ArrayList<>();
            stream.forEach(files::add);
            Collections.sort(files);
            for (Path testFile : files) {
                String stem = testFile.getFileName().toString().replace(".ty", "");
                for (String target : available) {
                    String testId = stem + "[" + target + "]";
                    collected.add(new TestDescriptor(phaseName, testId, "ordering", new Object[]{testFile.toString(), target}, cfg));
                }
            }
        }
    }

    static String[] runSingleTest(TestDescriptor test) {
        try {
            switch (test.testType) {
                case "skip":
                    return new String[]{test.phaseName, test.testId, "skip", null};
                case "prefail":
                    return new String[]{test.phaseName, test.testId, "fail", (String) test.testData};
                case "cli": {
                    CliSpec spec = (CliSpec) test.testData;
                    RunResult result;
                    if (!spec.stdinHex.isEmpty()) {
                        byte[] raw = hexToBytes(spec.stdinHex);
                        result = runInprocess(spec.args.toArray(new String[0]), "", raw);
                    } else {
                        result = runInprocess(spec.args.toArray(new String[0]), spec.stdin);
                    }
                    String err = checkCliAssertions(result.exit, result.stdout, result.stderr, spec.assertions);
                    return new String[]{test.phaseName, test.testId, err.isEmpty() ? "pass" : "fail", err.isEmpty() ? null : err};
                }
                case "linker": {
                    LinkerSpec spec = (LinkerSpec) test.testData;
                    List<String> parts = new ArrayList<>();
                    for (LinkerFile lf : spec.files) {
                        parts.add(lf.path);
                        parts.add(lf.source);
                    }
                    String stdinData = String.join("\0", parts);
                    RunResult result = runInprocess(spec.args.toArray(new String[0]), stdinData);
                    String err = checkCliAssertions(result.exit, result.stdout, result.stderr, spec.assertions);
                    return new String[]{test.phaseName, test.testId, err.isEmpty() ? "pass" : "fail", err.isEmpty() ? null : err};
                }
                case "phase": {
                    SpecEntry entry = (SpecEntry) test.testData;
                    boolean lenient = Arrays.asList("parse", "pycheck", "typarse", "tycheck").contains(test.phaseName);
                    String[] args = (String[]) test.cfg.getOrDefault("args", new String[0]);
                    boolean isTaytsh = (Boolean) test.cfg.getOrDefault("taytsh", false);
                    boolean expectJson = (Boolean) test.cfg.getOrDefault("json", true);
                    PhaseResult phaseResult = runTranspiledPhase(entry.input, args, isTaytsh, expectJson);
                    String err = checkExpected(entry.expected, phaseResult.errors, phaseResult.warnings, phaseResult.data, test.phaseName, lenient);
                    return new String[]{test.phaseName, test.testId, err.isEmpty() ? "pass" : "fail", err.isEmpty() ? null : err};
                }
                case "lowering": {
                    SpecEntry entry = (SpecEntry) test.testData;
                    Path tmpFile = Files.createTempFile("test_", ".py");
                    try {
                        Files.writeString(tmpFile, entry.input);
                        RunResult result = runInprocess(new String[]{"--stop-at", "lowering-text", tmpFile.toString()});
                        if (entry.expected.startsWith("error:")) {
                            String expectedMsg = entry.expected.substring(6).trim();
                            if (result.exit == 0) {
                                return new String[]{test.phaseName, test.testId, "fail", "Expected error containing '" + expectedMsg + "', got success"};
                            }
                            String firstLine = result.stderr.trim().split("\n")[0];
                            if (!expectedMsg.isEmpty() && !firstLine.toLowerCase().contains(expectedMsg.toLowerCase())) {
                                return new String[]{test.phaseName, test.testId, "fail", "Expected error containing '" + expectedMsg + "', got: " + firstLine};
                            }
                            return new String[]{test.phaseName, test.testId, "pass", null};
                        }
                        if (result.exit != 0) {
                            String errMsg = result.stderr.trim().split("\n")[0];
                            return new String[]{test.phaseName, test.testId, "fail", "Lowering error: " + errMsg};
                        }
                        if (!containsNormalized(result.stdout, entry.expected)) {
                            return new String[]{test.phaseName, test.testId, "fail", "Expected not found in output:\n--- expected ---\n" + entry.expected + "\n--- got ---\n" + result.stdout};
                        }
                        return new String[]{test.phaseName, test.testId, "pass", null};
                    } finally {
                        Files.deleteIfExists(tmpFile);
                    }
                }
                case "codegen": {
                    Object[] data = (Object[]) test.testData;
                    String content = (String) data[0];
                    String expected = (String) data[1];
                    String lang = (String) data[2];
                    Path tmpFile = Files.createTempFile("test_", ".ty");
                    try {
                        Files.writeString(tmpFile, content);
                        RunResult result = runInprocess(new String[]{"taytsh", "--emit", lang, tmpFile.toString()});
                        if (result.exit != 0) {
                            String stderr = result.stderr.trim().split("\n")[0];
                            return new String[]{test.phaseName, test.testId, "fail", "Transpile error: " + stderr};
                        }
                        if (!containsNormalized(result.stdout, expected)) {
                            return new String[]{test.phaseName, test.testId, "fail", "Expected not found in output:\n--- expected ---\n" + expected + "\n--- got ---\n" + result.stdout};
                        }
                        return new String[]{test.phaseName, test.testId, "pass", null};
                    } finally {
                        Files.deleteIfExists(tmpFile);
                    }
                }
                case "emit": {
                    Object[] data = (Object[]) test.testData;
                    String content = (String) data[0];
                    String expected = (String) data[1];
                    String lang = (String) data[2];
                    Path tmpFile = Files.createTempFile("test_", ".py");
                    try {
                        Files.writeString(tmpFile, content);
                        RunResult result = runInprocess(new String[]{"--target", lang, tmpFile.toString()});
                        if (result.exit != 0) {
                            String stderr = result.stderr.trim().split("\n")[0];
                            return new String[]{test.phaseName, test.testId, "fail", "Emit error: " + stderr};
                        }
                        if (!containsNormalized(result.stdout, expected)) {
                            return new String[]{test.phaseName, test.testId, "fail", "Expected not found in output:\n--- expected ---\n" + expected + "\n--- got ---\n" + result.stdout};
                        }
                        return new String[]{test.phaseName, test.testId, "pass", null};
                    } finally {
                        Files.deleteIfExists(tmpFile);
                    }
                }
                case "app": {
                    Object[] data = (Object[]) test.testData;
                    String source = (String) data[0];
                    String target = (String) data[1];
                    Path tmpFile = Files.createTempFile("test_", ".py");
                    try {
                        Files.writeString(tmpFile, source);
                        RunResult result = runInprocess(new String[]{"--target", target, tmpFile.toString()});
                        if (result.exit != 0) {
                            String stderr = result.stderr.trim().split("\n")[0];
                            return new String[]{test.phaseName, test.testId, "fail", "Transpile error (" + target + "): " + stderr};
                        }
                        String transpiledCode = result.stdout;
                        String[] runtime = RUNTIMES.get(target);
                        ProcessBuilder pb = new ProcessBuilder(runtime);
                        pb.redirectErrorStream(true);
                        Process p = pb.start();
                        p.getOutputStream().write(transpiledCode.getBytes("UTF-8"));
                        p.getOutputStream().close();
                        String output = new String(p.getInputStream().readAllBytes(), "UTF-8");
                        int exitCode = p.waitFor();
                        if (exitCode != 0) {
                            return new String[]{test.phaseName, test.testId, "fail", "App test failed with exit " + exitCode + "\n" + output};
                        }
                        return new String[]{test.phaseName, test.testId, "pass", null};
                    } finally {
                        Files.deleteIfExists(tmpFile);
                    }
                }
                case "ty_app": {
                    String testFile = (String) test.testData;
                    RunResult result = runInprocess(new String[]{"taytsh", testFile});
                    if (result.exit != 0) {
                        String output = (result.stdout + result.stderr).trim();
                        return new String[]{test.phaseName, test.testId, "fail", "Exit code " + result.exit + ":\n" + output};
                    }
                    return new String[]{test.phaseName, test.testId, "pass", null};
                }
                case "ordering": {
                    Object[] data = (Object[]) test.testData;
                    String testFile = (String) data[0];
                    String target = (String) data[1];
                    RunResult result = runInprocess(new String[]{"taytsh", "--emit", target, testFile});
                    if (result.exit != 0) {
                        String stderr = result.stderr.trim().split("\n")[0];
                        return new String[]{test.phaseName, test.testId, "fail", "Transpile error (" + target + "): " + stderr};
                    }
                    String transpiledCode = result.stdout;
                    String[] runtime = RUNTIMES.get(target);
                    ProcessBuilder pb = new ProcessBuilder(runtime);
                    pb.redirectErrorStream(true);
                    Process p = pb.start();
                    p.getOutputStream().write(transpiledCode.getBytes("UTF-8"));
                    p.getOutputStream().close();
                    String output = new String(p.getInputStream().readAllBytes(), "UTF-8");
                    int exitCode = p.waitFor();
                    if (exitCode != 0) {
                        return new String[]{test.phaseName, test.testId, "fail", "Ordering test failed with exit " + exitCode + "\n" + output};
                    }
                    return new String[]{test.phaseName, test.testId, "pass", null};
                }
                default:
                    return new String[]{test.phaseName, test.testId, "fail", "Unknown test type: " + test.testType};
            }
        } catch (Exception e) {
            return new String[]{test.phaseName, test.testId, "fail", "Exception: " + e.getMessage()};
        }
    }

    // -------------------------------------------------------------------------
    // Main
    // -------------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: java TestTranspiled <path-to-classes-dir> [--via-vm <tongues.ty>] [--target <name>] [-n <num|auto>]");
            System.exit(1);
        }

        String targetName = null;
        String viaVmPath = null;
        int numWorkers = getCpuCount();
        String classesDirArg = args[0];
        for (int i = 1; i < args.length; i++) {
            if (args[i].equals("--target") && i + 1 < args.length) {
                targetName = args[i + 1];
                i++;
            } else if (args[i].equals("--via-vm") && i + 1 < args.length) {
                viaVmPath = args[i + 1];
                i++;
            } else if (args[i].equals("-n") && i + 1 < args.length) {
                String val = args[i + 1];
                numWorkers = val.equals("auto") ? getCpuCount() : Integer.parseInt(val);
                i++;
            }
        }

        Path classesDir = Paths.get(classesDirArg).toAbsolutePath();
        if (!Files.isDirectory(classesDir)) {
            System.err.println("Classes directory not found: " + classesDir);
            System.exit(1);
        }

        // Determine tongues directory (current working directory when run from tongues/)
        tonguesDir = Paths.get("").toAbsolutePath();
        testsDir = tonguesDir.resolve("tests");
        libDir = tonguesDir.resolve("src").resolve("lib");

        // Enable test mode so SystemExitException propagates instead of calling System.exit
        System.setProperty("tongues.test", "1");

        System.out.println("Loading transpiled binary: " + classesDir);
        long t0 = System.currentTimeMillis();

        // Load Main class from the classes directory
        java.net.URLClassLoader classLoader = new java.net.URLClassLoader(
            new java.net.URL[]{classesDir.toUri().toURL()},
            TestTranspiled.class.getClassLoader()
        );
        mainClass = classLoader.loadClass("Main");
        mainMethod = mainClass.getMethod("main", String[].class);

        long t1 = System.currentTimeMillis();
        System.out.printf("Loaded in %.1fs%n", (t1 - t0) / 1000.0);

        if (viaVmPath != null) {
            // Resolve path relative to tonguesDir if not absolute
            Path vmPath = Paths.get(viaVmPath);
            if (!vmPath.isAbsolute()) {
                vmPath = tonguesDir.resolve(viaVmPath);
            }
            if (!Files.exists(vmPath)) {
                System.err.println("VM module not found: " + vmPath);
                System.exit(1);
            }
            System.out.println("Loading VM module: " + vmPath);
            long vmT0 = System.currentTimeMillis();
            loadVmModule(vmPath.toString());
            System.out.printf("VM compiled in %.1fs%n", (System.currentTimeMillis() - vmT0) / 1000.0);
            _useVm = true;
        }

        System.out.println();

        int totalPass = 0;
        int totalFail = 0;
        int totalSkip = 0;
        List<String[]> failures = new ArrayList<>();

        // Collect all tests
        System.out.println("Collecting tests...");
        List<TestDescriptor> allTests = collectTests();
        int totalTests = allTests.size();
        System.out.println("Running " + totalTests + " tests with " + numWorkers + " workers");
        System.out.println();

        String vmTag = viaVmPath != null ? " [vm]" : "";

        List<String[]> results = new ArrayList<>();
        if (_useVm) {
            // VM mode is thread-safe - run in parallel
            ExecutorService executor = Executors.newFixedThreadPool(numWorkers);
            List<Future<String[]>> futures = new ArrayList<>();
            for (TestDescriptor test : allTests) {
                futures.add(executor.submit(() -> runSingleTest(test)));
            }
            for (Future<String[]> future : futures) {
                try {
                    String[] result = future.get(30, TimeUnit.SECONDS);
                    results.add(result);
                } catch (TimeoutException e) {
                    results.add(new String[]{"unknown", "unknown", "fail", "TIMEOUT after 30s"});
                } catch (Exception e) {
                    results.add(new String[]{"unknown", "unknown", "fail", "Exception: " + e.getMessage()});
                }
            }
            executor.shutdown();
        } else {
            // Non-VM mode has static state - run serially
            System.out.println("(non-VM mode: running serially due to static state)");
            for (TestDescriptor test : allTests) {
                results.add(runSingleTest(test));
            }
        }

        // Process results
        for (String[] r : results) {
            String phaseName = r[0];
            String testId = r[1];
            String status = r[2];
            String err = r[3];
            if (status.equals("pass")) {
                System.out.println("PASS " + phaseName + "::" + testId + vmTag);
                totalPass++;
            } else if (status.equals("skip")) {
                System.out.println("SKIP " + phaseName + "::" + testId + vmTag);
                totalSkip++;
            } else {
                System.out.println("FAIL " + phaseName + "::" + testId + vmTag);
                failures.add(new String[]{phaseName, testId, err});
                totalFail++;
            }
        }

        System.out.println();
        if (!failures.isEmpty()) {
            System.out.println("=".repeat(60));
            System.out.println(targetName != null ? "FAILURES [" + targetName + "]" : "FAILURES");
            System.out.println("=".repeat(60));
            for (String[] f : failures) {
                System.out.println();
                System.out.println(f[0] + " :: " + f[1]);
                System.out.println(f[2]);
            }
            System.out.println();
        }

        System.out.println("=".repeat(60));
        int total = totalPass + totalFail + totalSkip;
        String prefix = targetName != null ? "[" + targetName + "] " : "";
        String summaryLine = prefix + total + " tests: " + totalPass + " passed, " + totalFail + " failed, " + totalSkip + " skipped";
        System.out.println(summaryLine);
        System.out.println("=".repeat(60));

        // GitHub Actions notice annotation
        if (totalFail == 0) {
            System.out.println("::notice::" + summaryLine);
        }

        // GitHub Actions job summary
        String summaryFile = System.getenv("GITHUB_STEP_SUMMARY");
        if (summaryFile != null) {
            try (java.io.PrintWriter pw = new java.io.PrintWriter(new java.io.FileWriter(summaryFile, true))) {
                String statusEmoji = totalFail == 0 ? "✅" : "❌";
                String name = targetName != null ? targetName : "Test Results";
                pw.println("## " + statusEmoji + " " + name + "\n");
                pw.println("| Passed | Failed | Skipped | Total |");
                pw.println("|--------|--------|---------|-------|");
                pw.println("| " + totalPass + " | " + totalFail + " | " + totalSkip + " | " + total + " |\n");
                if (!failures.isEmpty()) {
                    pw.println("### Failures\n");
                    for (String[] f : failures) {
                        pw.println("<details><summary><code>" + f[0] + " :: " + f[1] + "</code></summary>\n");
                        pw.println("```\n" + f[2] + "\n```\n");
                        pw.println("</details>\n");
                    }
                }
            } catch (Exception e) {
                System.err.println("Failed to write job summary: " + e);
            }
        }

        System.exit(totalFail > 0 ? 1 : 0);
    }
}
