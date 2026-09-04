import Foundation

/// Runs the repository's Python scripts out of process.
///
/// The Layer B tools are the upstream scripts verbatim — the app deliberately
/// does not reimplement them, so a fix landing upstream is a script update
/// rather than a new release of this app.
enum PythonRunner {
    struct Result {
        var exitCode: Int32
        var standardOutput: String
        var standardError: String

        var succeeded: Bool { exitCode == 0 }
    }

    enum RunError: LocalizedError {
        case interpreterNotFound
        case timedOut(seconds: Double)
        case launchFailed(String)

        var errorDescription: String? {
            switch self {
            case .interpreterNotFound:
                return "No python3 found. Install the Xcode command line tools "
                    + "(xcode-select --install) or Homebrew's python3, then try again."
            case .timedOut(let seconds):
                return "The rewrite did not finish within \(Int(seconds)) seconds. "
                    + "Raise the timeout in Settings, or try a shorter passage."
            case .launchFailed(let detail):
                return "Could not start python3: \(detail)"
            }
        }
    }

    /// The interpreters worth trying, most likely first. The scripts need only
    /// the standard library, so the system python3 is enough for every strategy
    /// except the masked-LM one.
    private static let candidatePaths = [
        "/usr/bin/python3",
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        "/opt/homebrew/opt/python3/libexec/bin/python3",
    ]

    static func findInterpreter() -> URL? {
        let fm = FileManager.default
        for path in candidatePaths where fm.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        // Fall back to whatever is on PATH, which covers pyenv and conda.
        guard let pathVariable = ProcessInfo.processInfo.environment["PATH"] else { return nil }
        for directory in pathVariable.split(separator: ":") {
            let candidate = URL(fileURLWithPath: String(directory))
                .appendingPathComponent("python3")
            if fm.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    /// Run `script` with `arguments`, feeding `input` on stdin.
    ///
    /// `environment` is merged over the process environment; this is how the
    /// API key reaches the script. `rewrite_text.py` has no `--api-key` flag on
    /// purpose — a key on argv shows up in `ps` — so the environment is the
    /// only channel, and it is never logged.
    static func run(script: URL,
                    arguments: [String],
                    input: String,
                    environment: [String: String],
                    workingDirectory: URL,
                    timeout: Double) throws -> Result {
        guard let interpreter = findInterpreter() else { throw RunError.interpreterNotFound }

        let process = Process()
        process.executableURL = interpreter
        process.arguments = [script.path] + arguments
        process.currentDirectoryURL = workingDirectory

        var merged = ProcessInfo.processInfo.environment
        // Keep the scripts' own output unbuffered and predictable.
        merged["PYTHONIOENCODING"] = "utf-8"
        merged["PYTHONUNBUFFERED"] = "1"
        merged["PYTHONDONTWRITEBYTECODE"] = "1"
        merged.merge(environment) { _, new in new }
        process.environment = merged

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            throw RunError.launchFailed(error.localizedDescription)
        }

        // Drain both pipes on background queues: a script that writes more than
        // a pipe buffer's worth would otherwise block forever while we wait.
        let outBox = OutputBox()
        let queue = DispatchQueue(label: "com.symbiola.Watermarker.python", attributes: .concurrent)
        let group = DispatchGroup()
        queue.async(group: group) {
            outBox.appendOut(stdoutPipe.fileHandleForReading.readDataToEndOfFile())
        }
        queue.async(group: group) {
            outBox.appendErr(stderrPipe.fileHandleForReading.readDataToEndOfFile())
        }

        if let data = input.data(using: .utf8) {
            try? stdinPipe.fileHandleForWriting.write(contentsOf: data)
        }
        try? stdinPipe.fileHandleForWriting.close()

        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if process.isRunning {
            process.terminate()
            // Give it a moment to die politely before reporting the timeout.
            let graceDeadline = Date().addingTimeInterval(2)
            while process.isRunning && Date() < graceDeadline {
                Thread.sleep(forTimeInterval: 0.05)
            }
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
            _ = group.wait(timeout: .now() + 5)
            throw RunError.timedOut(seconds: timeout)
        }

        process.waitUntilExit()
        _ = group.wait(timeout: .now() + 10)
        return Result(exitCode: process.terminationStatus,
                      standardOutput: outBox.stdoutText,
                      standardError: outBox.stderrText)
    }
}

/// A tiny lock around the two pipe buffers, since they fill from two queues.
private final class OutputBox: @unchecked Sendable {
    private let lock = NSLock()
    private var out = Data()
    private var err = Data()

    func appendOut(_ data: Data) { lock.lock(); out.append(data); lock.unlock() }
    func appendErr(_ data: Data) { lock.lock(); err.append(data); lock.unlock() }

    var stdoutText: String {
        lock.lock(); defer { lock.unlock() }
        return String(decoding: out, as: UTF8.self)
    }

    var stderrText: String {
        lock.lock(); defer { lock.unlock() }
        return String(decoding: err, as: UTF8.self)
    }
}
