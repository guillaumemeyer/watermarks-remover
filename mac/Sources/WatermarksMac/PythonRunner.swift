import Foundation

struct ProcessResult: Sendable {
    var exitCode: Int32
    var stdout: String
    var stderr: String
}

enum RunnerError: LocalizedError, Sendable {
    case missingPython
    case missingScripts(String)
    case missingScript(String)
    case timeout(TimeInterval)
    case launchFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingPython:
            return "Python 3 not found. Install it (Homebrew python3) or set the path in Settings."
        case .missingScripts(let path):
            return "Bundled scripts not found at \(path)."
        case .missingScript(let name):
            return "Missing script \(name)."
        case .timeout(let seconds):
            return "Timed out after \(Int(seconds))s."
        case .launchFailed(let message):
            return message
        }
    }
}

enum PythonRunner {
    static func findInterpreter(override: String) -> String? {
        let fm = FileManager.default
        let trimmed = override.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty, fm.isExecutableFile(atPath: trimmed) {
            return trimmed
        }
        let fallbacks = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]
        return fallbacks.first { fm.isExecutableFile(atPath: $0) }
    }

    static func run(
        script: String,
        arguments: [String],
        extraEnv: [String: String] = [:],
        pythonPath: String,
        timeout: TimeInterval
    ) async throws -> ProcessResult {
        try await Task.detached(priority: .userInitiated) {
            try runSync(
                script: script,
                arguments: arguments,
                extraEnv: extraEnv,
                pythonPath: pythonPath,
                timeout: timeout
            )
        }.value
    }

    private static func runSync(
        script: String,
        arguments: [String],
        extraEnv: [String: String],
        pythonPath: String,
        timeout: TimeInterval
    ) throws -> ProcessResult {
        guard let python = findInterpreter(override: pythonPath) else {
            throw RunnerError.missingPython
        }
        let scripts = try ScriptBundle.directory()
        let scriptURL = scripts.appendingPathComponent(script)
        guard FileManager.default.fileExists(atPath: scriptURL.path) else {
            throw RunnerError.missingScript(script)
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = [scriptURL.path] + arguments
        process.currentDirectoryURL = scripts

        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        for (key, value) in extraEnv {
            env[key] = value
        }
        process.environment = env

        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        process.standardInput = FileHandle.nullDevice

        do {
            try process.run()
        } catch {
            throw RunnerError.launchFailed(error.localizedDescription)
        }

        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if process.isRunning {
            process.terminate()
            throw RunnerError.timeout(timeout)
        }
        process.waitUntilExit()

        let stdout = String(data: outPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let stderr = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        return ProcessResult(exitCode: process.terminationStatus, stdout: stdout, stderr: redact(stderr, extraEnv: extraEnv))
    }

    private static func redact(_ text: String, extraEnv: [String: String]) -> String {
        var result = text
        if let key = extraEnv["WATERMARKS_REWRITE_API_KEY"], !key.isEmpty {
            result = result.replacingOccurrences(of: key, with: "***")
        }
        return result
    }
}
