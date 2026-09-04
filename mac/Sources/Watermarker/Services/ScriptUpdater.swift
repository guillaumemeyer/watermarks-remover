import Foundation

/// Pulls newer Layer B scripts from GitHub.
///
/// The app deliberately runs the upstream Python verbatim, so keeping current
/// means replacing files rather than shipping a new build. Downloads go to a
/// staging directory, get compiled with `python3 -m py_compile` before anything
/// is believed, and only then replace the live copy — a half-downloaded or
/// syntactically broken update can never become the active one.
struct ScriptUpdater: Sendable {
    struct Availability: Sendable {
        var commit: String
        var shortCommit: String
        var message: String
        var isNewer: Bool
    }

    enum UpdateError: LocalizedError {
        case badRepository(String)
        case network(String)
        case httpStatus(Int, String)
        case missingScript(String)
        case didNotCompile(String)

        var errorDescription: String? {
            switch self {
            case .badRepository(let value):
                return "\"\(value)\" is not an owner/repository pair, for example "
                    + "guillaumemeyer/watermarks-remover."
            case .network(let detail):
                return "Could not reach GitHub: \(detail)"
            case .httpStatus(let code, let path):
                return code == 404
                    ? "GitHub has no \(path) on that repository and branch."
                    : "GitHub answered \(code) for \(path)."
            case .missingScript(let name):
                return "The update is missing \(name), so it was not installed."
            case .didNotCompile(let detail):
                return "The downloaded scripts did not compile, so the working "
                    + "copy was left alone.\n\(detail)"
            }
        }
    }

    var repository: String
    var ref: String

    private var ownerAndName: (owner: String, name: String) {
        get throws {
            let parts = repository
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .split(separator: "/")
                .map(String.init)
            guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else {
                throw UpdateError.badRepository(repository)
            }
            return (parts[0], parts[1])
        }
    }

    private static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 30
        configuration.waitsForConnectivity = false
        return URLSession(configuration: configuration)
    }()

    // MARK: Checking

    /// Ask GitHub for the head commit of `ref` and compare it to what is
    /// installed.
    func check() async throws -> Availability {
        let (owner, name) = try ownerAndName
        let encodedRef = ref.addingPercentEncoding(
            withAllowedCharacters: .urlPathAllowed) ?? ref
        let url = URL(string: "https://api.github.com/repos/\(owner)/\(name)/commits/\(encodedRef)")!
        var request = URLRequest(url: url)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("Watermarker (macOS)", forHTTPHeaderField: "User-Agent")

        let data = try await fetch(request, describing: "the branch \(ref)")
        struct CommitResponse: Decodable {
            struct Commit: Decodable { var message: String }
            var sha: String
            var commit: Commit
        }
        guard let response = try? JSONDecoder().decode(CommitResponse.self, from: data) else {
            throw UpdateError.network("GitHub returned something this app could not read.")
        }
        let short = String(response.sha.prefix(7))
        let installed = ScriptBundle.loadManifest()?.commit
        return Availability(
            commit: response.sha,
            shortCommit: short,
            message: response.commit.message
                .split(separator: "\n").first.map(String.init) ?? response.commit.message,
            isNewer: installed != response.sha
        )
    }

    // MARK: Installing

    /// Download every required script at `commit`, verify it, and swap it in.
    /// Returns the manifest that is now live.
    func install(commit: String) async throws -> ScriptBundle.Manifest {
        let (owner, name) = try ownerAndName
        let staging = ScriptBundle.supportDirectory
            .appendingPathComponent("staging-\(UUID().uuidString)", isDirectory: true)
        let fm = FileManager.default
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: staging) }

        for script in ScriptBundle.requiredScripts {
            let raw = "https://raw.githubusercontent.com/\(owner)/\(name)/\(commit)/service/scripts/\(script)"
            guard let url = URL(string: raw) else { throw UpdateError.missingScript(script) }
            var request = URLRequest(url: url)
            request.setValue("Watermarker (macOS)", forHTTPHeaderField: "User-Agent")
            let data = try await fetch(request, describing: "service/scripts/\(script)")
            guard !data.isEmpty else { throw UpdateError.missingScript(script) }
            try data.write(to: staging.appendingPathComponent(script), options: .atomic)
        }

        try verifyCompiles(in: staging)

        // Swap: move the staged copy into place, keeping the previous one until
        // the new one has landed.
        let live = ScriptBundle.updatedDirectory
        let previous = ScriptBundle.supportDirectory
            .appendingPathComponent("PythonScripts.previous", isDirectory: true)
        try? fm.removeItem(at: previous)
        if fm.fileExists(atPath: live.path) {
            try fm.moveItem(at: live, to: previous)
        }
        do {
            try fm.moveItem(at: staging, to: live)
        } catch {
            // Put the working copy back rather than leaving the app with none.
            if fm.fileExists(atPath: previous.path) {
                try? fm.moveItem(at: previous, to: live)
            }
            throw error
        }
        try? fm.removeItem(at: previous)

        let manifest = ScriptBundle.Manifest(
            repository: repository,
            ref: ref,
            commit: commit,
            shortCommit: String(commit.prefix(7)),
            installedAt: Date()
        )
        try ScriptBundle.saveManifest(manifest)
        return manifest
    }

    /// Drop any downloaded scripts and go back to the copy inside the app.
    static func revertToBundled() throws {
        let fm = FileManager.default
        try? fm.removeItem(at: ScriptBundle.manifestURL)
        if fm.fileExists(atPath: ScriptBundle.updatedDirectory.path) {
            try fm.removeItem(at: ScriptBundle.updatedDirectory)
        }
    }

    // MARK: Helpers

    private func fetch(_ request: URLRequest, describing what: String) async throws -> Data {
        do {
            let (data, response) = try await Self.session.data(for: request)
            if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                throw UpdateError.httpStatus(http.statusCode, what)
            }
            return data
        } catch let error as UpdateError {
            throw error
        } catch {
            throw UpdateError.network(error.localizedDescription)
        }
    }

    /// `python3 -m py_compile` over the staged files. This is the gate that
    /// stops a truncated download or an upstream syntax error from bricking the
    /// Run button.
    private func verifyCompiles(in directory: URL) throws {
        guard let interpreter = PythonRunner.findInterpreter() else {
            throw PythonRunner.RunError.interpreterNotFound
        }
        let process = Process()
        process.executableURL = interpreter
        process.arguments = ["-m", "py_compile"]
            + ScriptBundle.requiredScripts.map { directory.appendingPathComponent($0).path }
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        process.environment = environment
        let errorPipe = Pipe()
        process.standardError = errorPipe
        process.standardOutput = Pipe()
        do {
            try process.run()
        } catch {
            throw PythonRunner.RunError.launchFailed(error.localizedDescription)
        }
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw UpdateError.didNotCompile(String(decoding: errorData, as: UTF8.self))
        }
    }
}
