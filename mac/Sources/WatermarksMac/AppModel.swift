import AppKit
import Foundation
import Observation
import UniformTypeIdentifiers

enum AppTab: String, CaseIterable, Identifiable {
    case files
    case text
    var id: String { rawValue }
}

enum JobPhase: String {
    case queued
    case inspecting
    case inspected
    case cleaning
    case cleaned
    case failed
}

struct FileJob: Identifiable {
    let id: UUID
    var url: URL
    var name: String
    var phase: JobPhase
    var inspect: InspectReport?
    var clean: CleanReport?
    var cleanedURL: URL?
    var error: String?
    var log: String
}

@MainActor
@Observable
final class AppModel {
    var tab: AppTab = .files
    var jobs: [FileJob] = []
    var selectedJobID: UUID?
    var isBusy = false
    var status = "Drop files you own, or paste text."
    var errorMessage: String?
    var showSettings = false
    var lastLog = ""

    var textInput = ""
    var textOutput = ""
    var textInspect: InspectReport?
    var textClean: CleanReport?
    var rewriteSkippedMessage: String?
    var textLog = ""

    let settings = SettingsStore()

    var selectedJob: FileJob? {
        jobs.first { $0.id == selectedJobID }
    }

    func addFiles(_ urls: [URL]) {
        errorMessage = nil
        for url in urls {
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
            if isDir.boolValue {
                errorMessage = "Folders are skipped in v1 — drop individual files."
                continue
            }
            if jobs.contains(where: { $0.url.path == url.path }) { continue }
            let job = FileJob(
                id: UUID(),
                url: url,
                name: url.lastPathComponent,
                phase: .queued,
                inspect: nil,
                clean: nil,
                cleanedURL: nil,
                error: nil,
                log: ""
            )
            jobs.append(job)
            if selectedJobID == nil { selectedJobID = job.id }
        }
        if !jobs.isEmpty {
            status = "\(jobs.count) file(s) ready"
        }
    }

    func openFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.begin { [weak self] response in
            guard response == .OK else { return }
            Task { @MainActor in
                self?.addFiles(panel.urls)
            }
        }
    }

    func inspectSelected() async {
        guard let id = selectedJobID else {
            await inspectAll()
            return
        }
        await inspect(id: id)
    }

    func inspectAll() async {
        for job in jobs where job.phase != .cleaning {
            await inspect(id: job.id)
        }
    }

    func inspect(id: UUID) async {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        isBusy = true
        errorMessage = nil
        jobs[index].phase = .inspecting
        jobs[index].error = nil
        status = "Inspecting \(jobs[index].name)…"
        let url = jobs[index].url
        do {
            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }
            let (report, result) = try await WatermarkService.inspect(
                file: url,
                pythonPath: settings.pythonPath
            )
            guard let idx = jobs.firstIndex(where: { $0.id == id }) else { return }
            jobs[idx].inspect = report
            jobs[idx].log = result.stderr
            jobs[idx].phase = .inspected
            lastLog = result.stderr
            status = report.summary
        } catch {
            if let idx = jobs.firstIndex(where: { $0.id == id }) {
                jobs[idx].phase = .failed
                jobs[idx].error = error.localizedDescription
            }
            errorMessage = error.localizedDescription
            status = "Inspect failed"
        }
        isBusy = false
    }

    func cleanSelected() async {
        guard let id = selectedJobID else {
            await cleanAll()
            return
        }
        await clean(id: id)
    }

    func cleanAll() async {
        for job in jobs {
            await clean(id: job.id)
        }
    }

    func clean(id: UUID) async {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        isBusy = true
        errorMessage = nil
        jobs[index].phase = .cleaning
        jobs[index].error = nil
        status = "Cleaning \(jobs[index].name)…"
        let url = jobs[index].url
        let dest = WatermarkService.siblingCleanedURL(for: url)
        do {
            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }
            let (report, result) = try await WatermarkService.clean(
                file: url,
                output: dest,
                pythonPath: settings.pythonPath
            )
            guard let idx = jobs.firstIndex(where: { $0.id == id }) else { return }
            jobs[idx].clean = report
            jobs[idx].cleanedURL = dest
            jobs[idx].log = result.stderr
            jobs[idx].phase = .cleaned
            lastLog = result.stderr
            if report.residual {
                status = "Cleaned with residual provenance — vendor detectors may still fire"
            } else if report.changed {
                status = "Wrote \(dest.lastPathComponent)"
            } else {
                status = "Already clean — wrote \(dest.lastPathComponent)"
            }
        } catch {
            if let idx = jobs.firstIndex(where: { $0.id == id }) {
                jobs[idx].phase = .failed
                jobs[idx].error = error.localizedDescription
            }
            errorMessage = error.localizedDescription
            status = "Clean failed"
        }
        isBusy = false
    }

    func revealCleaned() {
        guard let url = selectedJob?.cleanedURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func inspectText() async {
        let text = textInput
        guard !text.isEmpty else { return }
        isBusy = true
        errorMessage = nil
        rewriteSkippedMessage = nil
        status = "Inspecting text…"
        do {
            let temp = try writeTemp(text, suffix: ".txt")
            let (report, result) = try await WatermarkService.inspect(
                file: temp,
                pythonPath: settings.pythonPath
            )
            textInspect = report
            textLog = result.stderr
            lastLog = result.stderr
            status = report.summary
        } catch {
            errorMessage = error.localizedDescription
            status = "Inspect failed"
        }
        isBusy = false
    }

    func cleanText() async {
        let text = textInput
        guard !text.isEmpty else { return }
        isBusy = true
        errorMessage = nil
        rewriteSkippedMessage = nil
        status = "Cleaning text…"
        do {
            var source = try writeTemp(text, suffix: ".txt")
            let snap = settings.snapshot()
            if snap.rewriteEnabled {
                if snap.backendFlag == "openai-compatible", snap.apiKey.isEmpty {
                    rewriteSkippedMessage = "Rewrite skipped — add an API key in Settings. Layer A still ran."
                } else if snap.backendFlag == "ollama", snap.model.isEmpty {
                    rewriteSkippedMessage = "Rewrite skipped — set an Ollama model in Settings. Layer A still ran."
                } else {
                    status = "Rewriting text…"
                    let rewritten = FileManager.default.temporaryDirectory
                        .appendingPathComponent("watermarks-mac-\(UUID().uuidString).rewritten.txt")
                    do {
                        _ = try await WatermarkService.rewrite(file: source, output: rewritten, settings: snap)
                        source = rewritten
                    } catch {
                        rewriteSkippedMessage = "Rewrite skipped — \(error.localizedDescription) Layer A still ran."
                    }
                }
            }
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("watermarks-mac-\(UUID().uuidString).cleaned.txt")
            let (report, result) = try await WatermarkService.clean(
                file: source,
                output: dest,
                pythonPath: settings.pythonPath
            )
            textClean = report
            textOutput = (try? String(contentsOf: dest, encoding: .utf8)) ?? ""
            textLog = result.stderr
            lastLog = result.stderr
            status = report.changed ? "Text cleaned" : "No Layer A marks found"
        } catch {
            errorMessage = error.localizedDescription
            status = "Clean failed"
        }
        isBusy = false
    }

    func copyOutput() {
        let value: String
        if tab == .text {
            value = textOutput
        } else if let cleaned = selectedJob?.cleanedURL,
                  let contents = try? String(contentsOf: cleaned, encoding: .utf8) {
            value = contents
        } else {
            value = selectedJob?.inspect?.rawJSON ?? ""
        }
        guard !value.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
        status = "Copied"
    }

    func saveTextOutput() {
        guard !textOutput.isEmpty else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText, .text]
        panel.nameFieldStringValue = "cleaned.txt"
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url, let self else { return }
            try? self.textOutput.write(to: url, atomically: true, encoding: .utf8)
            Task { @MainActor in
                self.status = "Saved \(url.lastPathComponent)"
            }
        }
    }

    func clearFiles() {
        jobs.removeAll()
        selectedJobID = nil
        status = "Drop files you own, or paste text."
    }

    private func writeTemp(_ text: String, suffix: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("watermarks-mac-\(UUID().uuidString)\(suffix)")
        try text.write(to: url, atomically: true, encoding: .utf8)
        return url
    }
}
