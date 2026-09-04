import AppKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

/// The window's state: what the user typed or imported, what came back, and
/// what the tool is doing right now.
@MainActor
final class AppModel: ObservableObject {
    @Published var inputText: String = ""
    @Published var outputText: String = ""
    @Published var isRunning = false
    @Published var sourceName: String?
    /// Set when the imported file was converted to plain text on the way in.
    @Published var sourceWasConverted = false
    @Published var status: String = "Ready."
    @Published var errorMessage: String?
    @Published var log: String = ""
    /// Driven from both the header button and the App menu's Settings item.
    @Published var isShowingSettings = false
    @Published var lastStats: RewriteService.Stats?
    @Published private(set) var scriptSource: ScriptBundle.Source?

    private let rewriter = RewriteService()

    var canRun: Bool {
        !isRunning && !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var inputWordCount: Int {
        inputText.split { $0.isWhitespace || $0.isNewline }.count
    }

    var outputWordCount: Int {
        outputText.split { $0.isWhitespace || $0.isNewline }.count
    }

    init() {
        refreshScriptSource()
    }

    func refreshScriptSource() {
        scriptSource = ScriptBundle.active()?.source
    }

    // MARK: Input

    func presentOpenPanel() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = DocumentImporter.supportedTypes
        panel.message = "Choose a Markdown, plain text, or Word document."
        panel.prompt = "Import"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        importFile(at: url)
    }

    func importFile(at url: URL) {
        do {
            let imported = try DocumentImporter.load(from: url)
            inputText = imported.text
            sourceName = imported.sourceName
            sourceWasConverted = imported.wasConverted
            outputText = ""
            lastStats = nil
            errorMessage = nil
            status = imported.wasConverted
                ? "Imported \(imported.sourceName) as plain text."
                : "Imported \(imported.sourceName)."
        } catch {
            errorMessage = error.localizedDescription
            status = "Import failed."
        }
    }

    func clearInput() {
        inputText = ""
        outputText = ""
        sourceName = nil
        sourceWasConverted = false
        lastStats = nil
        log = ""
        errorMessage = nil
        status = "Ready."
    }

    // MARK: Running

    func run(settings: SettingsStore) async {
        guard canRun else { return }
        isRunning = true
        errorMessage = nil
        outputText = ""
        lastStats = nil
        status = "Rewriting with \(settings.model)…"
        defer { isRunning = false }

        let request = RewriteService.Request(
            text: inputText,
            model: settings.model,
            baseURL: settings.baseURL,
            apiKey: settings.apiKey,
            strategy: settings.strategy,
            temperature: settings.temperature,
            timeout: settings.timeoutSeconds
        )
        let rewriter = self.rewriter
        let started = Date()

        do {
            let outcome = try await Task.detached(priority: .userInitiated) {
                try rewriter.run(request)
            }.value
            outputText = outcome.text
            lastStats = outcome.stats
            log = outcome.log
            settings.rememberCurrentModel()
            let elapsed = Date().timeIntervalSince(started)
            status = String(format: "Done in %.1fs — %d words in, %d words out.",
                            elapsed, inputWordCount, outputWordCount)
        } catch {
            errorMessage = error.localizedDescription
            status = "The rewrite did not finish."
        }
    }

    // MARK: Output

    func copyOutput() {
        guard !outputText.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(outputText, forType: .string)
        status = "Copied the cleaned text to the clipboard."
    }

    func useOutputAsInput() {
        guard !outputText.isEmpty else { return }
        inputText = outputText
        outputText = ""
        lastStats = nil
        sourceName = nil
        status = "Moved the result back into the editor for another pass."
    }

    func saveOutput() {
        guard !outputText.isEmpty else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "md") ?? .plainText]
        panel.nameFieldStringValue = suggestedFileName()
        panel.message = "Save the cleaned text."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try outputText.write(to: url, atomically: true, encoding: .utf8)
            status = "Saved to \(url.lastPathComponent)."
        } catch {
            errorMessage = error.localizedDescription
            status = "Could not save the file."
        }
    }

    private func suggestedFileName() -> String {
        guard let sourceName else { return "cleaned.md" }
        let base = (sourceName as NSString).deletingPathExtension
        return "\(base).cleaned.md"
    }
}
