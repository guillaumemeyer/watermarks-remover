import AppKit
import SwiftUI

@main
struct WatermarkerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var settings = SettingsStore()
    @StateObject private var model = AppModel()

    var body: some Scene {
        Window("Watermarker", id: "main") {
            ContentView()
                .environmentObject(settings)
                .environmentObject(model)
                .frame(minWidth: 860, minHeight: 560)
        }
        .commands {
            CommandGroup(replacing: .appSettings) {
                Button("Settings…") { model.isShowingSettings = true }
                    .keyboardShortcut(",", modifiers: .command)
            }
            CommandGroup(replacing: .newItem) {
                Button("Open…") { model.presentOpenPanel() }
                    .keyboardShortcut("o", modifiers: .command)
            }
            CommandGroup(after: .saveItem) {
                Button("Save Cleaned Text…") { model.saveOutput() }
                    .keyboardShortcut("s", modifiers: .command)
                    .disabled(model.outputText.isEmpty)
            }
            CommandMenu("Tools") {
                Button("Remove Watermarks") {
                    Task { await model.run(settings: settings) }
                }
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!model.canRun || !settings.hasAPIKey)

                Divider()

                Button("Copy Result") { model.copyOutput() }
                    .disabled(model.outputText.isEmpty)
                Button("Reuse Result as Input") { model.useOutputAsInput() }
                    .disabled(model.outputText.isEmpty)

                Divider()

                Button("Clear") { model.clearInput() }
            }
        }
    }
}

/// A SwiftUI-only app on macOS still needs a delegate for the two AppKit
/// behaviours the framework does not cover: activating on launch, and quitting
/// when the single window closes.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
