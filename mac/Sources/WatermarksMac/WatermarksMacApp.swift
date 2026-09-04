import AppKit
import SwiftUI
import UniformTypeIdentifiers

@main
struct WatermarksMacApp: App {
    @State private var model = AppModel()

    init() {
        NSApplication.shared.setActivationPolicy(.regular)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(model)
                .onAppear {
                    NSApp.activate(ignoringOtherApps: true)
                }
        }
        .defaultSize(width: 920, height: 640)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("File") {
                Button("Open…") { model.openFiles() }
                    .keyboardShortcut("o")
                Button("Inspect") {
                    Task {
                        if model.tab == .text { await model.inspectText() }
                        else { await model.inspectSelected() }
                    }
                }
                .keyboardShortcut("i")
                Button("Clean") {
                    Task {
                        if model.tab == .text { await model.cleanText() }
                        else { await model.cleanSelected() }
                    }
                }
                .keyboardShortcut("k")
                Button("Reveal Cleaned") { model.revealCleaned() }
                    .disabled(model.selectedJob?.cleanedURL == nil)
                Divider()
                Button("Settings…") { model.showSettings = true }
                    .keyboardShortcut(",")
                Button("Quit") { NSApp.terminate(nil) }
                    .keyboardShortcut("q")
            }
        }
    }
}
