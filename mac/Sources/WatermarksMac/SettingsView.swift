import SwiftUI

struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var revealKey = false
    @State private var keyDraft = ""
    @State private var pythonResolved: String = "…"

    var body: some View {
        @Bindable var settings = model.settings

        VStack(alignment: .leading, spacing: 16) {
            Text("Settings")
                .font(.title2.weight(.semibold))

            GroupBox("Python") {
                VStack(alignment: .leading, spacing: 8) {
                    TextField("Interpreter path (blank = auto)", text: $settings.pythonPath)
                    Text("Using \(pythonResolved)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(6)
            }

            GroupBox("Layer B rewrite") {
                VStack(alignment: .leading, spacing: 10) {
                    Picker("Backend", selection: $settings.backend) {
                        ForEach(RewriteBackend.allCases) { backend in
                            Text(backend.label).tag(backend)
                        }
                    }
                    Picker("Tactic", selection: $settings.tactic) {
                        Text("Paraphrase").tag(RewriteTactic.paraphrase)
                        Text("Humanize").tag(RewriteTactic.humanize)
                    }
                    if settings.backend == .ollama {
                        TextField("Ollama model", text: $settings.ollamaModel)
                        TextField("Ollama URL", text: $settings.ollamaBaseURL)
                    }
                    if settings.backend == .openaiCompatible {
                        TextField("Model", text: $settings.openaiModel)
                        TextField("Base URL", text: $settings.openaiBaseURL)
                        HStack {
                            Group {
                                if revealKey {
                                    TextField("API key", text: $keyDraft)
                                } else {
                                    SecureField("API key", text: $keyDraft)
                                }
                            }
                            Button {
                                revealKey.toggle()
                            } label: {
                                Image(systemName: revealKey ? "eye.slash" : "eye")
                            }
                            .help(revealKey ? "Hide the key" : "Show the key")
                            .accessibilityLabel(revealKey ? "Hide the key" : "Show the key")
                        }
                    }
                    Text("Rewrite is off by default. Clean still strips Unicode and metadata without a model.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(6)
            }

            Spacer()
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Save") {
                    model.settings.apiKey = keyDraft
                    model.settings.saveAPIKey()
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .onAppear {
            keyDraft = model.settings.apiKey
            pythonResolved = PythonRunner.findInterpreter(override: model.settings.pythonPath) ?? "not found"
        }
    }
}
