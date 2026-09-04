import SwiftUI

/// The palette is lifted from the app icon: a deep navy ground, brushed-steel
/// chrome for controls, and the pale document blue for anything the user is
/// reading. Every colour lives here so the window, the sheets, and the icon
/// stay in agreement.
enum Theme {
    static let deepNavy = Color(red: 0.098, green: 0.165, blue: 0.235)
    static let navy = Color(red: 0.145, green: 0.239, blue: 0.333)
    static let midNavy = Color(red: 0.184, green: 0.298, blue: 0.408)
    static let slate = Color(red: 0.243, green: 0.361, blue: 0.475)

    static let steel = Color(red: 0.690, green: 0.745, blue: 0.788)
    static let steelBright = Color(red: 0.914, green: 0.937, blue: 0.953)

    static let paper = Color(red: 0.651, green: 0.773, blue: 0.886)
    static let accent = Color(red: 0.435, green: 0.639, blue: 0.824)

    static let ink = Color(red: 0.878, green: 0.918, blue: 0.957)
    static let inkDim = Color(red: 0.650, green: 0.718, blue: 0.784)

    static let danger = Color(red: 0.902, green: 0.451, blue: 0.404)
    static let success = Color(red: 0.478, green: 0.784, blue: 0.635)

    static let windowBackground = LinearGradient(
        colors: [midNavy, deepNavy],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    /// The recessed well every text view and list sits in.
    static let fieldBackground = Color.black.opacity(0.22)
    static let fieldBorder = steel.opacity(0.28)

    static let editorFont = Font.system(size: 13, design: .default)
    static let monoFont = Font.system(size: 12, design: .monospaced)
}

/// A filled, steel-rimmed button — the primary action in the window.
///
/// Both styles read `isEnabled` from a nested view rather than from the style
/// itself: a `ButtonStyle` is not part of the view hierarchy, so an
/// `@Environment` property declared on it never updates.
struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        StyledLabel(configuration: configuration)
    }

    private struct StyledLabel: View {
        let configuration: ButtonStyleConfiguration
        @Environment(\.isEnabled) private var isEnabled

        var body: some View {
            configuration.label
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(isEnabled ? Theme.deepNavy : Theme.deepNavy.opacity(0.45))
                .padding(.horizontal, 18)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(isEnabled ? Theme.steelBright : Theme.steel.opacity(0.4))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Color.white.opacity(0.55), lineWidth: 0.5)
                )
                .opacity(configuration.isPressed ? 0.75 : 1)
                .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
    }
}

/// An outlined button for everything that is not the primary action.
struct SecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        StyledLabel(configuration: configuration)
    }

    private struct StyledLabel: View {
        let configuration: ButtonStyleConfiguration
        @Environment(\.isEnabled) private var isEnabled

        var body: some View {
            configuration.label
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(isEnabled ? Theme.ink : Theme.inkDim.opacity(0.5))
                .padding(.horizontal, 14)
                .padding(.vertical, 7)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color.white.opacity(configuration.isPressed ? 0.16 : 0.07))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Theme.steel.opacity(isEnabled ? 0.45 : 0.2), lineWidth: 1)
                )
                .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
    }
}

extension View {
    /// The recessed panel treatment shared by the editor and the log.
    func watermarkerWell() -> some View {
        background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Theme.fieldBackground)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(Theme.fieldBorder, lineWidth: 1)
        )
    }
}
