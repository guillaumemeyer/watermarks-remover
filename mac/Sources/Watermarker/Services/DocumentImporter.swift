import Foundation
import UniformTypeIdentifiers

/// Turns a file the user picked into the plain text the Layer B rewrite works
/// on.
///
/// Markdown and plain text come through unchanged. A .docx is unzipped and its
/// WordprocessingML body is flattened to text — the rewrite operates on
/// wording, so styling and the rest of the package are deliberately dropped
/// rather than round-tripped.
enum DocumentImporter {
    static let supportedTypes: [UTType] = {
        var types: [UTType] = [.plainText, .text]
        if let markdown = UTType(filenameExtension: "md") { types.append(markdown) }
        if let markdown = UTType("net.daringfireball.markdown") { types.append(markdown) }
        if let docx = UTType(filenameExtension: "docx") { types.append(docx) }
        return types
    }()

    enum ImportError: LocalizedError {
        case unreadableEncoding(String)
        case emptyDocument(String)
        case unsupportedExtension(String)

        var errorDescription: String? {
            switch self {
            case .unreadableEncoding(let name):
                return "Could not read \(name) as text. It may not be UTF-8."
            case .emptyDocument(let name):
                return "\(name) has no body text to clean."
            case .unsupportedExtension(let ext):
                return ext.isEmpty
                    ? "That file has no extension; open a .md, .txt, or .docx."
                    : "Files of type .\(ext) are not supported. Open a .md, .txt, or .docx."
            }
        }
    }

    struct Imported {
        var text: String
        var sourceName: String
        /// True when the original was a .docx, so the UI can say the export
        /// will be plain text rather than Word.
        var wasConverted: Bool
    }

    static func load(from url: URL) throws -> Imported {
        let name = url.lastPathComponent
        let ext = url.pathExtension.lowercased()
        switch ext {
        case "docx":
            let data = try Data(contentsOf: url)
            let xml = try ZipArchive.extract("word/document.xml", from: data)
            let text = try DocxBody.text(from: xml)
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ImportError.emptyDocument(name)
            }
            return Imported(text: text, sourceName: name, wasConverted: true)
        case "md", "markdown", "mdown", "txt", "text", "":
            let data = try Data(contentsOf: url)
            guard let text = String(data: data, encoding: .utf8)
                ?? String(data: data, encoding: .isoLatin1)
            else { throw ImportError.unreadableEncoding(name) }
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ImportError.emptyDocument(name)
            }
            return Imported(text: text, sourceName: name, wasConverted: false)
        default:
            throw ImportError.unsupportedExtension(ext)
        }
    }
}

/// Flattens `word/document.xml` to text.
///
/// WordprocessingML puts runs of text in `w:t` inside paragraphs (`w:p`), with
/// `w:br` and `w:tab` for in-paragraph breaks. That is the whole subset that
/// matters for a rewrite; tables contribute their cell paragraphs in document
/// order, which reads acceptably as prose.
private final class DocxBody: NSObject, XMLParserDelegate {
    private var paragraphs: [String] = []
    private var current = ""
    private var capturingText = false
    /// Text inside `w:instrText` is a field code (a TOC switch, a page ref),
    /// never body prose, so it is skipped.
    private var suppressDepth = 0

    static func text(from xml: Data) throws -> String {
        let body = DocxBody()
        let parser = XMLParser(data: xml)
        parser.delegate = body
        parser.shouldProcessNamespaces = false
        guard parser.parse() else {
            throw parser.parserError ?? ZipArchive.ZipError.corruptEntry("word/document.xml")
        }
        body.endParagraph()
        return body.paragraphs.joined(separator: "\n\n")
    }

    private func endParagraph() {
        let trimmed = current.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty { paragraphs.append(trimmed) }
        current = ""
    }

    func parser(_ parser: XMLParser, didStartElement elementName: String,
                namespaceURI: String?, qualifiedName: String?,
                attributes: [String: String]) {
        switch elementName {
        case "w:t":
            capturingText = suppressDepth == 0
        case "w:instrText", "w:delText":
            suppressDepth += 1
        case "w:br", "w:cr":
            if suppressDepth == 0 { current += "\n" }
        case "w:tab":
            if suppressDepth == 0 { current += "\t" }
        default:
            break
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        guard capturingText else { return }
        current += string
    }

    func parser(_ parser: XMLParser, didEndElement elementName: String,
                namespaceURI: String?, qualifiedName: String?) {
        switch elementName {
        case "w:t":
            capturingText = false
        case "w:instrText", "w:delText":
            suppressDepth = max(0, suppressDepth - 1)
        case "w:p":
            endParagraph()
        default:
            break
        }
    }
}
