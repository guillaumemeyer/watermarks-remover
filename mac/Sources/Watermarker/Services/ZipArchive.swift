import Compression
import Foundation

/// The little of the ZIP format a .docx needs: find one member by name and
/// inflate it.
///
/// A .docx is a ZIP whose body text lives in `word/document.xml`, so rather
/// than shelling out to `unzip` or taking on a dependency, this reads the
/// central directory directly and hands the compressed bytes to `Compression`.
/// `COMPRESSION_ZLIB` there is raw DEFLATE, which is exactly what ZIP method 8
/// stores.
enum ZipArchive {
    enum ZipError: LocalizedError {
        case notAZipArchive
        case memberNotFound(String)
        case unsupportedCompression(UInt16)
        case corruptEntry(String)

        var errorDescription: String? {
            switch self {
            case .notAZipArchive:
                return "That file is not a ZIP archive, so it is not a .docx either."
            case .memberNotFound(let name):
                return "The archive has no \(name); it may not be a Word document."
            case .unsupportedCompression(let method):
                return "The archive uses compression method \(method), which this app "
                    + "cannot read. Re-save the document from Word and try again."
            case .corruptEntry(let name):
                return "The entry \(name) in the archive is damaged."
            }
        }
    }

    /// Read one member out of `data` by exact path.
    static func extract(_ name: String, from data: Data) throws -> Data {
        guard let directoryStart = endOfCentralDirectory(in: data) else {
            throw ZipError.notAZipArchive
        }
        guard let entry = findEntry(named: name, in: data, directoryStart: directoryStart) else {
            throw ZipError.memberNotFound(name)
        }
        return try read(entry: entry, name: name, from: data)
    }

    // MARK: Central directory

    private struct Entry {
        var method: UInt16
        var compressedSize: Int
        var uncompressedSize: Int
        var localHeaderOffset: Int
    }

    /// The offset of the central directory, found by scanning back for the
    /// end-of-central-directory signature (`PK\u{05}\u{06}`).
    private static func endOfCentralDirectory(in data: Data) -> Int? {
        let signature: UInt32 = 0x0605_4B50
        guard data.count >= 22 else { return nil }
        // The EOCD record is 22 bytes plus a comment of at most 65535.
        let lowest = max(0, data.count - 22 - 65_535)
        var index = data.count - 22
        while index >= lowest {
            if read32(data, index) == signature {
                return Int(read32(data, index + 16))
            }
            index -= 1
        }
        return nil
    }

    private static func findEntry(named name: String, in data: Data,
                                  directoryStart: Int) -> Entry? {
        let signature: UInt32 = 0x0201_4B50
        let target = Array(name.utf8)
        var offset = directoryStart
        while offset + 46 <= data.count, read32(data, offset) == signature {
            let method = read16(data, offset + 10)
            let compressedSize = Int(read32(data, offset + 20))
            let uncompressedSize = Int(read32(data, offset + 24))
            let nameLength = Int(read16(data, offset + 28))
            let extraLength = Int(read16(data, offset + 30))
            let commentLength = Int(read16(data, offset + 32))
            let localHeaderOffset = Int(read32(data, offset + 42))
            let nameStart = offset + 46
            guard nameStart + nameLength <= data.count else { return nil }
            let entryName = Array(data[data.startIndex + nameStart
                                       ..< data.startIndex + nameStart + nameLength])
            if entryName == target {
                return Entry(method: method,
                             compressedSize: compressedSize,
                             uncompressedSize: uncompressedSize,
                             localHeaderOffset: localHeaderOffset)
            }
            offset = nameStart + nameLength + extraLength + commentLength
        }
        return nil
    }

    private static func read(entry: Entry, name: String, from data: Data) throws -> Data {
        let header = entry.localHeaderOffset
        guard header + 30 <= data.count, read32(data, header) == 0x0403_4B50 else {
            throw ZipError.corruptEntry(name)
        }
        // Name and extra lengths are re-read from the local header: the central
        // directory's extra field is allowed to differ from the local one.
        let nameLength = Int(read16(data, header + 26))
        let extraLength = Int(read16(data, header + 28))
        let start = header + 30 + nameLength + extraLength
        guard start + entry.compressedSize <= data.count else {
            throw ZipError.corruptEntry(name)
        }
        let payload = data.subdata(in: (data.startIndex + start)
                                      ..< (data.startIndex + start + entry.compressedSize))
        switch entry.method {
        case 0:
            return payload
        case 8:
            guard let inflated = inflate(payload, hint: entry.uncompressedSize) else {
                throw ZipError.corruptEntry(name)
            }
            return inflated
        default:
            throw ZipError.unsupportedCompression(entry.method)
        }
    }

    // MARK: Raw DEFLATE

    private static func inflate(_ data: Data, hint: Int) -> Data? {
        guard !data.isEmpty else { return Data() }
        // The stored uncompressed size is only a hint: a ZIP64 or streamed
        // entry records 0 there, so never trust it as the sole bound.
        let capacity = max(hint > 0 ? hint : data.count * 6, 64 * 1024)
        var output = Data()
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: capacity)
        defer { buffer.deallocate() }

        var stream = compression_stream(dst_ptr: buffer, dst_size: capacity,
                                        src_ptr: buffer, src_size: 0, state: nil)
        guard compression_stream_init(&stream, COMPRESSION_STREAM_DECODE,
                                      COMPRESSION_ZLIB) == COMPRESSION_STATUS_OK
        else { return nil }
        defer { compression_stream_destroy(&stream) }

        let result: Data? = data.withUnsafeBytes { raw -> Data? in
            guard let base = raw.bindMemory(to: UInt8.self).baseAddress else { return nil }
            stream.src_ptr = base
            stream.src_size = raw.count
            while true {
                stream.dst_ptr = buffer
                stream.dst_size = capacity
                let status = compression_stream_process(&stream, Int32(COMPRESSION_STREAM_FINALIZE.rawValue))
                let produced = capacity - stream.dst_size
                if produced > 0 { output.append(buffer, count: produced) }
                switch status {
                case COMPRESSION_STATUS_END:
                    return output
                case COMPRESSION_STATUS_OK:
                    // No progress and no output means the stream is truncated.
                    if produced == 0 && stream.src_size == 0 { return nil }
                default:
                    return nil
                }
            }
        }
        return result
    }

    // MARK: Little-endian reads

    private static func read16(_ data: Data, _ offset: Int) -> UInt16 {
        let i = data.startIndex + offset
        guard i + 1 < data.endIndex else { return 0 }
        return UInt16(data[i]) | (UInt16(data[i + 1]) << 8)
    }

    private static func read32(_ data: Data, _ offset: Int) -> UInt32 {
        let i = data.startIndex + offset
        guard i + 3 < data.endIndex else { return 0 }
        return UInt32(data[i]) | (UInt32(data[i + 1]) << 8)
            | (UInt32(data[i + 2]) << 16) | (UInt32(data[i + 3]) << 24)
    }
}
