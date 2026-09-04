#!/usr/bin/env swift
//
// make_icon.swift — turn a square source image into Watermarker.icns.
//
//   swift mac/Scripts/make_icon.swift <source.png> <output.icns>
//
// The source is auto-cropped first: artwork exported from a design tool (or a
// chat) usually arrives with a uniform border of whitespace around it, and a
// macOS icon has to run edge to edge or it renders visibly smaller than every
// other icon in the Dock. Any uniform border — white, black, or transparent —
// is measured from the corners and trimmed, then the result is padded back to
// square and rendered into every size an .iconset needs. `iconutil` does the
// final packing.
//
// Uses only ImageIO/CoreGraphics, so the Xcode command line tools are the
// whole dependency list.

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Tolerance, per channel out of 255, for calling two pixels the same colour.
let borderTolerance = 10

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("make_icon: \(message)\n".utf8))
    exit(1)
}

func loadImage(_ path: String) -> CGImage {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else { fail("cannot read image at \(path)") }
    return image
}

/// Redraw into a known 8-bit premultiplied RGBA buffer so pixel reads are sane
/// regardless of what colour space or bit depth the source arrived in.
func rgbaBuffer(_ image: CGImage) -> (pixels: [UInt8], width: Int, height: Int) {
    let w = image.width, h = image.height
    var pixels = [UInt8](repeating: 0, count: w * h * 4)
    guard let ctx = pixels.withUnsafeMutableBytes({ raw in
        CGContext(data: raw.baseAddress,
                  width: w, height: h,
                  bitsPerComponent: 8, bytesPerRow: w * 4,
                  space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    }) else { fail("cannot allocate a \(w)x\(h) bitmap") }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    return (pixels, w, h)
}

/// The bounding box of everything that is not the uniform border colour.
/// Returns nil when the corners disagree, i.e. there is no border to trim.
func contentBox(_ buf: (pixels: [UInt8], width: Int, height: Int)) -> CGRect? {
    let (p, w, h) = buf
    func pixel(_ x: Int, _ y: Int) -> (Int, Int, Int, Int) {
        let i = (y * w + x) * 4
        return (Int(p[i]), Int(p[i + 1]), Int(p[i + 2]), Int(p[i + 3]))
    }
    func matches(_ a: (Int, Int, Int, Int), _ b: (Int, Int, Int, Int)) -> Bool {
        // Two fully transparent pixels match whatever their colour components
        // happen to be; premultiplied buffers leave those undefined.
        if a.3 <= 2 && b.3 <= 2 { return true }
        return abs(a.0 - b.0) <= borderTolerance && abs(a.1 - b.1) <= borderTolerance
            && abs(a.2 - b.2) <= borderTolerance && abs(a.3 - b.3) <= borderTolerance
    }

    let corners = [pixel(0, 0), pixel(w - 1, 0), pixel(0, h - 1), pixel(w - 1, h - 1)]
    guard corners.dropFirst().allSatisfy({ matches($0, corners[0]) }) else { return nil }
    let background = corners[0]

    var minX = w, minY = h, maxX = -1, maxY = -1
    for y in 0..<h {
        for x in 0..<w where !matches(pixel(x, y), background) {
            if x < minX { minX = x }
            if x > maxX { maxX = x }
            if y < minY { minY = y }
            if y > maxY { maxY = y }
        }
    }
    guard maxX >= minX, maxY >= minY else { return nil }
    if minX == 0 && minY == 0 && maxX == w - 1 && maxY == h - 1 { return nil }
    return CGRect(x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1)
}

/// Crop to `box` (in top-left pixel coordinates) and pad back out to a square.
func croppedSquare(_ image: CGImage, _ box: CGRect) -> CGImage {
    // CGImage.cropping uses the same top-left origin as the buffer above.
    guard let cropped = image.cropping(to: box) else { fail("crop failed") }
    let side = Int(max(box.width, box.height).rounded())
    if cropped.width == side && cropped.height == side { return cropped }
    guard let ctx = CGContext(data: nil, width: side, height: side,
                              bitsPerComponent: 8, bytesPerRow: side * 4,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { fail("cannot allocate the square canvas") }
    ctx.interpolationQuality = .high
    ctx.draw(cropped, in: CGRect(x: (side - cropped.width) / 2,
                                 y: (side - cropped.height) / 2,
                                 width: cropped.width, height: cropped.height))
    guard let out = ctx.makeImage() else { fail("cannot render the square canvas") }
    return out
}

func writePNG(_ image: CGImage, side: Int, to url: URL) {
    guard let ctx = CGContext(data: nil, width: side, height: side,
                              bitsPerComponent: 8, bytesPerRow: side * 4,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { fail("cannot allocate a \(side)px canvas") }
    ctx.interpolationQuality = .high
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
    guard let scaled = ctx.makeImage(),
          let dest = CGImageDestinationCreateWithURL(
              url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { fail("cannot write \(url.lastPathComponent)") }
    CGImageDestinationAddImage(dest, scaled, nil)
    guard CGImageDestinationFinalize(dest) else { fail("cannot finalize \(url.lastPathComponent)") }
}

// MARK: - main

let args = CommandLine.arguments
guard args.count == 3 else {
    fail("usage: make_icon.swift <source-image> <output.icns>")
}
let sourcePath = args[1]
let icnsPath = args[2]

let source = loadImage(sourcePath)
let box = contentBox(rgbaBuffer(source))
let artwork = box.map { croppedSquare(source, $0) } ?? source
if let box {
    print("make_icon: cropped border -> \(Int(box.width))x\(Int(box.height)) "
          + "at (\(Int(box.minX)),\(Int(box.minY))) of \(source.width)x\(source.height)")
} else {
    print("make_icon: no uniform border found; using the source as-is "
          + "(\(source.width)x\(source.height))")
}

let work = URL(fileURLWithPath: NSTemporaryDirectory())
    .appendingPathComponent("Watermarker-\(UUID().uuidString)")
let iconset = work.appendingPathComponent("AppIcon.iconset")
try? FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: work) }

// (base point size, scale) pairs iconutil expects in an .iconset.
for (points, scale) in [(16, 1), (16, 2), (32, 1), (32, 2), (128, 1), (128, 2),
                        (256, 1), (256, 2), (512, 1), (512, 2)] {
    let suffix = scale == 1 ? "" : "@2x"
    let name = "icon_\(points)x\(points)\(suffix).png"
    writePNG(artwork, side: points * scale, to: iconset.appendingPathComponent(name))
}

let out = URL(fileURLWithPath: icnsPath)
try? FileManager.default.createDirectory(
    at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
let iconutil = Process()
iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
iconutil.arguments = ["--convert", "icns", iconset.path, "--output", out.path]
try? iconutil.run()
iconutil.waitUntilExit()
guard iconutil.terminationStatus == 0 else { fail("iconutil failed") }
print("make_icon: wrote \(out.path)")
