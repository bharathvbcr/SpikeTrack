// Renders one 1280x720 PNG per segment of the demo video from a JSON spec (make.sh writes them).
// Spec fields, all optional except "out": title, subtitle, body (lines), mono (lines, monospaced),
// image (a picture fitted into the area between the header and the caption), labels (two strings
// drawn over the left and right halves of a side-by-side clip region), caption (bottom bar).
// usage: swiftc -O cards.swift -o cards && ./cards spec1.json [spec2.json ...]
import AppKit
import Foundation

let W: CGFloat = 1280
let H: CGFloat = 720
let bg = NSColor(calibratedRed: 0.07, green: 0.08, blue: 0.10, alpha: 1)
let fg = NSColor(calibratedWhite: 0.95, alpha: 1)
let dim = NSColor(calibratedWhite: 0.70, alpha: 1)
let accent = NSColor(calibratedRed: 0.40, green: 0.78, blue: 1.0, alpha: 1)

struct Spec: Decodable {
    let out: String
    let title: String?
    let subtitle: String?
    let body: [String]?
    let mono: [String]?
    let image: String?
    let labels: [String]?
    let caption: String?
}

func draw(_ s: String, font: NSFont, color: NSColor, in rect: NSRect, align: NSTextAlignment = .left) {
    let p = NSMutableParagraphStyle()
    p.alignment = align
    p.lineBreakMode = .byWordWrapping
    let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: color, .paragraphStyle: p]
    (s as NSString).draw(with: rect, options: [.usesLineFragmentOrigin], attributes: attrs)
}

func height(_ s: String, font: NSFont, width: CGFloat) -> CGFloat {
    let r = (s as NSString).boundingRect(
        with: NSSize(width: width, height: 10_000), options: [.usesLineFragmentOrigin],
        attributes: [.font: font])
    return ceil(r.height)
}

func render(_ spec: Spec) throws {
    guard let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: Int(W), pixelsHigh: Int(H), bitsPerSample: 8,
        samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
        bytesPerRow: 0, bitsPerPixel: 0)
    else { throw NSError(domain: "cards", code: 1) }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    // AppKit's origin is bottom-left; `top(y, h)` turns a top-down layout into its rects.
    func top(_ y: CGFloat, _ h: CGFloat, x: CGFloat = 64, w: CGFloat = W - 128) -> NSRect {
        NSRect(x: x, y: H - y - h, width: w, height: h)
    }
    bg.setFill()
    NSRect(x: 0, y: 0, width: W, height: H).fill()

    var y: CGFloat = 48
    if let t = spec.title {
        let f = NSFont.systemFont(ofSize: spec.image == nil && spec.labels == nil ? 56 : 34, weight: .bold)
        let h = height(t, font: f, width: W - 128)
        draw(t, font: f, color: fg, in: top(y, h))
        y += h + 28
    }
    if let t = spec.subtitle {
        let f = NSFont.systemFont(ofSize: 26, weight: .regular)
        let h = height(t, font: f, width: W - 128)
        draw(t, font: f, color: accent, in: top(y, h))
        y += h + 24
    }
    if let lines = spec.body {
        let f = NSFont.systemFont(ofSize: 28, weight: .regular)
        for l in lines {
            let s = "•  " + l
            let h = height(s, font: f, width: W - 160)
            draw(s, font: f, color: fg, in: top(y, h, x: 80, w: W - 160))
            y += h + 14
        }
    }
    if let lines = spec.mono {
        let f = NSFont.monospacedSystemFont(ofSize: 16, weight: .regular)
        let block = lines.joined(separator: "\n")
        let h = height(block, font: f, width: W - 160)
        NSColor(calibratedWhite: 0.0, alpha: 0.55).setFill()
        top(y - 12, h + 24, x: 64, w: W - 128).fill()
        draw(block, font: f, color: NSColor(calibratedRed: 0.75, green: 0.95, blue: 0.75, alpha: 1),
             in: top(y, h, x: 80, w: W - 160))
        y += h + 24
    }
    let captionH: CGFloat = spec.caption == nil ? 0 : 92
    if let path = spec.image, let img = NSImage(contentsOfFile: path) {
        let area = top(y, H - y - captionH - 16, x: 0, w: W)
        let scale = min(area.width / img.size.width, area.height / img.size.height)
        let w = img.size.width * scale, h = img.size.height * scale
        img.draw(in: NSRect(x: area.midX - w / 2, y: area.midY - h / 2, width: w, height: h))
    }
    if let l = spec.labels, l.count == 2 {
        // The side-by-side clip (1280x480) is overlaid below this band by make.sh, at y = 96.
        let f = NSFont.systemFont(ofSize: 26, weight: .semibold)
        draw(l[0], font: f, color: fg, in: top(40, 40, x: 0, w: W / 2), align: .center)
        draw(l[1], font: f, color: fg, in: top(40, 40, x: W / 2, w: W / 2), align: .center)
    }
    if let c = spec.caption {
        NSColor(calibratedWhite: 0.0, alpha: 0.75).setFill()
        NSRect(x: 0, y: 0, width: W, height: captionH).fill()
        let f = NSFont.systemFont(ofSize: 24, weight: .medium)
        draw(c, font: f, color: fg, in: NSRect(x: 48, y: 16, width: W - 96, height: captionH - 28))
    }
    NSGraphicsContext.restoreGraphicsState()
    guard let png = rep.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "cards", code: 2)
    }
    try png.write(to: URL(fileURLWithPath: spec.out))
}

var failed = false
for path in CommandLine.arguments.dropFirst() {
    do {
        let spec = try JSONDecoder().decode(Spec.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        try render(spec)
        print("wrote \(spec.out)")
    } catch {
        FileHandle.standardError.write("\(path): \(error)\n".data(using: .utf8)!)
        failed = true
    }
}
exit(failed ? 1 : 0)
