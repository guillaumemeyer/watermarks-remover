import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from container_meta import clean_svg


class TestSvgSecurityHardening(unittest.TestCase):
    def test_clean_svg_strips_doctype_entities(self):
        svg_with_entity = b"""<?xml version="1.0"?>
<!DOCTYPE svg [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;">
]>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <circle cx="50" cy="50" r="40" />
</svg>"""
        cleaned_bytes, actions = clean_svg(svg_with_entity)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertNotIn("<!DOCTYPE", cleaned_str)
        self.assertNotIn("<!ENTITY", cleaned_str)
        self.assertIn("<svg", cleaned_str)
        self.assertTrue(any("DOCTYPE" in a for a in actions))

    def test_clean_svg_consumes_external_identifier_with_quoted_gt(self):
        svg = b"""<?xml version="1.0"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" />
"""
        cleaned_bytes, actions = clean_svg(svg)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertNotIn("<!DOCTYPE", cleaned_str)
        self.assertIn("<svg", cleaned_str)
        self.assertTrue(any("DOCTYPE" in a for a in actions))

    def test_clean_svg_consumes_multiline_external_identifier(self):
        # Line breaks are legal inside quoted system/public literals.
        svg = b"""<?xml version="1.0"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG
1.1//EN" "http://www.w3.org/2000/svg">
<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" />
"""
        cleaned_bytes, actions = clean_svg(svg)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertNotIn("<!DOCTYPE", cleaned_str)
        self.assertIn("<svg", cleaned_str)
        self.assertTrue(any("DOCTYPE" in a for a in actions))

    def test_clean_svg_handles_gt_close_inside_dtd_comment(self):
        # ']>' inside a DTD comment must not close the internal subset early.
        svg = b"""<?xml version="1.0"?>
<!DOCTYPE svg [ <!-- ]> --> <!ENTITY x "y"> ]>
<svg xmlns="http://www.w3.org/2000/svg" />
"""
        cleaned_bytes, actions = clean_svg(svg)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertNotIn("<!DOCTYPE", cleaned_str)
        self.assertNotIn("<!ENTITY", cleaned_str)
        self.assertNotIn("]>", cleaned_str)
        self.assertIn("<svg", cleaned_str)
        self.assertTrue(any("DOCTYPE" in a for a in actions))

        cleaned_bytes, actions = clean_svg(svg)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertNotIn("<!DOCTYPE", cleaned_str)
        self.assertIn("<svg", cleaned_str)
        self.assertTrue(any("DOCTYPE" in a for a in actions))

    def test_clean_svg_consumes_nested_internal_subset(self):
        # A '>' inside a quoted entity value and a nested internal subset must
        # not terminate the DOCTYPE early.
        svg = b"""<?xml version="1.0"?>
<!DOCTYPE svg [ <!ENTITY a "x>y"> <!ENTITY b "b>c"> ]>
<svg xmlns="http://www.w3.org/2000/svg"><text>&a;</text></svg>
"""
        cleaned_bytes, actions = clean_svg(svg)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertNotIn("<!DOCTYPE", cleaned_str)
        self.assertNotIn("<!ENTITY", cleaned_str)
        self.assertIn("<svg", cleaned_str)
        self.assertTrue(any("DOCTYPE" in a for a in actions))

    def test_clean_svg_preserves_decl_like_text_in_cdata(self):
        svg = b"""<svg xmlns="http://www.w3.org/2000/svg">
<![CDATA[<!ENTITY legit "keep me"> <!DOCTYPE fake>]]>
<circle cx="1" cy="2" r="3" />
</svg>"""
        cleaned_bytes, actions = clean_svg(svg)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertIn("<!ENTITY legit", cleaned_str)
        self.assertIn("<!DOCTYPE fake>", cleaned_str)
        self.assertTrue(all("DOCTYPE" not in a for a in actions))

    def test_clean_svg_preserves_decl_like_text_in_comment(self):
        svg = b"""<svg xmlns="http://www.w3.org/2000/svg">
<!-- <!ENTITY legit "keep me"> -->
<circle cx="1" cy="2" r="3" />
</svg>"""
        cleaned_bytes, actions = clean_svg(svg)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertIn("<!ENTITY legit", cleaned_str)
        self.assertTrue(all("DOCTYPE" not in a for a in actions))

    def test_clean_svg_preserves_decl_like_text_in_quoted_attribute(self):
        svg = b"""<svg xmlns="http://www.w3.org/2000/svg" title="<!ENTITY legit &quot;keep me&quot;>">
<circle cx="1" cy="2" r="3" />
</svg>"""
        cleaned_bytes, actions = clean_svg(svg)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertIn("<svg", cleaned_str)
        self.assertIn("<!ENTITY legit", cleaned_str)
        self.assertTrue(all("DOCTYPE" not in a for a in actions))

    def test_clean_svg_leaves_unterminated_declaration_intact(self):
        # A declaration with no closing '>' is left as-is rather than deleted.
        svg = b"""<svg xmlns="http://www.w3.org/2000/svg"><text>hi</text></svg>
<!ENTITY unterminated "no close"""
        cleaned_bytes, actions = clean_svg(svg)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertIn("<!ENTITY unterminated", cleaned_str)
        self.assertIn("<svg", cleaned_str)
        self.assertTrue(all("DOCTYPE" not in a for a in actions))

    def test_clean_svg_removes_single_quoted_root_attrs(self):
        svg = b"<svg xmlns='http://www.w3.org/2000/svg' generator='tool' inkscape:version='1.0' sodipodi:docname='x.svg' width='10'/>"
        cleaned_bytes, actions = clean_svg(svg)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertNotIn("generator", cleaned_str)
        self.assertNotIn("inkscape:version", cleaned_str)
        self.assertNotIn("sodipodi:docname", cleaned_str)
        self.assertTrue(any("generator" in a for a in actions))

    def test_clean_svg_preserves_generator_like_text_content(self):
        # Commoned generator-like text in visible content must be preserved.
        svg = b'<svg xmlns="http://www.w3.org/2000/svg"><text>run generator="tool" now</text></svg>'
        cleaned_bytes, actions = clean_svg(svg)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertIn('generator="tool"', cleaned_str)
        self.assertTrue(all("generator" not in a for a in actions))

    def test_clean_svg_removes_generator_attrs_alongside_declarations(self):
        svg = b"""<?xml version="1.0"?>
<!DOCTYPE svg [
  <!ENTITY lol "lol">
]>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" generator="tool" inkscape:version="1.0" sodipodi:docname="x.svg" />"""
        cleaned_bytes, actions = clean_svg(svg)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertNotIn("<!DOCTYPE", cleaned_str)
        self.assertNotIn("generator=", cleaned_str)
        self.assertNotIn("inkscape:version", cleaned_str)
        self.assertNotIn("sodipodi:docname", cleaned_str)
        self.assertTrue(any("DOCTYPE" in a for a in actions))
        self.assertTrue(any("generator" in a for a in actions))

    def test_clean_svg_skips_processing_instruction_when_locating_root(self):
        # A processing instruction containing '<svg' must not be mistaken for
        # the root start tag, so the real root's generator attr is still removed.
        svg = b'<?xml-stylesheet type="text/css" href="<svg foo=1>"?>\n<svg xmlns="http://www.w3.org/2000/svg" generator="tool" width="10"/>'
        cleaned_bytes, actions = clean_svg(svg)
        cleaned_str = cleaned_bytes.decode("utf-8")
        self.assertIn("<?xml-stylesheet", cleaned_str)
        self.assertNotIn('generator="tool"', cleaned_str)
        self.assertTrue(any("generator" in a for a in actions))


if __name__ == "__main__":
    unittest.main()
