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


if __name__ == '__main__':
    unittest.main()
