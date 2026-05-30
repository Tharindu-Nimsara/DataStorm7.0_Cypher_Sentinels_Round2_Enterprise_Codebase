"""Render reports/report.md to a single self-contained HTML file with the
project's brand styling (dark navy + grey, Inter / Calibri fallback,
McKinsey-style minimalism).

Output: reports/report.html — open in any browser and use the print dialog
("Save as PDF") to produce the submission PDF. Page breaks are configured
to mark the 5 logical pages.
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MD_PATH = REPORTS / "report.md"
HTML_PATH = REPORTS / "report.html"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CSS = """
@page {
  size: A4;
  margin: 12mm 12mm 12mm 12mm;
}

:root {
  --navy: #1B2A4E;
  --grey: #6C757D;
  --light: #D6DBDF;
  --bg: #FFFFFF;
  --accent: #A4B7CB;
}

* { box-sizing: border-box; }

body {
  font-family: 'Inter', 'Calibri', 'Segoe UI', sans-serif;
  color: var(--navy);
  background: var(--bg);
  line-height: 1.25;
  font-size: 8.5pt;
  margin: 0;
  padding: 0;
}

h1 {
  color: var(--navy);
  font-size: 16pt;
  margin: 0 0 0.1em 0;
  font-weight: 700;
  letter-spacing: -0.4px;
}

h2 {
  color: var(--navy);
  font-size: 11pt;
  margin: 0.5em 0 0.15em 0;
  padding-bottom: 0.1em;
  border-bottom: 1.2px solid var(--navy);
  font-weight: 700;
  page-break-after: avoid;
}

h3 {
  color: var(--navy);
  font-size: 9.5pt;
  margin: 0.4em 0 0.15em 0;
  font-weight: 700;
  page-break-after: avoid;
}

p { margin: 0.2em 0; color: var(--navy); }
ul, ol { margin: 0.2em 0 0.2em 1em; padding-left: 0.6em; }
li { margin: 0.05em 0; }

strong { color: var(--navy); font-weight: 700; }
em { color: var(--grey); }

hr {
  border: none;
  border-top: 1px solid var(--light);
  margin: 0.4em 0;
}

a { color: var(--navy); text-decoration: none; border-bottom: 1px dotted var(--grey); }

img {
  max-width: 75%;
  max-height: 54mm;
  height: auto;
  display: block;
  margin: 0.15em auto;
  page-break-inside: avoid;
  object-fit: contain;
}

table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.3em 0;
  font-size: 7.5pt;
  page-break-inside: avoid;
}

th, td {
  padding: 2.5px 6px;
  text-align: left;
  border-bottom: 0.5px solid var(--light);
  color: var(--navy);
  line-height: 1.2;
}

th {
  background: var(--navy);
  color: white !important;
  font-weight: 600;
  letter-spacing: 0.2px;
}

th * { color: white !important; }

td:last-child, th:last-child { text-align: right; }

table tr:nth-child(even) { background: #F4F6F8; }

code, pre {
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 7.5pt;
  background: #F4F6F8;
  color: var(--navy);
}

pre {
  padding: 4px 6px;
  border-left: 2px solid var(--navy);
  white-space: pre-wrap;
  page-break-inside: avoid;
  margin: 0.3em 0;
  line-height: 1.2;
}

code { padding: 1px 3px; border-radius: 2px; }

blockquote {
  margin: 0.3em 0;
  padding-left: 8px;
  border-left: 2px solid var(--accent);
  color: var(--grey);
  font-style: italic;
}

@media print {
  body { background: white; }
}
"""


def embed_images(html: str, base_dir: Path) -> str:
    """Inline all relative <img src> as base64 so the HTML is fully portable."""
    def replace(m: re.Match) -> str:
        src = m.group(1)
        if src.startswith("data:") or src.startswith("http"):
            return m.group(0)
        path = (base_dir / src).resolve()
        if not path.exists():
            print(f"  WARN: image not found: {path}")
            return m.group(0)
        suffix = path.suffix.lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "svg": "image/svg+xml"}.get(suffix, "image/png")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img src="data:{mime};base64,{data}"'

    return re.sub(r'<img\s+[^>]*src="([^"]+)"', replace, html)


def main() -> None:
    print(f"Reading {MD_PATH}...")
    md_text = MD_PATH.read_text(encoding="utf-8")

    body_html = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )

    body_html = embed_images(body_html, REPORTS)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Latent Outlet Potential — Sri Lanka</title>
<style>{CSS}</style>
</head>
<body>
{body_html}
</body>
</html>
"""

    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {HTML_PATH}")
    print(f"  size: {HTML_PATH.stat().st_size / 1024:.0f} KB (images inlined)")
    print()
    print("To produce report.pdf:")
    print("  1. Open report.html in a browser (Chrome/Edge recommended)")
    print("  2. Print → Destination: 'Save as PDF' → Layout: Portrait → Save")
    print("  3. Verify it lands at ≤ 5 pages")


if __name__ == "__main__":
    main()
