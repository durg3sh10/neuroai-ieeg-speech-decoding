"""report.md (+ figures/) → report.html → report.pdf (markdown → HTML, then LibreOffice Writer headless)."""
import re, subprocess
from pathlib import Path

import markdown
from PIL import Image

CSS = """
body { font-family: 'Liberation Serif', serif; font-size: 11pt; line-height: 1.3; }
h1 { font-size: 17pt; } h2 { font-size: 13.5pt; margin-top: 14pt; } h3 { font-size: 12pt; }
table { border-collapse: collapse; font-size: 9.5pt; } th, td { border: 1px solid #999; padding: 2px 6px; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; }
p.caption { font-size: 9.5pt; color: #333; }
"""
WIDTH = 620                                                        # px at 96 dpi ≈ 16.4 cm, fits A4 text width


def sized_img(m):
    src = m.group(1)
    w, h = Image.open(src).size                                     # LibreOffice needs explicit width AND height
    return f'<img src="{src}" width="{WIDTH}" height="{round(WIDTH * h / w)}"'


md = Path('report.md').read_text(encoding='utf-8')
html = markdown.markdown(md, extensions=['tables'])
html = re.sub(r'<img alt="" src="([^"]+)"', sized_img, html)
html = re.sub(r'<p><em>(Figure.*?)</em></p>', r'<p class="caption"><em>\1</em></p>', html)
Path('report.html').write_text(f'<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{html}</body></html>',
                               encoding='utf-8')
subprocess.run(['soffice', '--headless', '--infilter=HTML (StarWriter)', '--convert-to', 'pdf:writer_pdf_Export',
                'report.html'], check=True)
Path('report.html').unlink()
print('wrote report.pdf')
