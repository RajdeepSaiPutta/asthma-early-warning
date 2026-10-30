"""save the risk report as a pdf.

fpdf2 keeps the dependency footprint small and bundles cleanly with
pyinstaller on all three platforms.
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos


def save_pdf(text: str, path: str | Path) -> Path:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(18, 18, 18)
    pdf.add_page()
    pdf.set_font("Courier", size=11)
    for line in text.splitlines():
        safe = line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 5.5, safe, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    target = Path(path)
    pdf.output(str(target))
    return target
