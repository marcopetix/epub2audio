"""Generate companion PDF per chapter with figures, code blocks, and math formulas.

All content is rendered in DOM order (as it appears in the chapter),
not grouped by type.
"""

import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup, NavigableString, Tag
from fpdf import FPDF
from PIL import Image

from pipeline.extractor import Chapter, CodeBlock, Figure, MathFormula

logger = logging.getLogger(__name__)

# Page layout constants
PAGE_W = 210  # A4 width mm
MARGIN = 15
CONTENT_W = PAGE_W - 2 * MARGIN
CODE_BG = (245, 245, 245)

# Font names (registered as Unicode TTF)
FONT_SANS = "dsans"
FONT_MONO = "dmono"


# ---------------------------------------------------------------------------
# MathML → LaTeX conversion
# ---------------------------------------------------------------------------

def _mathml_to_latex(el) -> str:
    """Convert a MathML element tree to a LaTeX string."""
    if isinstance(el, NavigableString):
        return str(el).strip()
    if not isinstance(el, Tag):
        return ""

    def child_tags(e):
        return [c for c in e.children if isinstance(c, Tag)]

    def recurse(e):
        return "".join(_mathml_to_latex(c) for c in e.children)

    tag = el.name
    if tag in ("math", "mrow", "mstyle", "semantics"):
        return recurse(el)
    elif tag == "mi":
        t = el.get_text()
        return rf"\mathrm{{{t}}}" if len(t) > 1 else t
    elif tag == "mo":
        t = el.get_text()
        ops = {
            "\u00d7": r"\times", "\u00b7": r"\cdot",
            "\u2264": r"\leq", "\u2265": r"\geq",
            "\u2211": r"\sum", "\u220f": r"\prod",
            "\u2016": r"\parallel", "\u2026": r"\ldots",
            "\u221e": r"\infty", "\u2260": r"\neq",
            "\u2208": r"\in", "\u2192": r"\rightarrow",
        }
        return ops.get(t, t)
    elif tag == "mn":
        return el.get_text()
    elif tag == "mtext":
        return rf"\text{{{el.get_text()}}}"
    elif tag == "mspace":
        return r"\;"
    elif tag == "mfrac":
        kids = child_tags(el)
        if len(kids) >= 2:
            return rf"\frac{{{_mathml_to_latex(kids[0])}}}{{{_mathml_to_latex(kids[1])}}}"
        return recurse(el)
    elif tag == "msup":
        kids = child_tags(el)
        if len(kids) >= 2:
            return rf"{_mathml_to_latex(kids[0])}^{{{_mathml_to_latex(kids[1])}}}"
        return recurse(el)
    elif tag == "msub":
        kids = child_tags(el)
        if len(kids) >= 2:
            return rf"{_mathml_to_latex(kids[0])}_{{{_mathml_to_latex(kids[1])}}}"
        return recurse(el)
    elif tag == "msubsup":
        kids = child_tags(el)
        if len(kids) >= 3:
            return (
                rf"{_mathml_to_latex(kids[0])}"
                rf"_{{{_mathml_to_latex(kids[1])}}}"
                rf"^{{{_mathml_to_latex(kids[2])}}}"
            )
        return recurse(el)
    elif tag == "msqrt":
        return rf"\sqrt{{{recurse(el)}}}"
    elif tag == "mover":
        kids = child_tags(el)
        if len(kids) >= 2:
            return rf"\overline{{{_mathml_to_latex(kids[0])}}}"
        return recurse(el)
    elif tag == "munder":
        kids = child_tags(el)
        if len(kids) >= 2:
            return rf"\underset{{{_mathml_to_latex(kids[1])}}}{{{_mathml_to_latex(kids[0])}}}"
        return recurse(el)
    elif tag == "munderover":
        kids = child_tags(el)
        if len(kids) >= 3:
            return (
                rf"{_mathml_to_latex(kids[0])}"
                rf"_{{{_mathml_to_latex(kids[1])}}}"
                rf"^{{{_mathml_to_latex(kids[2])}}}"
            )
        return recurse(el)
    elif tag == "mtable":
        rows = child_tags(el)
        latex_rows = []
        for row in rows:
            cells = child_tags(row)
            latex_rows.append(" & ".join(_mathml_to_latex(c) for c in cells))
        return r"\begin{matrix}" + r" \\ ".join(latex_rows) + r"\end{matrix}"
    else:
        return recurse(el)


def _render_latex_to_image(latex_str: str) -> bytes | None:
    """Render a LaTeX formula to a PNG image using matplotlib."""
    try:
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.text(
            0.5, 0.5, f"${latex_str}$",
            fontsize=18, ha="center", va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", dpi=150,
            bbox_inches="tight", facecolor="white",
        )
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Failed to render LaTeX: {e}")
        plt.close("all")
        return None


def _mathml_string_to_latex(mathml: str) -> str:
    """Parse MathML string and convert to LaTeX."""
    soup = BeautifulSoup(mathml, "lxml-xml")
    math = soup.find("math")
    if math:
        return _mathml_to_latex(math)
    return ""


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

def _find_dejavu_fonts() -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """Find DejaVu TTF fonts on the system."""
    search_dirs = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local/share/fonts",
    ]
    sans = sans_bold = mono = sans_italic = None
    for base in search_dirs:
        if not base.exists():
            continue
        for f in base.rglob("*.ttf"):
            name = f.name.lower()
            if "dejavusans-bold" in name and "mono" not in name:
                sans_bold = f
            elif "dejavusans-oblique" in name and "mono" not in name:
                sans_italic = f
            elif (
                "dejavusans" in name
                and "mono" not in name
                and "bold" not in name
                and "oblique" not in name
                and "condensed" not in name
                and "extra" not in name
            ):
                sans = f
            elif "dejavusansmono" in name and "bold" not in name and "oblique" not in name:
                mono = f
    return sans, sans_bold, sans_italic, mono


# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------

class CompanionPDF(FPDF):
    """Custom PDF with header/footer for companion documents."""

    def __init__(self, chapter_title: str, book_title: str):
        super().__init__()
        self.chapter_title = chapter_title
        self.book_title = book_title
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(MARGIN, MARGIN, MARGIN)

        # Register Unicode TTF fonts
        sans, sans_bold, sans_italic, mono = _find_dejavu_fonts()
        if sans:
            self.add_font(FONT_SANS, "", str(sans), uni=True)
        if sans_bold:
            self.add_font(FONT_SANS, "B", str(sans_bold), uni=True)
        if sans_italic:
            self.add_font(FONT_SANS, "I", str(sans_italic), uni=True)
        elif sans:
            self.add_font(FONT_SANS, "I", str(sans), uni=True)
        if mono:
            self.add_font(FONT_MONO, "", str(mono), uni=True)

        self._has_unicode = sans is not None
        self._has_mono = mono is not None
        self.add_page()

    def _font_sans(self, style: str = "", size: int = 10):
        if self._has_unicode:
            self.set_font(FONT_SANS, style, size)
        else:
            self.set_font("Helvetica", style, size)

    def _font_mono(self, size: int = 8):
        if self._has_mono:
            self.set_font(FONT_MONO, "", size)
        else:
            self.set_font("Courier", "", size)

    def header(self):
        self._font_sans("I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, f"{self.book_title} - Companion", align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self._font_sans("I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_separator(self):
        """Add a visual separator between elements."""
        self.set_draw_color(200, 200, 200)
        y = self.get_y()
        self.line(MARGIN, y, PAGE_W - MARGIN, y)
        self.ln(5)

    def element_label(self, label: str, context: str = ""):
        """Add element label and context."""
        self._font_sans("B", 12)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, label, new_x="LMARGIN", new_y="NEXT")
        if context:
            self._font_sans("I", 9)
            self.set_text_color(100, 100, 100)
            self.multi_cell(CONTENT_W, 5, f"Section: {context}")
        self.ln(2)

    def add_code_block(self, code: str, language: str = ""):
        """Render a code block with grey background and monospace font."""
        if language:
            self._font_sans("I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, f"Language: {language}", new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

        # Wrap long lines at ~90 chars
        lines = code.split("\n")
        wrapped_lines = []
        for line in lines:
            while len(line) > 90:
                wrapped_lines.append(line[:90])
                line = line[90:]
            wrapped_lines.append(line)

        self._font_mono(8)
        self.set_text_color(30, 30, 30)

        x = self.get_x()
        for wline in wrapped_lines:
            y = self.get_y()
            if y > 270:
                self.add_page()
                y = self.get_y()
            self.set_fill_color(*CODE_BG)
            self.rect(x, y, CONTENT_W, 4, style="F")
            self.cell(CONTENT_W, 4, wline, new_x="LMARGIN", new_y="NEXT")

        self.ln(3)


# ---------------------------------------------------------------------------
# Image and math rendering helpers
# ---------------------------------------------------------------------------

def _add_image_to_pdf(pdf: CompanionPDF, image_data: bytes, max_width: float = CONTENT_W):
    """Add an image to the PDF, scaled to fit."""
    try:
        img = Image.open(io.BytesIO(image_data))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        img_w, img_h = img.size
        aspect = img_h / img_w
        display_w = min(max_width, img_w * 0.264583)  # px to mm at 96 DPI
        display_w = min(display_w, max_width)
        display_h = display_w * aspect

        if pdf.get_y() + display_h > 270:
            pdf.add_page()

        pdf.image(buf, x=MARGIN, w=display_w)
        pdf.ln(3)
    except Exception as e:
        logger.warning(f"Failed to add image to PDF: {e}")
        pdf._font_sans("I", 10)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 8, "[Image could not be rendered]", new_x="LMARGIN", new_y="NEXT")


def _add_math_to_pdf(pdf: CompanionPDF, formula: MathFormula):
    """Render a math formula as a LaTeX image and add to PDF."""
    latex = _mathml_string_to_latex(formula.mathml)

    if latex:
        img_data = _render_latex_to_image(latex)
        if img_data:
            _add_image_to_pdf(pdf, img_data, max_width=CONTENT_W)
            return

    # Fallback: render alttext in monospace
    pdf._font_mono(10)
    pdf.set_text_color(30, 30, 30)
    text = formula.alttext or "[MathML formula - see original book]"
    pdf.set_fill_color(*CODE_BG)
    pdf.multi_cell(CONTENT_W, 6, text, fill=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_companion(
    chapter: Chapter,
    output_dir: Path,
    book_title: str = "Generative AI Design Patterns",
) -> Path:
    """Generate a companion PDF for a chapter.

    All visual content (figures, code, formulas) is rendered in the order
    it appears in the chapter, not grouped by type. Numbering matches
    the audio references.
    """
    safe_title = "".join(
        c if c.isalnum() or c in " -" else "_" for c in chapter.title
    )
    safe_title = "_".join(safe_title.split())
    pdf_filename = f"{chapter.number:02d}_{safe_title}_companion.pdf"
    pdf_path = output_dir / pdf_filename

    pdf = CompanionPDF(chapter.title, book_title)

    # --- Cover page ---
    pdf.ln(30)
    pdf._font_sans("B", 24)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(CONTENT_W, 12, f"Chapter {chapter.number}")
    pdf.ln(2)
    pdf._font_sans("", 18)
    pdf.multi_cell(CONTENT_W, 10, chapter.title)
    pdf.ln(10)
    pdf._font_sans("I", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(CONTENT_W, 7, "Companion PDF - visual content for audio listening")
    pdf.ln(5)
    pdf._font_sans("", 11)
    pdf.multi_cell(
        CONTENT_W, 7,
        f"This companion contains {len(chapter.figures)} figures, "
        f"{len(chapter.code_blocks)} code examples, "
        f"and {len(chapter.math_formulas)} formulas.",
    )
    pdf.ln(5)
    pdf._font_sans("", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(CONTENT_W, 6, book_title)

    # --- Content in DOM order ---
    # Merge all elements and sort by dom_position
    elements: list[Figure | CodeBlock | MathFormula] = []
    elements.extend(chapter.figures)
    elements.extend(chapter.code_blocks)
    elements.extend(chapter.math_formulas)
    elements.sort(key=lambda e: e.dom_position)

    if elements:
        pdf.add_page()

    for i, elem in enumerate(elements):
        # Page break check
        if pdf.get_y() > 240:
            pdf.add_page()

        # Separator between elements (not before first)
        if i > 0:
            pdf.section_separator()

        if isinstance(elem, Figure):
            pdf.element_label(elem.label, elem.context)

            # Add image
            if elem.src in chapter.images:
                _add_image_to_pdf(pdf, chapter.images[elem.src])
            else:
                pdf._font_sans("I", 10)
                pdf.set_text_color(200, 0, 0)
                pdf.cell(0, 8, f"[Image not available: {elem.src}]", new_x="LMARGIN", new_y="NEXT")

            # Caption (using multi_cell to wrap within page width)
            if elem.caption:
                pdf._font_sans("I", 10)
                pdf.set_text_color(60, 60, 60)
                pdf.multi_cell(CONTENT_W, 5, elem.caption)

            # Alt text (wrapped with multi_cell)
            if elem.alt and elem.alt != elem.caption:
                pdf._font_sans("", 8)
                pdf.set_text_color(120, 120, 120)
                pdf.multi_cell(CONTENT_W, 4, f"Alt: {elem.alt}")

            pdf.ln(3)

        elif isinstance(elem, CodeBlock):
            pdf.element_label(f"Code Example {elem.number}", elem.context)
            pdf.add_code_block(elem.code, elem.language)

        elif isinstance(elem, MathFormula):
            pdf.element_label(f"Formula {elem.number}", elem.context)
            _add_math_to_pdf(pdf, elem)
            pdf.ln(3)

    # Save
    pdf.output(str(pdf_path))
    logger.info(
        f"Chapter {chapter.number} companion: {pdf_path.name} - "
        f"{pdf_path.stat().st_size / 1024:.0f} KB"
    )

    return pdf_path
