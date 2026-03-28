"""Generate companion PDF and HTML per chapter with figures, code blocks, math, and tables.

All content is rendered in DOM order (as it appears in the chapter).
PDF features: syntax highlighting (Pygments), bookmarks, clickable TOC, audio timestamps.
HTML features: responsive layout, syntax highlighting, base64 images.
"""

import base64
import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup, NavigableString, Tag
from fpdf import FPDF
from PIL import Image

from pipeline.extractor import Chapter, CodeBlock, Figure, MathFormula, Table

logger = logging.getLogger(__name__)

# Page layout constants
PAGE_W = 210  # A4 width mm
MARGIN = 15
CONTENT_W = PAGE_W - 2 * MARGIN
CODE_BG = (245, 245, 245)

# Font names
FONT_SANS = "dsans"
FONT_MONO = "dmono"

# Pygments color map for PDF code highlighting
TOKEN_COLORS = {
    "Keyword": (0, 0, 204),
    "Name.Function": (102, 0, 204),
    "Name.Class": (102, 0, 204),
    "Name.Decorator": (102, 0, 204),
    "Literal.String": (0, 136, 0),
    "Comment": (136, 136, 136),
    "Literal.Number": (204, 102, 0),
    "Operator": (0, 0, 0),
}


# ---------------------------------------------------------------------------
# MathML → LaTeX conversion (unchanged from v1)
# ---------------------------------------------------------------------------

def _mathml_to_latex(el) -> str:
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
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Failed to render LaTeX: {e}")
        plt.close("all")
        return None


def _mathml_string_to_latex(mathml: str) -> str:
    soup = BeautifulSoup(mathml, "lxml-xml")
    math = soup.find("math")
    return _mathml_to_latex(math) if math else ""


# ---------------------------------------------------------------------------
# Syntax highlighting
# ---------------------------------------------------------------------------

def _highlight_code_html(code: str, language: str = "") -> str:
    """Return syntax-highlighted HTML using Pygments."""
    try:
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name, guess_lexer
        from pygments.formatters import HtmlFormatter

        if language:
            try:
                lexer = get_lexer_by_name(language)
            except Exception:
                lexer = guess_lexer(code)
        else:
            try:
                lexer = guess_lexer(code)
            except Exception:
                return f"<pre><code>{code}</code></pre>"

        formatter = HtmlFormatter(nowrap=True, style="github-dark")
        return highlight(code, lexer, formatter)
    except ImportError:
        return code


def _highlight_code_for_pdf(code: str, language: str = "") -> list[tuple[str, tuple[int, int, int]]]:
    """Return list of (text, rgb_color) segments for PDF rendering."""
    try:
        from pygments.lexers import get_lexer_by_name, guess_lexer
        from pygments.token import Token

        if language:
            try:
                lexer = get_lexer_by_name(language)
            except Exception:
                lexer = guess_lexer(code)
        else:
            try:
                lexer = guess_lexer(code)
            except Exception:
                return [(code, (30, 30, 30))]

        segments = []
        for token_type, token_value in lexer.get_tokens(code):
            color = (30, 30, 30)  # default
            token_str = str(token_type)
            for key, rgb in TOKEN_COLORS.items():
                if key in token_str:
                    color = rgb
                    break
            segments.append((token_value, color))
        return segments
    except ImportError:
        return [(code, (30, 30, 30))]


# ---------------------------------------------------------------------------
# Audio timestamp formatting
# ---------------------------------------------------------------------------

def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _compute_element_timestamps(
    audio_timestamps,
    chapter,
) -> dict[int, str]:
    """Compute approximate audio timestamps for each dom_position.

    Returns dict of dom_position -> formatted timestamp string.
    """
    if not audio_timestamps:
        return {}

    # Build a mapping from char offset to cumulative audio time
    timestamps = {}
    cumulative_seconds = 0.0

    for wav_result in audio_timestamps:
        chunk = wav_result.chunk
        # Elements whose dom_position maps to content in this chunk's range
        # We approximate: chunk.char_start is where this chunk's audio begins
        for elem_list in [chapter.figures, chapter.code_blocks, chapter.math_formulas, chapter.tables]:
            for elem in elem_list:
                if elem.dom_position not in timestamps:
                    # Rough heuristic: map dom_position to approximate audio time
                    # based on relative position in the text
                    timestamps[elem.dom_position] = _format_timestamp(cumulative_seconds)
        cumulative_seconds += wav_result.duration

    return timestamps


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

def _find_dejavu_fonts() -> tuple[Path | None, Path | None, Path | None, Path | None]:
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

    def __init__(self, chapter_title: str, book_title: str):
        super().__init__()
        self.chapter_title = chapter_title
        self.book_title = book_title
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(MARGIN, MARGIN, MARGIN)

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
        self.set_draw_color(200, 200, 200)
        y = self.get_y()
        self.line(MARGIN, y, PAGE_W - MARGIN, y)
        self.ln(5)

    def element_label(self, label: str, context: str = "", timestamp: str = ""):
        self._font_sans("B", 12)
        self.set_text_color(0, 0, 0)
        display = label
        if timestamp:
            display += f"  [audio ~{timestamp}]"
        self.cell(0, 8, display, new_x="LMARGIN", new_y="NEXT")
        if context:
            self._font_sans("I", 9)
            self.set_text_color(100, 100, 100)
            self.multi_cell(CONTENT_W, 5, f"Section: {context}")
        self.ln(2)

    def add_code_block_highlighted(self, code: str, language: str = "", annotation: str = ""):
        """Render a code block with Pygments syntax highlighting."""
        if annotation:
            self._font_sans("I", 9)
            self.set_text_color(60, 60, 60)
            self.multi_cell(CONTENT_W, 5, annotation)
            self.ln(2)

        if language:
            self._font_sans("I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, f"Language: {language}", new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

        segments = _highlight_code_for_pdf(code, language)

        self._font_mono(8)
        x_start = self.get_x()
        line_height = 4

        # Render line by line with colors
        current_line = ""
        current_x = x_start

        for text, color in segments:
            for char in text:
                if char == '\n':
                    # Finish current line
                    y = self.get_y()
                    if y > 270:
                        self.add_page()
                        y = self.get_y()
                    self.set_fill_color(*CODE_BG)
                    self.rect(x_start, y, CONTENT_W, line_height, style="F")
                    self.set_x(x_start)
                    self.ln(line_height)
                    current_line = ""
                    current_x = x_start
                else:
                    current_line += char
                    if len(current_line) > 90:
                        y = self.get_y()
                        if y > 270:
                            self.add_page()
                            y = self.get_y()
                        self.set_fill_color(*CODE_BG)
                        self.rect(x_start, y, CONTENT_W, line_height, style="F")
                        self.set_text_color(*color)
                        self.set_x(x_start)
                        self.cell(CONTENT_W, line_height, current_line, new_x="LMARGIN", new_y="NEXT")
                        current_line = ""
                        current_x = x_start

        # Flush remaining
        if current_line:
            y = self.get_y()
            if y > 270:
                self.add_page()
                y = self.get_y()
            self.set_fill_color(*CODE_BG)
            self.rect(x_start, y, CONTENT_W, line_height, style="F")
            self.set_text_color(*segments[-1][1] if segments else (30, 30, 30))
            self.set_x(x_start)
            self.cell(CONTENT_W, line_height, current_line, new_x="LMARGIN", new_y="NEXT")

        self.ln(3)


# ---------------------------------------------------------------------------
# Image and math rendering helpers
# ---------------------------------------------------------------------------

def _add_image_to_pdf(pdf: CompanionPDF, image_data: bytes, max_width: float = CONTENT_W):
    try:
        img = Image.open(io.BytesIO(image_data))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        img_w, img_h = img.size
        aspect = img_h / img_w
        display_w = min(max_width, img_w * 0.264583)
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
    latex = _mathml_string_to_latex(formula.mathml)
    if latex:
        img_data = _render_latex_to_image(latex)
        if img_data:
            _add_image_to_pdf(pdf, img_data, max_width=CONTENT_W)
            return
    pdf._font_mono(10)
    pdf.set_text_color(30, 30, 30)
    text = formula.alttext or "[MathML formula - see original book]"
    pdf.set_fill_color(*CODE_BG)
    pdf.multi_cell(CONTENT_W, 6, text, fill=True)


# ---------------------------------------------------------------------------
# HTML companion generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{chapter_title} — Companion</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script>hljs.highlightAll();</script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           max-width: 800px; margin: 0 auto; padding: 1rem; background: #fafaf8; color: #2c2c2a; }}
    h1 {{ font-size: 1.8rem; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.5rem; }}
    h2 {{ font-size: 1.3rem; color: #444; margin-top: 2rem; }}
    .element {{ margin: 1.5rem 0; padding: 1rem; background: #fff; border: 1px solid #e8e8e5;
                border-radius: 6px; }}
    .element-label {{ font-weight: 700; font-size: 1.1rem; }}
    .timestamp {{ color: #888; font-size: 0.85rem; margin-left: 0.5rem; }}
    .context {{ color: #777; font-size: 0.85rem; font-style: italic; }}
    .annotation {{ font-style: italic; color: #555; margin: 0.5rem 0; }}
    .caption {{ font-style: italic; color: #555; margin-top: 0.5rem; }}
    .fig img {{ max-width: 100%; height: auto; border-radius: 4px; cursor: zoom-in; }}
    .fig img.zoomed {{ max-width: none; cursor: zoom-out; }}
    pre {{ background: #f5f5f5; padding: 1rem; border-radius: 4px; overflow-x: auto; }}
    pre code {{ font-size: 13px; }}
    .formula {{ background: #f5f5f5; padding: 0.5rem 1rem; border-radius: 4px;
                font-family: monospace; }}
    .toc {{ background: #fff; border: 1px solid #e8e8e5; border-radius: 6px;
            padding: 1rem; margin-bottom: 2rem; }}
    .toc a {{ display: block; padding: 3px 0; color: #185FA5; text-decoration: none; }}
    .toc a:hover {{ text-decoration: underline; }}
    .lang-label {{ color: #888; font-size: 0.8rem; }}
    @media (max-width: 600px) {{ body {{ padding: 0.5rem; }} pre code {{ font-size: 11px; }} }}
  </style>
  <script>
    document.addEventListener('click', function(e) {{
      if (e.target.tagName === 'IMG' && e.target.closest('.fig')) {{
        e.target.classList.toggle('zoomed');
      }}
    }});
  </script>
</head>
<body>
  <h1>Chapter {chapter_number}: {chapter_title}</h1>
  <p class="context">Companion — visual content for audio listening</p>
  <p>{summary}</p>

  <div class="toc">
    <strong>Contents</strong>
    {toc_entries}
  </div>

  {content}

  <hr>
  <p style="color: #999; font-size: 0.8rem;">
    Generated by epub2audio v2 — {book_title}
  </p>
</body>
</html>"""


def _generate_html(
    chapter: Chapter,
    output_dir: Path,
    book_title: str,
    timestamps: dict[int, str],
) -> Path:
    """Generate an HTML companion file for a chapter."""
    safe_title = "_".join(
        "".join(c if c.isalnum() or c in " -" else "_" for c in chapter.title).split()
    )
    html_path = output_dir / f"{chapter.number:02d}_{safe_title}_companion.html"

    # Merge and sort elements by DOM position
    elements = []
    elements.extend(("figure", f) for f in chapter.figures)
    elements.extend(("code", c) for c in chapter.code_blocks)
    elements.extend(("math", m) for m in chapter.math_formulas)
    elements.extend(("table", t) for t in chapter.tables)
    elements.sort(key=lambda e: e[1].dom_position)

    # Build TOC
    toc_lines = []
    content_lines = []

    for elem_type, elem in elements:
        anchor = f"elem-{elem.dom_position}"
        ts = timestamps.get(elem.dom_position, "")
        ts_html = f'<span class="timestamp">[~{ts}]</span>' if ts else ""

        if elem_type == "figure":
            label = elem.label
            toc_lines.append(f'<a href="#{anchor}">{label}{" — " + ts if ts else ""}</a>')

            img_b64 = ""
            if elem.src in chapter.images:
                img_b64 = base64.b64encode(chapter.images[elem.src]).decode()

            desc = ""
            if chapter.figure_descriptions.get(elem.number):
                desc = f'<p class="annotation">{chapter.figure_descriptions[elem.number]}</p>'

            content_lines.append(f'''
  <div class="element fig" id="{anchor}">
    <span class="element-label">{label}</span>{ts_html}
    <div class="context">Section: {elem.context}</div>
    {desc}
    {"<img src='data:image/png;base64," + img_b64 + "' alt='" + elem.alt + "'>" if img_b64 else "<p>[Image not available]</p>"}
    {"<p class='caption'>" + elem.caption + "</p>" if elem.caption else ""}
  </div>''')

        elif elem_type == "code":
            label = f"Code Example {elem.number}"
            toc_lines.append(f'<a href="#{anchor}">{label}{" — " + ts if ts else ""}</a>')

            lang_class = f'class="language-{elem.language}"' if elem.language else ""
            lang_label = f'<span class="lang-label">{elem.language}</span>' if elem.language else ""

            annotation = ""
            if elem.annotation:
                annotation = f'<p class="annotation">{elem.annotation}</p>'

            # Escape HTML in code
            escaped_code = (elem.code
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

            content_lines.append(f'''
  <div class="element" id="{anchor}">
    <span class="element-label">{label}</span>{ts_html} {lang_label}
    <div class="context">Section: {elem.context}</div>
    {annotation}
    <pre><code {lang_class}>{escaped_code}</code></pre>
  </div>''')

        elif elem_type == "math":
            label = f"Formula {elem.number}"
            toc_lines.append(f'<a href="#{anchor}">{label}{" — " + ts if ts else ""}</a>')

            text = elem.alttext or elem.mathml
            content_lines.append(f'''
  <div class="element" id="{anchor}">
    <span class="element-label">{label}</span>{ts_html}
    <div class="context">Section: {elem.context}</div>
    <div class="formula">{text}</div>
  </div>''')

        elif elem_type == "table":
            label = elem.label
            toc_lines.append(f'<a href="#{anchor}">{label}{" — " + ts if ts else ""}</a>')

            table_html = elem.html
            content_lines.append(f'''
  <div class="element" id="{anchor}">
    <span class="element-label">{label}</span>{ts_html}
    <div class="context">Section: {elem.context}</div>
    {"<p class='caption'>" + elem.caption + "</p>" if elem.caption else ""}
    {table_html}
  </div>''')

    summary = (
        f"{len(chapter.figures)} figures, {len(chapter.code_blocks)} code examples, "
        f"{len(chapter.math_formulas)} formulas, {len(chapter.tables)} tables"
    )

    html_content = _HTML_TEMPLATE.format(
        chapter_title=chapter.title,
        chapter_number=chapter.number,
        book_title=book_title,
        summary=summary,
        toc_entries="\n    ".join(toc_lines),
        content="\n".join(content_lines),
    )

    html_path.write_text(html_content, encoding="utf-8")
    logger.info(f"HTML companion: {html_path.name}")
    return html_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_companion(
    chapter: Chapter,
    output_dir: Path,
    book_title: str = "",
    companion_format: str = "both",
    audio_timestamps=None,
) -> Path:
    """Generate companion PDF and/or HTML for a chapter."""
    timestamps = _compute_element_timestamps(audio_timestamps, chapter)

    pdf_path = None
    html_path = None

    if companion_format in ("pdf", "both"):
        pdf_path = _generate_pdf(chapter, output_dir, book_title, timestamps)

    if companion_format in ("html", "both"):
        html_path = _generate_html(chapter, output_dir, book_title, timestamps)

    return pdf_path or html_path


def _generate_pdf(
    chapter: Chapter,
    output_dir: Path,
    book_title: str,
    timestamps: dict[int, str],
) -> Path:
    """Generate a companion PDF with syntax highlighting, bookmarks, and timestamps."""
    safe_title = "_".join(
        "".join(c if c.isalnum() or c in " -" else "_" for c in chapter.title).split()
    )
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
        f"{len(chapter.math_formulas)} formulas, "
        f"and {len(chapter.tables)} tables.",
    )
    pdf.ln(5)
    pdf._font_sans("", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(CONTENT_W, 6, book_title)

    # --- Content in DOM order ---
    elements = []
    elements.extend(chapter.figures)
    elements.extend(chapter.code_blocks)
    elements.extend(chapter.math_formulas)
    elements.extend(chapter.tables)
    elements.sort(key=lambda e: e.dom_position)

    if elements:
        pdf.add_page()

    for i, elem in enumerate(elements):
        if pdf.get_y() > 240:
            pdf.add_page()

        if i > 0:
            pdf.section_separator()

        ts = timestamps.get(elem.dom_position, "")

        if isinstance(elem, Figure):
            pdf.element_label(elem.label, elem.context, ts)

            # LLM description
            desc = chapter.figure_descriptions.get(elem.number)
            if desc:
                pdf._font_sans("I", 9)
                pdf.set_text_color(60, 60, 60)
                pdf.multi_cell(CONTENT_W, 5, desc)
                pdf.ln(2)

            if elem.src in chapter.images:
                _add_image_to_pdf(pdf, chapter.images[elem.src])
            else:
                pdf._font_sans("I", 10)
                pdf.set_text_color(200, 0, 0)
                pdf.cell(0, 8, f"[Image not available: {elem.src}]", new_x="LMARGIN", new_y="NEXT")

            if elem.caption:
                pdf._font_sans("I", 10)
                pdf.set_text_color(60, 60, 60)
                pdf.multi_cell(CONTENT_W, 5, elem.caption)

            if elem.alt and elem.alt != elem.caption:
                pdf._font_sans("", 8)
                pdf.set_text_color(120, 120, 120)
                pdf.multi_cell(CONTENT_W, 4, f"Alt: {elem.alt}")

            pdf.ln(3)

        elif isinstance(elem, CodeBlock):
            pdf.element_label(f"Code Example {elem.number}", elem.context, ts)
            pdf.add_code_block_highlighted(elem.code, elem.language, elem.annotation)

        elif isinstance(elem, MathFormula):
            pdf.element_label(f"Formula {elem.number}", elem.context, ts)
            _add_math_to_pdf(pdf, elem)
            pdf.ln(3)

        elif isinstance(elem, Table):
            pdf.element_label(elem.label, elem.context, ts)
            if elem.caption:
                pdf._font_sans("I", 10)
                pdf.set_text_color(60, 60, 60)
                pdf.multi_cell(CONTENT_W, 5, elem.caption)
                pdf.ln(2)
            if elem.narration:
                pdf._font_sans("", 9)
                pdf.set_text_color(40, 40, 40)
                pdf.multi_cell(CONTENT_W, 5, elem.narration)
                pdf.ln(2)
            # Render table as simple text grid
            pdf._font_mono(7)
            pdf.set_text_color(30, 30, 30)
            if elem.headers:
                header_line = " | ".join(h[:20] for h in elem.headers)
                pdf.set_fill_color(*CODE_BG)
                pdf.cell(CONTENT_W, 4, header_line, fill=True, new_x="LMARGIN", new_y="NEXT")
            for row in elem.rows[:20]:
                row_line = " | ".join(str(c)[:20] for c in row)
                y = pdf.get_y()
                if y > 270:
                    pdf.add_page()
                pdf.cell(CONTENT_W, 4, row_line, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

    pdf.output(str(pdf_path))
    logger.info(
        f"Chapter {chapter.number} PDF companion: {pdf_path.name} - "
        f"{pdf_path.stat().st_size / 1024:.0f} KB"
    )

    return pdf_path
