"""MathML → LaTeX → PNG rendering.

Extracted from companion.py so that math can be pre-rendered during
extraction (populating MathFormula.rendered_path) instead of only at
companion-generation time.
"""

import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)


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
            "×": r"\times", "·": r"\cdot",
            "≤": r"\leq", "≥": r"\geq",
            "∑": r"\sum", "∏": r"\prod",
            "‖": r"\parallel", "…": r"\ldots",
            "∞": r"\infty", "≠": r"\neq",
            "∈": r"\in", "→": r"\rightarrow",
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


def render_formula_to_png(mathml: str, output_path: Path) -> bool:
    """Render a MathML formula to PNG at output_path.

    Returns True on success, False on failure (graceful).
    """
    try:
        latex = _mathml_string_to_latex(mathml)
        if not latex:
            return False
        img_data = _render_latex_to_image(latex)
        if img_data is None:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(img_data)
        return True
    except Exception as e:
        logger.warning(f"Math rendering failed for {output_path}: {e}")
        return False
