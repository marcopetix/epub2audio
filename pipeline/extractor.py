"""EPUB → structured Chapter objects with figures, code blocks, and math formulas."""

import logging
import re
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


@dataclass
class Figure:
    number: int           # Sequential within chapter (1-based)
    label: str            # "Figure 1-1" (from <span class="label">)
    src: str              # "assets/gaid_0101.png"
    alt: str              # Alt text from <img>
    caption: str          # Full caption text
    context: str          # Last heading before this figure
    dom_position: int = 0 # Position in DOM order among all visual elements


@dataclass
class CodeBlock:
    number: int           # Sequential within chapter (1-based)
    language: str         # From data-code-language attribute, "" if unknown
    code: str             # Raw text content (tags stripped)
    context: str          # Last heading before this code block
    dom_position: int = 0 # Position in DOM order among all visual elements


@dataclass
class MathFormula:
    number: int           # Sequential within chapter (1-based)
    alttext: str          # From alttext attribute
    mathml: str           # Raw MathML markup
    context: str          # Last heading before this formula
    dom_position: int = 0 # Position in DOM order among all visual elements


@dataclass
class Chapter:
    number: int           # 1-based
    filename: str         # "ch01.html"
    title: str            # From <h1>
    raw_html: str         # Full HTML content
    figures: list[Figure] = field(default_factory=list)
    code_blocks: list[CodeBlock] = field(default_factory=list)
    math_formulas: list[MathFormula] = field(default_factory=list)
    images: dict[str, bytes] = field(default_factory=dict)  # src -> image data


def _get_current_heading(tag: Tag) -> str:
    """Walk backwards from tag to find the nearest preceding heading."""
    for prev in tag.find_all_previous(re.compile(r'^h[1-3]$')):
        text = prev.get_text(strip=True)
        if text:
            return text
    return ""


def _extract_all_elements(soup: BeautifulSoup) -> tuple[list[Figure], list[CodeBlock], list[MathFormula]]:
    """Extract figures, code blocks, and math formulas in DOM order.

    Each element gets a dom_position tracking its order relative to all
    other visual elements in the chapter.
    """
    figures = []
    code_blocks = []
    math_formulas = []

    fig_counter = 0
    code_counter = 0
    math_counter = 0

    # Walk all visual elements in DOM order
    all_elements = soup.find_all(["figure", "pre", "math"])
    for dom_pos, element in enumerate(all_elements):
        context = _get_current_heading(element)

        if element.name == "figure":
            img = element.find("img")
            if not img:
                continue

            fig_counter += 1
            src = img.get("src", "")
            alt = img.get("alt", "")

            caption = ""
            label = ""
            h6 = element.find("h6")
            if h6:
                label_span = h6.find("span", class_="label")
                if label_span:
                    label = label_span.get_text(strip=True).rstrip(". ")
                caption = h6.get_text(strip=True)
                if label_span:
                    label_text = label_span.get_text(strip=True)
                    caption = caption.replace(label_text, "", 1).strip()

            if not label:
                label = f"Figure {fig_counter}"

            figures.append(Figure(
                number=fig_counter,
                label=label,
                src=src,
                alt=alt,
                caption=caption,
                context=context,
                dom_position=dom_pos,
            ))

        elif element.name == "pre":
            code_counter += 1
            code = element.get_text()
            language = element.get("data-code-language", "")

            code_blocks.append(CodeBlock(
                number=code_counter,
                language=language,
                code=code,
                context=context,
                dom_position=dom_pos,
            ))

        elif element.name == "math":
            math_counter += 1
            alttext = element.get("alttext", "")
            mathml = str(element)

            math_formulas.append(MathFormula(
                number=math_counter,
                alttext=alttext,
                mathml=mathml,
                context=context,
                dom_position=dom_pos,
            ))

    return figures, code_blocks, math_formulas


def _extract_chapter(
    zf: zipfile.ZipFile,
    chapter_file: str,
    chapter_number: int,
) -> Chapter:
    """Parse a single chapter HTML file from the EPUB ZIP."""
    html = zf.read(chapter_file).decode("utf-8")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        soup = BeautifulSoup(html, "lxml-xml")
    # Fall back if XML parser drops HTML structure
    if not soup.find("body") and not soup.find("section"):
        soup = BeautifulSoup(html, "lxml")

    # Title from first <h1>
    h1 = soup.find("h1")
    title = ""
    if h1:
        title = h1.get_text(strip=True)
        # Strip "Chapter N. " prefix if present
        label_span = h1.find("span", class_="label")
        if label_span:
            label_text = label_span.get_text(strip=True)
            title = title.replace(label_text, "", 1).strip()

    filename = chapter_file.split("/")[-1]

    # Extract all structured elements in DOM order
    figures, code_blocks, math_formulas = _extract_all_elements(soup)

    # Extract image data from ZIP
    images = {}
    for fig in figures:
        img_path = f"OEBPS/{fig.src}" if not fig.src.startswith("OEBPS/") else fig.src
        try:
            images[fig.src] = zf.read(img_path)
        except KeyError:
            logger.warning(f"Image not found in EPUB: {img_path}")

    logger.info(
        f"Chapter {chapter_number}: {title} — "
        f"{len(figures)} figures, {len(code_blocks)} code blocks, "
        f"{len(math_formulas)} math formulas"
    )

    return Chapter(
        number=chapter_number,
        filename=filename,
        title=title,
        raw_html=html,
        figures=figures,
        code_blocks=code_blocks,
        math_formulas=math_formulas,
        images=images,
    )


def extract_chapters(epub_path: Path) -> list[Chapter]:
    """Extract all chapters from an EPUB file.

    Returns a list of Chapter objects sorted by chapter number.
    """
    chapters = []

    with zipfile.ZipFile(epub_path, "r") as zf:
        # Find chapter files (ch01.html, ch02.html, ...)
        chapter_files = sorted(
            name for name in zf.namelist()
            if re.match(r"OEBPS/ch\d+\.x?html$", name)
        )

        if not chapter_files:
            raise ValueError(f"No chapter files found in {epub_path}")

        logger.info(f"Found {len(chapter_files)} chapters in {epub_path.name}")

        for ch_file in chapter_files:
            # Extract chapter number from filename
            match = re.search(r"ch(\d+)", ch_file)
            if not match:
                continue
            ch_num = int(match.group(1))

            chapter = _extract_chapter(zf, ch_file, ch_num)
            chapters.append(chapter)

    return chapters


def extract_cover(epub_path: Path) -> bytes | None:
    """Extract cover image from EPUB if available."""
    with zipfile.ZipFile(epub_path, "r") as zf:
        for candidate in ["OEBPS/assets/cover.png", "OEBPS/cover.png"]:
            try:
                return zf.read(candidate)
            except KeyError:
                continue
    return None
