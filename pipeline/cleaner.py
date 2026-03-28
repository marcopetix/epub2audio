"""HTML → TTS-friendly plain text with section markers for M4B chapter markers.

Transforms raw chapter HTML into clean text optimized for text-to-speech,
replacing code blocks, figures, and math with numbered references to the
companion PDF. Returns section markers with char offsets for M4B assembly.
"""

import html
import json
import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)


@dataclass
class SectionMarker:
    """A heading position in the cleaned text, used for M4B chapter markers."""
    title: str
    level: int           # 1=h1, 2=h2, 3=h3
    char_offset: int     # Position in the cleaned text string


# Acronyms that should be spelled out letter by letter for TTS
ACRONYMS = {
    "GPT": "G P T",
    "LLM": "L L M",
    "LLMs": "L L Ms",
    "API": "A P I",
    "APIs": "A P Is",
    "RLHF": "R L H F",
    "RAG": "R A G",
    "NLP": "N L P",
    "SQL": "S Q L",
    "JSON": "J son",
    "YAML": "Y A M L",
    "HTTP": "H T T P",
    "HTTPS": "H T T P S",
    "URL": "U R L",
    "URLs": "U R Ls",
    "CLI": "C L I",
    "SDK": "S D K",
    "IDE": "I D E",
    "CPU": "C P U",
    "GPU": "G P U",
    "TPU": "T P U",
    "GCP": "G C P",
    "AWS": "A W S",
    "SLM": "S L M",
    "SLMs": "S L Ms",
    "MLOps": "M L Ops",
    "GenAI": "Gen A I",
    "AI": "A I",
    "ML": "M L",
    "DPO": "D P O",
    "SFT": "S F T",
    "KV": "K V",
    "GRPO": "G R P O",
    "CoT": "C o T",
    "ToT": "T o T",
    "DSPy": "D S Py",
    "MoE": "M o E",
    "BLEU": "B L E U",
    "ROUGE": "rouge",
    "BERT": "bert",
    "LSTM": "L S T M",
    "CNN": "C N N",
    "HTML": "H T M L",
    "CSS": "C S S",
    "REST": "rest",
    "gRPC": "G R P C",
    "OAuth": "O Auth",
    "JWT": "J W T",
    "RBAC": "R back",
    "PII": "P I I",
    "GDPR": "G D P R",
    "OWASP": "O W A S P",
}

# Abbreviations to expand
ABBREVIATIONS = {
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "et cetera",
    "vs.": "versus",
    "Fig.": "Figure",
    "fig.": "figure",
    "approx.": "approximately",
}


def _load_pronunciation(pronunciation_file: str) -> dict[str, str]:
    """Load custom pronunciation dictionary from JSON file."""
    path = Path(pronunciation_file)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        # Support both flat dict and {"terms": {...}} format
        if "terms" in data:
            return data["terms"]
        return data
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load pronunciation file {path}: {e}")
        return {}


def clean_chapter(
    raw_html: str,
    pronunciation_file: str = "pronunciation.json",
) -> tuple[str, list[SectionMarker]]:
    """Convert chapter HTML to TTS-friendly plain text.

    Returns:
        tuple of (cleaned text, list of SectionMarker with char offsets)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        soup = BeautifulSoup(raw_html, "lxml-xml")

    # If lxml-xml drops the body, fall back to html parser
    if not soup.find("body") and not soup.find("section"):
        soup = BeautifulSoup(raw_html, "lxml")

    # Find the main content body
    body = soup.find("body")
    if not body:
        body = soup

    # --- Phase 1: Remove invisible / unwanted elements ---

    # Remove index term anchors (invisible)
    for tag in body.find_all("a", attrs={"data-type": "indexterm"}):
        tag.decompose()

    # Remove footnote reference markers (superscript numbers)
    for sup in body.find_all("sup"):
        a_tag = sup.find("a")
        if a_tag and a_tag.get("data-type") == "noteref":
            sup.decompose()

    # --- Phase 2: Replace structured elements in document order ---

    code_counter = 0
    math_counter = 0

    # Process code blocks: <pre> → numbered reference
    for pre in body.find_all("pre"):
        code_counter += 1
        replacement = (
            f"\n\nThe author provides code example number {code_counter} here. "
            f"You can find it in the companion PDF for this chapter.\n\n"
        )
        pre.replace_with(NavigableString(replacement))

    # Process math: <math> → numbered reference
    for math_tag in body.find_all("math"):
        math_counter += 1
        replacement = (
            f"The author presents a mathematical formula here, "
            f"formula number {math_counter}. "
            f"See the companion PDF for the notation."
        )
        math_tag.replace_with(NavigableString(replacement))

    # Process equation divs that may wrap math
    for eq_div in body.find_all("div", attrs={"data-type": "equation"}):
        text = eq_div.get_text(strip=True)
        if text:
            eq_div.replace_with(NavigableString(f"\n\n{text}\n\n"))
        else:
            eq_div.decompose()

    # Process figures: <figure> → reference with caption
    for fig in body.find_all("figure"):
        label = ""
        caption = ""
        alt = ""

        h6 = fig.find("h6")
        if h6:
            label_span = h6.find("span", class_="label")
            if label_span:
                label = label_span.get_text(strip=True).rstrip(".")
            caption = h6.get_text(strip=True)
            if label_span:
                caption = caption.replace(label_span.get_text(strip=True), "", 1).strip()

        img = fig.find("img")
        if img:
            alt = img.get("alt", "")

        if not label:
            label = "a figure"

        parts = [f"\n\nSee {label} in the companion PDF"]
        desc = caption or alt
        if desc:
            parts.append(f": {desc}")
        parts.append(".\n\n")
        fig.replace_with(NavigableString("".join(parts)))

    # Process tables
    for table in body.find_all("table"):
        label = ""
        caption_text = ""
        caption_tag = table.find("caption")
        if caption_tag:
            label_span = caption_tag.find("span", class_="label")
            if label_span:
                label = label_span.get_text(strip=True).rstrip(".")
            caption_text = caption_tag.get_text(strip=True)
            if label_span:
                caption_text = caption_text.replace(
                    label_span.get_text(strip=True), "", 1
                ).strip()

        rows = table.find_all("tr")
        if len(rows) <= 6:
            # Small table: read as text
            parts = []
            if label:
                parts.append(f"\n\n{label}.")
            if caption_text:
                parts.append(f" {caption_text}.")

            headers = []
            thead = table.find("thead")
            if thead:
                for th in thead.find_all("th"):
                    headers.append(th.get_text(strip=True))

            tbody = table.find("tbody") or table
            for row in tbody.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue
                cell_texts = []
                for j, cell in enumerate(cells):
                    cell_text = cell.get_text(strip=True)
                    if headers and j < len(headers):
                        cell_texts.append(f"{headers[j]}: {cell_text}")
                    else:
                        cell_texts.append(cell_text)
                parts.append(" " + ", ".join(cell_texts) + ".")

            parts.append("\n\n")
            table.replace_with(NavigableString("".join(parts)))
        else:
            # Large table: reference companion
            desc = f"\n\nThe author presents {label}" if label else "\n\nThe author presents a table"
            if caption_text:
                desc += f", {caption_text}"
            desc += ". See the companion PDF for details.\n\n"
            table.replace_with(NavigableString(desc))

    # Process notes, tips, warnings
    for note_type in ["note", "tip", "warning"]:
        for div in body.find_all("div", attrs={"data-type": note_type}):
            h6 = div.find("h6")
            if h6:
                h6.decompose()
            text = div.get_text(strip=True)
            prefix = note_type.capitalize()
            div.replace_with(NavigableString(
                f"\n\n{prefix} from the author: {text}\n\n"
            ))

    # Process sidebars
    for aside in body.find_all("aside", attrs={"data-type": "sidebar"}):
        sidebar_title = ""
        h1 = aside.find("h1")
        if h1:
            sidebar_title = h1.get_text(strip=True)
            h1.decompose()
        text = aside.get_text(strip=True)
        prefix = f"Sidebar, {sidebar_title}. " if sidebar_title else "Sidebar. "
        aside.replace_with(NavigableString(f"\n\n{prefix}{text}\n\n"))

    # Process footnote sections
    for fn_div in body.find_all("div", attrs={"data-type": "footnotes"}):
        parts = ["\n\n"]
        for fn_p in fn_div.find_all("p", attrs={"data-type": "footnote"}):
            for sup in fn_p.find_all("sup"):
                sup.decompose()
            text = fn_p.get_text(strip=True)
            if text:
                parts.append(f"Footnote: {text} ")
        parts.append("\n\n")
        fn_div.replace_with(NavigableString("".join(parts)))

    # --- Phase 3: Process headings for natural pauses ---
    # Collect heading info BEFORE replacing, so we can build section markers later

    heading_texts = []  # (tag_name, text) in DOM order
    for heading_name in ["h1", "h2", "h3"]:
        for h_tag in body.find_all(heading_name):
            text = h_tag.get_text(strip=True)
            if text:
                heading_texts.append((heading_name, text))

    for h1 in body.find_all("h1"):
        text = h1.get_text(strip=True)
        h1.replace_with(NavigableString(f"\n\n\n{text}.\n\n"))

    for h2 in body.find_all("h2"):
        text = h2.get_text(strip=True)
        h2.replace_with(NavigableString(f"\n\n{text}.\n\n"))

    for h3 in body.find_all("h3"):
        text = h3.get_text(strip=True)
        h3.replace_with(NavigableString(f"\n\n{text}.\n\n"))

    for h6 in body.find_all("h6"):
        text = h6.get_text(strip=True)
        h6.replace_with(NavigableString(f"\n{text}\n"))

    # --- Phase 4: Handle inline elements ---

    for code in body.find_all("code"):
        code.replace_with(NavigableString(code.get_text()))

    for a in body.find_all("a"):
        a.replace_with(NavigableString(a.get_text()))

    for tag_name in ["em", "strong", "b", "i", "span"]:
        for tag in body.find_all(tag_name):
            tag.replace_with(NavigableString(tag.get_text()))

    # --- Phase 5: Extract final text ---
    text = body.get_text()

    # --- Phase 6: Post-processing ---
    pronunciation = _load_pronunciation(pronunciation_file)
    text = _post_process(text, pronunciation)

    # --- Build section markers with char offsets ---
    section_markers = _build_section_markers(text, heading_texts)

    return text, section_markers


def _build_section_markers(text: str, heading_texts: list[tuple[str, str]]) -> list[SectionMarker]:
    """Find heading positions in the cleaned text to build section markers."""
    markers = []
    search_start = 0

    for tag_name, heading_text in heading_texts:
        level = int(tag_name[1])
        # The heading was replaced with "{text}." — search for that pattern
        search_text = f"{heading_text}."
        idx = text.find(search_text, search_start)
        if idx >= 0:
            markers.append(SectionMarker(
                title=heading_text,
                level=level,
                char_offset=idx,
            ))
            search_start = idx + len(search_text)

    return markers


def _post_process(text: str, pronunciation: dict[str, str] | None = None) -> str:
    """Clean up and normalize text for TTS."""

    # Decode any remaining HTML entities
    text = html.unescape(text)

    # Expand abbreviations (before acronym processing)
    for abbr, expansion in ABBREVIATIONS.items():
        text = text.replace(abbr, expansion)

    # Apply custom pronunciation dictionary
    if pronunciation:
        for term, phonetic in pronunciation.items():
            text = re.sub(
                rf'\b{re.escape(term)}\b',
                phonetic,
                text,
                flags=re.IGNORECASE,
            )

    # Expand acronyms (whole word only)
    for acronym, expansion in ACRONYMS.items():
        text = re.sub(
            rf'\b{re.escape(acronym)}\b',
            expansion,
            text,
        )

    # Handle versioned model names
    text = re.sub(r'\bG P T-(\d\w*)\b', r'G P T \1', text)

    # Remove long URLs
    text = re.sub(r'https?://\S+', '', text)

    # Clean up figure cross-references
    text = re.sub(
        r'\(see (Figure \d+-\d+)\)',
        r', as shown in \1 in the companion PDF,',
        text,
        flags=re.IGNORECASE,
    )

    # Clean up table cross-references
    text = re.sub(
        r'\(see (Table \d+-\d+)\)',
        r', as shown in \1,',
        text,
        flags=re.IGNORECASE,
    )

    # Normalize dashes for TTS
    text = text.replace("—", " — ")
    text = text.replace("–", " — ")

    # Remove backticks
    text = text.replace("`", "")

    # Normalize whitespace: collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalize newlines: collapse 3+ into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    text = text.strip()

    return text
