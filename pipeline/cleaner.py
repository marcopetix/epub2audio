"""HTML → TTS-friendly plain text.

Transforms raw chapter HTML into clean text optimized for text-to-speech,
replacing code blocks, figures, and math with numbered references to the
companion PDF.
"""

import html
import logging
import re
import warnings

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)

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


def clean_chapter(raw_html: str) -> str:
    """Convert chapter HTML to TTS-friendly plain text.

    Code blocks, figures, and math are replaced with numbered references
    that correspond to the companion PDF.
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
    # We'll inline the footnote text instead
    for sup in body.find_all("sup"):
        a_tag = sup.find("a")
        if a_tag and a_tag.get("data-type") == "noteref":
            sup.decompose()

    # --- Phase 2: Replace structured elements in document order ---
    # We must process these in DOM order so numbering is sequential

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
        # If it still contains text (math already replaced above), keep it
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

        # Get label from <span class="label">
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

        # Build replacement text
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

        # Count rows to decide read-aloud vs reference
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
            # Remove the <h6> header (e.g., "Tip", "Note")
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
            # Remove superscript number
            for sup in fn_p.find_all("sup"):
                sup.decompose()
            text = fn_p.get_text(strip=True)
            if text:
                parts.append(f"Footnote: {text} ")
        parts.append("\n\n")
        fn_div.replace_with(NavigableString("".join(parts)))

    # --- Phase 3: Process headings for natural pauses ---

    for h1 in body.find_all("h1"):
        text = h1.get_text(strip=True)
        h1.replace_with(NavigableString(f"\n\n\n{text}.\n\n"))

    for h2 in body.find_all("h2"):
        text = h2.get_text(strip=True)
        h2.replace_with(NavigableString(f"\n\n{text}.\n\n"))

    for h3 in body.find_all("h3"):
        text = h3.get_text(strip=True)
        h3.replace_with(NavigableString(f"\n\n{text}.\n\n"))

    # h6 remaining (not inside figures/tables, already processed)
    for h6 in body.find_all("h6"):
        text = h6.get_text(strip=True)
        h6.replace_with(NavigableString(f"\n{text}\n"))

    # --- Phase 4: Handle inline elements ---

    # Strip inline <code> tags but keep text
    for code in body.find_all("code"):
        code.replace_with(NavigableString(code.get_text()))

    # Strip links but keep text
    for a in body.find_all("a"):
        a.replace_with(NavigableString(a.get_text()))

    # Strip emphasis/strong but keep text
    for tag_name in ["em", "strong", "b", "i", "span"]:
        for tag in body.find_all(tag_name):
            tag.replace_with(NavigableString(tag.get_text()))

    # --- Phase 5: Extract final text ---
    text = body.get_text()

    # --- Phase 6: Post-processing ---
    text = _post_process(text)

    return text


def _post_process(text: str) -> str:
    """Clean up and normalize text for TTS."""

    # Decode any remaining HTML entities
    text = html.unescape(text)

    # Expand abbreviations (before acronym processing)
    for abbr, expansion in ABBREVIATIONS.items():
        text = text.replace(abbr, expansion)

    # Expand acronyms (whole word only)
    for acronym, expansion in ACRONYMS.items():
        text = re.sub(
            rf'\b{re.escape(acronym)}\b',
            expansion,
            text,
        )

    # Handle versioned model names: "GPT-4" already handled by acronym expansion
    # but handle patterns like "GPT-4o", "GPT-3.5"
    text = re.sub(r'\bG P T-(\d\w*)\b', r'G P T \1', text)

    # Remove long URLs (anything with http:// or https://)
    text = re.sub(r'https?://\S+', '', text)

    # Clean up figure cross-references: "(see Figure 3-1)" → "as shown in Figure 3-1"
    text = re.sub(
        r'\(see (Figure \d+-\d+)\)',
        r', as shown in \1 in the companion PDF,',
        text,
        flags=re.IGNORECASE,
    )

    # Clean up table cross-references similarly
    text = re.sub(
        r'\(see (Table \d+-\d+)\)',
        r', as shown in \1,',
        text,
        flags=re.IGNORECASE,
    )

    # Normalize dashes for TTS
    text = text.replace("—", " — ")
    text = text.replace("–", " — ")

    # Remove backticks (from any remaining inline code)
    text = text.replace("`", "")

    # Normalize whitespace: collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalize newlines: collapse 3+ into 2 (paragraph separator)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text
