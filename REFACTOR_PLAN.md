# Refactor: NarratedElement + math pre-rendering

## Goal

Two coordinated refactors in one pass:

1. Replace 4 separate dataclasses (Figure, CodeBlock, MathFormula, Table)
   with a common base class NarratedElement. Replace the 4 lists in
   Chapter with a single `elements: list[NarratedElement]` ordered by
   dom_position.

2. Pre-render math formulas to PNG during extraction (move logic out of
   companion.py into a new math_renderer module). Pre-write Figure
   images to disk during extraction. Populate `rendered_path` on
   elements that have a visual asset, ready for downstream vision-based
   enrichment.

3. Eliminate accumulated workarounds in the data model:
   - Remove `Figure.src` (replaced by `rendered_path`)
   - Remove `Chapter.images` dict (images go straight to disk)
   - Remove `Chapter.figure_descriptions` (narration lives on the element)

The three changes share the same dataclass restructuring, so they are
done together to avoid multiple review/test cycles.

## Operating constraints for the AI assistant

- **Work file by file, not all at once.** After each file is modified,
  pause and report what changed. Do not proceed to the next file until
  acknowledged.
- **Order of files matters.** Follow the order in "Implementation order"
  below. Do not skip ahead.
- **Preserve behavior.** Audio output, companion PDFs, companion HTML,
  and manifest.json must be byte-equivalent (or as close as the
  randomness of LLM enrichment allows) before and after the refactor,
  when running with --no-llm.
- **Don't introduce new dependencies.** All changes use existing
  libraries already in requirements.txt.
- **Don't refactor things outside this scope.** No drive-by improvements
  on unrelated code. If something is ugly but works, leave it.
- **When in doubt, ask.** Better to pause and confirm than to make
  silent assumptions.

## NarratedElement base class

```python
@dataclass(kw_only=True)
class NarratedElement:
    number: int                        # 1-based sequential per element type
    context: str                       # last heading before this element
    dom_position: int                  # position in DOM order
    narration: str = ""                # LLM-generated description (filled by enricher)
    rendered_path: Path | None = None  # PNG asset for vision enrichment
```

`kw_only=True` is required because subclasses add fields without defaults
after fields with defaults in the base. Python's dataclass inheritance
forbids that ordering unless kw_only is used.

## Subclasses

```python
@dataclass(kw_only=True)
class Figure(NarratedElement):
    label: str
    alt: str
    caption: str
    # NB: src field is removed — rendered_path is now the canonical reference

@dataclass(kw_only=True)
class CodeBlock(NarratedElement):
    language: str
    code: str
    # NB: previously had `annotation` field — now uses `narration` from base class

@dataclass(kw_only=True)
class MathFormula(NarratedElement):
    alttext: str
    mathml: str

@dataclass(kw_only=True)
class Table(NarratedElement):
    label: str
    caption: str
    html: str
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    # NB: previously had `narration` field — now inherited from base class
```

All subclass instantiations elsewhere in the code must use keyword
arguments. Update all call sites that currently use positional fields.

### Naming unification: `narration` for all enriched content

Previously:
- `CodeBlock.annotation` — LLM-generated description
- `Table.narration` — LLM-generated description
- `Chapter.figure_descriptions[fig.number]` — LLM-generated description (external dict)

After this refactor, ALL enriched content uses `element.narration`.
This is a breaking change for the cleaner and the enricher. Both files
must be updated to read/write `narration` consistently.

## rendered_path semantics

- **Figure.rendered_path**: populated by extractor. For each figure in
  the EPUB, the extractor reads the image bytes from the EPUB zip,
  writes them to disk under `assets_dir / Path(epub_relative_path).name`,
  and sets `Figure.rendered_path` to that Path. Fallback: None if the
  image was not found in the EPUB (with a warning logged).

- **MathFormula.rendered_path**: populated by extractor by calling
  `math_renderer.render_formula_to_png(mathml, output_path)`. Output
  path: `assets_dir / "math" / f"ch{chapter_num:02d}_math{math_num:02d}.png"`.
  Fallback: None if rendering failed (logged as warning).

- **CodeBlock.rendered_path**: always None for now.

- **Table.rendered_path**: always None for now.

The extractor receives `assets_dir: Path` as a new parameter. Update
the `extract_chapters()` signature accordingly, and update
`epub2audio.py` to pass `config.assets_dir`.

## Chapter refactor

Replace these fields:

```python
@dataclass
class Chapter:
    figures: list[Figure] = field(default_factory=list)
    code_blocks: list[CodeBlock] = field(default_factory=list)
    math_formulas: list[MathFormula] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figure_descriptions: dict[int, str] = field(default_factory=dict)
    images: dict[str, bytes] = field(default_factory=dict)
```

With:

```python
@dataclass
class Chapter:
    # ... other unchanged fields (number, filename, title, raw_html,
    # sections, intro) ...
    elements: list[NarratedElement] = field(default_factory=list)
```

Provide backward-compatibility helper properties on Chapter:

```python
@property
def figures(self) -> list[Figure]:
    return [e for e in self.elements if isinstance(e, Figure)]

@property
def code_blocks(self) -> list[CodeBlock]:
    return [e for e in self.elements if isinstance(e, CodeBlock)]

@property
def math_formulas(self) -> list[MathFormula]:
    return [e for e in self.elements if isinstance(e, MathFormula)]

@property
def tables(self) -> list[Table]:
    return [e for e in self.elements if isinstance(e, Table)]
```

These properties allow consumers (companion, assembler, manifest) to
keep working with minimal changes, while internally we have a single
ordered list.

`figure_descriptions` is REMOVED. The narration now lives directly on
each Figure element.

`images` is REMOVED. Images are written to disk during extraction; consumers
read them via `figure.rendered_path` instead of from an in-memory dict.

## Image handling: end-to-end change

Currently:
1. Extractor reads images from EPUB zip into `chapter.images: dict[str, bytes]`.
2. Main loop in `epub2audio.py` writes those bytes to `config.assets_dir` during
   the companion phase.
3. Companion reads from `chapter.images[elem.src]` for HTML base64 embedding
   and PDF rendering.

After refactor:
1. Extractor reads images from EPUB zip and writes them to disk **immediately**
   under `assets_dir`. Sets `figure.rendered_path` to the disk path.
2. Main loop no longer writes images (that step is removed).
3. Companion reads images from `elem.rendered_path` (Path.read_bytes) for both
   HTML and PDF.

The path inside the EPUB zip (e.g. `"OEBPS/assets/foo.png"`) is used
only as a local variable inside `_extract_chapter` to read from the
zip. After reading and writing to disk, this string is discarded.

The `Figure.src` field is therefore no longer needed and is removed.

## Files to modify

### Implementation order

1. **pipeline/math_renderer.py** (NEW)
2. **pipeline/extractor.py** (modify dataclasses + extraction logic + image writing)
3. **pipeline/cleaner.py** (update enrichment lookups, use unified `narration`)
4. **pipeline/llm_enricher.py** (iterate over chapter.elements, write to `narration`)
5. **pipeline/companion.py** (use rendered_path with fallback, remove math rendering)
6. **pipeline/assembler.py** (verify section_markers logic still works)
7. **epub2audio.py** (pass assets_dir to extract_chapters; remove image-writing loop)
8. **tests/test_chunker.py** (verify nothing broke)

### File 1: pipeline/math_renderer.py (NEW)

Extract from pipeline/companion.py:
- `_mathml_to_latex(el)` (function)
- `_render_latex_to_image(latex_str)` (function)
- `_mathml_string_to_latex(mathml)` (function)

Add new convenience function:

```python
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
```

The existing `_mathml_to_latex` and `_render_latex_to_image` functions
should be exported as module-level functions so companion.py can still
use them as a fallback path.

### File 2: pipeline/extractor.py

- Replace dataclasses (Figure, CodeBlock, MathFormula, Table) with new
  versions inheriting from NarratedElement. Remove the `src` field
  from Figure. Remove the `annotation` field from CodeBlock (uses
  inherited `narration`). Remove the explicit `narration` field from
  Table (uses inherited `narration`).

- Update Chapter dataclass: remove `figures`, `code_blocks`,
  `math_formulas`, `tables`, `figure_descriptions`, `images`. Add
  `elements: list[NarratedElement]`. Add the four `@property` helpers.

- Update `_extract_table` to use keyword arguments for the new dataclass.

- Update `_extract_all_elements` to:
  - Build a single ordered list, not 4 separate lists
  - Return that single list (not a 5-tuple)

- Update `_extract_chapter` to:
  - Accept `assets_dir: Path` as a parameter
  - After collecting elements, for each Figure: read image bytes from
    EPUB zip, write to `assets_dir / Path(epub_path).name`, populate
    `rendered_path`. If image not found in zip, log warning and leave
    `rendered_path = None`.
  - For each MathFormula: call `render_formula_to_png(mathml, output_path)`
    where `output_path = assets_dir / "math" / f"ch{N}_math{M}.png"`.
    Populate `rendered_path` on success, leave None on failure.
  - Build Chapter with `elements=...` only (no images dict, no
    figures/code_blocks/etc separate lists).

- Update `extract_chapters(epub_path: Path, assets_dir: Path)` signature.

### File 3: pipeline/cleaner.py

In `clean_chapter()`, the enrichment lookups currently read from:
- `chapter.code_blocks` — works via helper property, no change
- `chapter.figures` — works via helper property, no change
- `chapter.tables` — works via helper property, no change
- `chapter.figure_descriptions[fig.number]` — REMOVED

Update enrichment lookups to read `narration` directly from each element:

```python
# Old
for cb in chapter.code_blocks:
    if cb.annotation:
        code_annotations[cb.number] = cb.annotation

# New
for cb in chapter.code_blocks:
    if cb.narration:
        code_annotations[cb.number] = cb.narration
```

```python
# Old
for fig in chapter.figures:
    desc = chapter.figure_descriptions.get(fig.number, "")
    if desc:
        fig_desc_by_label[fig.label] = desc

# New
for fig in chapter.figures:
    if fig.narration:
        fig_desc_by_label[fig.label] = fig.narration
```

```python
# Old
for tbl in chapter.tables:
    if tbl.narration:
        table_narrations[tbl.number] = tbl.narration

# New (unchanged — tbl.narration is now inherited but still exists)
for tbl in chapter.tables:
    if tbl.narration:
        table_narrations[tbl.number] = tbl.narration
```

The rest of `clean_chapter` is unchanged.

### File 4: pipeline/llm_enricher.py

Update `enrich_chapter()` to iterate over `chapter.elements` and
dispatch by type. The methods `_annotate_code`, `_describe_figure`,
`_narrate_table` should remain functionally equivalent — they take
the element and return the narration string. The assignment to
`element.narration` happens in `enrich_chapter`.

```python
def enrich_chapter(self, chapter: Chapter) -> None:
    if not self.available:
        return

    logger.info(f"Enriching chapter {chapter.number}: {chapter.title}")

    # 1. Generate chapter intro
    chapter.intro = self._generate_intro(chapter)

    # 2. Iterate elements and dispatch by type
    for elem in chapter.elements:
        if isinstance(elem, CodeBlock):
            elem.narration = self._annotate_code(elem)
        elif isinstance(elem, Figure):
            elem.narration = self._describe_figure(elem)
        elif isinstance(elem, Table) and elem.row_count <= 6:
            elem.narration = self._narrate_table(elem)
        # MathFormula skipped (no enrichment in this version)

    # Logging
    code_count = sum(1 for e in chapter.elements if isinstance(e, CodeBlock) and e.narration)
    fig_count = sum(1 for e in chapter.elements if isinstance(e, Figure) and e.narration)
    table_count = sum(1 for e in chapter.elements if isinstance(e, Table) and e.narration)
    logger.info(
        f"  Enriched: intro, {code_count} code annotations, "
        f"{fig_count} figure descriptions, {table_count} table narrations"
    )
```

The internal helper methods (`_annotate_code`, `_describe_figure`,
`_narrate_table`) keep their current signatures and prompts. Just
rename references inside their bodies if they used `code_block.annotation`
or similar.

`simplify_paragraph` was already stubbed in the previous step. Leave it.

### File 5: pipeline/companion.py

- Remove `_mathml_to_latex`, `_render_latex_to_image`,
  `_mathml_string_to_latex` (moved to math_renderer.py). Import them
  from math_renderer if needed for the fallback path.

- Update `_add_math_to_pdf` to:
  - Check if `formula.rendered_path` exists and the file is on disk
  - If yes, load PNG via `_add_image_to_pdf` directly
  - If no, fall back to current re-rendering via math_renderer functions

- Update HTML generation similarly: prefer `rendered_path`, fall back
  to text rendering.

- Update image embedding for figures:

```python
# Old
if elem.src in chapter.images:
    img_b64 = base64.b64encode(chapter.images[elem.src]).decode()

# New
if elem.rendered_path and elem.rendered_path.exists():
    img_b64 = base64.b64encode(elem.rendered_path.read_bytes()).decode()
```

- The merge-and-sort logic at the start of `_generate_html` and
  `_generate_pdf`:

```python
# Old
elements = []
elements.extend(("figure", f) for f in chapter.figures)
elements.extend(("code", c) for c in chapter.code_blocks)
elements.extend(("math", m) for m in chapter.math_formulas)
elements.extend(("table", t) for t in chapter.tables)
elements.sort(key=lambda e: e[1].dom_position)
```

This still works because helper properties return filtered lists.
Optionally simplify to iterate `chapter.elements` directly:

```python
# Optional simpler version (if not too disruptive)
for elem in chapter.elements:
    if isinstance(elem, Figure):
        # render figure
    elif isinstance(elem, CodeBlock):
        # render code
    elif isinstance(elem, MathFormula):
        # render formula
    elif isinstance(elem, Table):
        # render table
```

If the existing code is easier to keep working with helper properties,
do that. Avoid disruptive rewrites.

### File 6: pipeline/assembler.py

Verify that nothing references removed fields. The marker assignment
relies on `wav_result.chunk.section_markers`, which is unaffected.
Probably no changes needed, but verify by reading the file.

### File 7: epub2audio.py

- Update `extract_chapters(config.epub_path)` call to also pass
  `config.assets_dir`. The directory must exist before this call
  (already ensured by `config.ensure_dirs()`).

- Remove the loop in the companion phase that writes images to disk:

```python
# REMOVE this block (in the Phase 6: companion section):
for src, img_data in chapter.images.items():
    img_path = config.assets_dir / Path(src).name
    img_path.write_bytes(img_data)
```

The extractor now writes images directly during extraction. This loop
is redundant and would fail anyway since `chapter.images` is removed.

- Verify manifest building still works:

```python
"figures": len(chapter.figures),         # works via helper property
"code_blocks": len(chapter.code_blocks), # works via helper property
"math_formulas": len(chapter.math_formulas),  # works via helper property
"tables": len(chapter.tables),           # works via helper property
```

These continue to work via the helper properties. No changes needed.

### File 8: tests/test_chunker.py

Should pass unchanged because the chunker doesn't depend on the
element model. Verify by running pytest.

## Validation

### Checkpoint after each file

After modifying each file, run:

```bash
python -c "from pipeline.{module} import *"
```

To verify no syntax errors / import errors.

### Checkpoint after files 1-2

```bash
python -c "
from pathlib import Path
from pipeline.extractor import extract_chapters

chapters = extract_chapters(Path('book.epub'), Path('/tmp/test_assets'))
print(f'Extracted {len(chapters)} chapters')
ch1 = chapters[0]
print(f'Ch1 has {len(ch1.elements)} elements')
print(f'Ch1 figures (via property): {len(ch1.figures)}')
print(f'Ch1 first 3 elements:')
for e in ch1.elements[:3]:
    print(f'  - {type(e).__name__} #{e.number} (dom_pos={e.dom_position}, rendered={e.rendered_path})')
"
```

Verify:
- Extractor produces Chapter objects with elements list
- Helper properties work
- Some Figure elements have rendered_path set (pointing to existing files)
- Some MathFormula elements have rendered_path set

### Checkpoint after file 3

```bash
python -c "
from pathlib import Path
from pipeline.cleaner import clean_chapter
from pipeline.extractor import extract_chapters

chapters = extract_chapters(Path('book.epub'), Path('/tmp/test_assets'))
text, markers = clean_chapter(chapters[0].raw_html, chapter=chapters[0])
print(f'Clean text: {len(text)} chars, {len(markers)} markers')
"
```

### Checkpoint after files 4-5

Quick check that imports resolve and basic methods don't crash:

```bash
python -c "from pipeline.llm_enricher import LLMEnricher; print('enricher OK')"
python -c "from pipeline.companion import generate_companion; print('companion OK')"
```

### Final E2E checkpoint

```bash
# Dry run on chapter 1
python epub2audio.py book.epub --chapters 1 --dry-run
```

Expected output: same as before refactor — same number of figures,
code blocks, math, tables; same chunk counts.

```bash
# Full run on chapter 1, no LLM (deterministic comparison)
python epub2audio.py book.epub --chapters 1 --no-llm
```

Compare outputs to baseline:
- Same audio file size and structure
- Same companion PDF/HTML files generated
- Same manifest.json structure (modulo reordering)

### Tests

```bash
pytest tests/test_chunker.py -v
```

Must pass.

## Constraints recap

- Maintain backward-compatible behavior on output (audio, companion,
  manifest).
- Remove `figure_descriptions`, `images`, `Figure.src`,
  `CodeBlock.annotation`. All replaced by unified model.
- Tests in tests/test_chunker.py must still pass.
- All enriched content (code, figures, tables, future formulas) uses
  `narration` field uniformly.
- Do not change requirements.txt.
- Do not change the public API of epub2audio.py CLI.

## What is OUT of scope for this refactor

- Vision-based enrichment (handled in next phase).
- Pipeline class extraction (handled in next phase).
- Inspect tool (handled in next phase).
- README updates (handled in next phase).
- datapizza-ai integration (handled in next phase).
- Any new dependencies.
- Any change to TTS, chunking, or assembly logic.
- Any change to the synthesizer threading/lock model.