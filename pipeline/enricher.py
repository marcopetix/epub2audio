"""LLM enrichment using datapizza-ai for audiobook enhancement.

Generates chapter intros, code annotations, table narrations, figure descriptions,
and (optionally) vision-based descriptions of figures and math formulas.

Two backends:
- Text: Ollama via OpenAILikeClient (local, no API cost)
- Vision: OpenRouter via OpenAILikeClient (cloud, opt-in via --vision)

Caching is provided by datapizza-ai's @cacheable decorator with a custom
FileCache backend. Tracing is automatic via OpenTelemetry spans.
"""

import logging

import requests

from datapizza.clients.openai_like import OpenAILikeClient
from datapizza.core.cache import Cache
from datapizza.type import Media, MediaBlock, TextBlock

from pipeline.extractor import (
    Chapter,
    CodeBlock,
    Figure,
    MathFormula,
    Table,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

INTRO_SYSTEM = (
    "You are a technical book narrator. Generate a 2-3 sentence introduction "
    "for an audiobook chapter that tells the listener what they will learn. "
    "Be concise and engaging."
)

CODE_SYSTEM = (
    "You are a technical book narrator. Write a 1-2 sentence description "
    "of what this code does, suitable for someone listening to an audiobook. "
    "Be concise."
)

FIGURE_TEXT_SYSTEM = (
    "You are narrating a technical book. Based on the caption, alt text and "
    "section context, generate a 1-2 sentence audio description of what "
    "this figure likely shows. Be concise."
)

FIGURE_VISION_PROMPT = (
    "You are narrating a technical book for an audiobook listener who cannot "
    "see the figures. Describe this figure in 2-3 sentences, focusing on:\n"
    "- What kind of diagram/chart/illustration it is\n"
    "- The main components or data shown\n"
    "- The conceptual point the figure makes\n"
    "Be concrete and specific."
)

FORMULA_TEXT_SYSTEM = (
    "You are narrating a technical book for an audiobook listener. "
    "Read the formula naturally and explain what it represents in 1-2 sentences. "
    "Focus on the mathematical meaning, not the visual notation."
)

FORMULA_VISION_PROMPT = (
    "You are narrating a technical book for an audiobook listener who cannot "
    "see the formulas. This image shows a mathematical formula. "
    "Read it aloud naturally and explain what it represents in 1-2 sentences. "
    "Focus on the mathematical meaning, not the visual notation."
)

TABLE_SYSTEM = (
    "Convert this table data into a natural language description suitable "
    "for audio listening. Be concise but include the key data points."
)
class Enricher:
    """Orchestrates LLM enrichment of a Chapter using datapizza-ai.

    Uses two clients:
    - text_client: Ollama (local) for intro, code narrations, table narrations,
      figure/formula text-only fallback
    - vision_client: OpenRouter (cloud, opt-in) for figure and formula vision-
      based descriptions

    Both clients share the same FileCache backend for persistent caching of
    LLM responses across pipeline runs.
    """

    def __init__(
        self,
        *,
        ollama_url: str,
        ollama_model: str,
        cache: Cache | None = None,
        vision_enabled: bool = False,
        vision_api_key: str | None = None,
        vision_base_url: str = "https://openrouter.ai/api/v1",
        vision_model: str = "google/gemini-2.5-flash",
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model

        self.text_client = OpenAILikeClient(
            api_key="ollama",  # required by constructor but unused by Ollama
            model=ollama_model,
            base_url=f"{self.ollama_url}/v1",
            cache=cache,
        )

        self.vision_client = None
        if vision_enabled:
            if not vision_api_key:
                logger.warning(
                    "Vision enrichment requested but no API key provided; "
                    "falling back to text-only enrichment."
                )
            else:
                self.vision_client = OpenAILikeClient(
                    api_key=vision_api_key,
                    model=vision_model,
                    base_url=vision_base_url,
                    cache=cache,
                )
                logger.info(
                    "Vision enrichment enabled: model=%s, base_url=%s",
                    vision_model, vision_base_url,
                )

        self.available = self._verify_text_connection()
    
    # ---- Connection / lifecycle ----

    def _verify_text_connection(self) -> bool:
        """Check Ollama is running and the configured model is available."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                logger.warning("Ollama returned status %d", resp.status_code)
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            base = self.ollama_model.split(":")[0]
            if not any(base in m for m in models):
                logger.warning(
                    "Model '%s' not in Ollama. Available: %s. "
                    "Run: ollama pull %s",
                    self.ollama_model, models, self.ollama_model,
                )
                return False
            logger.info(
                "Ollama connected, model '%s' available", self.ollama_model,
            )
            return True
        except requests.ConnectionError:
            logger.warning("Ollama not running at %s", self.ollama_url)
            return False
        except Exception as e:
            logger.warning("Ollama check failed: %s", e)
            return False

    def unload(self):
        """Tell Ollama to unload the model from VRAM."""
        try:
            requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_model, "keep_alive": 0},
                timeout=10,
            )
            logger.info(
                "Ollama model '%s' unloaded from VRAM", self.ollama_model,
            )
        except Exception as e:
            logger.warning("Failed to unload Ollama model: %s", e)

    # ---- Public entry point ----

    def enrich_chapter(self, chapter: Chapter) -> None:
        """Enrich a chapter with LLM-generated content (modifies in place).

        Populates:
        - chapter.intro
        - For each CodeBlock: cb.narration
        - For each Figure: fig.narration (vision-based if enabled, else text-only)
        - For each Table with row_count <= 6: tbl.narration
        - For each MathFormula: f.narration (vision-based if enabled, else text-only)
        """
        if not self.available:
            return

        logger.info(
            "Enriching chapter %d: %s", chapter.number, chapter.title,
        )

        chapter.intro = self._generate_intro(chapter)

        for cb in chapter.code_blocks:
            cb.narration = self._annotate_code(cb)

        for fig in chapter.figures:
            fig.narration = self._describe_figure(fig)

        for tbl in chapter.tables:
            if tbl.row_count <= 6:
                tbl.narration = self._narrate_table(tbl)

        for formula in chapter.math_formulas:
            formula.narration = self._describe_formula(formula)

        logger.info(
            "  Enriched: intro, %d code, %d figures, %d formulas, "
            "%d table narrations",
            len(chapter.code_blocks),
            len(chapter.figures),
            len(chapter.math_formulas),
            sum(1 for t in chapter.tables if t.narration),
        )

    # ---- Per-task methods ----

    def _generate_intro(self, chapter: Chapter) -> str:
        section_titles = [s.title for s in chapter.sections if s.level <= 2]
        sections_str = ", ".join(section_titles[:10]) if section_titles else "various topics"
        prompt = (
            f"Generate an audio introduction for this chapter:\n"
            f"Title: {chapter.title}\n"
            f"Sections covered: {sections_str}\n"
            f"Output just the introduction text, no labels."
        )
        response = self.text_client.invoke(
            input=prompt,
            system_prompt=INTRO_SYSTEM,
            temperature=0.3,
            max_tokens=512,
        )
        return response.text

    def _annotate_code(self, cb: CodeBlock) -> str:
        prompt = (
            f"Describe this code block in 1-2 sentences:\n"
            f"Section: {cb.context}\n"
            f"Language: {cb.language or 'unknown'}\n"
            f"Code:\n{cb.code[:1000]}"
        )
        response = self.text_client.invoke(
            input=prompt,
            system_prompt=CODE_SYSTEM,
            temperature=0.3,
            max_tokens=512,
        )
        return response.text

    def _narrate_table(self, table: Table) -> str:
        rows_text = ""
        if table.headers:
            rows_text += " | ".join(table.headers) + "\n"
        for row in table.rows[:10]:
            rows_text += " | ".join(row) + "\n"
        prompt = (
            f"Narrate this table for audio listening:\n"
            f"Caption: {table.caption}\n"
            f"Data:\n{rows_text[:2000]}"
        )
        response = self.text_client.invoke(
            input=prompt,
            system_prompt=TABLE_SYSTEM,
            temperature=0.3,
            max_tokens=300,
        )
        return response.text

    # ---- Figure: vision dispatch ----

    def _describe_figure(self, figure: Figure) -> str:
        if (
            self.vision_client
            and figure.rendered_path
            and figure.rendered_path.exists()
        ):
            return self._describe_figure_vision(figure)
        return self._describe_figure_text(figure)

    def _describe_figure_text(self, figure: Figure) -> str:
        prompt = (
            f"Describe this figure for an audio listener:\n"
            f"Label: {figure.label}\n"
            f"Alt text: {figure.alt}\n"
            f"Caption: {figure.caption}\n"
            f"Section: {figure.context}"
        )
        response = self.text_client.invoke(
            input=prompt,
            system_prompt=FIGURE_TEXT_SYSTEM,
            temperature=0.3,
            max_tokens=512,
        )
        return response.text

    def _describe_figure_vision(self, figure: Figure) -> str:
        text_block = TextBlock(content=(
            f"{FIGURE_VISION_PROMPT}\n\n"
            f"Figure label: {figure.label}\n"
            f"Caption: {figure.caption}\n"
            f"Alt text: {figure.alt}\n"
            f"Section: {figure.context}"
        ))
        media_block = MediaBlock(media=Media(
            media_type="image",
            source_type="path",
            source=str(figure.rendered_path),  # str() critical: see notebook bug
            extension="png",
        ))
        response = self.vision_client.invoke(
            input=[text_block, media_block],
            temperature=0.3,
            max_tokens=300,
        )
        return response.text

    # ---- Math formula: vision dispatch ----

    def _describe_formula(self, formula: MathFormula) -> str:
        if (
            self.vision_client
            and formula.rendered_path
            and formula.rendered_path.exists()
        ):
            return self._describe_formula_vision(formula)
        return self._describe_formula_text(formula)

    def _describe_formula_text(self, formula: MathFormula) -> str:
        prompt = (
            f"Describe this mathematical formula for an audio listener:\n"
            f"Section: {formula.context}\n"
            f"Formula (text): {formula.alttext}\n"
            f"MathML (raw): {formula.mathml[:300]}"
        )
        response = self.text_client.invoke(
            input=prompt,
            system_prompt=FORMULA_TEXT_SYSTEM,
            temperature=0.3,
            max_tokens=200,
        )
        return response.text

    def _describe_formula_vision(self, formula: MathFormula) -> str:
        text_block = TextBlock(content=(
            f"{FORMULA_VISION_PROMPT}\n\n"
            f"Section: {formula.context}\n"
            f"Alt text (if available): {formula.alttext}"
        ))
        media_block = MediaBlock(media=Media(
            media_type="image",
            source_type="path",
            source=str(formula.rendered_path),
            extension="png",
        ))
        response = self.vision_client.invoke(
            input=[text_block, media_block],
            temperature=0.3,
            max_tokens=200,
        )
        return response.text