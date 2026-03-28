"""LLM enrichment using Qwen3-8B via Ollama for audiobook enhancement.

Generates chapter intros, code annotations, table narrations, figure descriptions,
and rewrites dense paragraphs for better audio listening experience.
"""

import hashlib
import json
import logging
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

from pipeline.extractor import Chapter

logger = logging.getLogger(__name__)


def _flesch_score(text: str) -> float:
    """Calculate Flesch Reading Ease score (approximate).

    Higher = easier to read. < 40 = very dense/academic.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 100.0

    words = text.split()
    if not words:
        return 100.0

    # Approximate syllable count
    syllable_count = 0
    for word in words:
        word = word.lower().strip(".,;:!?\"'()-")
        if not word:
            continue
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in "aeiou"
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        syllable_count += max(count, 1)

    num_sentences = len(sentences)
    num_words = len(words)

    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (syllable_count / num_words)
    return score


class LLMEnricher:
    """Enriches chapter content using a local LLM via Ollama."""

    def __init__(self, model: str, ollama_url: str, cache_dir: Path):
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.available = self._verify_connection()

    def _verify_connection(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                logger.warning(f"Ollama returned status {resp.status_code}")
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check if model (with or without tag) is available
            model_base = self.model.split(":")[0]
            found = any(model_base in m for m in models)
            if not found:
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available: {models}. Run: ollama pull {self.model}"
                )
                return False
            logger.info(f"Ollama connected, model '{self.model}' available")
            return True
        except requests.ConnectionError:
            logger.warning("Ollama not running at {self.ollama_url}")
            return False
        except Exception as e:
            logger.warning(f"Ollama check failed: {e}")
            return False

    def _cache_key(self, prompt: str, system: str = "") -> str:
        """Generate a cache key from model + prompt."""
        content = f"{self.model}:{system}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _cached_call(self, prompt: str, system: str = "", max_tokens: int = 512) -> str:
        """Call LLM with file-based caching."""
        key = self._cache_key(prompt, system)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                return data["response"]
            except (json.JSONDecodeError, KeyError):
                pass

        response = self._call_ollama(prompt, system, max_tokens)

        # Cache the result
        cache_file.write_text(json.dumps({
            "model": self.model,
            "prompt": prompt[:200],
            "system": system[:100],
            "response": response,
        }, ensure_ascii=False))

        return response

    def _call_ollama(
        self, prompt: str, system: str = "", max_tokens: int = 512, retries: int = 3
    ) -> str:
        """Call Ollama API with retry logic."""
        for attempt in range(retries):
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": max_tokens,
                    },
                }
                if system:
                    payload["system"] = system

                resp = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                return resp.json()["response"].strip()
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Ollama call failed (attempt {attempt + 1}): {e}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"Ollama call failed after {retries} attempts: {e}")
                    return ""

    def enrich_chapter(self, chapter: Chapter) -> None:
        """Enrich a chapter with LLM-generated content (modifies in place)."""
        if not self.available:
            return

        logger.info(f"Enriching chapter {chapter.number}: {chapter.title}")

        # 1. Generate chapter intro
        chapter.intro = self._generate_intro(chapter)

        # 2. Annotate code blocks
        for cb in tqdm(chapter.code_blocks, desc=f"  Ch{chapter.number} code", leave=False):
            cb.annotation = self._annotate_code(cb)

        # 3. Describe figures
        for fig in tqdm(chapter.figures, desc=f"  Ch{chapter.number} figures", leave=False):
            chapter.figure_descriptions[fig.number] = self._describe_figure(fig)

        # 4. Narrate small tables
        for table in chapter.tables:
            if table.row_count <= 6:
                table.narration = self._narrate_table(table)

        logger.info(
            f"  Enriched: intro, {len(chapter.code_blocks)} code annotations, "
            f"{len(chapter.figures)} figure descriptions, "
            f"{sum(1 for t in chapter.tables if t.narration)} table narrations"
        )

    def _generate_intro(self, chapter: Chapter) -> str:
        """Generate a 2-3 sentence audio introduction for a chapter."""
        section_titles = [s.title for s in chapter.sections if s.level <= 2]
        sections_str = ", ".join(section_titles[:10]) if section_titles else "various topics"

        system = (
            "You are a technical book narrator. Generate a 2-3 sentence introduction "
            "for an audiobook chapter that tells the listener what they will learn. "
            "Be concise and engaging."
        )
        prompt = (
            f"Generate an audio introduction for this chapter:\n"
            f"Title: {chapter.title}\n"
            f"Sections covered: {sections_str}\n"
            f"Output just the introduction text, no labels."
        )
        return self._cached_call(prompt, system)

    def _annotate_code(self, code_block) -> str:
        """Generate a 1-2 sentence description of a code block."""
        system = (
            "You are a technical book narrator. Write a 1-2 sentence description "
            "of what this code does, suitable for someone listening to an audiobook. "
            "Be concise."
        )
        prompt = (
            f"Describe this code block in 1-2 sentences:\n"
            f"Section: {code_block.context}\n"
            f"Language: {code_block.language or 'unknown'}\n"
            f"Code:\n{code_block.code[:1000]}"
        )
        return self._cached_call(prompt, system)

    def _describe_figure(self, figure) -> str:
        """Generate a 1-2 sentence audio description of a figure."""
        system = (
            "You are narrating a technical book. Based on the caption/alt text and "
            "section context, generate a 1-2 sentence audio description of what "
            "this figure likely shows. Be concise."
        )
        prompt = (
            f"Describe this figure for an audio listener:\n"
            f"Label: {figure.label}\n"
            f"Alt text: {figure.alt}\n"
            f"Caption: {figure.caption}\n"
            f"Section: {figure.context}"
        )
        return self._cached_call(prompt, system)

    def _narrate_table(self, table) -> str:
        """Convert a small table into a natural language narration."""
        system = (
            "Convert this table data into a natural language description suitable "
            "for audio listening. Be concise but include the key data points."
        )

        # Build a text representation
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
        return self._cached_call(prompt, system, max_tokens=300)

    def simplify_paragraph(self, text: str) -> str:
        """Rewrite a dense paragraph for audio listening (Flesch < 40)."""
        if _flesch_score(text) >= 40:
            return text

        system = (
            "Rewrite this paragraph for audio listening. Keep ALL technical content "
            "but use shorter sentences, explicit transitions, and repeat key terms "
            "instead of pronouns. Do NOT summarize — maintain 100% of the information."
        )
        result = self._cached_call(text, system, max_tokens=1024)
        return result if result else text

    def unload(self):
        """Tell Ollama to unload the model from VRAM."""
        try:
            requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=10,
            )
            logger.info(f"Ollama model '{self.model}' unloaded from VRAM")
        except Exception as e:
            logger.warning(f"Failed to unload Ollama model: {e}")
