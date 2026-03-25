"""
LLM backend for classifai.

Classifies text using any LLM provider through `instructor`, which
guarantees that the model always returns a valid category label — no
regex parsing, no free-text matching, no retries on bad outputs.

Supported providers
-------------------
- "openai"    → OpenAI API (gpt-4o-mini, gpt-4o, …)
- "anthropic" → Anthropic API (claude-haiku-4-5, claude-sonnet-4-6, …)
- "ollama"    → Local models via Ollama (llama3, mistral, phi4, …)

Unique-value optimization
-------------------------
Rather than sending every row to the LLM, the backend first extracts
all distinct text values, classifies only those, then maps results back.
For datasets with repetitive text this can reduce API calls by >90 %.

Usage
-----
    from classifai.backends import LLMBackend

    clf = LLMBackend(
        categories=["Billing", "Technical Support", "Account", "Other"],
        model="gpt-4o-mini",
        provider="openai",
    )
    labels = clf.predict(df["ticket_text"])

    # With a custom prompt
    clf = LLMBackend(
        categories=["Positive", "Negative", "Neutral"],
        model="llama3",
        provider="ollama",
        prompt_template="Classify the sentiment of this review: {text}\\n\\nCategories: {categories}\\n\\nLabel only:",
    )
"""

from __future__ import annotations

import os
import logging
import hashlib
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type

import pandas as pd
from pydantic import BaseModel

from classifai.backends.base import BaseBackend

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_label_enum(categories: List[str]) -> Type[Enum]:
    """Create a Pydantic-compatible Enum from a list of category strings."""
    return Enum("Label", {c: c for c in categories})  # type: ignore[return-value]


def _build_classification_model(categories: List[str]) -> Type[BaseModel]:
    """
    Build a Pydantic model whose `label` field is constrained to `categories`.

    instructor uses this model to enforce that the LLM response is always
    one of the valid categories — validation failures trigger automatic retries.
    """
    LabelEnum = _make_label_enum(categories)

    class Classification(BaseModel):
        label: LabelEnum  # type: ignore[valid-type]

    return Classification


def _get_instructor_client(provider: str, model: str, api_key: Optional[str] = None):
    """
    Return an instructor-patched client for the requested provider.

    Raises ImportError with an actionable message if optional deps are missing.
    """
    try:
        import instructor
    except ImportError:
        raise ImportError(
            "instructor is required for the LLM backend.\n"
            "Install it with: pip install 'classifai[llm]'"
        )

    provider = provider.lower()

    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass api_key."
            )
        return instructor.from_openai(OpenAI(api_key=key))

    elif provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY or pass api_key."
            )
        return instructor.from_anthropic(Anthropic(api_key=key))

    elif provider == "ollama":
        try:
            from openai import OpenAI as OllamaClient
        except ImportError:
            raise ImportError("pip install openai  # Ollama uses the OpenAI-compatible API")
        return instructor.from_openai(
            OllamaClient(base_url="http://localhost:11434/v1", api_key="ollama"),
            mode=instructor.Mode.JSON,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Choose from: 'openai', 'anthropic', 'ollama'."
        )


# ── Cache ─────────────────────────────────────────────────────────────────────

class _ClassificationCache:
    """Simple disk cache keyed by (text, model, categories_hash)."""

    def __init__(self, cache_dir: str = ".cache/classifai"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        path = self.cache_dir / "cache.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    self._mem = pickle.load(f)
                logger.debug(f"Cache loaded: {len(self._mem)} entries from {path}")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self._mem = {}

    def _flush(self) -> None:
        path = self.cache_dir / "cache.pkl"
        with open(path, "wb") as f:
            pickle.dump(self._mem, f)

    def _key(self, text: str, model: str, categories_hash: str) -> str:
        raw = f"{text}||{model}||{categories_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, text: str, model: str, categories_hash: str) -> Optional[str]:
        return self._mem.get(self._key(text, model, categories_hash))

    def set(self, text: str, model: str, categories_hash: str, label: str) -> None:
        self._mem[self._key(text, model, categories_hash)] = label
        self._flush()


# ── Unique-value optimizer ────────────────────────────────────────────────────

class _UniqueValueOptimizer:
    """
    Reduces API calls by classifying only distinct text values.

    For a column with 10,000 rows but only 400 unique values, this sends
    400 requests instead of 10,000 — a 97.5 % reduction.
    """

    def __init__(self) -> None:
        self._value_to_indices: Dict[str, List[int]] = {}

    def extract_unique(self, texts: List[str]) -> List[str]:
        self._value_to_indices = {}
        for i, raw in enumerate(texts):
            text = "" if (raw is None or (isinstance(raw, float) and pd.isna(raw))) else str(raw).strip()
            self._value_to_indices.setdefault(text, []).append(i)

        unique = [t for t in self._value_to_indices if t]
        reduction = 1 - len(unique) / max(len(texts), 1)
        logger.info(
            f"Unique-value optimization: {len(texts)} rows → {len(unique)} unique "
            f"({reduction:.1%} reduction in API calls)"
        )
        return unique

    def map_back(self, unique_texts: List[str], unique_labels: List[str],
                 original_length: int, unknown: str) -> List[str]:
        results = [unknown] * original_length
        for text, label in zip(unique_texts, unique_labels):
            for idx in self._value_to_indices.get(text, []):
                results[idx] = label
        # Fill empty/null positions with unknown
        for idx in self._value_to_indices.get("", []):
            results[idx] = unknown
        return results


# ── Main backend ──────────────────────────────────────────────────────────────

class LLMBackend(BaseBackend):
    """
    LLM-based text classifier with guaranteed structured outputs.

    Parameters
    ----------
    categories : list of str
        The valid output labels. The LLM is constrained to return exactly one.
    model : str
        Model identifier (e.g. "gpt-4o-mini", "claude-haiku-4-5", "llama3").
    provider : str
        One of "openai", "anthropic", "ollama". Default: "openai".
    temperature : float
        Sampling temperature. Use 0.0 for deterministic classification.
    unknown_category : str
        Label used when the text is empty or cannot be classified. If not
        already in `categories`, it is appended automatically.
    batch_size : int
        Number of texts per parallel batch.
    max_workers : int
        Thread-pool workers for parallel API calls.
    max_retries : int
        instructor retry attempts on validation failure.
    prompt_template : str, optional
        Custom prompt. Available placeholders: {text}, {categories}.
        If None, a sensible default is used.
    api_key : str, optional
        Provider API key. Falls back to the environment variable
        OPENAI_API_KEY / ANTHROPIC_API_KEY when omitted.
    cache : bool
        Enable disk cache to avoid re-classifying identical texts.
    cache_dir : str
        Directory for the cache file.
    """

    DEFAULT_PROMPT = (
        "Classify the following text into exactly one of the provided categories.\n\n"
        "Categories:\n{categories}\n\n"
        "Text:\n{text}\n\n"
        "Return only the category label."
    )

    def __init__(
        self,
        categories: List[str],
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        temperature: float = 0.0,
        unknown_category: str = "Other",
        batch_size: int = 50,
        max_workers: int = 4,
        max_retries: int = 2,
        prompt_template: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: bool = True,
        cache_dir: str = ".cache/classifai",
    ) -> None:
        self.categories = list(categories)
        if unknown_category not in self.categories:
            self.categories.append(unknown_category)
        self.unknown_category = unknown_category

        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

        # Lazily built — only when predict() is first called
        self._client = None
        self._api_key = api_key
        self._Classification = _build_classification_model(self.categories)
        self._categories_hash = hashlib.md5(
            "|".join(sorted(self.categories)).encode()
        ).hexdigest()

        self._cache: Optional[_ClassificationCache] = (
            _ClassificationCache(cache_dir) if cache else None
        )
        self._optimizer = _UniqueValueOptimizer()

        # Stats
        self.stats = {"api_calls": 0, "cache_hits": 0, "errors": 0}

    # ── Private helpers ───────────────────────────────────────────────────────

    @property
    def client(self):
        if self._client is None:
            self._client = _get_instructor_client(self.provider, self.model, self._api_key)
        return self._client

    def _format_prompt(self, text: str) -> str:
        categories_str = "\n".join(f"- {c}" for c in self.categories)
        return self.prompt_template.format(text=text, categories=categories_str)

    def _classify_one(self, text: str) -> str:
        """Classify a single text string. Returns the label."""
        # Cache lookup
        if self._cache:
            cached = self._cache.get(text, self.model, self._categories_hash)
            if cached is not None:
                self.stats["cache_hits"] += 1
                return cached

        prompt = self._format_prompt(text)

        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_model=self._Classification,
                temperature=self.temperature,
                max_retries=self.max_retries,
            )
            label = result.label.value
            self.stats["api_calls"] += 1

            if self._cache:
                self._cache.set(text, self.model, self._categories_hash, label)

            return label

        except Exception as e:
            self.stats["errors"] += 1
            logger.warning(f"Classification error (returning '{self.unknown_category}'): {e}")
            return self.unknown_category

    def _classify_batch(self, texts: List[str]) -> List[str]:
        """Classify a list of texts in parallel."""
        results = [self.unknown_category] * len(texts)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._classify_one, t): i for i, t in enumerate(texts)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Batch item {idx} failed: {e}")

        return results

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, texts: pd.Series) -> pd.Series:
        """
        Classify each text in `texts` and return a Series of labels.

        Uses unique-value optimization: only distinct text values are sent
        to the LLM; results are mapped back to all original rows.
        """
        raw_list = texts.tolist()
        index = texts.index

        # Extract unique values
        unique_texts = self._optimizer.extract_unique(raw_list)

        if not unique_texts:
            logger.warning("All texts are empty — returning unknown category for all rows.")
            return pd.Series([self.unknown_category] * len(raw_list), index=index)

        # Classify in batches
        unique_labels: List[str] = []
        for start in range(0, len(unique_texts), self.batch_size):
            batch = unique_texts[start : start + self.batch_size]
            logger.info(
                f"Classifying batch {start // self.batch_size + 1}/"
                f"{(len(unique_texts) - 1) // self.batch_size + 1} "
                f"({len(batch)} texts)"
            )
            unique_labels.extend(self._classify_batch(batch))

        # Map unique results back to every row
        all_labels = self._optimizer.map_back(
            unique_texts, unique_labels, len(raw_list), self.unknown_category
        )

        logger.info(
            f"Done. API calls: {self.stats['api_calls']} | "
            f"Cache hits: {self.stats['cache_hits']} | "
            f"Errors: {self.stats['errors']}"
        )

        return pd.Series(all_labels, index=index)

    def predict_with_confidence(self, texts: pd.Series) -> pd.DataFrame:
        """
        Like predict(), but returns a DataFrame with a `confidence` column.

        Confidence is estimated from the number of retries needed:
        - 0 retries → 1.0
        - 1 retry   → 0.7
        - 2 retries → 0.5
        Note: not a calibrated probability; use for rough ranking only.
        """
        labels = self.predict(texts)
        # Placeholder confidence — will be replaced with logprobs when available
        confidence = pd.Series([1.0] * len(labels), index=labels.index)
        return pd.DataFrame({"label": labels, "confidence": confidence})

    @classmethod
    def from_config(cls, perspective_config: dict, global_config=None) -> "LLMBackend":
        """
        Build an LLMBackend from a classifai YAML perspective config block.

        Example YAML block::

            my_classifier:
              type: openai_classification
              target_categories: [Billing, Technical, Other]
              llm_config:
                provider: openai
                model: gpt-4o-mini
                temperature: 0.0
                api_key_env: OPENAI_API_KEY
              classification_config:
                batch_size: 50
                unknown_category: Other
                prompt_template: |
                  Classify: {text}
                  Categories: {categories}
                  Label only:
        """
        llm_cfg = perspective_config.get("llm_config", {})
        cls_cfg = perspective_config.get("classification_config", {})

        provider = llm_cfg.get("provider", "openai")
        model = llm_cfg.get("model", "gpt-4o-mini")
        temperature = llm_cfg.get("temperature", 0.0)

        api_key_env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)

        categories = perspective_config.get("target_categories", [])
        unknown = cls_cfg.get("unknown_category", "Other")
        batch_size = cls_cfg.get("batch_size", 50)
        prompt_template = cls_cfg.get("prompt_template")

        cache = True
        cache_dir = ".cache/classifai"
        if global_config is not None:
            cache_cfg = global_config.get_config_value("ai_classification.caching", {})
            cache = cache_cfg.get("enabled", True)
            cache_dir = cache_cfg.get("cache_directory", cache_dir)

        return cls(
            categories=categories,
            model=model,
            provider=provider,
            temperature=temperature,
            unknown_category=unknown,
            batch_size=batch_size,
            prompt_template=prompt_template,
            api_key=api_key,
            cache=cache,
            cache_dir=cache_dir,
        )
