from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_DICTIONARY_PATH = Path("config/translation_dictionary.json")


class DictionaryError(ValueError):
    """Raised when the terminology configuration is invalid."""


@dataclass(frozen=True)
class TranslationDictionary:
    skip_translation: tuple[str, ...]
    keep_english: tuple[str, ...]
    fixed_translation: dict[str, str]

    @classmethod
    def load(cls, path: Path = DEFAULT_DICTIONARY_PATH) -> TranslationDictionary:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DictionaryError(f"Unable to read translation dictionary '{path}': {error}") from error
        if not isinstance(payload, dict):
            raise DictionaryError("Translation dictionary must be a JSON object")
        required = {"skip_translation", "keep_english", "fixed_translation"}
        if set(payload) != required:
            raise DictionaryError(f"Dictionary keys must be exactly: {sorted(required)}")
        skip = _terms(payload["skip_translation"], "skip_translation")
        keep = _terms(payload["keep_english"], "keep_english")
        fixed_value = payload["fixed_translation"]
        if not isinstance(fixed_value, dict):
            raise DictionaryError("fixed_translation must be an object")
        fixed: dict[str, str] = {}
        for term, translation in fixed_value.items():
            if not isinstance(term, str) or not term.strip():
                raise DictionaryError("fixed_translation terms must be non-empty strings")
            if not isinstance(translation, str) or not translation.strip():
                raise DictionaryError(f"Fixed translation for {term!r} must be a non-empty string")
            fixed[term] = translation
        _validate_conflicts(skip, keep, fixed)
        return cls(tuple(skip), tuple(keep), fixed)

    def skip_matches(self, source: str) -> bool:
        return source in self.skip_translation

    def matching_terms(self, source: str, category: str) -> list[str]:
        terms: Iterable[str]
        if category == "keep_english":
            terms = self.keep_english
        elif category == "fixed_translation":
            terms = self.fixed_translation.keys()
        else:
            raise DictionaryError(f"Unknown terminology category: {category}")
        return _matching_phrases(source, terms)

    def context(self) -> str:
        lines = ["TERMINOLOGY RULES", "KEEP IN ENGLISH:"]
        lines.extend(f"- {term}" for term in sorted(self.keep_english, key=len, reverse=True))
        lines.append("FIXED TRANSLATIONS:")
        lines.extend(
            f"- {term} -> {self.fixed_translation[term]}"
            for term in sorted(self.fixed_translation, key=len, reverse=True)
        )
        lines.extend([
            "Rules:",
            "- Keep KEEP IN ENGLISH terms unchanged.",
            "- Use the specified translation for FIXED TRANSLATIONS.",
            "- Do not invent alternative terminology.",
            "- Translate the surrounding sentence naturally.",
        ])
        return "\n".join(lines)


def _terms(value: object, category: str) -> list[str]:
    if not isinstance(value, list):
        raise DictionaryError(f"{category} must be an array")
    result: list[str] = []
    for term in value:
        if not isinstance(term, str) or not term.strip():
            raise DictionaryError(f"{category} terms must be non-empty strings")
        if term not in result:
            result.append(term)
    return result


def _validate_conflicts(skip: list[str], keep: list[str], fixed: dict[str, str]) -> None:
    categories = {
        "skip_translation": {term.casefold() for term in skip},
        "keep_english": {term.casefold() for term in keep},
        "fixed_translation": {term.casefold() for term in fixed},
    }
    names = list(categories)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            overlap = sorted(categories[first] & categories[second])
            if overlap:
                raise DictionaryError(f"Terminology conflict between {first} and {second}: {overlap}")


def _matching_phrases(source: str, terms: Iterable[str]) -> list[str]:
    matches: list[tuple[int, int, str]] = []
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(source):
            if not any(match.start() < end and match.end() > start for start, end, _ in matches):
                matches.append((match.start(), match.end(), term))
    matches.sort(key=lambda item: item[0])
    return [term for _, _, term in matches]
