from __future__ import annotations

import argparse
import json
import logging
import os
from dotenv import load_dotenv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

from .filters import should_translate

LOGGER = logging.getLogger(__name__)
DEFAULT_EXTRACTION_PATH = Path("artifacts/extraction/extraction.json")
DEFAULT_OUTPUT_PATH = Path("artifacts/translation/translation.json")
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_RETRIES = 2

SYSTEM_PROMPT = """You are a technical document translator.
Translate English to professional technical Indonesian.
Preserve meaning exactly. Do not summarize, explain, add, remove, or rewrite unnecessarily.
Preserve technical terminology, acronyms, identifiers, product names, filenames, paths,
IP addresses, numbers, units, parameter names, URLs, symbols, and code-like values.
Do not translate part numbers, document IDs, URLs, or values that should remain unchanged.
The input units are already logically grouped. Do not split, merge, create, or delete units.
Return ONLY valid JSON in exactly this shape:
{"translations":[{"id":1,"source":"exact input source","translation":"Indonesian translation"}]}
The source field is an opaque immutable string copied from the input JSON. Copy it character-for-character,
including whitespace, punctuation, and incomplete-looking endings. Some source strings may intentionally end
mid-sentence because the PDF extractor split adjacent blocks; never complete, normalize, or infer source text.
Every input ID must appear exactly once.
Do not output Chinese or commentary outside the JSON."""


class TranslationClient(Protocol):
    def chat_completion(self, messages: list[dict[str, str]], model: str) -> str:
        ...


class TranslationError(RuntimeError):
    """Raised when a translation batch cannot produce a validated result."""


class BatchValidationError(TranslationError):
    """Raised when a model response violates the batch contract."""


class TranslationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    source: str
    translation: str = Field(min_length=1)


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    translations: list[TranslationItem]


@dataclass(frozen=True)
class TranslationConfig:
    base_url: str
    api_key: str
    model: str
    batch_size: int = DEFAULT_BATCH_SIZE
    max_retries: int = DEFAULT_MAX_RETRIES

    @classmethod
    def from_environment(cls) -> TranslationConfig:
        base_url = os.getenv("LLM_BASE_URL", "").strip()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("MODEL", "").strip()
        if not base_url:
            raise TranslationError("LLM_BASE_URL is required")
        if not api_key:
            raise TranslationError("OPENAI_API_KEY is required")
        if not model:
            raise TranslationError("MODEL is required")
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            batch_size=_positive_environment_int("TRANSLATION_BATCH_SIZE", DEFAULT_BATCH_SIZE),
            max_retries=_nonnegative_environment_int("TRANSLATION_MAX_RETRIES", DEFAULT_MAX_RETRIES),
        )


def _positive_environment_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise TranslationError(f"{name} must be an integer") from error
    if parsed < 1:
        raise TranslationError(f"{name} must be at least 1")
    return parsed


def _nonnegative_environment_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise TranslationError(f"{name} must be an integer") from error
    if parsed < 0:
        raise TranslationError(f"{name} must be non-negative")
    return parsed


def load_translation_units(extraction_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = json.loads(extraction_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TranslationError(f"Unable to read extraction JSON '{extraction_path}': {error}") from error
    units = [
        {"id": unit["id"], "unit_type": unit["unit_type"], "source": unit["source"]}
        for page in payload.get("pages", [])
        for unit in page.get("units", [])
        if unit.get("translate") is True
    ]
    return payload, units


def extract_json_response(response: str) -> dict[str, Any]:
    candidate = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = candidate.find("{")
        if start < 0:
            raise BatchValidationError("Model response does not contain a JSON object")
        try:
            parsed, _ = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError as error:
            raise BatchValidationError(f"Invalid model JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise BatchValidationError("Model response must be a JSON object")
    return parsed


def _clearly_translatable(source: str) -> bool:
    words = re.findall(r"[A-Za-z]+", source)
    return len(words) >= 2 and any(any(character.islower() for character in word) for word in words)


def _validate_translation_item(item: Any, expected: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise BatchValidationError("Each translation must be an object")
    if set(item) != {"id", "source", "translation"}:
        raise BatchValidationError("Each translation must contain only id, source, and translation")
    unit_id = item.get("id")
    if not isinstance(unit_id, int) or isinstance(unit_id, bool):
        raise BatchValidationError(f"Invalid translation ID: {unit_id!r}")
    if unit_id not in expected:
        raise BatchValidationError(f"Unexpected translation ID: {unit_id}")
    if item["source"] != expected[unit_id]["source"]:
        raise BatchValidationError(f"Source mismatch for ID {unit_id}")
    translation = item["translation"]
    if not isinstance(translation, str) or not translation.strip():
        raise BatchValidationError(f"Empty translation for ID {unit_id}")
    if translation == item["source"] and _clearly_translatable(item["source"]):
        raise BatchValidationError(f"Translation is unchanged for clearly translatable ID {unit_id}")
    return item


def validate_response(payload: dict[str, Any], expected_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_translations = payload.get("translations")
    if not isinstance(raw_translations, list):
        raise BatchValidationError("Response must contain a translations list")
    expected = {unit["id"]: unit for unit in expected_units}
    returned_ids = [item.get("id") if isinstance(item, dict) else None for item in raw_translations]
    duplicate_ids = sorted({unit_id for unit_id in returned_ids if returned_ids.count(unit_id) > 1})
    if duplicate_ids:
        raise BatchValidationError(f"Duplicate translation IDs: {duplicate_ids}")
    validated = [_validate_translation_item(item, expected) for item in raw_translations]
    expected_ids = set(expected)
    returned_id_set = set(returned_ids)
    missing = sorted(expected_ids - returned_id_set)
    unexpected = sorted(returned_id_set - expected_ids)
    if missing:
        raise BatchValidationError(f"Missing translation IDs: {missing}")
    if unexpected:
        raise BatchValidationError(f"Unexpected translation IDs: {unexpected}")
    return validated


def _request_payload(units: list[dict[str, Any]]) -> str:
    return json.dumps(units, ensure_ascii=False, indent=2)


def translate_batches(
    units: list[dict[str, Any]], client: TranslationClient, model: str,
    batch_size: int = DEFAULT_BATCH_SIZE, max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise TranslationError("batch_size must be at least 1")
    batches = [units[index:index + batch_size] for index in range(0, len(units), batch_size)]
    results: list[dict[str, Any]] = []
    for batch_number, batch in enumerate(batches, start=1):
        ids = [unit["id"] for unit in batch]
        print(f"Batch {batch_number}/{len(batches)}")
        print(f"IDs: {ids[0]}-{ids[-1]}")
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _request_payload(batch)},
            ]
            if last_error is not None:
                messages.append({
                    "role": "user",
                    "content": (
                        "Correction for the previous response: return the source field exactly as provided "
                        "in the input JSON. Do not normalize whitespace or punctuation. The previous response "
                        f"failed validation: {last_error}. Re-emit the complete batch JSON."
                    ),
                })
            try:
                response = client.chat_completion(messages, model)
                validated = validate_response(extract_json_response(response), batch)
                results.extend(validated)
                print("Status: OK")
                break
            except (BatchValidationError, ValueError, TypeError) as error:
                last_error = error
                LOGGER.warning("Batch %s failed on attempt %s: %s", batch_number, attempt + 1, error)
                LOGGER.debug("Batch %s response was: %s", batch_number, locals().get("response", "<no response>"))
        else:
            raise TranslationError(
                f"Batch {batch_number} failed after {max_retries + 1} attempts; IDs={ids}: {last_error}"
            ) from last_error
    return validate_response({"translations": results}, units)


def save_translation(
    output_path: Path, extraction_path: Path, extraction_payload: dict[str, Any],
    translations: list[dict[str, Any]], model: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "source_file": extraction_payload.get("source_file", extraction_path.name),
        "source_extraction": str(extraction_path),
        "source_language": "English",
        "target_language": "Indonesian",
        "model": model,
        "translations": translations,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


class OpenAITranslationClient:
    def __init__(self, base_url: str, api_key: str):
        try:
            from openai import OpenAI
        except ImportError as error:
            raise TranslationError("The openai package is required for translation") from error
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat_completion(self, messages: list[dict[str, str]], model: str) -> str:
        response = self.client.responses.parse(
            model=model,
            input=messages,
            text_format=LLMResponse,
            max_output_tokens=2048,
            temperature=0,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise TranslationError("Model returned no structured response")
        return parsed.model_dump_json()


def run_translation(
    extraction_path: Path = DEFAULT_EXTRACTION_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    config: TranslationConfig | None = None,
    client: TranslationClient | None = None,
) -> Path:
    config = config or TranslationConfig.from_environment()
    extraction_payload, units = load_translation_units(extraction_path)
    active_client = client or OpenAITranslationClient(config.base_url, config.api_key)
    print("Translation\n-----------")
    print(f"Input: {extraction_path}")
    print(f"Units to translate: {len(units)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Batches: {(len(units) + config.batch_size - 1) // config.batch_size}")
    print(f"Model: {config.model}")
    translations = translate_batches(units, active_client, config.model, config.batch_size, config.max_retries)
    print("\nValidation\n----------")
    print(f"Expected translations: {len(units)}")
    print(f"Returned translations: {len(translations)}")
    print("Missing IDs: 0\nUnexpected IDs: 0\nDuplicate IDs: 0\nSource mismatches: 0")
    save_translation(output_path, extraction_path, extraction_payload, translations, config.model)
    print(f"\nOutput: {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate extracted PDF units using a local OpenAI-compatible model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_EXTRACTION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    try:
        run_translation(args.input, args.output)
    except TranslationError as error:
        print(f"Translation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
