from __future__ import annotations

import argparse
import json
import logging
import os
from dotenv import load_dotenv
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

from .filters import should_translate
from .dictionary import DEFAULT_DICTIONARY_PATH, TranslationDictionary
from .identity import payload_sha256

LOGGER = logging.getLogger(__name__)
DEFAULT_EXTRACTION_PATH = Path("artifacts/extraction/extraction.json")
DEFAULT_OUTPUT_PATH = Path("artifacts/translation/translation.json")
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_RETRIES = 2
MAX_OUTPUT_TOKENS = 10000

SYSTEM_PROMPT = """You are a technical document translator.
Translate English to professional technical Indonesian.
Preserve meaning exactly. Do not summarize, explain, add, remove, or rewrite unnecessarily.
Preserve technical terminology, acronyms, identifiers, product names, filenames, paths,
IP addresses, numbers, units, parameter names, URLs, symbols, and code-like values.
Do not translate part numbers, document IDs, URLs, or values that should remain unchanged.
The input units are already logically grouped. Do not split, merge, create, or delete units.
Return ONLY valid JSON in exactly this shape:
{"translations":[{"id":1,"translation":"Indonesian translation"}]}
Do not return a source field. The source text is an opaque immutable value supplied in the input JSON;
the caller will associate it with the returned ID. Some source strings may intentionally end mid-sentence
because the PDF extractor split adjacent blocks; never complete, normalize, or infer source text.
Every input ID must appear exactly once.
For normal English words and sentences, you MUST actually translate them to Indonesian.
Do not copy the source text unchanged unless it is an acronym, identifier, product name, code-like value,
or a term explicitly protected by the terminology context.
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
    translation: str 


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


def load_translation_units(
    extraction_path: Path, dictionary: TranslationDictionary | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = json.loads(extraction_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TranslationError(f"Unable to read extraction JSON '{extraction_path}': {error}") from error
    units = [
        {"id": unit["id"], "unit_type": unit["unit_type"], "source": unit["source"]}
        for page in payload.get("pages", [])
        for unit in page.get("units", [])
        if unit.get("translate") is True
        or (dictionary is not None and dictionary.skip_matches(unit.get("source", "")))
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
    if set(item) not in ({"id", "translation"}, {"id", "source", "translation"}):
        raise BatchValidationError("Each translation must contain id, translation, and optional source only")
    unit_id = item.get("id")
    if not isinstance(unit_id, int) or isinstance(unit_id, bool):
        raise BatchValidationError(f"Invalid translation ID: {unit_id!r}")
    if unit_id not in expected:
        raise BatchValidationError(f"Unexpected translation ID: {unit_id}")
    if "source" in item and item["source"] != expected[unit_id]["source"]:
        raise BatchValidationError(f"Source mismatch for ID {unit_id}")
    translation = item["translation"]
    source = expected[unit_id]["source"]
    if not isinstance(translation, str):
        raise BatchValidationError(f"Translation for ID {unit_id} must be a string")
    if not translation.strip():
        LOGGER.warning("Unit %s returned an empty translation; using source text", unit_id)
        translation = source
    return {
        "id": unit_id,
        "source": source,
        "translation": translation,
    }


def validate_response(payload: dict[str, Any], expected_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if set(payload) != {"translations"}:
        raise BatchValidationError("Response must contain only the translations field")
    raw_translations = payload.get("translations")
    if not isinstance(raw_translations, list):
        raise BatchValidationError("Response must contain a translations list")
    expected_ids_list = [unit.get("id") for unit in expected_units]
    if any(not isinstance(unit_id, int) or isinstance(unit_id, bool) for unit_id in expected_ids_list):
        raise BatchValidationError("Expected unit IDs must be integers")
    if len(expected_ids_list) != len(set(expected_ids_list)):
        raise BatchValidationError("Expected unit IDs must be unique")
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


def _plain_text(value: str) -> str:
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    return re.sub(r"</?(?:b|i|strong|em)>", "", value, flags=re.IGNORECASE)


@dataclass(frozen=True)
class SanitizationResult:
    text: str
    issues: tuple[str, ...] = ()


_META_PREFIX = re.compile(
    r"^\s*(?:here is (?:the )?translation|translation|terjemahan|certainly|as an ai)\s*[:,-]",
    re.IGNORECASE,
)
_META_PARENTHETICAL = re.compile(
    r"\((?:no changes? (?:are )?needed|tidak ada perubahan|as an ai|karena kalimat ini)[^)]*\)",
    re.IGNORECASE,
)


def _private_use_characters(text: str) -> tuple[str, ...]:
    return tuple(character for character in text if unicodedata.category(character) == "Co")


def _normalized_sentences(text: str) -> list[str]:
    return [
        " ".join(part.casefold().split())
        for part in re.split(r"(?<=[.!?])\s+|\n+", text)
        if len(" ".join(part.split())) >= 20
    ]


def sanitize_translation(source: str, translation: str) -> SanitizationResult:
    if not translation.strip():
        return SanitizationResult(source)
    issues: list[str] = []
    if _META_PREFIX.search(translation) or _META_PARENTHETICAL.search(translation):
        issues.append("model commentary")
    source_sentences = _normalized_sentences(source)
    translated_sentences = _normalized_sentences(translation)
    for left, right in zip(translated_sentences, translated_sentences[1:]):
        if left == right and not any(
            first == second == left for first, second in zip(source_sentences, source_sentences[1:])
        ):
            issues.append("repeated sentence")
            break
    compact_source = "".join(source.split())
    compact_translation = "".join(translation.split())
    if len(compact_translation) > max(len(compact_source) * 4, len(compact_source) + 160):
        issues.append("extreme unexplained length growth")
    if _private_use_characters(source) != _private_use_characters(translation):
        issues.append("private-use glyph sequence changed")
    return SanitizationResult(translation, tuple(issues))


def _normalized_source(source: str) -> str:
    source = unicodedata.normalize("NFC", source).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(" ".join(line.split()) for line in source.splitlines()).strip()


def _translation_key(
    unit: dict[str, Any], dictionary: TranslationDictionary | None,
    source_language: str, target_language: str,
) -> tuple[str, str, str, str, str]:
    return (
        source_language,
        target_language,
        dictionary.context_key() if dictionary is not None else "no-dictionary",
        str(unit.get("unit_type", "text")),
        _normalized_source(str(unit.get("source", ""))),
    )


def _replace_corresponding_phrase(text: str, source: str, term: str, replacement: str) -> str:
    pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)
    source_start = source.casefold().find(term.casefold())
    source_ratio = source_start / max(len(source), 1)
    words = list(re.finditer(r"\S+", text))
    if not words:
        return replacement
    start_index = min(len(words) - 1, round(source_ratio * len(words)))
    term_word_count = max(1, len(term.split()))
    end_index = min(len(words), start_index + term_word_count)
    start = words[start_index].start()
    end = words[end_index - 1].end()
    return text[:start] + replacement + text[end:]


def enforce_terminology(source: str, translation: str, dictionary: TranslationDictionary) -> tuple[str, list[str]]:
    result = _plain_text(translation)
    corrections: list[str] = []
    for term in dictionary.matching_terms(source, "fixed_translation"):
        expected = dictionary.fixed_translation[term]
        corrected = _replace_corresponding_phrase(result, source, term, expected)
        if corrected != result:
            corrections.append(f"{term} -> {expected}")
            result = corrected
    for term in dictionary.matching_terms(source, "keep_english"):
        corrected = _replace_corresponding_phrase(result, source, term, term)
        if corrected != result:
            corrections.append(f"{term} -> KEEP_ENGLISH")
            result = corrected
    return result, corrections


def translate_batches(
    units: list[dict[str, Any]], client: TranslationClient, model: str,
    batch_size: int = DEFAULT_BATCH_SIZE, max_retries: int = DEFAULT_MAX_RETRIES,
    dictionary: TranslationDictionary | None = None,
    warning_sink: list[dict[str, Any]] | None = None,
    source_language: str = "English",
    target_language: str = "Indonesian",
) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise TranslationError("batch_size must be at least 1")

    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for unit in units:
        key = _translation_key(unit, dictionary, source_language, target_language)
        groups.setdefault(key, []).append(unit)
    representatives = [group[0] for group in groups.values()]
    batches = [
        representatives[index:index + batch_size]
        for index in range(0, len(representatives), batch_size)
    ]
    representative_results: dict[int, dict[str, Any]] = {}
    source_fallback_ids: set[int] = set()
    warnings = warning_sink if warning_sink is not None else []

    def messages_for(batch: list[dict[str, Any]], correction: str | None = None) -> list[dict[str, str]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _request_payload(batch)},
        ]
        if dictionary is not None:
            messages.insert(1, {"role": "system", "content": dictionary.context()})
        if correction:
            messages.append({"role": "user", "content": correction})
        return messages

    for batch_number, batch in enumerate(batches, start=1):
        ids = [unit["id"] for unit in batch]
        print(f"Batch {batch_number}/{len(batches)}")
        print(f"IDs: {ids[0]}-{ids[-1]}")

        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            correction = None
            if last_error is not None:
                correction = (
                        "Correction for the previous response: return the complete batch "
                        "with exactly one translation object per input ID. "
                        "Return only valid JSON matching the required schema. "
                        f"Previous validation error: {last_error}"
                    )

            try:
                response = client.chat_completion(messages_for(batch, correction), model)
                payload = extract_json_response(response)
                validated = validate_response(payload, batch)

                suspicious = [
                    (item, sanitize_translation(item["source"], item["translation"]))
                    for item in validated
                ]
                retry_items = [item for item, result in suspicious if result.issues]
                clean_items = [
                    {**item, "translation": result.text}
                    for item, result in suspicious if not result.issues
                ]
                if retry_items:
                    issue_summary = "; ".join(
                        f"ID {item['id']}: {', '.join(result.issues)}"
                        for item, result in suspicious if result.issues
                    )
                    retry_batch = [
                        next(unit for unit in batch if unit["id"] == item["id"])
                        for item in retry_items
                    ]
                    try:
                        retry_response = client.chat_completion(
                            messages_for(
                                retry_batch,
                                "The previous translations contained invalid commentary, repetition, "
                                "length growth, or altered private-use glyphs. Return clean translations "
                                f"only. Problems: {issue_summary}",
                            ),
                            model,
                        )
                        retry_validated = validate_response(
                            extract_json_response(retry_response), retry_batch
                        )
                    except (BatchValidationError, ValueError, TypeError, TranslationError, OSError) as error:
                        LOGGER.warning("Content retry failed: %s", error)
                        retry_validated = []
                    retry_by_id = {item["id"]: item for item in retry_validated}
                    for original in retry_items:
                        retry_item = retry_by_id.get(original["id"])
                        retry_result = (
                            sanitize_translation(retry_item["source"], retry_item["translation"])
                            if retry_item is not None else None
                        )
                        if retry_item is not None and retry_result is not None and not retry_result.issues:
                            clean_items.append({**retry_item, "translation": retry_result.text})
                            continue
                        source_fallback_ids.add(original["id"])
                        issues = list(retry_result.issues) if retry_result is not None else ["invalid content retry"]
                        warning = {
                            "id": original["id"],
                            "source": original["source"],
                            "translation": original["source"],
                            "note": f"Source fallback after sanitation: {', '.join(issues)}",
                        }
                        warnings.append(warning)
                        LOGGER.warning("Unit %s source fallback: %s", original["id"], warning["note"])
                        clean_items.append({**original, "translation": original["source"]})

                for item in clean_items:
                    if dictionary is not None and item["id"] not in source_fallback_ids and item["translation"] != item["source"]:
                        item["translation"], corrections = enforce_terminology(
                            item["source"], item["translation"], dictionary
                        )
                        for terminology_correction in corrections:
                            LOGGER.info(
                                "Unit %s terminology correction: %s",
                                item["id"], terminology_correction,
                            )
                    representative_results[item["id"]] = item

                print("Status: OK")
                break

            except (BatchValidationError, ValueError, TypeError, TranslationError) as error:
                last_error = error
                LOGGER.warning(
                    "Batch %s failed on attempt %s: %s",
                    batch_number,
                    attempt + 1,
                    error,
                )
                LOGGER.debug(
                    "Batch %s response was: %s",
                    batch_number,
                    locals().get("response", "<no response>"),
                )

        else:
            raise TranslationError(
                f"Batch {batch_number} failed after {max_retries + 1} attempts; "
                f"IDs={ids}: {last_error}"
            ) from last_error

    results: list[dict[str, Any]] = []
    for group in groups.values():
        representative = group[0]
        translated = representative_results[representative["id"]]
        representative_fallback = representative["id"] in source_fallback_ids
        for unit in group:
            translation = unit["source"] if representative_fallback else translated["translation"]
            if dictionary is not None and not representative_fallback and translation != unit["source"]:
                translation, _ = enforce_terminology(unit["source"], translation, dictionary)
            results.append({
                "id": unit["id"],
                "source": unit["source"],
                "translation": translation,
            })
    results.sort(key=lambda item: item["id"])
    return validate_response({"translations": results}, units)

def save_translation(
    output_path: Path, extraction_path: Path, extraction_payload: dict[str, Any],
    translations: list[dict[str, Any]], model: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    modern_identity = isinstance(extraction_payload.get("source_sha256"), str)
    output = {
        "schema_version": 2 if modern_identity else 1,
        "artifact_type": "translation",
        "source_file": extraction_payload.get("source_file", extraction_path.name),
        "source_extraction": str(extraction_path),
        "source_extraction_sha256": payload_sha256(extraction_payload),
        "source_language": "English",
        "target_language": "Indonesian",
        "model": model,
        "translations": translations,
    }
    if modern_identity:
        output["source_sha256"] = extraction_payload["source_sha256"]
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
            max_output_tokens=MAX_OUTPUT_TOKENS,
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
    dictionary_path: Path = DEFAULT_DICTIONARY_PATH,
) -> Path:
    config = config or TranslationConfig.from_environment()
    dictionary = TranslationDictionary.load(dictionary_path)
    extraction_payload, units = load_translation_units(extraction_path, dictionary)
    skipped = [unit for unit in units if dictionary.skip_matches(unit["source"])]
    translatable = [unit for unit in units if not dictionary.skip_matches(unit["source"])]
    skipped_results = [
        {"id": unit["id"], "source": unit["source"], "translation": unit["source"]}
        for unit in skipped
    ]
    active_client = client or OpenAITranslationClient(config.base_url, config.api_key)
    print("Translation\n-----------")
    print(f"Input: {extraction_path}")
    print(f"Units to translate: {len(translatable)}")
    print("\nTranslation Dictionary")
    print("----------------------")
    print(f"Skip translation terms: {len(dictionary.skip_translation)}")
    print(f"Keep English terms: {len(dictionary.keep_english)}")
    print(f"Fixed translations: {len(dictionary.fixed_translation)}")
    print(f"Units skipped: {len(skipped)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Batches: {(len(translatable) + config.batch_size - 1) // config.batch_size}")
    print(f"Model: {config.model}")
    warnings: list[dict[str, Any]] = []
    translations = translate_batches(
        translatable,
        active_client,
        config.model,
        config.batch_size,
        config.max_retries,
        dictionary,
        warning_sink=warnings,
    )
    translations = skipped_results + translations

    if warnings:
        print("\nTranslation Warnings")
        print("--------------------")
        print(f"Sanitation fallbacks: {len(warnings)}")
        for warning in warnings:
            print(f"- ID {warning['id']}: {warning['note']}")
    else:
        print("\nTranslation Warnings")
        print("--------------------")
        print("None")
    translations.sort(key=lambda item: item["id"])
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
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY_PATH)
    args = parser.parse_args()
    try:
        run_translation(args.input, args.output, dictionary_path=args.dictionary)
    except TranslationError as error:
        print(f"Translation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
