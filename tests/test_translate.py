import json
from pathlib import Path

import pytest

from pdf_translator.translate import (
    BatchValidationError,
    TranslationConfig,
    TranslationError,
    extract_json_response,
    load_translation_units,
    run_translation,
    translate_batches,
    validate_response,
)


UNITS = [
    {"id": 1, "unit_type": "text", "source": "Server Configuration"},
    {"id": 2, "unit_type": "table_cell", "source": "Parameter designation"},
    {"id": 3, "unit_type": "text", "source": "WARNING"},
]


def valid_response(units=UNITS):
    return json.dumps({
        "translations": [
            {"id": unit["id"], "source": unit["source"], "translation": f"Terjemahan {unit['id']}"}
            for unit in units
        ]
    })


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat_completion(self, messages, model):
        self.calls.append((messages, model))
        return next(self.responses)


def test_valid_response_and_translate_false_filtering(tmp_path: Path):
    extraction = {
        "source_file": "test.pdf",
        "pages": [{"units": [
            {"id": 1, "unit_type": "text", "source": "Server Configuration", "translate": True},
            {"id": 2, "unit_type": "text", "source": "12345", "translate": False},
        ]}],
    }
    path = tmp_path / "extraction.json"
    path.write_text(json.dumps(extraction), encoding="utf-8")
    _, units = load_translation_units(path)
    assert units == [{"id": 1, "unit_type": "text", "source": "Server Configuration"}]


def test_validation_rejects_missing_unexpected_duplicate_source_and_empty():
    with pytest.raises(BatchValidationError, match="Missing"):
        validate_response({"translations": []}, UNITS)
    with pytest.raises(BatchValidationError, match="Unexpected"):
        validate_response({"translations": [
            {"id": 99, "source": "x", "translation": "x"}
        ]}, UNITS)
    with pytest.raises(BatchValidationError, match="Duplicate"):
        validate_response({"translations": [
            {"id": 1, "source": UNITS[0]["source"], "translation": "A"},
            {"id": 1, "source": UNITS[0]["source"], "translation": "B"},
            {"id": 2, "source": UNITS[1]["source"], "translation": "C"},
            {"id": 3, "source": UNITS[2]["source"], "translation": "D"},
        ]}, UNITS)
    with pytest.raises(BatchValidationError, match="Source mismatch"):
        validate_response({"translations": [
            {"id": 1, "source": "changed", "translation": "A"},
            {"id": 2, "source": UNITS[1]["source"], "translation": "B"},
            {"id": 3, "source": UNITS[2]["source"], "translation": "C"},
        ]}, UNITS)
    with pytest.raises(BatchValidationError, match="Empty"):
        validate_response({"translations": [
            {"id": 1, "source": UNITS[0]["source"], "translation": " "},
            {"id": 2, "source": UNITS[1]["source"], "translation": "B"},
            {"id": 3, "source": UNITS[2]["source"], "translation": "C"},
        ]}, UNITS)


def test_identical_translation_rejected_for_clear_english():
    with pytest.raises(BatchValidationError, match="unchanged"):
        validate_response({"translations": [
            {"id": 1, "source": "Server Configuration", "translation": "Server Configuration"},
            {"id": 2, "source": "Parameter designation", "translation": "Penamaan parameter"},
            {"id": 3, "source": "WARNING", "translation": "WARNING"},
        ]}, UNITS)


def test_json_parser_accepts_code_fences_and_surrounding_text():
    assert extract_json_response('```json\n{"translations": []}\n```') == {"translations": []}
    assert extract_json_response('Here is the result:\n{"translations": []}\nDone') == {"translations": []}


def test_invalid_json_retries_then_succeeds():
    client = FakeClient(["not json", valid_response()])
    result = translate_batches(UNITS, client, "mistral", batch_size=3, max_retries=1)
    assert [item["id"] for item in result] == [1, 2, 3]
    assert len(client.calls) == 2


def test_multiple_batches_combine_and_preserve_ids():
    client = FakeClient([valid_response(UNITS[:2]), valid_response(UNITS[2:])])
    result = translate_batches(UNITS, client, "mistral", batch_size=2, max_retries=0)
    assert [item["id"] for item in result] == [1, 2, 3]
    assert len(client.calls) == 2
    sent = json.loads(client.calls[0][0][1]["content"])
    assert list(sent[0]) == ["id", "unit_type", "source"]


def test_persistent_invalid_batch_raises():
    client = FakeClient(["bad", "still bad"])
    with pytest.raises(TranslationError, match="Batch 1 failed"):
        translate_batches(UNITS, client, "mistral", batch_size=3, max_retries=1)


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "local-key")
    monkeypatch.setenv("MODEL", "mistral")
    monkeypatch.setenv("TRANSLATION_BATCH_SIZE", "7")
    monkeypatch.setenv("TRANSLATION_MAX_RETRIES", "4")
    config = TranslationConfig.from_environment()
    assert config.batch_size == 7
    assert config.max_retries == 4


def test_run_translation_writes_separate_output_without_mutating_extraction(tmp_path: Path):
    extraction = {
        "source_file": "test.pdf",
        "pages": [{"units": [
            {"id": 1, "unit_type": "text", "source": "Server Configuration", "translate": True},
            {"id": 2, "unit_type": "text", "source": "12345", "translate": False},
        ]}],
    }
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps(extraction), encoding="utf-8")
    original = extraction_path.read_bytes()
    output_path = tmp_path / "translation.json"
    config = TranslationConfig("http://local", "key", "mistral", batch_size=1, max_retries=0)
    run_translation(extraction_path, output_path, config, FakeClient([valid_response(UNITS[:1])]))
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["translations"][0]["id"] == 1
    assert extraction_path.read_bytes() == original
