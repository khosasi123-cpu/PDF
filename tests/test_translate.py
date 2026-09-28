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
    sanitize_translation,
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


def test_validation_rejects_structural_errors_and_empty_falls_back_to_source():
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
    result = validate_response({"translations": [
        {"id": 1, "source": UNITS[0]["source"], "translation": " "},
        {"id": 2, "source": UNITS[1]["source"], "translation": "B"},
        {"id": 3, "source": UNITS[2]["source"], "translation": "C"},
    ]}, UNITS)
    assert result[0]["translation"] == UNITS[0]["source"]


def test_identical_translation_is_accepted_for_protected_terms():
    result = validate_response({"translations": [
        {"id": 1, "source": "Server Configuration", "translation": "Server Configuration"},
        {"id": 2, "source": "Parameter designation", "translation": "Penamaan parameter"},
        {"id": 3, "source": "WARNING", "translation": "WARNING"},
    ]}, UNITS)
    assert result[0]["translation"] == "Server Configuration"
    assert result[2]["translation"] == "WARNING"


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


def test_commentary_is_retried_once_then_clean_translation_is_used():
    bad = json.dumps({"translations": [{
        "id": 1, "translation": "Terjemahan: Konfigurasi server",
    }]})
    good = json.dumps({"translations": [{
        "id": 1, "translation": "Konfigurasi server",
    }]})
    client = FakeClient([bad, good])

    result = translate_batches(UNITS[:1], client, "mistral", max_retries=0)

    assert result[0]["translation"] == "Konfigurasi server"
    assert len(client.calls) == 2


def test_content_retry_only_resends_suspicious_units():
    first = json.dumps({"translations": [
        {"id": 1, "translation": "Translation: Konfigurasi server"},
        {"id": 2, "translation": "Penamaan parameter"},
    ]})
    retry = json.dumps({"translations": [
        {"id": 1, "translation": "Konfigurasi server"},
    ]})
    client = FakeClient([first, retry])

    result = translate_batches(UNITS[:2], client, "mistral", max_retries=0)

    assert [item["translation"] for item in result] == ["Konfigurasi server", "Penamaan parameter"]
    retry_payload = json.loads(client.calls[1][0][1]["content"])
    assert [item["id"] for item in retry_payload] == [1]


def test_empty_translation_falls_back_without_retry():
    client = FakeClient([json.dumps({"translations": [{"id": 1, "translation": " "}]})])
    result = translate_batches(UNITS[:1], client, "mistral", max_retries=0)
    assert result[0]["translation"] == UNITS[0]["source"]
    assert len(client.calls) == 1


def test_persistent_commentary_falls_back_to_source():
    response = json.dumps({"translations": [{
        "id": 1, "translation": "Terjemahan: Konfigurasi server",
    }]})
    warnings = []

    result = translate_batches(
        UNITS[:1], FakeClient([response, response]), "mistral",
        max_retries=0, warning_sink=warnings,
    )

    assert result[0]["translation"] == UNITS[0]["source"]
    assert "sanitation" in warnings[0]["note"]


def test_repeated_sentence_and_extreme_growth_are_detected_conservatively():
    source = "This document must not be copied."
    repeated = "Dokumen ini tidak boleh disalin. Dokumen ini tidak boleh disalin."
    assert "repeated sentence" in sanitize_translation(source, repeated).issues
    assert sanitize_translation("USB USB", "USB USB").issues == ()
    assert "extreme unexplained length growth" in sanitize_translation("Short", "long " * 100).issues


def test_identical_normalized_source_is_translated_once_per_document():
    units = [
        {"id": 1, "unit_type": "text", "source": "Repeated heading"},
        {"id": 2, "unit_type": "text", "source": "Repeated   heading"},
    ]
    client = FakeClient([json.dumps({"translations": [{
        "id": 1, "translation": "Judul berulang",
    }]})])

    result = translate_batches(units, client, "mistral", max_retries=0)

    assert len(client.calls) == 1
    sent = json.loads(client.calls[0][0][1]["content"])
    assert [item["id"] for item in sent] == [1]
    assert [item["translation"] for item in result] == ["Judul berulang", "Judul berulang"]


def test_same_source_with_different_unit_types_is_not_reused():
    units = [
        {"id": 1, "unit_type": "text", "source": "Status"},
        {"id": 2, "unit_type": "table_cell", "source": "Status"},
    ]
    client = FakeClient([
        json.dumps({"translations": [{"id": 1, "translation": "Status teks"}]}),
        json.dumps({"translations": [{"id": 2, "translation": "Status tabel"}]}),
    ])
    result = translate_batches(units, client, "mistral", batch_size=1, max_retries=0)
    assert [item["translation"] for item in result] == ["Status teks", "Status tabel"]
    assert len(client.calls) == 2


def test_private_use_glyph_loss_is_rejected():
    result = sanitize_translation("\ue000 Open menu", "Buka menu")
    assert result.issues == ("private-use glyph sequence changed",)


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
