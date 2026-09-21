import json
from pathlib import Path

import pytest

from pdf_translator.dictionary import DictionaryError, TranslationDictionary
from pdf_translator.translate import TranslationConfig, enforce_terminology, run_translation


class RecordingClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def chat_completion(self, messages, model):
        self.calls += 1
        return self.response


def write_dictionary(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "dictionary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phrase_matching_prefers_longer_and_respects_boundaries(tmp_path: Path):
    dictionary = TranslationDictionary.load(write_dictionary(tmp_path, {
        "skip_translation": [],
        "keep_english": ["reading", "Medium reading"],
        "fixed_translation": {},
    }))
    assert dictionary.matching_terms("Medium reading station", "keep_english") == ["Medium reading"]
    assert dictionary.matching_terms("breadings", "keep_english") == []


def test_dictionary_conflicts_and_structure_are_rejected(tmp_path: Path):
    for payload in [
        {"skip_translation": ["EDMS"], "keep_english": ["EDMS"], "fixed_translation": {}},
        {"skip_translation": [], "keep_english": ["EDMS"], "fixed_translation": {"edms": "EDMS"}},
        {"skip_translation": "EDMS", "keep_english": [], "fixed_translation": {}},
    ]:
        with pytest.raises(DictionaryError):
            TranslationDictionary.load(write_dictionary(tmp_path, payload))


def test_fixed_and_keep_terms_are_enforced_deterministically(tmp_path: Path):
    dictionary = TranslationDictionary.load(write_dictionary(tmp_path, {
        "skip_translation": [],
        "keep_english": ["Flight View"],
        "fixed_translation": {"Free entry field": "Kolom isian bebas"},
    }))
    corrected, corrections = enforce_terminology(
        "Free entry field for Flight View",
        "Bidang input bebas untuk tampilan Penerbangan",
        dictionary,
    )
    assert corrected == "Kolom isian bebas untuk Flight View"
    assert len(corrections) == 2


def test_skip_unit_is_not_sent_to_llm_and_is_written_unchanged(tmp_path: Path):
    dictionary_path = write_dictionary(tmp_path, {
        "skip_translation": ["Copyright:"],
        "keep_english": [],
        "fixed_translation": {},
    })
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({"source_file": "test.pdf", "pages": [{"units": [
        {"id": 1, "unit_type": "text", "source": "Copyright:", "translate": True},
        {"id": 2, "unit_type": "text", "source": "Server Configuration", "translate": True},
    ]}]}), encoding="utf-8")
    client = RecordingClient(json.dumps({"translations": [{
        "id": 2, "source": "Server Configuration", "translation": "Konfigurasi Server"
    }]}))
    output_path = tmp_path / "translation.json"
    run_translation(
        extraction_path, output_path,
        TranslationConfig("http://local", "key", "model", batch_size=20, max_retries=0),
        client,
        dictionary_path,
    )
    assert client.calls == 1
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert {item["id"] for item in output["translations"]} == {1, 2}
    assert next(item for item in output["translations"] if item["id"] == 1)["translation"] == "Copyright:"


def test_context_excludes_skip_terms_and_includes_other_terms(tmp_path: Path):
    dictionary = TranslationDictionary.load(write_dictionary(tmp_path, {
        "skip_translation": ["Copyright:"],
        "keep_english": ["EDMS"],
        "fixed_translation": {"Free entry field": "Kolom isian bebas"},
    }))
    context = dictionary.context()
    assert "Copyright:" not in context
    assert "EDMS" in context
    assert "Free entry field -> Kolom isian bebas" in context
