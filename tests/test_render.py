import json
from pathlib import Path

import fitz

from pdf_translator.render import render_pdf


def _make_inputs(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    page.insert_text((20, 30), "Hello")
    document.save(pdf_path)
    document.close()
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "units": [{
            "id": 1, "unit_type": "text", "source": "Hello", "bbox": [15, 15, 55, 35],
            "fontsize": 11, "flags": 0, "translate": True,
        }, {
            "id": 2, "unit_type": "text", "source": "Skip", "bbox": [10, 40, 30, 50],
            "fontsize": 10, "flags": 0, "translate": False,
        }]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{"id": 1, "source": "Hello", "translation": "Halo"}],
    }), encoding="utf-8")
    return pdf_path, extraction_path, translation_path


def test_render_preserves_pages_and_skips_untranslated_units(tmp_path: Path):
    pdf_path, extraction_path, translation_path = _make_inputs(tmp_path)
    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")
    assert output_path.exists()
    assert stats.total_units == 2
    assert stats.translated_units == 1
    assert stats.skipped_units == 1
    assert stats.rendered_units == 1
    rendered = fitz.open(output_path)
    assert rendered.page_count == 1
    assert "Halo" in rendered[0].get_text()
    rendered.close()


def test_render_reports_missing_translation_without_crashing(tmp_path: Path):
    pdf_path, extraction_path, translation_path = _make_inputs(tmp_path)
    payload = json.loads(extraction_path.read_text(encoding="utf-8"))
    payload["pages"][0]["units"][0]["id"] = 99
    extraction_path.write_text(json.dumps(payload), encoding="utf-8")
    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")
    assert output_path.exists()
    assert stats.missing_translations == 1
    assert any("Unit 99" in warning for warning in stats.warnings)


def test_translation_markup_becomes_plain_text(tmp_path: Path):
    pdf_path, extraction_path, translation_path = _make_inputs(tmp_path)
    payload = json.loads(translation_path.read_text(encoding="utf-8"))
    payload["translations"][0]["translation"] = "<b>Halo</b><br>semua"
    translation_path.write_text(json.dumps(payload), encoding="utf-8")
    output_path, _, = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")
    text = fitz.open(output_path)[0].get_text()
    assert "Halo" in text
    assert "semua" in text
