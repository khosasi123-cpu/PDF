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


def test_multiline_table_cell_translation_is_rendered(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    page.insert_text((25, 35), "English One", fontsize=8)
    document.save(pdf_path)
    document.close()

    source = "English One\nEnglish Two\nEnglish Three\n1"
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "units": [{
            "id": 1, "unit_type": "table_cell", "source": source,
            "bbox": [20, 20, 180, 38], "fontsize": 8, "flags": 0, "translate": True,
        }]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{
            "id": 1, "source": source,
            "translation": "Indonesian One\nIndonesian Two\nIndonesian Three\n1",
        }],
    }), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")

    assert stats.rendered_units == 1
    assert not stats.warnings
    assert "Indonesian One Indonesian Two Indonesian Three 1" in fitz.open(output_path)[0].get_text()


def test_compact_borderless_structured_text_is_rendered(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=100)
    page.insert_text((20, 30), "English One", fontsize=8)
    document.save(pdf_path)
    document.close()

    source = "English One\nEnglish Two\nEnglish Three"
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "units": [{
            "id": 1, "unit_type": "text", "source": source,
            "bbox": [20, 20, 280, 31], "fontsize": 8, "line_count": 3,
            "flags": 0, "translate": True,
        }]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{
            "id": 1, "source": source,
            "translation": "Indonesian One\nIndonesian Two\nIndonesian Three",
        }],
    }), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")

    assert stats.rendered_units == 1
    assert not stats.warnings
    assert "Indonesian One Indonesian Two Indonesian Three" in fitz.open(output_path)[0].get_text()


def test_borderless_structured_lines_keep_original_x_positions(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=100)
    for x, text in ((20, "One"), (100, "Two"), (180, "Three")):
        page.insert_text((x, 30), text, fontsize=8)
    document.save(pdf_path)
    document.close()

    source = "One\nTwo\nThree"
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "units": [{
            "id": 1, "unit_type": "text", "source": source,
            "bbox": [20, 20, 210, 32.5], "fontsize": 9, "line_count": 3,
            "flags": 0, "translate": True,
        }]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{
            "id": 1, "source": source,
            "translation": "Satu\nDua\nTiga",
        }],
    }), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")

    spans = [
        span
        for block in fitz.open(output_path)[0].get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span["text"] in {"Satu", "Dua", "Tiga"}
    ]
    assert stats.rendered_units == 1
    assert not stats.warnings
    assert [(span["text"], round(span["bbox"][0])) for span in spans] == [
        ("Satu", 20), ("Dua", 100), ("Tiga", 180)
    ]
    assert min(span["size"] for span in spans) >= 7.5


def test_structured_translation_uses_space_before_next_column(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=100)
    for x, text in ((20, "GROUND POWER UNIT"), (210, "Code"), (280, "1")):
        page.insert_text((x, 30), text, fontsize=8)
    document.save(pdf_path)
    document.close()

    source = "GROUND POWER UNIT\nCode\n1"
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "units": [{
            "id": 1, "unit_type": "text", "source": source,
            "bbox": [20, 20, 295, 32.5], "fontsize": 8.5, "line_count": 3,
            "flags": 0, "translate": True,
        }]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{
            "id": 1, "source": source,
            "translation": "UNIT DAYA DARAT UNTUK OPERASIONAL\nKode\n1",
        }],
    }), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")
    translated_spans = [
        span
        for block in fitz.open(output_path)[0].get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span["text"] == "UNIT DAYA DARAT UNTUK OPERASIONAL"
    ]

    assert stats.rendered_units == 1
    assert not stats.warnings
    assert len(translated_spans) == 1
    assert translated_spans[0]["bbox"][0] == 20
    assert translated_spans[0]["size"] >= 6.5
