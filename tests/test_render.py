import json
from pathlib import Path

import fitz

from pdf_translator.render import _plain_text, render_pdf


def test_symbol_normalization_preserves_normal_text():
    assert _plain_text("● Test") == "• Test"
    assert _plain_text("○ • –") == "○ • –"
    assert _plain_text("Normal text") == "Normal text"


def test_bullet_translation_uses_a_visible_symbol_font(tmp_path: Path):
    pdf_path, extraction_path, translation_path = _make_inputs(tmp_path)
    payload = json.loads(translation_path.read_text(encoding="utf-8"))
    payload["translations"][0]["translation"] = "● Test"
    translation_path.write_text(json.dumps(payload), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")
    rendered = fitz.open(output_path)

    assert stats.rendered_units == 1
    assert not stats.warnings
    assert any("Symbols" in font[3] for font in rendered[0].get_fonts(full=True))
    assert "?" not in rendered[0].get_text()
    rendered.close()


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


def test_table_cell_span_mismatch_uses_logged_logical_cell_fallback(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    page.insert_text((25, 30), "English", fontsize=8)
    page.insert_text((25, 45), "continued", fontsize=8)
    document.save(pdf_path)
    document.close()

    source = "English continued"
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "units": [{
            "id": 1, "unit_type": "table_cell", "source": source,
            "bbox": [20, 20, 180, 60], "fontsize": 8, "flags": 0,
            "line_count": 2, "translate": True,
        }]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{"id": 1, "source": source, "translation": "Terjemahan panjang"}],
    }), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")
    warning = stats.warnings[0]

    assert stats.rendered_units == 1
    assert "Terjemahan panjang" in fitz.open(output_path)[0].get_text()
    assert all(field in warning for field in (
        "Unit 1", "source=", "translation=", "span_count=2",
        "translated_line_count=1", "source_span_bboxes=", "cell_bbox=",
        "chosen_render_rects=",
    ))


def test_multiline_fallback_translation_is_visible(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=240, height=120)
    page.insert_text((20, 30), "Source", fontsize=10)
    document.save(pdf_path)
    document.close()

    source = "\n".join(f"Source {index}" for index in range(15))
    translation = "\n".join(f"Translated {index}" for index in range(15))
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "units": [{
            "id": 1, "unit_type": "text", "source": source,
            "bbox": [20, 20, 220, 90], "fontsize": 10, "flags": 0,
            "line_count": 15, "translate": True,
        }]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{"id": 1, "source": source, "translation": translation}],
    }), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")

    output_text = fitz.open(output_path)[0].get_text()
    assert stats.rendered_units == 1
    assert not stats.warnings
    assert "Translated 14" in output_text


def test_compact_borderless_structured_text_is_rendered(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=100)
    for x, text in ((20, "English One"), (110, "English Two"), (200, "English Three")):
        page.insert_text((x, 30), text, fontsize=8)
    document.save(pdf_path)
    document.close()

    source = "English One\nEnglish Two\nEnglish Three"
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "units": [{
            "id": 1, "unit_type": "text", "source": source,
            "bbox": [20, 20, 280, 34], "fontsize": 8, "line_count": 3,
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
    output_text = fitz.open(output_path)[0].get_text()
    assert all(value in output_text for value in ("Indonesian One", "Indonesian Two", "Indonesian Three"))


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


def test_structured_five_column_translation_preserves_x_positions(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=450, height=80)
    for x, text in ((20, "Manufacturer"), (100, "Reference"), (180, "Designation"),
                    (300, "Quantity"), (380, "Type")):
        page.insert_text((x, 30), text, fontsize=8)
    document.save(pdf_path)
    document.close()

    source = "Manufacturer\nReference\nDesignation\nQuantity\nType"
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "units": [{
            "id": 1, "unit_type": "table_cell", "source": source,
            "bbox": [20, 20, 440, 34], "fontsize": 8, "flags": 0,
            "line_count": 5, "translate": True,
        }]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{
            "id": 1, "source": source,
            "translation": "Manufaktur\nReferensi\nPenunjukan\nJumlah\nJenis",
        }],
    }), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")
    spans = {
        span["text"]: round(span["bbox"][0])
        for block in fitz.open(output_path)[0].get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span["text"] in {"Manufaktur", "Referensi", "Penunjukan", "Jumlah", "Jenis"}
    }

    assert stats.rendered_units == 1
    assert spans == {
        "Manufaktur": 20, "Referensi": 100, "Penunjukan": 180,
        "Jumlah": 300, "Jenis": 380,
    }


def test_structured_rows_keep_number_and_time_columns(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=360, height=100)
    values = (
        (20, 30, "T : ALL"), (240, 30, "1"), (300, 30, "0.2 h"),
        (20, 50, "ARMAMENT / 2320"), (240, 50, "1"), (300, 50, "0.2 h"),
    )
    for x, y, value in values:
        page.insert_text((x, y), value, fontsize=8)
    document.save(pdf_path)
    document.close()

    source = "T : ALL\n1\n0.2 h\nARMAMENT / 2320\n1\n0.2 h"
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "units": [{
            "id": 1, "unit_type": "text", "source": source,
            "bbox": [20, 20, 330, 53], "fontsize": 8, "flags": 0,
            "line_count": 6, "translate": True,
        }]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{
            "id": 1, "source": source,
            "translation": "T : SEMUA\n1\n0.2 jam\nARMAMEN / 2320\n1\n0.2 jam",
        }],
    }), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")
    translated = [
        span
        for block in fitz.open(output_path)[0].get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span["text"] in {"T : SEMUA", "ARMAMEN / 2320", "1", "0.2 jam"}
    ]

    assert stats.rendered_units == 1
    assert not stats.warnings
    assert {(span["text"], round(span["bbox"][0])) for span in translated} == {
        ("T : SEMUA", 20), ("ARMAMEN / 2320", 20),
        ("1", 240), ("0.2 jam", 300),
    }


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


def test_overflow_translation_uses_safe_expansion_and_remains_visible(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=100)
    page.insert_text((20, 30), "Required Time", fontsize=8)
    page.insert_text((250, 30), "Other", fontsize=8)
    document.save(pdf_path)
    document.close()

    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "units": [
            {"id": 1, "unit_type": "text", "source": "Required Time",
             "bbox": [20, 20, 100, 32], "fontsize": 8, "flags": 0, "translate": True},
            {"id": 2, "unit_type": "text", "source": "Other",
             "bbox": [250, 20, 280, 32], "fontsize": 8, "flags": 0, "translate": False},
        ]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{"id": 1, "source": "Required Time", "translation": "Waktu yang Dibutuhkan"}],
    }), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")
    output_text = fitz.open(output_path)[0].get_text()

    assert stats.rendered_units == 1
    assert "Waktu yang Dibutuhkan" in output_text
    assert not any("Unit 1" in warning for warning in stats.warnings)


def test_safe_expansion_covers_the_original_area_before_retry(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=100)
    page.insert_text((20, 25), "Original text that must be covered", fontsize=8)
    document.save(pdf_path)
    document.close()

    source = "Original text that must be covered"
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "units": [{
            "id": 1, "unit_type": "text", "source": source,
            "bbox": [20, 12, 150, 28], "fontsize": 8, "flags": 0, "translate": True,
        }]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{
            "id": 1,
            "source": source,
            "translation": "Teks Indonesia yang sangat panjang dan membutuhkan ruang tambahan",
        }],
    }), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")
    output_text = " ".join(fitz.open(output_path)[0].get_text().split())

    assert stats.rendered_units == 1
    assert "Teks Indonesia yang sangat panjang dan membutuhkan ruang tambahan" in output_text


def test_overflow_translation_uses_html_last_resort(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=100, height=50)
    page.insert_text((20, 20), "Short", fontsize=8)
    document.save(pdf_path)
    document.close()

    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "units": [{
            "id": 1, "unit_type": "text", "source": "Short",
            "bbox": [20, 10, 25, 21], "fontsize": 8, "flags": 0, "translate": True,
        }]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{"id": 1, "source": "Short", "translation": "A much longer valid translation"}],
    }), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")

    assert stats.rendered_units == 1
    output_text = " ".join(fitz.open(output_path)[0].get_text().split())
    assert "A much longer valid translation" in output_text
