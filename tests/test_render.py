import json
from pathlib import Path

import fitz
import pytest

from pdf_translator.layout import LayoutPlan
from pdf_translator.identity import payload_sha256, sha256_file
from pdf_translator.render import _extracted_structured_spans, _plain_text, render_pdf


def test_symbol_normalization_preserves_normal_text():
    assert _plain_text("● Test") == "• Test"
    assert _plain_text("○ • –") == "○ • –"
    assert _plain_text("Normal text") == "Normal text"


def test_extracted_spans_keep_translation_unit_source_order(tmp_path: Path):
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    unit = {
        "source_ids": ["title", "classification"],
    }
    page_data = {"source_objects": [
        {"id": "classification", "kind": "span", "bbox": [140, 20, 190, 30]},
        {"id": "title", "kind": "span", "bbox": [20, 35, 120, 50]},
    ]}

    spans = _extracted_structured_spans(
        page, unit, page_data, fitz.Rect(20, 20, 190, 50)
    )

    assert [tuple(span.source_rect) for span in spans] == [
        (20.0, 35.0, 120.0, 50.0),
        (140.0, 20.0, 190.0, 30.0),
    ]
    document.close()


def test_extracted_span_expansion_respects_column_right_limit():
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    unit = {"source_ids": ["left"]}
    page_data = {"source_objects": [{
        "id": "left", "kind": "span", "bbox": [20, 20, 60, 30],
    }]}

    spans = _extracted_structured_spans(
        page,
        unit,
        page_data,
        fitz.Rect(20, 20, 60, 30),
        expand_last_column=True,
        right_limit=99,
    )

    assert spans[0].render_rect.x1 == 99
    document.close()


def test_bullet_translation_uses_visible_unicode_without_question_mark(tmp_path: Path):
    pdf_path, extraction_path, translation_path = _make_inputs(tmp_path)
    payload = json.loads(translation_path.read_text(encoding="utf-8"))
    payload["translations"][0]["translation"] = "● Test"
    translation_path.write_text(json.dumps(payload), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")
    rendered = fitz.open(output_path)

    assert stats.rendered_units == 1
    assert not stats.warnings
    assert "?" not in rendered[0].get_text()
    assert "•" in rendered[0].get_text()
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
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction["pages"][0]["units"][0]["bbox"] = [15, 15, 100, 55]
    extraction_path.write_text(json.dumps(extraction), encoding="utf-8")
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

    identity_dir = tmp_path / "identity"
    output_path, stats = render_pdf(
        pdf_path, extraction_path, translation_path, tmp_path / "out.pdf",
        identity_debug_dir=identity_dir,
    )
    warning = stats.warnings[0]
    identity = json.loads(
        (identity_dir / "render_identity.json").read_text(encoding="utf-8")
    )["units"][0]

    assert stats.rendered_units == 1
    assert "Terjemahan panjang" in fitz.open(output_path)[0].get_text()
    assert all(field in warning for field in (
        "Unit 1", "source=", "translation=", "span_count=2",
        "translated_line_count=1", "source_span_bboxes=", "cell_bbox=",
        "chosen_render_rects=",
    ))
    assert identity["render_status"] == "rendered"
    assert identity["final_source_ids"] == identity["source_ids"]
    x0, y0, x1, y1 = identity["final_bbox"]
    assert 20 <= x0 < x1 <= 180
    assert 20 <= y0 < y1 <= 60


def test_multiline_below_typography_floor_preserves_source(tmp_path: Path):
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
    assert any("typography floor" in warning for warning in stats.warnings)
    assert "Source" in output_text
    assert "Translated 14" not in output_text


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


def test_text_cover_does_not_erase_inline_vector_graphic(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=80)
    page.insert_text((20, 30), "Source", fontsize=9)
    page.draw_rect(
        fitz.Rect(70, 15, 90, 35), color=(1, 0, 0), fill=(1, 0, 0),
    )
    document.save(pdf_path)
    document.close()
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{
            "page_number": 1,
            "width": 200,
            "height": 80,
            "source_objects": [{
                "id": "p0001/b0001/l0001/s0001",
                "kind": "span",
                "bbox": [20, 20, 50, 32],
                "text": "Source",
            }],
            "units": [{
                "id": 1,
                "unit_type": "text",
                "source": "Source",
                "bbox": [20, 10, 120, 40],
                "fontsize": 9,
                "flags": 0,
                "translate": True,
                "source_ids": ["p0001/b0001/l0001/s0001"],
            }],
        }],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{
            "id": 1, "source": "Source", "translation": "Terjemahan",
        }],
    }), encoding="utf-8")

    output_path, stats = render_pdf(
        pdf_path, extraction_path, translation_path, tmp_path / "out.pdf"
    )
    pixmap = fitz.open(output_path)[0].get_pixmap(alpha=False)
    pixel = pixmap.pixel(80, 25)

    assert stats.rendered_units == 1
    assert pixel[0] > 200 and pixel[1] < 50 and pixel[2] < 50


def test_single_line_uses_source_baseline_without_textbox_shrink(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=80)
    page.insert_text((20, 30), "Heading", fontsize=12)
    document.save(pdf_path)
    document.close()
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{
            "page_number": 1,
            "source_objects": [{
                "id": "span", "kind": "span", "bbox": [20, 20, 70, 32],
                "text": "Heading", "metadata": {"origin": [20, 30]},
            }],
            "units": [{
                "id": 1, "unit_type": "text", "source": "Heading",
                "bbox": [20, 20, 70, 32], "fontsize": 12, "flags": 16,
                "line_count": 1, "translate": True, "source_ids": ["span"],
            }],
        }],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{
            "id": 1, "source": "Heading", "translation": "Judul",
        }],
    }), encoding="utf-8")

    output_path, stats = render_pdf(
        pdf_path, extraction_path, translation_path, tmp_path / "out.pdf"
    )
    spans = [
        span for block in fitz.open(output_path)[0].get_text("dict")["blocks"]
        if block.get("type") == 0 for line in block.get("lines", [])
        for span in line.get("spans", []) if span["text"] == "Judul"
    ]

    assert stats.rendered_units == 1
    assert len(spans) == 1
    assert spans[0]["size"] == 12


def test_overflow_below_typography_floor_preserves_source_without_masking(tmp_path: Path):
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

    identity_dir = tmp_path / "identity"
    output_path, stats = render_pdf(
        pdf_path,
        extraction_path,
        translation_path,
        tmp_path / "out.pdf",
        identity_debug_dir=identity_dir,
    )

    assert stats.rendered_units == 1
    output_text = " ".join(fitz.open(output_path)[0].get_text().split())
    identity = json.loads(
        (identity_dir / "render_identity.json").read_text(encoding="utf-8")
    )["units"][0]
    assert output_text == "Short"
    assert identity["render_status"] == "source_fallback"
    assert identity["font_size_ratio"] == 1
    assert identity["mask_rectangles"] == []
    assert identity["fallback_reason"] == "typography_floor"


def test_structured_span_mismatch_preserves_source_composition(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    page.insert_text((20, 25), "First", fontsize=8)
    page.insert_text((20, 40), "Second", fontsize=8)
    document.save(pdf_path)
    document.close()
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "width": 200, "height": 100, "units": [{
            "id": 1, "unit_type": "text", "source": "First\nSecond",
            "bbox": [20, 15, 80, 42], "fontsize": 8, "flags": 0,
            "line_count": 2, "translate": True,
        }]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{
            "id": 1, "source": "First\nSecond", "translation": "Terjemahan gabungan",
        }],
    }), encoding="utf-8")
    layout = LayoutPlan.model_validate({
        "page_number": 1, "width": 200, "height": 100, "confidence": 0.9,
        "source": "vision", "regions": [{
            "id": "structured", "type": "structured", "bbox": [20, 15, 80, 42],
            "reading_order": 0, "confidence": 0.9,
            "recommended_strategy": "structured_region", "unit_ids": [1],
        }],
    })

    identity_dir = tmp_path / "identity"
    output_path, stats = render_pdf(
        pdf_path, extraction_path, translation_path, tmp_path / "out.pdf",
        layout_plans={1: layout},
        identity_debug_dir=identity_dir,
    )
    output_text = " ".join(fitz.open(output_path)[0].get_text().split())
    identity = json.loads(
        (identity_dir / "render_identity.json").read_text(encoding="utf-8")
    )["units"][0]

    assert stats.rendered_units == 1
    assert "First Second" in output_text
    assert "Terjemahan gabungan" not in output_text
    assert any("structured span mismatch" in warning for warning in stats.warnings)
    assert identity["render_status"] == "source_fallback"
    assert identity["render_method"] == "source_preserved"
    assert identity["font_size_ratio"] == 1
    assert identity["mask_rectangles"] == []
    assert identity["fallback_reason"] == "uncertain_fragment_mapping"
    assert identity["visual_fallback_reason"] == "uncertain_fragment_mapping"


def test_render_plan_debug_artifacts_are_written(tmp_path: Path):
    pdf_path, extraction_path, translation_path = _make_inputs(tmp_path)
    debug_dir = tmp_path / "layout"

    render_pdf(
        pdf_path, extraction_path, translation_path, tmp_path / "out.pdf",
        render_plan_debug_dir=debug_dir,
        identity_debug_dir=debug_dir / "identity",
    )

    payload = json.loads((debug_dir / "page_001_render_plan.json").read_text(encoding="utf-8"))
    assert payload["regions"][0]["strategy"] == "reflow_region"
    assert (debug_dir / "page_001_render_plan_debug.png").exists()
    identity = json.loads(
        (debug_dir / "identity" / "render_identity.json").read_text(encoding="utf-8")
    )
    assert identity["units"][0]["translation_unit"] == 1
    assert identity["units"][0]["source_ids"] == identity["units"][0]["final_source_ids"]
    final_bbox = identity["units"][0]["final_bbox"]
    assert 15 <= final_bbox[0] < final_bbox[2] <= 55
    assert 15 <= final_bbox[1] < final_bbox[3] <= 35
    assert identity["units"][0]["source_font_size"] == 11
    assert identity["units"][0]["rendered_font_size"] is not None
    assert identity["units"][0]["mask_rectangles"]
    assert identity["schema_version"] == 2
    assert len(identity["source_sha256"]) == 64
    assert len(identity["source_extraction_sha256"]) == 64
    assert len(identity["source_translation_sha256"]) == 64
    assert len(identity["output_sha256"]) == 64
    assert (debug_dir / "identity" / "source_page_001_overlay.png").exists()
    assert (debug_dir / "identity" / "output_page_001_overlay.png").exists()


def test_render_rejects_cross_document_artifacts(tmp_path: Path):
    pdf_path, extraction_path, translation_path = _make_inputs(tmp_path)
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction["source_file"] = "different.pdf"
    extraction_path.write_text(json.dumps(extraction), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Extraction source mismatch"):
        render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")


def test_render_rejects_same_name_pdf_with_different_content(tmp_path: Path):
    pdf_path, extraction_path, translation_path = _make_inputs(tmp_path)
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction.update({
        "schema_version": 3,
        "artifact_type": "extraction",
        "source_file": pdf_path.name,
        "source_sha256": "0" * 64,
    })
    extraction_path.write_text(json.dumps(extraction), encoding="utf-8")

    with pytest.raises(RuntimeError, match="source hash mismatch"):
        render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")


def test_render_rejects_translation_from_stale_extraction(tmp_path: Path):
    pdf_path, extraction_path, translation_path = _make_inputs(tmp_path)
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction.update({
        "schema_version": 3,
        "artifact_type": "extraction",
        "source_file": pdf_path.name,
        "source_sha256": sha256_file(pdf_path),
    })
    extraction_path.write_text(json.dumps(extraction), encoding="utf-8")
    translation = json.loads(translation_path.read_text(encoding="utf-8"))
    translation.update({
        "schema_version": 2,
        "artifact_type": "translation",
        "source_file": pdf_path.name,
        "source_sha256": sha256_file(pdf_path),
        "source_extraction_sha256": "0" * 64,
    })
    translation_path.write_text(json.dumps(translation), encoding="utf-8")

    with pytest.raises(RuntimeError, match="extraction hash mismatch"):
        render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")


def test_render_rejects_layout_plan_from_another_document(tmp_path: Path):
    pdf_path, extraction_path, translation_path = _make_inputs(tmp_path)
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction.update({
        "schema_version": 3,
        "artifact_type": "extraction",
        "source_file": pdf_path.name,
        "source_sha256": sha256_file(pdf_path),
    })
    extraction_path.write_text(json.dumps(extraction), encoding="utf-8")
    translation = json.loads(translation_path.read_text(encoding="utf-8"))
    translation.update({
        "schema_version": 2,
        "artifact_type": "translation",
        "source_file": pdf_path.name,
        "source_sha256": sha256_file(pdf_path),
        "source_extraction_sha256": payload_sha256(extraction),
    })
    translation_path.write_text(json.dumps(translation), encoding="utf-8")
    layout = LayoutPlan.model_validate({
        "source_file": pdf_path.name,
        "source_sha256": "f" * 64,
        "source_extraction_sha256": payload_sha256(extraction),
        "page_number": 1, "width": 200, "height": 100,
        "regions": [{
            "id": "r1", "type": "paragraph", "bbox": [15, 15, 55, 35],
            "reading_order": 0, "confidence": 1,
            "recommended_strategy": "reflow_region", "unit_ids": [1],
        }],
    })

    with pytest.raises(RuntimeError, match="LayoutPlan page 1 source hash mismatch"):
        render_pdf(
            pdf_path, extraction_path, translation_path, tmp_path / "out.pdf",
            layout_plans={1: layout},
        )


def test_render_rejects_duplicate_translation_ids(tmp_path: Path):
    pdf_path, extraction_path, translation_path = _make_inputs(tmp_path)
    translation = json.loads(translation_path.read_text(encoding="utf-8"))
    translation["translations"].append(dict(translation["translations"][0]))
    translation_path.write_text(json.dumps(translation), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Duplicate translation ID"):
        render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")


def test_common_unicode_survives_rendering(tmp_path: Path):
    pdf_path, extraction_path, translation_path = _make_inputs(tmp_path)
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction["pages"][0]["units"][0]["bbox"] = [15, 15, 150, 50]
    extraction_path.write_text(json.dumps(extraction), encoding="utf-8")
    payload = json.loads(translation_path.read_text(encoding="utf-8"))
    payload["translations"][0]["translation"] = "Café • arah → selesai"
    translation_path.write_text(json.dumps(payload), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")
    output_text = fitz.open(output_path)[0].get_text()

    assert stats.rendered_units == 1
    assert "Café" in output_text
    assert "•" in output_text
    assert "→" in output_text
    assert "?" not in output_text


def test_private_use_glyph_preserves_original_visual_unit(tmp_path: Path):
    pdf_path, extraction_path, translation_path = _make_inputs(tmp_path)
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction["pages"][0]["units"][0]["source"] = "\ue000 Hello"
    extraction_path.write_text(json.dumps(extraction), encoding="utf-8")
    translation = json.loads(translation_path.read_text(encoding="utf-8"))
    translation["translations"][0].update({
        "source": "\ue000 Hello", "translation": "\ue000 Halo",
    })
    translation_path.write_text(json.dumps(translation), encoding="utf-8")

    output_path, stats = render_pdf(pdf_path, extraction_path, translation_path, tmp_path / "out.pdf")
    output_text = fitz.open(output_path)[0].get_text()

    assert "Hello" in output_text
    assert "Halo" not in output_text
    assert any("private-use glyph preserved" in warning for warning in stats.warnings)


def test_toc_title_wrap_keeps_page_number_anchor_and_regenerates_leader(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=220, height=100)
    page.insert_text((20, 30), "Short title", fontsize=9)
    page.insert_text((110, 30), "........", fontsize=9)
    page.insert_text((185, 30), "49", fontsize=9)
    document.save(pdf_path)
    document.close()
    extraction_path = tmp_path / "extraction.json"
    extraction_path.write_text(json.dumps({
        "pages": [{"page_number": 1, "width": 220, "height": 100, "units": [{
            "id": 1, "unit_type": "toc_entry", "source": "Short title",
            "bbox": [20, 20, 80, 32], "fontsize": 9, "flags": 0,
            "line_count": 1, "translate": True, "metadata": {
                "toc_page_number_bbox": [185, 20, 198, 32],
                "toc_hierarchy_level": 0, "toc_column": 0,
            },
        }]}],
    }), encoding="utf-8")
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(json.dumps({
        "translations": [{
            "id": 1, "source": "Short title",
            "translation": "Judul terjemahan panjang yang memerlukan baris kedua",
        }],
    }), encoding="utf-8")

    output_path, stats = render_pdf(
        pdf_path, extraction_path, translation_path, tmp_path / "out.pdf"
    )
    output_page = fitz.open(output_path)[0]
    spans = [
        span for block in output_page.get_text("dict")["blocks"] if block.get("type") == 0
        for line in block.get("lines", []) for span in line.get("spans", [])
    ]

    assert stats.rendered_units == 1
    assert any(span["text"] == "49" and round(span["bbox"][0]) == 185 for span in spans)
    assert "." in output_page.get_text()
    assert "Judul" in output_page.get_text()
