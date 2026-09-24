import json
from pathlib import Path

from pdf_translator.extractor import block_to_unit, detect_table_regions, extract_page_units, extract_pdf, save_extraction


def span(text, bbox, size=10, font="Arial", flags=0, color=0):
    return {"text": text, "bbox": bbox, "size": size, "font": font, "flags": flags, "color": color}


def representative_block():
    return {
        "type": 0,
        "lines": [{
            "dir": [1.0, 0.0],
            "spans": [
                span("Setting ", [10, 10, 50, 20]),
                span("Configuration", [50, 10, 120, 20], size=12, font="Arial-Bold", flags=16, color=5),
            ],
        }],
    }


def test_empty_spans_and_non_text_blocks_are_ignored():
    assert block_to_unit({"type": 0, "lines": [{"spans": [span("   ", [0, 0, 1, 1])]}]}, 1, 1) is None
    assert block_to_unit({"type": 1, "lines": []}, 1, 1) is None


def test_non_text_blocks_are_excluded_from_page_units():
    class MockPage:
        def get_text(self, mode, **options):
            return {"blocks": [{"type": 1}, {"type": 0, "lines": [{"spans": [span("Text", [0, 0, 20, 10])]}]}]}

    units = extract_page_units(MockPage(), 1, 1)
    assert [unit.source for unit in units] == ["Text"]


def test_block_reconstruction_metadata_and_one_based_page_number():
    unit = block_to_unit(representative_block(), 3, 8)
    assert unit is not None
    assert unit.page_number == 3
    assert unit.unit_type == "text"
    assert unit.source == "Setting Configuration"
    assert unit.direction == (1.0, 0.0)
    assert unit.color == 5
    assert unit.line_count == 1
    assert unit.fontname == "Arial-Bold"
    assert unit.fontsize == 12
    assert unit.flags == 16


def test_multiple_lines_and_bbox_union():
    block = {
        "type": 0,
        "lines": [
            {"dir": [1, 0], "spans": [span("First", [10, 10, 30, 20])]},
            {"dir": [1, 0], "spans": [span("second", [5, 20, 45, 30])]},
        ],
    }
    unit = block_to_unit(block, 1, 1)
    assert unit is not None
    assert unit.source == "First\nsecond"
    assert unit.bbox == (5, 10, 45, 30)
    assert unit.line_count == 2


def test_page_units_preserve_pymupdf_block_order_instead_of_global_y_x_sort():
    class MockPage:
        def get_text(self, mode, **options):
            return {"blocks": [
                {"type": 0, "lines": [{"spans": [span("First column", [100, 20, 160, 30])]}]},
                {"type": 0, "lines": [{"spans": [span("Second column", [10, 10, 80, 20])]}]},
            ]}

    units = extract_page_units(MockPage(), 1)

    assert [unit.source for unit in units] == ["First column", "Second column"]


def test_two_columns_are_ordered_left_to_right_then_top_to_bottom():
    class MockPage:
        rect = type("Rect", (), {"width": 600})()

        def get_text(self, mode, **options):
            return {"blocks": [
                {"type": 0, "lines": [{"spans": [span("Right top", [310, 10, 380, 20])]}]},
                {"type": 0, "lines": [{"spans": [span("Left top", [40, 30, 100, 40])]}]},
                {"type": 0, "lines": [{"spans": [span("Right bottom", [310, 40, 390, 50])]}]},
                {"type": 0, "lines": [{"spans": [span("Left bottom", [40, 50, 110, 60])]}]},
            ]}

    units = extract_page_units(MockPage(), 1)

    assert [unit.source for unit in units] == [
        "Left top", "Left bottom", "Right top", "Right bottom"
    ]


def test_wide_block_does_not_trigger_column_grouping():
    class MockPage:
        rect = type("Rect", (), {"width": 600})()

        def get_text(self, mode, **options):
            return {"blocks": [
                {"type": 0, "lines": [{"spans": [span("Left", [40, 20, 100, 30])]}]},
                {"type": 0, "lines": [{"spans": [span("Wide heading", [40, 40, 580, 50])]}]},
                {"type": 0, "lines": [{"spans": [span("Right", [310, 20, 370, 30])]}]},
                {"type": 0, "lines": [{"spans": [span("Left later", [40, 60, 110, 70])]}]},
            ]}

    units = extract_page_units(MockPage(), 1)

    assert [unit.source for unit in units] == ["Left", "Right", "Wide heading", "Left later"]


def test_representative_font_uses_non_whitespace_character_count():
    block = {
        "type": 0,
        "lines": [{"spans": [
            span("A", [0, 0, 5, 5], font="Short"),
            span("Long representative", [5, 0, 50, 5], font="Long"),
        ]}],
    }
    unit = block_to_unit(block, 1, 1)
    assert unit is not None
    assert unit.fontname == "Long"


def test_multispan_spacing_does_not_join_words():
    block = {"type": 0, "lines": [{"spans": [
        span("Server", [0, 0, 40, 10]), span("Configuration", [45, 0, 100, 10])
    ]}]}
    unit = block_to_unit(block, 1, 1)
    assert unit is not None
    assert unit.source == "Server Configuration"


def test_extraction_json_is_valid(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "input.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Hello PDF")
    document.save(pdf_path)
    document.close()
    result = extract_pdf(pdf_path)
    output = tmp_path / "extraction.json"
    save_extraction(result, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["text_extraction_options"] == {"sort": True}
    assert payload["pages"][0]["page_number"] == 1
    assert payload["pages"][0]["units"][0]["translate"] is True


def test_skipped_units_remain_in_extraction_json(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "input.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "12345")
    document.save(pdf_path)
    document.close()
    result = extract_pdf(pdf_path)
    output = tmp_path / "extraction.json"
    save_extraction(result, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["pages"][0]["units"][0]["source"] == "12345"
    assert payload["pages"][0]["units"][0]["translate"] is False


def _table_page(blocks):
    import fitz

    class MockPage:
        def get_text(self, mode, **options):
            return {"blocks": blocks}

        def get_drawings(self):
            lines = []
            for x in (10, 50, 90):
                lines.append({"items": [("re", fitz.Rect(x, 10, x + 0.5, 50), -1)]})
            for y in (10, 30, 50):
                lines.append({"items": [("re", fitz.Rect(10, y, 90, y + 0.5), -1)]})
            return lines

    return MockPage()


def test_detected_table_cells_merge_blocks_and_keep_normal_text():
    blocks = [
        {"type": 0, "lines": [{"dir": [1, 0], "spans": [span("Header", [15, 15, 35, 25])]}]},
        {"type": 0, "lines": [{"dir": [1, 0], "spans": [span("continued", [15, 25, 40, 29])]}]},
        {"type": 0, "lines": [{"dir": [1, 0], "spans": [span("Other", [55, 15, 75, 25])]}]},
        {"type": 0, "lines": [{"dir": [1, 0], "spans": [span("Paragraph", [100, 60, 150, 70])]}]},
    ]
    page = _table_page(blocks)
    regions = detect_table_regions(page)
    assert len(regions) == 1
    assert len(regions[0].cells) == 4
    units = extract_page_units(page, 2)
    assert [unit.page_number for unit in units] == [2] * 3
    cell_units = [unit for unit in units if unit.unit_type == "table_cell"]
    assert len(cell_units) == 2
    assert any(unit.source == "Header continued" for unit in cell_units)
    assert any(unit.unit_type == "text" and unit.source == "Paragraph" for unit in units)
    assert [unit.id for unit in units] == [1, 2, 3]


def test_spans_in_one_source_line_are_assigned_to_their_own_cells():
    blocks = [{"type": 0, "lines": [{"dir": [1, 0], "spans": [
        span("Left", [15, 15, 35, 25]), span("Right", [55, 15, 75, 25])
    ]}]}]

    units = extract_page_units(_table_page(blocks), 1)

    assert [(unit.unit_type, unit.source, unit.bbox) for unit in units] == [
        ("table_cell", "Left", (10.25, 10.25, 50.25, 30.25)),
        ("table_cell", "Right", (50.25, 10.25, 90.25, 30.25)),
    ]


def test_ambiguous_line_is_not_forced_into_a_cell():
    blocks = [{"type": 0, "lines": [{"dir": [1, 0], "spans": [
        span("crossing", [35, 15, 65, 25])
    ]}]}]
    units = extract_page_units(_table_page(blocks), 1)
    assert len(units) == 1
    assert units[0].unit_type == "text"
    assert units[0].source == "crossing"


def test_small_bbox_overflow_uses_strongest_cell_overlap():
    blocks = [{"type": 0, "lines": [{"dir": [1, 0], "spans": [
        span("Overflow", [15, 15, 52, 25])
    ]}]}]
    units = extract_page_units(_table_page(blocks), 1)
    assert len(units) == 1
    assert units[0].unit_type == "table_cell"
    assert units[0].source == "Overflow"


def test_smoke_pdf_without_table_geometry_remains_text(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "smoke.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Normal text")
    document.save(pdf_path)
    document.close()
    result = extract_pdf(pdf_path)
    assert result.detected_tables == 0
    assert all(unit.unit_type == "text" for unit in result.units)


def test_real_hums_table_assignments_if_fixture_is_available():
    pdf_path = Path("data/input/test.pdf")
    if not pdf_path.exists():
        return
    result = extract_pdf(pdf_path)
    by_source = {unit.source: unit for unit in result.units}
    for source in [
        "EDMS maximum timeout",
        "Type of field",
        "Possible values",
        "Time format",
        "Free entry fields: numbers only",
        "Local time = UTC time +/-",
    ]:
        assert by_source[source].unit_type == "table_cell"
    expected_cell_text = {
        1: [
            "EDMS maximum timeout",
            "Answer granted time to EDMS",
            "Name or IP address of the server hosting EDMS",
            "Reading of data recorded on the MDC",
        ],
        2: [
            "Local time = UTC time +/-",
            "Defines the time difference between local time and UTC time according to the operated Time Zone",
            "Time delay in seconds before a message exchange with MMS is declared timeout",
            "The three letters displayed in the top left corner of the Flight View",
        ],
        3: [
            "Indicates the directory on the server in which the Flight Data will be stored on the server",
            "Indicates the directory on the server in which the log files will be stored",
            "Indicates the directory where messages sent by the MMS are saved",
            "Indicates the directory where messages transmitted to the MMS are saved",
        ],
    }
    for page_number, expected_sources in expected_cell_text.items():
        cell_sources = [
            unit.source for unit in result.pages[page_number - 1].units
            if unit.unit_type == "table_cell"
        ]
        for expected_source in expected_sources:
            assert any(expected_source in source for source in cell_sources)
    assert by_source["Setting data modification for backup of dialogs originating from MMS entails restarting the\nserver application."].unit_type == "text"
    assert by_source["Figure 1: Constants of the application"].unit_type == "text"
