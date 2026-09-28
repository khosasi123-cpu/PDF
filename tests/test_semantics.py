from pdf_translator.models import ExtractionResult, PageExtraction, SourceObject, TranslationUnit
from pdf_translator.semantics import analyze_document_semantics


def make_unit(unit_id, page, source, bbox, source_ids):
    return TranslationUnit(
        id=unit_id, page_number=page, unit_type="text", source=source, bbox=bbox,
        fontsize=9, fontname="Arial", flags=0, color=0, direction=(1, 0),
        line_count=1, translate=True, source_ids=source_ids,
    )


def test_repeated_headers_and_footers_are_grouped_without_swapping_fields():
    pages = []
    next_id = 1
    for page_number in range(1, 5):
        units = [
            make_unit(next_id, page_number, f"Manual {page_number}", (10, 5, 80, 15), []),
            make_unit(next_id + 1, page_number, f"Manual {page_number}", (120, 5, 190, 15), []),
            make_unit(next_id + 2, page_number, str(page_number), (95, 185, 105, 195), []),
        ]
        next_id += 3
        pages.append(PageExtraction(page_number, 200, 200, 0, units))
    result = analyze_document_semantics(ExtractionResult("doc.pdf", "test", pages))

    left_groups = {page.units[0].template_group_id for page in result.pages}
    right_groups = {page.units[1].template_group_id for page in result.pages}
    footer_groups = {page.units[2].template_group_id for page in result.pages}
    assert len(left_groups) == len(right_groups) == len(footer_groups) == 1
    assert left_groups != right_groups
    assert all(page.units[0].semantic_role == "running_header" for page in result.pages)
    assert all(page.units[2].semantic_role == "running_footer" for page in result.pages)
    assert all(page.units[0].recurrence_count == 4 for page in result.pages)


def _toc_page(two_columns=False):
    units = []
    sources = []
    for index in range(4):
        column = index % 2 if two_columns else 0
        x_offset = 100 * column
        y = 20 + (index // 2 if two_columns else index) * 20
        title_x = x_offset + 10 + (5 if index == 1 else 0)
        number_x = x_offset + 85
        title_id = f"p0001/b{index + 1:04d}/l0001/s0001"
        leader_id = f"p0001/b{index + 1:04d}/l0001/s0002"
        number_id = f"p0001/b{index + 1:04d}/l0001/s0003"
        unit_id = index + 1
        title = f"Section {index + 1}"
        sources.extend([
            SourceObject(title_id, "span", (title_x, y, title_x + 35, y + 9), title, unit_ids=[unit_id]),
            SourceObject(leader_id, "span", (title_x + 38, y, number_x - 3, y + 9), ".....", unit_ids=[unit_id]),
            SourceObject(number_id, "span", (number_x, y, number_x + 8, y + 9), str(index + 1), unit_ids=[unit_id]),
        ])
        units.append(make_unit(
            unit_id, 1, f"{title} ..... {index + 1}",
            (title_x, y, number_x + 8, y + 9), [title_id, leader_id, number_id],
        ))
    return PageExtraction(1, 200, 150, 0, units, source_objects=sources)


def test_toc_entries_preserve_title_hierarchy_page_anchor_and_columns():
    page = _toc_page(two_columns=True)
    result = analyze_document_semantics(ExtractionResult("doc.pdf", "test", [page]))

    assert len(result.pages[0].toc_entries) == 4
    assert all(unit.unit_type == "toc_entry" for unit in result.pages[0].units)
    assert [unit.source for unit in result.pages[0].units] == [
        "Section 1", "Section 2", "Section 3", "Section 4"
    ]
    assert {entry.column for entry in result.pages[0].toc_entries} == {0, 1}
    assert result.pages[0].toc_entries[1].hierarchy_level == 1
    assert all(entry.page_label == str(index) for index, entry in enumerate(result.pages[0].toc_entries, 1))
    assert all(entry.page_number_bbox[0] >= entry.title_bbox[2] for entry in result.pages[0].toc_entries)


def test_repeated_x_anchors_mark_borderless_key_value_rows_structured():
    units = []
    sources = []
    for index, y in enumerate((20, 40, 60), start=1):
        left_id = f"p0001/b{index:04d}/l0001/s0001"
        right_id = f"p0001/b{index:04d}/l0001/s0002"
        sources.extend([
            SourceObject(left_id, "span", (10, y, 45, y + 9), f"Key {index}", unit_ids=[index]),
            SourceObject(right_id, "span", (100, y, 160, y + 9), f"Value {index}", unit_ids=[index]),
        ])
        units.append(make_unit(index, 1, f"Key {index}\nValue {index}", (10, y, 160, y + 9), [left_id, right_id]))
    page = PageExtraction(1, 200, 100, 0, units, source_objects=sources)

    result = analyze_document_semantics(ExtractionResult("doc.pdf", "test", [page]))

    assert len(result.pages[0].units) == 6
    assert all(unit.unit_type == "borderless_structured" for unit in result.pages[0].units)
    assert [round(unit.bbox[0]) for unit in result.pages[0].units] == [10, 100, 10, 100, 10, 100]
    assert all(unit.metadata["geometry_source"] == "inferred_borderless_grid" for unit in result.pages[0].units)
