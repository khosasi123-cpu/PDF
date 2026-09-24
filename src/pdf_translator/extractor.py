from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import logging

import fitz

from .filters import should_translate
from .models import ExtractionResult, PageExtraction, TranslationUnit


EXTRACTION_OPTIONS = {"sort": True}
LINE_TOLERANCE = 1.5
MIN_TABLE_CELLS = 2
MIN_LINE_LENGTH = 12.0
MIN_OVERLAP_RATIO = 0.75
DEBUG_NEAR_TABLE_DISTANCE = 3.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableCell:
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class TableDetection:
    cells: tuple[TableCell, ...]
    bbox: tuple[float, float, float, float]


@dataclass
class _Segment:
    start: float
    end: float
    position: float


def _meaningful_spans(block: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        span
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span.get("text", "").strip()
    ]


def _reconstruct_line(line: dict[str, Any]) -> str:
    text_parts: list[str] = []
    previous_span: dict[str, Any] | None = None
    for span in line.get("spans", []):
        text = span.get("text", "")
        if not text.strip():
            continue
        if previous_span is not None:
            previous_text = previous_span.get("text", "")
            previous_bbox = previous_span.get("bbox", [0, 0, 0, 0])
            current_bbox = span.get("bbox", [0, 0, 0, 0])
            has_gap = current_bbox[0] - previous_bbox[2] > 0.5
            if not previous_text[-1:].isspace() and not text[0].isspace() and has_gap:
                text_parts.append(" ")
        text_parts.append(text)
        previous_span = span
    return "".join(text_parts).strip()


def _reconstruct_block(block: dict[str, Any]) -> str:
    lines = [_reconstruct_line(line) for line in block.get("lines", [])]
    return "\n".join(line for line in lines if line)


def _union_bbox(spans: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    boxes = [span["bbox"] for span in spans]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _representative_span(spans: list[dict[str, Any]]) -> dict[str, Any]:
    return max(spans, key=lambda span: len("".join(span.get("text", "").split())))


def _direction(block: dict[str, Any]) -> tuple[float, float] | None:
    for line in block.get("lines", []):
        direction = line.get("dir")
        if direction is not None:
            return tuple(float(value) for value in direction)
    return None


def _unit_from_block(
    block: dict[str, Any], page_number: int, unit_id: int, unit_type: str = "text"
) -> TranslationUnit | None:
    spans = _meaningful_spans(block)
    if not spans:
        return None
    representative = _representative_span(spans)
    source = _reconstruct_block(block)
    return TranslationUnit(
        id=unit_id,
        page_number=page_number,
        unit_type=unit_type,
        source=source,
        bbox=_union_bbox(spans),
        fontsize=float(representative.get("size", 0.0)),
        fontname=str(representative.get("font", "")),
        flags=int(representative.get("flags", 0)),
        color=representative.get("color"),
        direction=_direction(block),
        line_count=sum(1 for line in block.get("lines", []) if _reconstruct_line(line)),
        translate=should_translate(source),
    )


def block_to_unit(
    block: dict[str, Any], page_number: int, unit_id: int, unit_type: str = "text"
) -> TranslationUnit | None:
    return _unit_from_block(block, page_number, unit_id, unit_type)


def _cluster(values: list[float]) -> list[float]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or abs(value - sum(clusters[-1]) / len(clusters[-1])) > LINE_TOLERANCE:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _drawing_segments(page: Any) -> tuple[list[_Segment], list[_Segment]]:
    horizontal: list[_Segment] = []
    vertical: list[_Segment] = []
    for drawing in page.get_drawings() if hasattr(page, "get_drawings") else []:
        for item in drawing.get("items", []):
            if item[0] == "re":
                rect = fitz.Rect(item[1])
                if rect.width >= MIN_LINE_LENGTH and rect.height <= LINE_TOLERANCE:
                    horizontal.append(_Segment(rect.x0, rect.x1, (rect.y0 + rect.y1) / 2))
                elif rect.height >= MIN_LINE_LENGTH and rect.width <= LINE_TOLERANCE:
                    vertical.append(_Segment(rect.y0, rect.y1, (rect.x0 + rect.x1) / 2))
            elif item[0] == "l":
                start, end = item[1], item[2]
                if abs(start.y - end.y) <= LINE_TOLERANCE and abs(start.x - end.x) >= MIN_LINE_LENGTH:
                    horizontal.append(_Segment(min(start.x, end.x), max(start.x, end.x), (start.y + end.y) / 2))
                elif abs(start.x - end.x) <= LINE_TOLERANCE and abs(start.y - end.y) >= MIN_LINE_LENGTH:
                    vertical.append(_Segment(min(start.y, end.y), max(start.y, end.y), (start.x + end.x) / 2))
    return horizontal, vertical


def _covers(segments: list[_Segment], position: float, start: float, end: float) -> bool:
    return any(
        abs(segment.position - position) <= LINE_TOLERANCE
        and segment.start <= start + LINE_TOLERANCE
        and segment.end >= end - LINE_TOLERANCE
        for segment in segments
    )


def _touching(a: TableCell, b: TableCell) -> bool:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    horizontal_touch = abs(ax1 - bx0) <= LINE_TOLERANCE or abs(bx1 - ax0) <= LINE_TOLERANCE
    vertical_overlap = min(ay1, by1) - max(ay0, by0) > LINE_TOLERANCE
    vertical_touch = abs(ay1 - by0) <= LINE_TOLERANCE or abs(by1 - ay0) <= LINE_TOLERANCE
    horizontal_overlap = min(ax1, bx1) - max(ax0, bx0) > LINE_TOLERANCE
    return (horizontal_touch and vertical_overlap) or (vertical_touch and horizontal_overlap)


def detect_table_regions(page: Any) -> list[TableDetection]:
    horizontal, vertical = _drawing_segments(page)
    y_values = _cluster([segment.position for segment in horizontal])
    cells: list[TableCell] = []
    for y0, y1 in zip(y_values, y_values[1:]):
        active_vertical = [
            segment.position for segment in vertical
            if segment.start <= y0 + LINE_TOLERANCE and segment.end >= y1 - LINE_TOLERANCE
        ]
        x_values = _cluster(active_vertical)
        for x0, x1 in zip(x_values, x_values[1:]):
            if (
                _covers(horizontal, y0, x0, x1)
                and _covers(horizontal, y1, x0, x1)
                and _covers(vertical, x0, y0, y1)
                and _covers(vertical, x1, y0, y1)
            ):
                cells.append(TableCell((x0, y0, x1, y1)))

    remaining = set(range(len(cells)))
    regions: list[TableDetection] = []
    while remaining:
        component = {remaining.pop()}
        changed = True
        while changed:
            changed = False
            for index in tuple(remaining):
                if any(_touching(cells[index], cells[other]) for other in component):
                    remaining.remove(index)
                    component.add(index)
                    changed = True
        component_cells = tuple(cells[index] for index in sorted(component))
        x_count = len({round(cell.bbox[0], 1) for cell in component_cells})
        y_count = len({round(cell.bbox[1], 1) for cell in component_cells})
        if len(component_cells) >= MIN_TABLE_CELLS and (x_count > 1 or y_count > 1):
            regions.append(TableDetection(
                cells=component_cells,
                bbox=(
                    min(cell.bbox[0] for cell in component_cells),
                    min(cell.bbox[1] for cell in component_cells),
                    max(cell.bbox[2] for cell in component_cells),
                    max(cell.bbox[3] for cell in component_cells),
                ),
            ))
    return sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0]))


def _center(span: dict[str, Any]) -> tuple[float, float]:
    box = span["bbox"]
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _cell_for_span(span: dict[str, Any], cells: tuple[TableCell, ...]) -> TableCell | None:
    x, y = _center(span)
    span_x0, span_y0, span_x1, span_y1 = span["bbox"]
    matches = [
        cell for cell in cells
        if cell.bbox[0] < x < cell.bbox[2]
        and cell.bbox[1] < y < cell.bbox[3]
        and span_x0 >= cell.bbox[0] - LINE_TOLERANCE
        and span_y0 >= cell.bbox[1] - LINE_TOLERANCE
        and span_x1 <= cell.bbox[2] + LINE_TOLERANCE
        and span_y1 <= cell.bbox[3] + LINE_TOLERANCE
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return None

    span_area = max(0.0, span_x1 - span_x0) * max(0.0, span_y1 - span_y0)
    if not span_area:
        return None
    candidates: list[tuple[float, TableCell]] = []
    for cell in cells:
        overlap_width = max(0.0, min(span_x1, cell.bbox[2]) - max(span_x0, cell.bbox[0]))
        overlap_height = max(0.0, min(span_y1, cell.bbox[3]) - max(span_y0, cell.bbox[1]))
        overlap_ratio = (overlap_width * overlap_height) / span_area
        if overlap_ratio >= MIN_OVERLAP_RATIO:
            candidates.append((overlap_ratio, cell))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if candidates and (len(candidates) == 1 or candidates[0][0] > candidates[1][0]):
        return candidates[0][1]
    return None


def _block_cell(block: dict[str, Any], regions: list[TableDetection]) -> tuple[TableCell | None, bool]:
    spans = _meaningful_spans(block)
    table_cells = tuple(cell for region in regions for cell in region.cells)
    matches = [_cell_for_span(span, table_cells) for span in spans]
    if not any(match is not None for match in matches):
        overlaps_table = any(
            span["bbox"][2] > cell.bbox[0]
            and span["bbox"][0] < cell.bbox[2]
            and span["bbox"][3] > cell.bbox[1]
            and span["bbox"][1] < cell.bbox[3]
            for span in spans
            for cell in table_cells
        )
        return None, overlaps_table
    if any(match is None for match in matches) or len({match for match in matches}) != 1:
        return None, True
    return matches[0], False


def _log_assignment(
    block: dict[str, Any], cell: TableCell, regions: list[TableDetection], page_number: int
) -> None:
    table_id = next(index for index, region in enumerate(regions, start=1) if cell in region.cells)
    region = regions[table_id - 1]
    row = len({round(other.bbox[1], 1) for other in region.cells if other.bbox[1] < cell.bbox[1]}) + 1
    column = len({round(other.bbox[0], 1) for other in region.cells if other.bbox[0] < cell.bbox[0]}) + 1
    text_bbox = _union_bbox(_meaningful_spans(block))
    fully_contained = all(
        span["bbox"][0] >= cell.bbox[0] - LINE_TOLERANCE
        and span["bbox"][1] >= cell.bbox[1] - LINE_TOLERANCE
        and span["bbox"][2] <= cell.bbox[2] + LINE_TOLERANCE
        and span["bbox"][3] <= cell.bbox[3] + LINE_TOLERANCE
        for span in _meaningful_spans(block)
    )
    text_area = max(0.0, text_bbox[2] - text_bbox[0]) * max(0.0, text_bbox[3] - text_bbox[1])
    overlap_width = max(0.0, min(text_bbox[2], cell.bbox[2]) - max(text_bbox[0], cell.bbox[0]))
    overlap_height = max(0.0, min(text_bbox[3], cell.bbox[3]) - max(text_bbox[1], cell.bbox[1]))
    overlap_ratio = (overlap_width * overlap_height / text_area) if text_area else 0.0
    logger.debug(
        "[TABLE MATCH]\npage=%s\ntext=%r\ntext_bbox=%s\ntable_id=%s\n"
        "row=%s\ncolumn=%s\ncell_bbox=%s\nmethod=%s\noverlap_ratio=%.3f",
        page_number,
        _reconstruct_block(block), text_bbox, table_id, row, column, cell.bbox,
        "full_containment" if fully_contained else "overlap", overlap_ratio,
    )


def _near_table(block: dict[str, Any], regions: list[TableDetection]) -> bool:
    bbox = _union_bbox(_meaningful_spans(block))
    return any(
        bbox[2] >= region.bbox[0] - DEBUG_NEAR_TABLE_DISTANCE
        and bbox[0] <= region.bbox[2] + DEBUG_NEAR_TABLE_DISTANCE
        and bbox[3] >= region.bbox[1] - DEBUG_NEAR_TABLE_DISTANCE
        and bbox[1] <= region.bbox[3] + DEBUG_NEAR_TABLE_DISTANCE
        for region in regions
    )


def _log_near_miss(block: dict[str, Any], page_number: int, regions: list[TableDetection]) -> None:
    if _near_table(block, regions):
        bbox = _union_bbox(_meaningful_spans(block))
        candidate_cells = sum(
            bbox[2] >= cell.bbox[0] - DEBUG_NEAR_TABLE_DISTANCE
            and bbox[0] <= cell.bbox[2] + DEBUG_NEAR_TABLE_DISTANCE
            and bbox[3] >= cell.bbox[1] - DEBUG_NEAR_TABLE_DISTANCE
            and bbox[1] <= cell.bbox[3] + DEBUG_NEAR_TABLE_DISTANCE
            for region in regions
            for cell in region.cells
        )
        logger.debug(
            "[TABLE MISS]\npage=%s\ntext=%r\ntext_bbox=%s\n"
            "candidate_cells=%s\nreason=ambiguous_or_insufficient_overlap",
            page_number, _reconstruct_block(block), bbox, candidate_cells,
        )


def _block_bbox(block: dict[str, Any]) -> tuple[float, float, float, float]:
    return _union_bbox(_meaningful_spans(block))


def _has_vertical_overlap(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> bool:
    left_top = min(_block_bbox(block)[1] for block in left)
    left_bottom = max(_block_bbox(block)[3] for block in left)
    right_top = min(_block_bbox(block)[1] for block in right)
    right_bottom = max(_block_bbox(block)[3] for block in right)
    return min(left_bottom, right_bottom) > max(left_top, right_top)


def _column_ordered_blocks(page: Any, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order genuine two-column segments left-to-right without moving
    full-width separators into either column."""
    page_width = float(page.rect.width)
    midpoint = page_width / 2

    def side(block: dict[str, Any]) -> str:
        x0, _, x1, _ = _block_bbox(block)
        if x1 <= midpoint:
            return "left"
        if x0 >= midpoint:
            return "right"
        return "wide"

    left = [block for block in blocks if side(block) == "left"]
    right = [block for block in blocks if side(block) == "right"]
    if not left or not right or not _has_vertical_overlap(left, right):
        return blocks

    def order_segment(segment: list[dict[str, Any]]) -> list[dict[str, Any]]:
        segment_left = [block for block in segment if side(block) == "left"]
        segment_right = [block for block in segment if side(block) == "right"]
        if not segment_left or not segment_right or not _has_vertical_overlap(segment_left, segment_right):
            return segment
        position = lambda block: (_block_bbox(block)[1], _block_bbox(block)[0])
        return sorted(segment_left, key=position) + sorted(segment_right, key=position)

    ordered: list[dict[str, Any]] = []
    segment: list[dict[str, Any]] = []
    for block in sorted(blocks, key=lambda item: (_block_bbox(item)[1], _block_bbox(item)[0])):
        if side(block) != "wide":
            segment.append(block)
            continue
        ordered.extend(order_segment(segment))
        segment = []
        ordered.append(block)
    ordered.extend(order_segment(segment))
    return ordered


def _cell_unit(
    lines: list[dict[str, Any]], cell: TableCell, page_number: int
) -> TranslationUnit | None:
    spans = [span for line in lines for span in line.get("spans", []) if span.get("text", "").strip()]
    if not spans:
        return None
    lines = [
        {**line, "spans": sorted(line["spans"], key=lambda span: (span["bbox"][0], span["bbox"][1]))}
        for line in lines
        if _reconstruct_line(line)
    ]
    lines.sort(
        key=lambda line: (
            min(span["bbox"][1] for span in line["spans"]),
            min(span["bbox"][0] for span in line["spans"]),
        )
    )
    representative = _representative_span(spans)
    # A cell is the translation boundary. Visual line breaks are layout
    # artifacts here, so preserve reading order while joining with one space.
    source = " ".join(_reconstruct_line(line) for line in lines)
    x0, y0, x1, y1 = cell.bbox
    return TranslationUnit(
        id=0,
        page_number=page_number,
        unit_type="table_cell",
        source=source,
        bbox=cell.bbox,
        fontsize=float(representative.get("size", 0.0)),
        fontname=str(representative.get("font", "")),
        flags=int(representative.get("flags", 0)),
        color=representative.get("color"),
        direction=_direction({"lines": lines}),
        line_count=len(lines),
        translate=should_translate(source),
    )


def _extract_page_units_with_stats(
    page: Any, page_number: int, debug_assignments: bool = False
) -> tuple[list[TranslationUnit], int, int]:
    blocks = [block for block in page.get_text("dict", **EXTRACTION_OPTIONS).get("blocks", []) if block.get("type") == 0]
    regions = detect_table_regions(page)
    if not regions and hasattr(page, "rect"):
        blocks = _column_ordered_blocks(page, blocks)
    assigned: dict[TableCell, list[dict[str, Any]]] = {cell: [] for region in regions for cell in region.cells}

    # Keep the selected page reading order while table spans are assigned.
    ordered_items: list[tuple[str, TableCell | dict[str, Any]]] = []
    seen_cells: set[TableCell] = set()
    normal_blocks: list[dict[str, Any]] = []
    ambiguous = 0
    for block in blocks:
        if not _meaningful_spans(block):
            continue
        assigned_spans = 0
        fallback_lines: list[dict[str, Any]] = []
        for line in block.get("lines", []):
            line_spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
            if not line_spans:
                continue
            spans_by_cell: dict[TableCell, list[dict[str, Any]]] = {}
            fallback_spans: list[dict[str, Any]] = []
            for span in line_spans:
                span_block = {"lines": [{"spans": [span]}]}
                cell, is_ambiguous = _block_cell(span_block, regions)
                if cell is None:
                    fallback_spans.append(span)
                    ambiguous += int(is_ambiguous)
                    if debug_assignments:
                        _log_near_miss(span_block, page_number, regions)
                    continue
                spans_by_cell.setdefault(cell, []).append(span)
                assigned_spans += 1
                if debug_assignments:
                    _log_assignment(span_block, cell, regions, page_number)

            # Retain source-line grouping within each cell so span geometry can
            # restore word spacing. A line split across cells is split into
            # cell-specific fragments and never merged across boundaries.
            for cell, cell_spans in spans_by_cell.items():
                if cell not in seen_cells:
                    seen_cells.add(cell)
                    ordered_items.append(("cell", cell))
                assigned[cell].append({"dir": line.get("dir"), "spans": cell_spans})
            if fallback_spans:
                fallback_lines.append({"dir": line.get("dir"), "spans": fallback_spans})
        if fallback_lines:
            block_item = {"type": 0, "lines": fallback_lines}
            normal_blocks.append(block_item)
            ordered_items.append(("text", block_item))
        elif not assigned_spans:
            normal_blocks.append(block)
            ordered_items.append(("text", block))

    units: list[TranslationUnit] = []
    for item_type, item in ordered_items:
        if item_type == "cell":
            cell = item
            unit = _cell_unit(assigned[cell], cell, page_number)
        else:
            unit = block_to_unit(item, page_number, 0)
        if unit is not None:
            units.append(unit)
    for unit_id, unit in enumerate(units, start=1):
        unit.id = unit_id
    return units, len(regions), ambiguous


def extract_page_units(
    page: Any, page_number: int, first_unit_id: int = 1, debug_assignments: bool = False
) -> list[TranslationUnit]:
    units, _, _ = _extract_page_units_with_stats(page, page_number, debug_assignments)
    for offset, unit in enumerate(units):
        unit.id = first_unit_id + offset
    return units


def extract_pdf(pdf_path: Path, debug_assignments: bool = False) -> ExtractionResult:
    try:
        document = fitz.open(pdf_path)
    except (fitz.FileDataError, OSError) as error:
        raise RuntimeError(f"Unable to open PDF '{pdf_path}': {error}") from error
    if document.needs_pass:
        document.close()
        raise RuntimeError(f"Unable to open PDF '{pdf_path}': the file is encrypted and requires a password")

    pages: list[PageExtraction] = []
    next_unit_id = 1
    detected_tables = 0
    ambiguous_assignments = 0
    try:
        for page_index in range(document.page_count):
            page = document[page_index]
            units, table_count, ambiguous = _extract_page_units_with_stats(
                page, page_index + 1, debug_assignments
            )
            for offset, unit in enumerate(units):
                unit.id = next_unit_id + offset
            next_unit_id += len(units)
            detected_tables += table_count
            ambiguous_assignments += ambiguous
            pages.append(PageExtraction(
                page_number=page_index + 1,
                width=float(page.rect.width),
                height=float(page.rect.height),
                rotation=int(page.rotation),
                units=units,
            ))
    finally:
        document.close()
    return ExtractionResult(
        source_file=Path(pdf_path).name,
        pymupdf_version=str(getattr(fitz, "VersionBind", "unknown")),
        pages=pages,
        detected_tables=detected_tables,
        ambiguous_table_assignments=ambiguous_assignments,
    )


def save_extraction(result: ExtractionResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
