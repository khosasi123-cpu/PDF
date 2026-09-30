from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz

from .layout import LayoutPlan, RegionType, RenderingStrategy, fallback_layout_plan, resolve_layout_plan_geometry
from .render_plan import PageGeometry, RenderPlan, RenderPlanner
from .identity import (
    complete_identity_record,
    identity_record,
    payload_sha256,
    validate_artifact_identity,
    validate_layout_identity,
    validate_source_ownership,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_EXTRACTION_PATH = Path("artifacts/extraction/extraction.json")
DEFAULT_TRANSLATION_PATH = Path("artifacts/translation/translation.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/rendered")
DEFAULT_LAYOUT_DIR = Path("artifacts/layout")
MIN_FONT_SIZE = 5.5
STRUCTURED_MIN_FONT_SIZE = 4.5
MIN_FONT_RATIO = 0.75
STRUCTURED_Y_TOLERANCE = 1.5
PADDING = 0.75
CELL_BORDER_INSET = 1.5
_SYMBOL_MAP = {
    "\u25cf": "\u2022",
}


@dataclass
class RenderStats:
    total_units: int = 0
    translated_units: int = 0
    skipped_units: int = 0
    rendered_units: int = 0
    missing_translations: int = 0
    warnings: list[str] = field(default_factory=list)
    identity_records: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _StructuredSpan:
    source_rect: fitz.Rect
    render_rect: fitz.Rect
    origin: fitz.Point


@dataclass(frozen=True)
class _TextRenderResult:
    fits: bool
    font_size: float | None
    method: str


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read JSON '{path}': {error}") from error


def load_layout_plans(directory: Path) -> dict[int, LayoutPlan]:
    plans: dict[int, LayoutPlan] = {}
    if not directory.exists():
        return plans
    for path in sorted(directory.glob("page_[0-9][0-9][0-9].json")):
        try:
            plan = LayoutPlan.model_validate(_load_json(path))
        except ValueError as error:
            raise RuntimeError(f"Invalid LayoutPlan '{path}': {error}") from error
        if plan.page_number in plans:
            raise RuntimeError(f"Duplicate LayoutPlan for page {plan.page_number}")
        plans[plan.page_number] = plan
    return plans


def _translation_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    translations = payload.get("translations", [])
    result: dict[int, dict[str, Any]] = {}
    for item in translations:
        unit_id = item.get("id")
        if isinstance(unit_id, int):
            if unit_id in result:
                raise RuntimeError(f"Duplicate translation ID: {unit_id}")
            result[unit_id] = item
    return result


def _plain_text(value: str) -> str:
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"</?(?:b|i|strong|em)>", "", value, flags=re.IGNORECASE)
    value = value.replace("**", "").replace("*", "")
    return "".join(_SYMBOL_MAP.get(character, character) for character in value)


def _has_private_use(text: str) -> bool:
    return any(unicodedata.category(character) == "Co" for character in text)


def _needs_unicode_fallback(text: str) -> bool:
    return any(ord(character) > 127 for character in text)


def _font_name(flags: int) -> str:
    bold = bool(flags & 16)
    italic = bool(flags & 2)
    if bold and italic:
        return "helv"
    if bold:
        return "hebo"
    if italic:
        return "heit"
    return "helv"


def _unit_rect(unit: dict[str, Any]) -> fitz.Rect | None:
    bbox = unit.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        rect = fitz.Rect(*(float(value) for value in bbox))
    except (TypeError, ValueError):
        return None
    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        return None
    return rect


def _owned_source_span_rects(
    unit: dict[str, Any], page_data: dict[str, Any]
) -> list[fitz.Rect]:
    source_ids = set(unit.get("source_ids", []))
    return [
        fitz.Rect(source["bbox"])
        for source in page_data.get("source_objects", [])
        if source.get("id") in source_ids
        and source.get("kind") == "span"
        and isinstance(source.get("bbox"), (list, tuple))
        and len(source["bbox"]) == 4
    ]


def _metadata_source_rects(
    unit: dict[str, Any], page_data: dict[str, Any], metadata_key: str
) -> list[fitz.Rect]:
    source_ids = set(unit.get("metadata", {}).get(metadata_key, []))
    return [
        fitz.Rect(source["bbox"])
        for source in page_data.get("source_objects", [])
        if source.get("id") in source_ids
        and isinstance(source.get("bbox"), (list, tuple))
        and len(source["bbox"]) == 4
    ]


def _source_origin(
    unit: dict[str, Any], page_data: dict[str, Any], fallback: fitz.Point
) -> fitz.Point:
    source_ids = set(unit.get("source_ids", []))
    for source in page_data.get("source_objects", []):
        if source.get("id") not in source_ids or source.get("kind") != "span":
            continue
        origin = source.get("metadata", {}).get("origin")
        if isinstance(origin, (list, tuple)) and len(origin) == 2:
            return fitz.Point(float(origin[0]), float(origin[1]))
    return fallback


def _unit_color(unit: dict[str, Any]) -> tuple[float, float, float]:
    color = unit.get("color")
    if isinstance(color, int):
        return fitz.sRGB_to_pdf(color)
    return (0.0, 0.0, 0.0)


def _render_single_line_baseline(
    page: fitz.Page,
    unit: dict[str, Any],
    page_data: dict[str, Any],
    text: str,
    rect: fitz.Rect,
    fontsize: float,
    flags: int,
    minimum_ratio: float = MIN_FONT_RATIO,
    draw: bool = True,
) -> tuple[_TextRenderResult, fitz.Rect] | None:
    if (
        "\n" in text
        or int(unit.get("line_count", 1)) != 1
        or _needs_unicode_fallback(text)
    ):
        return None
    fontname = _font_name(flags)
    source_origin = _source_origin(
        unit, page_data, fitz.Point(rect.x0, rect.y0 + fontsize)
    )
    origin = fitz.Point(
        max(rect.x0, min(source_origin.x, rect.x1)),
        max(rect.y0 + min(fontsize, rect.height), min(source_origin.y, rect.y1)),
    )
    minimum_size = max(MIN_FONT_SIZE, fontsize * minimum_ratio)
    size = _measure_structured_line(
        text,
        max(rect.x1 - origin.x, 1.0),
        fontsize,
        fontname,
        minimum_size,
    )
    if size is None:
        return None
    if draw:
        page.insert_text(
            origin,
            text,
            fontsize=size,
            fontname=fontname,
            color=_unit_color(unit),
            overlay=True,
        )
    drawn = fitz.Rect(
        origin.x,
        origin.y - size,
        min(rect.x1, origin.x + fitz.get_text_length(text, fontname=fontname, fontsize=size)),
        origin.y + size * 0.25,
    )
    return _TextRenderResult(True, size, "source_baseline"), drawn


def _set_visual_diagnostics(
    identity: dict[str, Any],
    source_font_size: float,
    rendered_font_size: float | None,
    method: str | None,
    obstacles: list[fitz.Rect],
    masks: list[fitz.Rect],
    safe_rect: fitz.Rect | None,
    reason: str | None = None,
) -> None:
    identity["rendered_font_size"] = rendered_font_size
    identity["font_size_ratio"] = (
        rendered_font_size / source_font_size
        if rendered_font_size is not None and source_font_size > 0
        else None
    )
    identity["render_method"] = method
    identity["detected_visual_obstacles"] = [list(rect) for rect in obstacles]
    identity["mask_rectangles"] = [list(rect) for rect in masks]
    identity["safe_render_bbox"] = list(safe_rect) if safe_rect is not None else None
    identity["visual_fallback_reason"] = reason


def _complete_visual_source_fallback(
    identity: dict[str, Any],
    rect: fitz.Rect,
    source_text: str,
    fontsize: float,
    obstacles: list[fitz.Rect],
    reason: str,
) -> None:
    _set_visual_diagnostics(
        identity,
        fontsize,
        fontsize,
        "source_preserved",
        obstacles,
        [],
        rect,
        reason,
    )
    complete_identity_record(
        identity, [rect], source_text, "source_fallback", reason
    )


def _cover_rect(unit: dict[str, Any], rect: fitz.Rect) -> fitz.Rect:
    if unit.get("unit_type") == "table_cell":
        return fitz.Rect(
            rect.x0 + CELL_BORDER_INSET,
            rect.y0 + CELL_BORDER_INSET,
            rect.x1 - CELL_BORDER_INSET,
            rect.y1 - CELL_BORDER_INSET,
        )
    return fitz.Rect(
        rect.x0 - PADDING,
        rect.y0 - PADDING,
        rect.x1 + PADDING,
        rect.y1 + PADDING,
    )


def _is_compact_structured_unit(unit: dict[str, Any], rect: fitz.Rect) -> bool:
    if unit.get("unit_type") == "table_cell":
        return True
    line_count = unit.get("line_count")
    fontsize = unit.get("fontsize")
    return (
        isinstance(line_count, int)
        and line_count > 1
        and isinstance(fontsize, (int, float))
        and rect.height <= float(fontsize) * 1.5
    )


def _available_right_edge(
    page: fitz.Page, source_rect: fitz.Rect, right_limit: float | None = None
) -> float:
    right = min(page.rect.x1, right_limit) if right_limit is not None else page.rect.x1
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                bbox = span.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                candidate = fitz.Rect(bbox)
                if (
                    candidate.x0 > source_rect.x1 + PADDING
                    and candidate.y1 > source_rect.y0
                    and candidate.y0 < source_rect.y1
                ):
                    right = min(right, candidate.x0 - PADDING)
    for obstacle in _page_obstacle_rects(page):
        if (
            obstacle.x0 > source_rect.x1 + PADDING
            and obstacle.y1 > source_rect.y0
            and obstacle.y0 < source_rect.y1
        ):
            right = min(right, obstacle.x0 - PADDING)
    return right


def _structured_spans(
    page: fitz.Page,
    rect: fitz.Rect,
    expand_last_column: bool = False,
    right_limit: float | None = None,
) -> list[_StructuredSpan]:
    source_spans: list[tuple[fitz.Rect, fitz.Point]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                bbox = span.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                span_rect = fitz.Rect(*(float(value) for value in bbox))
                if span.get("text", "").strip() and rect.contains(span_rect):
                    origin = span.get("origin")
                    point = (
                        fitz.Point(float(origin[0]), float(origin[1]))
                        if isinstance(origin, (list, tuple)) and len(origin) == 2
                        else fitz.Point(span_rect.x0, span_rect.y1)
                    )
                    source_spans.append((span_rect, point))
    source_spans.sort(key=lambda item: (item[0].y0, item[0].x0))
    rows: list[list[tuple[fitz.Rect, fitz.Point]]] = []
    for span in source_spans:
        if rows and abs(span[0].y0 - rows[-1][0][0].y0) <= STRUCTURED_Y_TOLERANCE:
            rows[-1].append(span)
        else:
            rows.append([span])
    result: list[_StructuredSpan] = []
    for row_index, row in enumerate(rows):
        row.sort(key=lambda item: item[0].x0)
        next_row_y = rows[row_index + 1][0][0].y0 if row_index + 1 < len(rows) else rect.y1
        for index, (span_rect, origin) in enumerate(row):
            if index + 1 < len(row):
                right = row[index + 1][0].x0 - PADDING
            elif expand_last_column:
                right = _available_right_edge(page, span_rect)
            else:
                right = rect.x1
            bottom = max(span_rect.y1, next_row_y - PADDING)
            result.append(_StructuredSpan(
                source_rect=span_rect,
                render_rect=fitz.Rect(span_rect.x0, span_rect.y0, max(span_rect.x1, right), bottom),
                origin=origin,
            ))
    return result


def _extracted_structured_spans(
    page: fitz.Page,
    unit: dict[str, Any],
    page_data: dict[str, Any],
    rect: fitz.Rect,
    expand_last_column: bool = False,
    right_limit: float | None = None,
) -> list[_StructuredSpan]:
    ordered_source_ids = [
        source_id for source_id in unit.get("source_ids", [])
        if isinstance(source_id, str)
    ]
    source_ids = set(ordered_source_ids)
    values: list[tuple[str, fitz.Rect, fitz.Point]] = []
    for source in page_data.get("source_objects", []):
        if source.get("id") not in source_ids or source.get("kind") != "span":
            continue
        bbox = source.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        span_rect = fitz.Rect(bbox)
        origin = source.get("metadata", {}).get("origin")
        point = (
            fitz.Point(float(origin[0]), float(origin[1]))
            if isinstance(origin, (list, tuple)) and len(origin) == 2
            else fitz.Point(span_rect.x0, span_rect.y1)
        )
        values.append((source["id"], span_rect, point))
    if not values:
        return _structured_spans(page, rect, expand_last_column, right_limit)
    values.sort(key=lambda item: (item[1].y0, item[1].x0))
    rows: list[list[tuple[str, fitz.Rect, fitz.Point]]] = []
    for value in values:
        if rows and abs(value[1].y0 - rows[-1][0][1].y0) <= STRUCTURED_Y_TOLERANCE:
            rows[-1].append(value)
        else:
            rows.append([value])
    result_by_id: dict[str, _StructuredSpan] = {}
    for row_index, row in enumerate(rows):
        row.sort(key=lambda item: item[1].x0)
        next_y = rows[row_index + 1][0][1].y0 if row_index + 1 < len(rows) else rect.y1
        for index, (source_id, span_rect, origin) in enumerate(row):
            if index + 1 < len(row):
                right = row[index + 1][1].x0 - PADDING
            elif expand_last_column:
                right = _available_right_edge(page, span_rect, right_limit)
            else:
                right = rect.x1
            result_by_id[source_id] = _StructuredSpan(
                source_rect=span_rect,
                render_rect=fitz.Rect(
                    span_rect.x0, span_rect.y0, max(span_rect.x1, right),
                    max(span_rect.y1, next_y - PADDING),
                ),
                origin=origin,
            )
    return [result_by_id[source_id] for source_id in ordered_source_ids if source_id in result_by_id]


def _structured_span_rects(page: fitz.Page, rect: fitz.Rect) -> list[fitz.Rect]:
    return [span.render_rect for span in _structured_spans(page, rect)]


def _has_multi_column_rows(span_rects: list[fitz.Rect]) -> bool:
    rows: list[list[fitz.Rect]] = []
    for span in span_rects:
        if rows and abs(span.y0 - rows[-1][0].y0) <= STRUCTURED_Y_TOLERANCE:
            rows[-1].append(span)
        else:
            rows.append([span])
    return any(len(row) > 1 for row in rows)


def _has_repeated_column_geometry(span_rects: list[fitz.Rect]) -> bool:
    rows: list[list[fitz.Rect]] = []
    for span in span_rects:
        if rows and abs(span.y0 - rows[-1][0].y0) <= STRUCTURED_Y_TOLERANCE:
            rows[-1].append(span)
        else:
            rows.append([span])
    multi_column_rows = [
        row for row in rows
        if len(row) > 1
        and max(
            right.x0 - left.x0
            for left, right in zip(row, row[1:])
        ) >= max(span.height for span in row) * 3
    ]
    for index, row in enumerate(multi_column_rows):
        for other in multi_column_rows[index + 1:]:
            aligned = sum(
                any(abs(span.x0 - candidate.x0) <= 12 for candidate in other)
                for span in row
            )
            if aligned >= 2:
                return True
    return False


def _page_obstacle_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Bboxes of raster images and vector drawings on the page. These must
    be treated as occupied space: growing a text rect into 'empty' space
    must never grow into an image, or the white cover we draw before
    re-inserting text will blank the image out."""
    obstacles: list[fitz.Rect] = []
    try:
        for image_info in page.get_image_info():
            bbox = image_info.get("bbox")
            if bbox:
                obstacles.append(fitz.Rect(bbox))
    except Exception:  # pragma: no cover - defensive, keep rendering going
        LOGGER.debug("get_image_info failed on page %s", page.number, exc_info=True)
    try:
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect:
                obstacles.append(fitz.Rect(rect))
    except Exception:  # pragma: no cover - defensive, keep rendering going
        LOGGER.debug("get_drawings failed on page %s", page.number, exc_info=True)
    return obstacles


def _page_image_rects(page: fitz.Page) -> list[fitz.Rect]:
    images: list[fitz.Rect] = []
    seen: set[tuple[float, float, float, float]] = set()
    try:
        for image_info in page.get_image_info():
            bbox = image_info.get("bbox")
            if bbox:
                rect = fitz.Rect(bbox)
                key = tuple(round(coordinate, 3) for coordinate in rect)
                if key not in seen:
                    seen.add(key)
                    images.append(rect)
    except Exception:  # pragma: no cover - defensive, keep rendering going
        LOGGER.debug("get_image_info failed on page %s", page.number, exc_info=True)
    return images


def _page_graphic_rects(page: fitz.Page) -> list[fitz.Rect]:
    graphics: list[fitz.Rect] = []
    try:
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect:
                graphics.append(fitz.Rect(rect))
    except Exception:  # pragma: no cover - defensive, keep rendering going
        LOGGER.debug("get_drawings failed on page %s", page.number, exc_info=True)
    return graphics


def _page_filled_rects(page: fitz.Page) -> list[tuple[fitz.Rect, tuple[float, ...]]]:
    filled: list[tuple[fitz.Rect, tuple[float, ...]]] = []
    try:
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            color = drawing.get("fill")
            if rect and color:
                filled.append((fitz.Rect(rect), tuple(color)))
    except Exception:  # pragma: no cover - defensive, keep rendering going
        LOGGER.debug("get_drawings failed on page %s", page.number, exc_info=True)
    return filled


def _background_color(
    page: fitz.Page,
    rect: fitz.Rect,
    filled_rects: list[tuple[fitz.Rect, tuple[float, ...]]],
) -> tuple[float, ...]:
    clip = rect & page.rect
    if not clip.is_empty:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), clip=clip, alpha=False)
        if pixmap.width > 0 and pixmap.height > 0:
            samples: list[tuple[int, int, int]] = []
            for x in range(pixmap.width):
                samples.append(tuple(pixmap.pixel(x, 0)[:3]))
                if pixmap.height > 1:
                    samples.append(tuple(pixmap.pixel(x, pixmap.height - 1)[:3]))
            for y in range(1, max(1, pixmap.height - 1)):
                samples.append(tuple(pixmap.pixel(0, y)[:3]))
                if pixmap.width > 1:
                    samples.append(tuple(pixmap.pixel(pixmap.width - 1, y)[:3]))
            if samples:
                quantized = [tuple(round(channel / 16) * 16 for channel in sample) for sample in samples]
                red, green, blue = Counter(quantized).most_common(1)[0][0]
                return (
                    min(red, 255) / 255,
                    min(green, 255) / 255,
                    min(blue, 255) / 255,
                )
    center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
    candidates = [
        (filled.width * filled.height, color)
        for filled, color in filled_rects
        if filled.contains(center)
    ]
    return min(candidates, default=(0.0, (1.0, 1.0, 1.0)))[1]


def _rect_area(rect: fitz.Rect) -> float:
    return max(rect.width, 0.0) * max(rect.height, 0.0)


def _intersection_area(left: fitz.Rect, right: fitz.Rect) -> float:
    intersection = left & right
    return 0.0 if intersection.is_empty else _rect_area(intersection)


def _protected_visual_obstacles(
    candidate: fitz.Rect,
    source_rects: list[fitz.Rect],
    image_rects: list[fitz.Rect],
    graphic_rects: list[fitz.Rect],
) -> list[fitz.Rect]:
    result: list[fitz.Rect] = []
    candidate_area = max(_rect_area(candidate), 1.0)
    for obstacle in [*image_rects, *graphic_rects]:
        if obstacle.is_empty or _intersection_area(candidate, obstacle) <= 0:
            continue
        is_graphic = obstacle in graphic_rects
        if is_graphic and (
            obstacle.width <= 2.0
            or obstacle.height <= 2.0
            or _rect_area(obstacle) >= candidate_area * 0.75
        ):
            continue
        if any(
            obstacle.contains(
                fitz.Point((source.x0 + source.x1) / 2, (source.y0 + source.y1) / 2)
            )
            for source in source_rects
        ):
            continue
        key = tuple(round(value, 2) for value in obstacle)
        if not any(tuple(round(value, 2) for value in existing) == key for existing in result):
            result.append(obstacle)
    return result


def _safe_render_area(
    candidate: fitz.Rect,
    source_rects: list[fitz.Rect],
    obstacles: list[fitz.Rect],
) -> fitz.Rect:
    choices = [candidate]
    for obstacle in obstacles:
        next_choices: list[fitz.Rect] = []
        padded = fitz.Rect(
            obstacle.x0 - PADDING,
            obstacle.y0 - PADDING,
            obstacle.x1 + PADDING,
            obstacle.y1 + PADDING,
        )
        for choice in choices:
            if _intersection_area(choice, padded) <= 0:
                next_choices.append(choice)
                continue
            next_choices.extend([
                fitz.Rect(choice.x0, choice.y0, min(choice.x1, padded.x0), choice.y1),
                fitz.Rect(max(choice.x0, padded.x1), choice.y0, choice.x1, choice.y1),
                fitz.Rect(choice.x0, choice.y0, choice.x1, min(choice.y1, padded.y0)),
                fitz.Rect(choice.x0, max(choice.y0, padded.y1), choice.x1, choice.y1),
            ])
        choices = [
            choice for choice in next_choices
            if choice.width >= 12.0 and choice.height >= MIN_FONT_SIZE * 1.2
        ]
        if not choices:
            return candidate
        choices.sort(
            key=lambda choice: (
                sum(_intersection_area(choice, source) for source in source_rects),
                _rect_area(choice),
            ),
            reverse=True,
        )
        choices = choices[:16]
    return choices[0] if choices else candidate


def _safe_expanded_rect(
    rect: fitz.Rect,
    page_rect: fitz.Rect,
    other_rects: list[fitz.Rect],
) -> fitz.Rect:
    right = page_rect.x1
    bottom = page_rect.y1
    for other in other_rects:
        if other == rect:
            continue
        if other.y1 > rect.y0 and other.y0 < rect.y1 and other.x0 > rect.x1:
            right = min(right, other.x0 - PADDING)
        if other.x1 > rect.x0 and other.x0 < rect.x1 and other.y0 > rect.y1:
            bottom = min(bottom, other.y0 - PADDING)
    return fitz.Rect(rect.x0, rect.y0, max(rect.x1, right), max(rect.y1, bottom))


def _measure_fit(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontsize: float,
    fontname: str,
    minimum_size: float,
) -> float | None:
    """Return the largest font size (down to minimum_size) at which `text`
    fits in `rect`, or None if it doesn't fit even at minimum_size.
    Uses an uncommitted Shape so measurement never adds hidden text to the
    PDF text layer."""
    minimum_size = max(float(minimum_size), 1.0)
    current_size = max(float(fontsize), minimum_size)
    while True:
        result = page.new_shape().insert_textbox(
            rect,
            text,
            fontsize=current_size,
            fontname=fontname,
            color=(0, 0, 0),
            align=0,
        )
        if result >= 0:
            return current_size
        if current_size <= minimum_size:
            break
        current_size = max(minimum_size, current_size - 0.5)
    return None


def _draw_text(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontsize: float,
    fontname: str,
    color: tuple[float, float, float] = (0, 0, 0),
) -> None:
    page.insert_textbox(
        rect,
        text,
        fontsize=fontsize,
        fontname=fontname,
        color=color,
        align=0,
        overlay=True,
    )


def _measure_structured_line(
    text: str, width: float, fontsize: float, fontname: str, minimum_size: float
) -> float | None:
    current_size = max(float(fontsize), float(minimum_size), 1.0)
    minimum_size = max(float(minimum_size), 1.0)
    while True:
        if fitz.get_text_length(text, fontname=fontname, fontsize=current_size) <= width:
            return current_size
        if current_size <= minimum_size:
            break
        current_size = max(minimum_size, current_size - 0.5)
    return None


def _draw_structured_line(
    page: fitz.Page,
    span: _StructuredSpan,
    text: str,
    fontsize: float,
    fontname: str,
    color: tuple[float, float, float] = (0, 0, 0),
) -> None:
    page.insert_text(
        span.origin,
        text,
        fontsize=fontsize,
        fontname=fontname,
        color=color,
        overlay=True,
    )


def fit_text_to_rect(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontsize: float,
    flags: int = 0,
    minimum_size: float = MIN_FONT_SIZE,
) -> tuple[float, bool]:
    """Measure the best-fitting size for `text` in `rect`, then draw it
    exactly once (at that size if it fits, otherwise at minimum_size)."""
    fontname = _font_name(flags)
    size = _measure_fit(page, rect, text, fontsize, fontname, minimum_size)
    fits = size is not None
    draw_size = size if fits else max(float(minimum_size), 1.0)
    _draw_text(page, rect, text, draw_size, fontname)
    return draw_size, fits


def _insert_html_fallback(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontsize: float,
    flags: int,
    minimum_size: float = MIN_FONT_SIZE,
    color: tuple[float, float, float] = (0, 0, 0),
) -> _TextRenderResult:
    escaped_lines = [html.escape(line) for line in text.splitlines()]
    lines = "<br>".join(escaped_lines)
    weight = "bold" if flags & 16 else "normal"
    base_size = max(float(fontsize), minimum_size)
    red, green, blue = (round(channel * 255) for channel in color)
    style = (
        f"font-family: sans-serif; font-size: {base_size}pt; font-weight: {weight}; "
        f"color: rgb({red}, {green}, {blue});"
    )
    minimum_scale = min(1.0, minimum_size / base_size)
    spare_height, scale = page.insert_htmlbox(
        rect,
        f'<div style="{style}">{lines}</div>',
        scale_low=minimum_scale,
        overlay=True,
    )
    fits = spare_height >= 0 and scale > 0
    return _TextRenderResult(
        fits=fits,
        font_size=base_size * scale if fits else None,
        method="html",
    )


def _preflight_html_fallback(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontsize: float,
    flags: int,
    minimum_size: float,
    color: tuple[float, float, float],
) -> _TextRenderResult:
    scratch = fitz.open()
    try:
        scratch_page = scratch.new_page(width=page.rect.width, height=page.rect.height)
        return _insert_html_fallback(
            scratch_page, rect, text, fontsize, flags, minimum_size, color
        )
    finally:
        scratch.close()


def _preflight_text_render(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontsize: float,
    flags: int,
    minimum_size: float,
    color: tuple[float, float, float],
) -> bool:
    if not _needs_unicode_fallback(text):
        fontname = _font_name(flags)
        if _measure_fit(page, rect, text, fontsize, fontname, minimum_size) is not None:
            return True
    return _preflight_html_fallback(
        page, rect, text, fontsize, flags, minimum_size, color
    ).fits


def _preflight_source_spans(
    page: fitz.Page,
    spans: list[_StructuredSpan],
    translated_lines: list[str],
    fontsize: float,
    flags: int,
    color: tuple[float, float, float],
) -> bool:
    minimum_size = max(STRUCTURED_MIN_FONT_SIZE, fontsize * MIN_FONT_RATIO)
    fontname = _font_name(flags)
    for span, translated_line in zip(spans, translated_lines):
        if _needs_unicode_fallback(translated_line):
            if not _preflight_html_fallback(
                page,
                span.render_rect,
                translated_line,
                fontsize,
                flags,
                minimum_size,
                color,
            ).fits:
                return False
        elif _measure_structured_line(
            translated_line,
            span.render_rect.width,
            fontsize,
            fontname,
            minimum_size,
        ) is None:
            return False
    return True


def _fit_text_with_fallbacks(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontsize: float,
    flags: int,
    obstacle_rects: list[fitz.Rect],
    minimum_size: float = MIN_FONT_SIZE,
    allow_expand: bool = True,
    color: tuple[float, float, float] = (0, 0, 0),
) -> _TextRenderResult:
    """Try, in order: (1) the original rect, (2) the rect grown into
    surrounding whitespace (never into another unit or an image), (3) an
    HTML box as a last resort. Exactly one of these ends up drawing text,
    so the unit is never rendered twice."""
    fontname = _font_name(flags)

    if _needs_unicode_fallback(text):
        return _insert_html_fallback(
            page, rect, text, fontsize, flags, minimum_size, color
        )

    size = _measure_fit(page, rect, text, fontsize, fontname, minimum_size)
    if size is not None:
        _draw_text(page, rect, text, size, fontname, color)
        return _TextRenderResult(True, size, "textbox")

    expanded = (
        _safe_expanded_rect(rect=rect, page_rect=page.rect, other_rects=obstacle_rects)
        if allow_expand
        else rect
    )
    if allow_expand and expanded != rect:
        size = _measure_fit(page, expanded, text, fontsize, fontname, minimum_size)
        if size is not None:
            page.draw_rect(expanded, color=(1, 1, 1), fill=(1, 1, 1), width=0, overlay=True)
            _draw_text(page, expanded, text, size, fontname, color)
            return _TextRenderResult(True, size, "expanded_textbox")

    # The caller has already covered the owned source glyphs. Covering the
    # whole allocation here would erase unrelated inline graphics or text.
    fallback_rect = expanded if expanded != rect else rect
    return _insert_html_fallback(
        page, fallback_rect, text, fontsize, flags, minimum_size, color
    )


@dataclass
class RenderingPrimitives:
    """Small drawing toolbox used by strategy dispatch; it never selects strategy."""

    page: fitz.Page
    obstacle_rects: list[fitz.Rect]
    filled_rects: list[tuple[fitz.Rect, tuple[float, ...]]]

    def cover(self, rect: fitz.Rect, background_aware: bool = False) -> None:
        color = (
            _background_color(self.page, rect, self.filled_rects)
            if background_aware else (1, 1, 1)
        )
        self.page.draw_rect(rect, color=color, fill=color, width=0, overlay=True)

    def render_textbox(
        self,
        rect: fitz.Rect,
        text: str,
        fontsize: float,
        flags: int,
        minimum_size: float = MIN_FONT_SIZE,
        color: tuple[float, float, float] = (0, 0, 0),
    ) -> _TextRenderResult:
        return _fit_text_with_fallbacks(
            self.page,
            rect,
            text,
            fontsize,
            flags,
            self.obstacle_rects,
            minimum_size,
            allow_expand=False,
            color=color,
        )

    def render_source_spans(
        self,
        spans: list[_StructuredSpan],
        translated_lines: list[str],
        fontsize: float,
        flags: int,
        color: tuple[float, float, float] = (0, 0, 0),
    ) -> _TextRenderResult:
        fits = True
        sizes: list[float] = []
        methods: set[str] = set()
        fontname = _font_name(flags)
        minimum_size = max(STRUCTURED_MIN_FONT_SIZE, fontsize * MIN_FONT_RATIO)
        for span, translated_line in zip(spans, translated_lines):
            source_cover = fitz.Rect(
                span.source_rect.x0 - 0.5,
                span.source_rect.y0,
                span.source_rect.x1 + 0.5,
                span.source_rect.y1,
            )
            self.cover(source_cover, background_aware=True)
            if _needs_unicode_fallback(translated_line):
                result = _insert_html_fallback(
                    self.page,
                    span.render_rect,
                    translated_line,
                    fontsize,
                    flags,
                    minimum_size,
                    color,
                )
                fits = result.fits and fits
                if result.font_size is not None:
                    sizes.append(result.font_size)
                methods.add(result.method)
                continue
            size = _measure_structured_line(
                translated_line,
                span.render_rect.width,
                fontsize,
                fontname,
                minimum_size,
            )
            if size is not None:
                _draw_structured_line(
                    self.page, span, translated_line, size, fontname, color
                )
                sizes.append(size)
                methods.add("source_span")
            else:
                result = _insert_html_fallback(
                    self.page,
                    span.render_rect,
                    translated_line,
                    fontsize,
                    flags,
                    minimum_size,
                    color,
                )
                fits = result.fits and fits
                if result.font_size is not None:
                    sizes.append(result.font_size)
                methods.add(result.method)
        return _TextRenderResult(
            fits,
            min(sizes) if sizes else None,
            "+".join(sorted(methods)) or "source_span",
        )


def _structured_mismatch_diagnostic(
    unit: dict[str, Any], translation: str, spans: list[_StructuredSpan], rect: fitz.Rect,
    fallback_rect: fitz.Rect | None,
) -> str:
    chosen = [] if fallback_rect is None else [tuple(fallback_rect)]
    return (
        f"Unit {unit.get('id')}: structured span mismatch; "
        f"unit_type={unit.get('unit_type')!r}; source={unit.get('source')!r}; "
        f"translation={translation!r}; span_count={len(spans)}; "
        f"translated_line_count={len(translation.splitlines())}; "
        f"source_span_bboxes={[tuple(span.source_rect) for span in spans]!r}; "
        f"cell_bbox={tuple(rect)!r}; chosen_render_rects={chosen!r}"
    )


def _bbox_tuple(rect: fitz.Rect) -> tuple[float, float, float, float]:
    return (rect.x0, rect.y0, rect.x1, rect.y1)


def _planning_geometry(
    page: fitz.Page,
    page_data: dict[str, Any],
    unit_rects: list[fitz.Rect],
    image_rects: list[fitz.Rect],
    graphic_rects: list[fitz.Rect],
) -> PageGeometry:
    source_spans: dict[int, tuple[tuple[float, float, float, float], ...]] = {}
    for unit in page_data.get("units", []):
        unit_id = unit.get("id")
        rect = _unit_rect(unit)
        if not isinstance(unit_id, int) or rect is None:
            continue
        source_spans[unit_id] = tuple(
            _bbox_tuple(span.source_rect)
            for span in _extracted_structured_spans(page, unit, page_data, rect)
        )

    def fit_checker(
        unit: dict[str, Any], bbox: tuple[float, float, float, float], text: str, size: float
    ) -> bool:
        fontname = _font_name(int(unit.get("flags", 0)))
        return _measure_fit(page, fitz.Rect(bbox), text, size, fontname, size) is not None

    return PageGeometry(
        page_bbox=_bbox_tuple(page.rect),
        text_obstacles=tuple(_bbox_tuple(rect) for rect in unit_rects),
        image_obstacles=tuple(_bbox_tuple(rect) for rect in image_rects),
        graphic_obstacles=tuple(_bbox_tuple(rect) for rect in graphic_rects),
        source_spans=source_spans,
        fit_checker=fit_checker,
    )


def render_pdf(
    pdf_path: Path,
    extraction_path: Path = DEFAULT_EXTRACTION_PATH,
    translation_path: Path = DEFAULT_TRANSLATION_PATH,
    output_path: Path | None = None,
    layout_plans: dict[int, LayoutPlan] | None = None,
    render_plan_debug_dir: Path | None = None,
    identity_debug_dir: Path | None = None,
) -> tuple[Path, RenderStats]:
    extraction = _load_json(extraction_path)
    translation_payload = _load_json(translation_path)
    source_sha256, extraction_sha256 = validate_artifact_identity(
        pdf_path, extraction, translation_payload
    )
    translation_sha256 = payload_sha256(translation_payload)
    translations = _translation_map(translation_payload)
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{pdf_path.stem}_id.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = RenderStats()
    document = fitz.open(pdf_path)
    original_page_count = document.page_count
    try:
        for page_data in extraction.get("pages", []):
            page_number = page_data.get("page_number")
            if not isinstance(page_number, int) or not 1 <= page_number <= document.page_count:
                stats.warnings.append(f"Invalid page number for page entry: {page_number!r}")
                continue
            page = document[page_number - 1]
            page_data = {
                **page_data,
                "width": float(page_data.get("width", page.rect.width)),
                "height": float(page_data.get("height", page.rect.height)),
                "source_file": extraction.get("source_file", pdf_path.name),
                "source_sha256": source_sha256,
                "source_extraction_sha256": extraction_sha256,
                "source_translation_sha256": translation_sha256,
            }
            unit_rects = [
                candidate
                for candidate in (_unit_rect(unit) for unit in page_data.get("units", []))
                if candidate is not None
            ]
            image_rects = _page_image_rects(page)
            graphic_rects = _page_graphic_rects(page)
            obstacle_rects = unit_rects + image_rects + graphic_rects
            filled_rects = _page_filled_rects(page)
            supplied_layout_plan = (
                layout_plans.get(page_number)
                if layout_plans is not None
                else None
            )
            if supplied_layout_plan is not None and (
                extraction.get("schema_version", 0) >= 3
                or supplied_layout_plan.source_sha256 is not None
                or supplied_layout_plan.source_extraction_sha256 is not None
            ):
                validate_layout_identity(
                    supplied_layout_plan, source_sha256, extraction_sha256
                )
            layout_plan = supplied_layout_plan or fallback_layout_plan(
                page_data, (_bbox_tuple(rect) for rect in image_rects)
            )
            if supplied_layout_plan is None:
                layout_plan.source_file = str(page_data["source_file"])
                layout_plan.source_sha256 = source_sha256
                layout_plan.source_extraction_sha256 = extraction_sha256
            layout_plan = resolve_layout_plan_geometry(layout_plan, page_data)
            render_plan = RenderPlanner().plan(
                layout_plan,
                page_data,
                translations,
                _planning_geometry(
                    page, page_data, unit_rects, image_rects, graphic_rects
                ),
            )
            stats.warnings.extend(render_plan.warnings)
            if render_plan_debug_dir is not None:
                from .render_debug import save_render_plan_debug

                save_render_plan_debug(
                    document, page_number - 1, render_plan, render_plan_debug_dir
                )
            primitives = RenderingPrimitives(page, obstacle_rects, filled_rects)
            region_by_id = {region.id: region for region in layout_plan.regions}
            for unit in page_data.get("units", []):
                stats.total_units += 1
                unit_id = unit.get("id")
                rect = _unit_rect(unit)
                instruction = render_plan.instruction_for_unit(unit_id)
                if rect is None:
                    stats.warnings.append(f"Unit {unit_id}: invalid bounding box")
                    continue
                if instruction is None:
                    stats.warnings.append(f"Unit {unit_id}: no executable RenderPlan instruction")
                    continue
                validate_source_ownership(unit, instruction)
                translation_item = translations.get(unit_id)
                raw_translation = (
                    str(translation_item.get("translation", ""))
                    if translation_item is not None else None
                )
                identity = identity_record(
                    page_number,
                    unit,
                    _plain_text(raw_translation) if raw_translation is not None else None,
                    instruction,
                    region_by_id.get(instruction.region_id),
                )
                stats.identity_records.append(identity)
                source_text = str(unit.get("source", ""))
                if unit.get("translate") is not True:
                    stats.skipped_units += 1
                    complete_identity_record(
                        identity, [rect], source_text, "skipped", "unit is not translatable"
                    )
                    continue
                if translation_item is None:
                    stats.missing_translations += 1
                    stats.warnings.append(f"Unit {unit_id}: missing translation")
                    complete_identity_record(
                        identity, [rect], source_text, "missing_translation",
                        "translation artifact has no matching unit",
                    )
                    continue
                if translation_item.get("source") != unit.get("source"):
                    stats.warnings.append(f"Unit {unit_id}: translation source mismatch")
                    complete_identity_record(
                        identity, [rect], source_text, "source_mismatch",
                        "translation source does not match extraction",
                    )
                    continue
                stats.translated_units += 1
                text = _plain_text(str(translation_item.get("translation", "")))
                structured_text = text
                if _is_compact_structured_unit(unit, rect):
                    text = " ".join(text.splitlines())
                if not text.strip():
                    stats.warnings.append(f"Unit {unit_id}: empty translation")
                    complete_identity_record(
                        identity, [rect], source_text, "source_preserved", "empty translation"
                    )
                    continue
                fontsize = float(unit.get("fontsize", 10.0))
                flags = int(unit.get("flags", 0))
                identity["translation"] = structured_text
                if text == source_text:
                    _set_visual_diagnostics(
                        identity, fontsize, fontsize, "source_preserved", [], [], rect
                    )
                    complete_identity_record(
                        identity, [rect], source_text, "source_preserved",
                        "translation equals source",
                    )
                    stats.rendered_units += 1
                    continue
                if _has_private_use(source_text) or _has_private_use(text):
                    stats.warnings.append(
                        f"Unit {unit_id}: private-use glyph preserved with original visual content"
                    )
                    _set_visual_diagnostics(
                        identity, fontsize, fontsize, "source_preserved", [], [], rect,
                        "private-use glyph preserved",
                    )
                    complete_identity_record(
                        identity, [rect], source_text, "source_preserved",
                        "private-use glyph preserved",
                    )
                    stats.rendered_units += 1
                    continue
                strategy = instruction.strategy

                structured_lines = None
                structured_mismatch = False
                structured_cell_fallback = False
                fits = True
                rendered_rects: list[fitz.Rect] = []
                rendered_text = text
                rendered_font_size: float | None = None
                render_method: str | None = None
                mask_rects: list[fitz.Rect] = []
                source_covers = _owned_source_span_rects(unit, page_data)
                toc_leader_covers: list[fitz.Rect] = []
                planned_rect = fitz.Rect(instruction.bbox)
                visual_reason = None
                span_right_limit = None
                parent_region = region_by_id.get(instruction.parent_region_id)
                if (
                    parent_region is not None
                    and parent_region.type == RegionType.MULTI_COLUMN
                    and parent_region.bbox is not None
                ):
                    midpoint = (parent_region.bbox[0] + parent_region.bbox[2]) / 2
                    span_right_limit = (
                        midpoint - PADDING
                        if (rect.x0 + rect.x1) / 2 <= midpoint
                        else parent_region.bbox[2]
                    )
                    parent_rect = fitz.Rect(parent_region.bbox)
                    if parent_rect.contains(rect):
                        bounded = planned_rect & parent_rect
                        if not bounded.is_empty:
                            planned_rect = bounded
                    else:
                        visual_reason = "semantic parent does not contain source bbox"
                visual_obstacles = _protected_visual_obstacles(
                    planned_rect,
                    source_covers,
                    image_rects,
                    graphic_rects,
                )
                if strategy in {
                    RenderingStrategy.STRUCTURED_REGION,
                    RenderingStrategy.SOURCE_SPAN_MAPPING,
                } and unit.get("line_count", 0) > 1:
                    structured_spans = _extracted_structured_spans(
                        page, unit, page_data, rect,
                        expand_last_column=unit.get("unit_type") != "table_cell",
                        right_limit=span_right_limit,
                    )
                    span_rects = [span.render_rect for span in structured_spans]
                    translated_lines = structured_text.splitlines()
                    is_table_cell = unit.get("unit_type") == "table_cell"
                    has_multi_column_rows = _has_multi_column_rows(span_rects)
                    is_structured = strategy in {
                        RenderingStrategy.STRUCTURED_REGION,
                        RenderingStrategy.SOURCE_SPAN_MAPPING,
                    } or (
                        is_table_cell
                        or _is_compact_structured_unit(unit, rect)
                        or has_multi_column_rows
                    )
                    has_geometry_match = (
                        len(structured_spans) == len(translated_lines) and len(structured_spans) > 1
                    )
                    if has_geometry_match and is_structured:
                        structured_lines = list(zip(structured_spans, translated_lines))
                    elif (
                        strategy == RenderingStrategy.STRUCTURED_REGION
                        or is_table_cell
                        or _is_compact_structured_unit(unit, rect)
                        or (
                            len(structured_spans) == unit.get("line_count")
                            and _has_repeated_column_geometry(span_rects)
                        )
                    ):
                        structured_mismatch = True
                        fallback_rect = None
                        if is_table_cell:
                            fallback_rect = fitz.Rect(
                                rect.x0 + CELL_BORDER_INSET,
                                rect.y0 + CELL_BORDER_INSET,
                                rect.x1 - CELL_BORDER_INSET,
                                rect.y1 - CELL_BORDER_INSET,
                            )
                            structured_cell_fallback = fallback_rect.width > 0 and fallback_rect.height > 0
                        diagnostic = _structured_mismatch_diagnostic(
                            unit, structured_text, structured_spans, rect,
                            fallback_rect if structured_cell_fallback else None,
                        )
                        stats.warnings.append(diagnostic)
                        LOGGER.warning(diagnostic)

                if structured_lines is not None:
                    rendered_rects = [item[0].render_rect for item in structured_lines]
                    rendered_text = structured_text
                    if not _preflight_source_spans(
                        page,
                        [item[0] for item in structured_lines],
                        [item[1] for item in structured_lines],
                        fontsize,
                        flags,
                        _unit_color(unit),
                    ):
                        stats.warnings.append(
                            f"Unit {unit_id}: source preserved at typography floor"
                        )
                        _complete_visual_source_fallback(
                            identity,
                            rect,
                            source_text,
                            fontsize,
                            visual_obstacles,
                            "typography_floor",
                        )
                        stats.rendered_units += 1
                        continue
                    mask_rects = [
                        fitz.Rect(
                            item[0].source_rect.x0 - 0.75,
                            item[0].source_rect.y0 - 0.25,
                            item[0].source_rect.x1 + 0.75,
                            item[0].source_rect.y1 + 0.25,
                        )
                        for item in structured_lines
                    ]
                    render_result = primitives.render_source_spans(
                        [item[0] for item in structured_lines],
                        [item[1] for item in structured_lines],
                        fontsize,
                        flags,
                        _unit_color(unit),
                    )
                    fits = render_result.fits
                    rendered_font_size = render_result.font_size
                    render_method = render_result.method
                elif structured_cell_fallback:
                    cover = _cover_rect(unit, rect)
                    text_rect = fitz.Rect(
                        cover.x0 + PADDING,
                        cover.y0 + PADDING,
                        cover.x1 - PADDING,
                        cover.y1 - PADDING,
                    )
                    rendered_rects = [text_rect]
                    fontname = _font_name(flags)
                    minimum_size = max(
                        STRUCTURED_MIN_FONT_SIZE, fontsize * MIN_FONT_RATIO
                    )
                    if not _preflight_text_render(
                        page,
                        text_rect,
                        text,
                        fontsize,
                        flags,
                        minimum_size,
                        _unit_color(unit),
                    ):
                        stats.warnings.append(
                            f"Unit {unit_id}: source preserved at typography floor"
                        )
                        _complete_visual_source_fallback(
                            identity,
                            rect,
                            source_text,
                            fontsize,
                            visual_obstacles,
                            "typography_floor",
                        )
                        stats.rendered_units += 1
                        continue
                    primitives.cover(cover, background_aware=True)
                    mask_rects = [cover]
                    size = _measure_fit(
                        page, text_rect, text, fontsize, fontname, minimum_size
                    )
                    if size is not None:
                        _draw_text(
                            page, text_rect, text, size, fontname, _unit_color(unit)
                        )
                        rendered_font_size = size
                        render_method = "structured_cell_textbox"
                    else:
                        render_result = _insert_html_fallback(
                            page,
                            text_rect,
                            text,
                            fontsize,
                            flags,
                            minimum_size,
                            _unit_color(unit),
                        )
                        fits = render_result.fits
                        rendered_font_size = render_result.font_size
                        render_method = render_result.method
                elif structured_mismatch:
                    # Without one-to-one field evidence, reflowing all translated
                    # text into the parent can detach labels from their values and
                    # visual children. Preserve the intact source composition.
                    fallback_reason = "uncertain_fragment_mapping"
                    _complete_visual_source_fallback(
                        identity,
                        rect,
                        source_text,
                        fontsize,
                        visual_obstacles,
                        fallback_reason,
                    )
                    stats.rendered_units += 1
                    continue
                else:
                    cover = _cover_rect(unit, rect)
                    if cover.width <= 0 or cover.height <= 0:
                        stats.warnings.append(f"Unit {unit_id}: invalid cover rectangle")
                        complete_identity_record(
                            identity, [], source_text, "failed", "invalid cover rectangle"
                        )
                        continue
                    toc_anchor: tuple[float, float, float, float] | None = None
                    if strategy == RenderingStrategy.TOC_REGION:
                        toc_leader_covers = _metadata_source_rects(
                            unit, page_data, "toc_leader_source_ids"
                        )
                        anchor = instruction.page_number_anchor
                        toc_anchor = anchor
                        text_rect = rect
                        if anchor is not None:
                            next_top = min(
                                (
                                    other.y0 for other in unit_rects
                                    if other.y0 > rect.y0 + PADDING
                                    and other.x1 > rect.x0 and other.x0 < anchor[0]
                                ),
                                default=rect.y0 + rect.height * 2.2,
                            )
                            text_rect = fitz.Rect(
                                rect.x0, rect.y0, max(rect.x1, anchor[0] - PADDING),
                                max(rect.y1, min(next_top - PADDING, rect.y0 + rect.height * 2.2)),
                            )
                    elif strategy == RenderingStrategy.EXPAND_REGION:
                        text_rect = planned_rect
                    else:
                        text_rect = fitz.Rect(
                            cover.x0 + PADDING,
                            cover.y0 + PADDING,
                            cover.x1 - PADDING,
                            cover.y1 - PADDING,
                        )
                    if (
                        strategy != RenderingStrategy.TOC_REGION
                        and "\n" not in text
                        and int(unit.get("line_count", 1)) == 1
                        and parent_region is not None
                        and parent_region.type != RegionType.MULTI_COLUMN
                        and parent_region.bbox is not None
                        and fitz.Rect(parent_region.bbox).contains(rect)
                        and _measure_structured_line(
                            text,
                            text_rect.width,
                            fontsize,
                            _font_name(flags),
                            max(MIN_FONT_SIZE, fontsize * 0.75),
                        ) is None
                    ):
                        expanded = _safe_expanded_rect(
                            text_rect,
                            fitz.Rect(parent_region.bbox),
                            [*unit_rects, *image_rects, *graphic_rects],
                        )
                        if expanded != text_rect:
                            text_rect = expanded
                            visual_reason = "single-line text expanded inside semantic parent"
                    if strategy != RenderingStrategy.TOC_REGION:
                        visual_obstacles = _protected_visual_obstacles(
                            text_rect,
                            source_covers,
                            image_rects,
                            graphic_rects,
                        )
                        safe_text_rect = _safe_render_area(
                            text_rect, source_covers, visual_obstacles
                        )
                        if safe_text_rect != text_rect:
                            text_rect = safe_text_rect
                            visual_reason = "render area reduced around protected visual obstacles"
                    minimum_size = max(MIN_FONT_SIZE, fontsize * MIN_FONT_RATIO)
                    baseline_preflight = _render_single_line_baseline(
                        page,
                        unit,
                        page_data,
                        text,
                        text_rect,
                        fontsize,
                        flags,
                        draw=False,
                    )
                    if baseline_preflight is None and not _preflight_text_render(
                        page,
                        text_rect,
                        text,
                        fontsize,
                        flags,
                        minimum_size,
                        _unit_color(unit),
                    ):
                        stats.warnings.append(
                            f"Unit {unit_id}: source preserved at typography floor"
                        )
                        _complete_visual_source_fallback(
                            identity,
                            rect,
                            source_text,
                            fontsize,
                            visual_obstacles,
                            "typography_floor",
                        )
                        stats.rendered_units += 1
                        continue
                    if source_covers:
                        for source_cover in source_covers:
                            mask = fitz.Rect(
                                source_cover.x0 - 0.75,
                                source_cover.y0 - 0.25,
                                source_cover.x1 + 0.75,
                                source_cover.y1 + 0.25,
                            )
                            mask_rects.append(mask)
                            primitives.cover(mask, background_aware=True)
                    else:
                        primitives.cover(cover, background_aware=True)
                        mask_rects = [cover]
                    for leader_cover in toc_leader_covers:
                        leader_mask = fitz.Rect(
                            leader_cover.x0 - 0.5,
                            leader_cover.y0 - 0.25,
                            leader_cover.x1 + 0.5,
                            leader_cover.y1 + 0.25,
                        )
                        mask_rects.append(leader_mask)
                        primitives.cover(leader_mask, background_aware=True)
                    if baseline_preflight is not None:
                        baseline_render = _render_single_line_baseline(
                            page,
                            unit,
                            page_data,
                            text,
                            text_rect,
                            fontsize,
                            flags,
                        )
                        assert baseline_render is not None
                        render_result, rendered_rect = baseline_render
                        fits = render_result.fits
                        rendered_font_size = render_result.font_size
                        render_method = render_result.method
                        rendered_rects = [rendered_rect]
                        if toc_anchor is not None:
                            leader_start = rendered_rect.x1 + PADDING
                            leader_end = toc_anchor[0] - PADDING
                            dot_size = min(
                                rendered_font_size or fontsize,
                                max(MIN_FONT_SIZE, fontsize),
                            )
                            dot_width = max(
                                fitz.get_text_length(
                                    ".", fontname="helv", fontsize=dot_size
                                ),
                                1.0,
                            )
                            count = int(max(0.0, leader_end - leader_start) / dot_width)
                            if count >= 2:
                                baseline = min(toc_anchor[3] - 1.0, rendered_rect.y1)
                                page.insert_text(
                                    (leader_start, baseline),
                                    "." * count,
                                    fontsize=dot_size,
                                    fontname="helv",
                                    color=_unit_color(unit),
                                    overlay=True,
                                )
                    else:
                        render_result = primitives.render_textbox(
                            text_rect,
                            text,
                            fontsize,
                            flags,
                            minimum_size,
                            _unit_color(unit),
                        )
                        fits = render_result.fits
                        rendered_font_size = render_result.font_size
                        render_method = render_result.method
                        rendered_rects = [text_rect]
                _set_visual_diagnostics(
                    identity,
                    fontsize,
                    rendered_font_size,
                    render_method,
                    visual_obstacles,
                    mask_rects,
                    rendered_rects[0] if rendered_rects else None,
                    visual_reason,
                )
                if not fits:
                    stats.warnings.append(f"Unit {unit_id}: text overflow at minimum font size")
                    status = "rendered_overflow"
                    fallback_reason = "text overflow at minimum font size"
                else:
                    status = "rendered"
                    fallback_reason = None
                if strategy == RenderingStrategy.FALLBACK_ORIGINAL_BBOX:
                    fallback_reason = instruction.reason
                complete_identity_record(
                    identity, rendered_rects, rendered_text, status, fallback_reason
                )
                stats.rendered_units += 1
    finally:
        document.save(output_path, garbage=4, deflate=True)
        document.close()

    try:
        rendered = fitz.open(output_path)
        if rendered.page_count != original_page_count:
            stats.warnings.append(
                f"Page count changed: original={original_page_count}, rendered={rendered.page_count}"
            )
        rendered.close()
    except (fitz.FileDataError, OSError) as error:
        stats.warnings.append(f"Output PDF validation failed: {error}")
    if identity_debug_dir is not None:
        from .identity import save_identity_artifacts

        save_identity_artifacts(
            pdf_path,
            output_path,
            stats.identity_records,
            identity_debug_dir,
            extraction_path,
            translation_path,
        )
    return output_path, stats


def print_summary(pdf_path: Path, output_path: Path, page_count: int, stats: RenderStats) -> None:
    print("PDF Rendering Summary")
    print("---------------------")
    print(f"Input: {pdf_path}")
    print(f"Output: {output_path}")
    print(f"Pages: {page_count}")
    print(f"Total units: {stats.total_units}")
    print(f"Translated units: {stats.translated_units or stats.rendered_units}")
    print(f"Skipped units: {stats.skipped_units}")
    print(f"Rendered units: {stats.rendered_units}")
    print(f"Missing translations: {stats.missing_translations}")
    print(f"Warnings: {len(stats.warnings)}")
    if stats.warnings:
        print("\nWarnings:")
        for warning in stats.warnings:
            print(f"- {warning}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render validated translations over the original PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--extraction", type=Path, default=DEFAULT_EXTRACTION_PATH)
    parser.add_argument("--translation", type=Path, default=DEFAULT_TRANSLATION_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--layout-dir", type=Path, default=None)
    parser.add_argument("--render-plan-debug-dir", type=Path, default=None)
    parser.add_argument("--identity-debug-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        layout_plans = load_layout_plans(args.layout_dir) if args.layout_dir else None
        debug_dir = args.render_plan_debug_dir or args.layout_dir
        identity_dir = args.identity_debug_dir or (
            args.layout_dir / "identity" if args.layout_dir else None
        )
        output, stats = render_pdf(
            args.pdf,
            args.extraction,
            args.translation,
            args.output,
            layout_plans=layout_plans,
            render_plan_debug_dir=debug_dir,
            identity_debug_dir=identity_dir,
        )
        with fitz.open(args.pdf) as original:
            print_summary(args.pdf, output, original.page_count, stats)
    except (RuntimeError, fitz.FileDataError, OSError) as error:
        print(f"PDF rendering failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
