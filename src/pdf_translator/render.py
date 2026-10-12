from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import logging
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz

from .layout import LayoutPlan, RegionType, RenderingStrategy, fallback_layout_plan, resolve_layout_plan_geometry
from .render_plan import (
    PageGeometry,
    RenderPlan,
    RenderPlanner,
)
from .identity import (
    complete_identity_record,
    identity_record,
    payload_sha256,
    validate_artifact_identity,
    validate_layout_identity,
    validate_source_ownership,
)
from .translate import (
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    TranslationError,
    language_pair_key,
    language_scoped_translation_path,
    validate_language_pair,
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


@dataclass(frozen=True)
class _StructuredFieldFragment:
    source_spans: tuple[_StructuredSpan, ...]
    field_rect: fitz.Rect
    render_rect: fitz.Rect
    text: str


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read JSON '{path}': {error}") from error


def _default_rendered_output(pdf_path: Path, translation: dict[str, Any]) -> Path:
    source_language = translation.get("source_language")
    target_language = translation.get("target_language")
    if not isinstance(source_language, str) or not isinstance(target_language, str):
        return DEFAULT_OUTPUT_DIR / f"{pdf_path.stem}_id.pdf"
    try:
        source_language, target_language = validate_language_pair(
            source_language, target_language
        )
    except TranslationError as error:
        raise RuntimeError(f"Invalid translation language metadata: {error}") from error
    if (
        source_language.casefold() == DEFAULT_SOURCE_LANGUAGE.casefold()
        and target_language.casefold() == DEFAULT_TARGET_LANGUAGE.casefold()
    ):
        return DEFAULT_OUTPUT_DIR / f"{pdf_path.stem}_id.pdf"
    return DEFAULT_OUTPUT_DIR / (
        f"{pdf_path.stem}_{language_pair_key(source_language, target_language)}.pdf"
    )


def _render_debug_directories(
    pdf_path: Path,
    layout_dir: Path | None,
    render_plan_dir: Path | None,
    identity_dir: Path | None,
    debug_artifacts: bool,
    source_language: str,
    target_language: str,
) -> tuple[Path | None, Path | None]:
    default_language_pair = (
        source_language.casefold() == DEFAULT_SOURCE_LANGUAGE.casefold()
        and target_language.casefold() == DEFAULT_TARGET_LANGUAGE.casefold()
    )
    pair_directory = f"render_{language_pair_key(source_language, target_language)}"
    debug_dir = render_plan_dir
    if debug_dir is None and layout_dir is not None:
        debug_dir = layout_dir if default_language_pair else layout_dir / pair_directory
    if debug_artifacts and debug_dir is None:
        base_dir = DEFAULT_LAYOUT_DIR / pdf_path.stem
        debug_dir = base_dir if default_language_pair else base_dir / pair_directory
    resolved_identity_dir = identity_dir or (
        debug_dir / "identity" if debug_dir is not None else None
    )
    return debug_dir, resolved_identity_dir


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


def _validate_page_source_ownership(page_data: dict[str, Any]) -> None:
    source_index = {
        source.get("id"): source
        for source in page_data.get("source_objects", [])
        if isinstance(source.get("id"), str)
    }
    toc_leaders_by_unit = {
        entry.get("unit_id"): set(entry.get("leader_source_ids", []))
        for entry in page_data.get("toc_entries", [])
        if isinstance(entry.get("unit_id"), int)
    }
    span_owners: dict[str, list[tuple[int, bool]]] = {}
    for unit in page_data.get("units", []):
        unit_id = unit.get("id")
        source_ids = unit.get("source_ids", [])
        if len(source_ids) != len(set(source_ids)):
            raise RuntimeError(f"Unit {unit_id}: duplicate source ownership IDs")
        missing = [source_id for source_id in source_ids if source_id not in source_index]
        if missing:
            raise RuntimeError(f"Unit {unit_id}: unknown source ownership IDs: {missing}")
        if not isinstance(unit_id, int):
            continue
        leader_ids = unit.get("metadata", {}).get("toc_leader_source_ids", [])
        if len(leader_ids) != len(set(leader_ids)):
            raise RuntimeError(f"Unit {unit_id}: duplicate TOC leader source IDs")
        unauthorized_leaders = [
            source_id
            for source_id in leader_ids
            if source_id not in source_index
            or source_id not in toc_leaders_by_unit.get(unit_id, set())
        ]
        if unauthorized_leaders:
            raise RuntimeError(
                f"Unit {unit_id}: unauthorized TOC leader source IDs: "
                f"{unauthorized_leaders}"
            )
        for source_id in source_ids:
            if source_index[source_id].get("kind") == "span":
                declared_owners = source_index[source_id].get("unit_ids")
                if not isinstance(declared_owners, list) or unit_id not in declared_owners:
                    raise RuntimeError(
                        f"Unit {unit_id}: source object {source_id!r} does not declare this owner"
                    )
                span_owners.setdefault(source_id, []).append(
                    (unit_id, unit.get("translate") is True)
                )
    duplicates = {
        source_id: owners
        for source_id, owners in span_owners.items()
        if len(owners) > 1 and any(translated for _, translated in owners)
    }
    if duplicates:
        raise RuntimeError(f"Duplicate translated source-span ownership: {duplicates}")


def _page_source_rect(page: fitz.Page) -> fitz.Rect:
    """PyMuPDF extraction and drawing APIs use unrotated page coordinates."""
    return fitz.Rect(0, 0, page.cropbox.width, page.cropbox.height)


def _source_mask_rect(source_rect: fitz.Rect, page_rect: fitz.Rect) -> fitz.Rect:
    return fitz.Rect(
        source_rect.x0 - 0.5,
        source_rect.y0 - 0.5,
        source_rect.x1 + 0.5,
        source_rect.y1 + 0.5,
    ) & page_rect


def _supports_text_direction(unit: dict[str, Any]) -> bool:
    direction = unit.get("direction")
    if direction is None:
        return True
    if not isinstance(direction, (list, tuple)) or len(direction) != 2:
        return False
    try:
        dx, dy = (float(value) for value in direction)
    except (TypeError, ValueError):
        return False
    return dx > 0.999 and abs(dy) < 0.001


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
    minimum_size = max(MIN_FONT_SIZE, fontsize * minimum_ratio)
    size = _measure_structured_line(
        text,
        max(rect.x1 - max(rect.x0, min(source_origin.x, rect.x1)), 1.0),
        fontsize,
        fontname,
        minimum_size,
    )
    if size is None:
        return None
    font = fitz.Font(fontname)
    ascender = max(float(font.ascender), 0.0) * size
    descender = max(-float(font.descender), 0.0) * size
    if ascender + descender > rect.height + 0.25:
        return None
    origin = fitz.Point(
        max(rect.x0, min(source_origin.x, rect.x1)),
        max(rect.y0 + ascender, min(source_origin.y, rect.y1 - descender)),
    )
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
        origin.y - ascender,
        min(rect.x1, origin.x + fitz.get_text_length(text, fontname=fontname, fontsize=size)),
        origin.y + descender,
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
    page_right = _page_source_rect(page).x1
    right = min(page_right, right_limit) if right_limit is not None else page_right
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
                right = _available_right_edge(page, span_rect, right_limit)
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
    center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
    candidates = [
        (filled.width * filled.height, color)
        for filled, color in filled_rects
        if filled.contains(center)
    ]
    if candidates:
        return min(candidates)[1]
    clip = rect & _page_source_rect(page)
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
                red, green, blue = Counter(samples).most_common(1)[0][0]
                return red / 255, green / 255, blue / 255
    return (1.0, 1.0, 1.0)


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
    for obstacle in [*image_rects, *graphic_rects]:
        if obstacle.is_empty or _intersection_area(candidate, obstacle) <= 0:
            continue
        is_graphic = obstacle in graphic_rects
        if is_graphic and (
            obstacle.width <= 2.0
            or obstacle.height <= 2.0
        ):
            continue
        if is_graphic and any(
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
) -> fitz.Rect | None:
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
            return None
        choices.sort(
            key=lambda choice: (
                sum(_intersection_area(choice, source) for source in source_rects),
                _rect_area(choice),
            ),
            reverse=True,
        )
        choices = choices[:16]
    return choices[0] if choices else None


def _structured_field_fragments(
    source_text: str,
    translation: str,
    spans: list[_StructuredSpan],
    rect: fitz.Rect,
    obstacles: list[fitz.Rect],
) -> list[_StructuredFieldFragment] | None:
    source_lines = [line.strip() for line in source_text.splitlines()]
    translated_lines = [line.strip() for line in translation.splitlines()]
    if (
        len(source_lines) != len(spans)
        or not source_lines
        or any(not line for line in source_lines)
        or any(not line for line in translated_lines)
    ):
        return None

    tolerance = max(2.0, min(span.source_rect.height for span in spans) * 0.5)
    anchor_groups: list[list[float]] = []
    for x in sorted(span.source_rect.x0 for span in spans):
        if anchor_groups and abs(x - sum(anchor_groups[-1]) / len(anchor_groups[-1])) <= tolerance:
            anchor_groups[-1].append(x)
        else:
            anchor_groups.append([x])
    if len(anchor_groups) != 2:
        return None
    label_anchor, value_anchor = (
        sum(group) / len(group) for group in anchor_groups
    )

    source_labels = [
        index for index, line in enumerate(source_lines) if line.endswith(":")
    ]
    translated_labels = [
        index for index, line in enumerate(translated_lines) if line.endswith(":")
    ]
    if len(source_labels) < 2 or len(source_labels) != len(translated_labels):
        return None
    if any(
        abs(spans[index].source_rect.x0 - label_anchor) > tolerance
        for index in source_labels
    ):
        return None
    if any(
        abs(span.source_rect.x0 - value_anchor) > tolerance
        for index, span in enumerate(spans)
        if index not in source_labels
    ):
        return None

    fragments: list[_StructuredFieldFragment] = []

    def add_value_fragment(
        source_group: list[_StructuredSpan],
        lines: list[str],
        top: float,
        bottom: float,
    ) -> bool:
        if not source_group or not lines or bottom <= top:
            return False
        candidate = fitz.Rect(value_anchor, top, rect.x1, bottom)
        local_obstacles = [
            obstacle for obstacle in obstacles
            if _intersection_area(candidate, obstacle) > 0
        ]
        safe_rect = _safe_render_area(
            candidate,
            [span.source_rect for span in source_group],
            local_obstacles,
        )
        if safe_rect is None:
            return False
        fragments.append(_StructuredFieldFragment(
            tuple(source_group), candidate, safe_rect, "\n".join(lines)
        ))
        return True

    first_source_label = source_labels[0]
    first_translated_label = translated_labels[0]
    if bool(first_source_label) != bool(first_translated_label):
        return None
    if first_source_label and not add_value_fragment(
        spans[:first_source_label],
        translated_lines[:first_translated_label],
        min(span.source_rect.y0 for span in spans[:first_source_label]),
        spans[first_source_label].source_rect.y0 - PADDING,
    ):
        return None

    for field_index, source_label_index in enumerate(source_labels):
        translated_label_index = translated_labels[field_index]
        next_source_index = (
            source_labels[field_index + 1]
            if field_index + 1 < len(source_labels)
            else len(spans)
        )
        next_translated_index = (
            translated_labels[field_index + 1]
            if field_index + 1 < len(translated_labels)
            else len(translated_lines)
        )
        label_span = spans[source_label_index]
        value_spans = spans[source_label_index + 1:next_source_index]
        translated_values = translated_lines[
            translated_label_index + 1:next_translated_index
        ]
        bottom = (
            spans[next_source_index].source_rect.y0 - PADDING
            if next_source_index < len(spans)
            else rect.y1
        )
        label_rect = fitz.Rect(
            label_span.source_rect.x0,
            label_span.source_rect.y0,
            value_anchor - PADDING,
            bottom,
        )
        if label_rect.width <= 0 or label_rect.height <= 0:
            return None
        fragments.append(_StructuredFieldFragment(
            (label_span,), label_rect, label_rect,
            translated_lines[translated_label_index],
        ))
        if not add_value_fragment(
            value_spans,
            translated_values,
            label_span.source_rect.y0,
            bottom,
        ):
            return None

    if "\n".join(fragment.text for fragment in fragments).splitlines() != translated_lines:
        return None
    return fragments


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
    """Measure and draw text only when it fits the approved rectangle."""
    fontname = _font_name(flags)
    size = _measure_fit(page, rect, text, fontsize, fontname, minimum_size)
    if size is None:
        return max(float(minimum_size), 1.0), False
    _draw_text(page, rect, text, size, fontname)
    return size, True


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
        page_rect = _page_source_rect(page)
        scratch_page = scratch.new_page(width=page_rect.width, height=page_rect.height)
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


def _obstacle_free_row_slots(
    candidate: fitz.Rect,
    obstacles: list[fitz.Rect],
    minimum_height: float,
) -> list[fitz.Rect]:
    padded_obstacles = [
        fitz.Rect(
            obstacle.x0 - PADDING,
            obstacle.y0 - PADDING,
            obstacle.x1 + PADDING,
            obstacle.y1 + PADDING,
        ) & candidate
        for obstacle in obstacles
        if _intersection_area(candidate, obstacle) > 0
    ]
    y_edges = sorted({
        candidate.y0,
        candidate.y1,
        *(
            coordinate
            for obstacle in padded_obstacles
            for coordinate in (obstacle.y0, obstacle.y1)
            if candidate.y0 < coordinate < candidate.y1
        ),
    })
    slots: list[fitz.Rect] = []
    for top, bottom in zip(y_edges, y_edges[1:]):
        if bottom - top < minimum_height:
            continue
        intervals = [(candidate.x0, candidate.x1)]
        for obstacle in padded_obstacles:
            if obstacle.y1 <= top or obstacle.y0 >= bottom:
                continue
            next_intervals: list[tuple[float, float]] = []
            for left, right in intervals:
                if obstacle.x1 <= left or obstacle.x0 >= right:
                    next_intervals.append((left, right))
                    continue
                if obstacle.x0 - left >= 12.0:
                    next_intervals.append((left, obstacle.x0))
                if right - obstacle.x1 >= 12.0:
                    next_intervals.append((obstacle.x1, right))
            intervals = next_intervals
        slots.extend(
            fitz.Rect(left, top, right, bottom)
            for left, right in intervals
            if right - left >= 12.0
        )
    return sorted(slots, key=lambda slot: (slot.y0, slot.x0))


def _flow_text_across_slots(
    page: fitz.Page,
    text: str,
    slots: list[fitz.Rect],
    fontsize: float,
    flags: int,
    minimum_size: float,
    color: tuple[float, float, float],
) -> list[tuple[fitz.Rect, str]] | None:
    words = text.split()
    if not words or len(slots) < 2:
        return None

    def assign(slot_index: int, word_index: int) -> list[tuple[fitz.Rect, str]] | None:
        if word_index == len(words):
            return []
        if slot_index == len(slots):
            return None
        for end in range(len(words), word_index, -1):
            candidate_text = " ".join(words[word_index:end])
            if not _preflight_text_render(
                page,
                slots[slot_index],
                candidate_text,
                fontsize,
                flags,
                minimum_size,
                color,
            ):
                continue
            remainder = assign(slot_index + 1, end)
            if remainder is not None:
                return [(slots[slot_index], candidate_text), *remainder]
        return assign(slot_index + 1, word_index)

    return assign(0, 0)


def _resolve_structured_field_slots(
    page: fitz.Page,
    fragments: list[_StructuredFieldFragment],
    obstacles: list[fitz.Rect],
    fontsize: float,
    flags: int,
    minimum_size: float,
    color: tuple[float, float, float],
) -> list[_StructuredFieldFragment] | None:
    resolved: list[_StructuredFieldFragment] = []
    for fragment in fragments:
        if _preflight_text_render(
            page,
            fragment.render_rect,
            fragment.text,
            fontsize,
            flags,
            minimum_size,
            color,
        ):
            resolved.append(fragment)
            continue
        if len(fragment.source_spans) < 2:
            return None
        local_obstacles = [
            obstacle for obstacle in obstacles
            if _intersection_area(fragment.field_rect, obstacle) > 0
        ]
        slots = _obstacle_free_row_slots(
            fragment.field_rect,
            local_obstacles,
            minimum_size * 1.2,
        )
        assignment = _flow_text_across_slots(
            page,
            fragment.text,
            slots,
            fontsize,
            flags,
            minimum_size,
            color,
        )
        if assignment is None:
            return None
        resolved.extend(
            _StructuredFieldFragment(
                fragment.source_spans, slot, slot, assigned_text
            )
            for slot, assigned_text in assignment
        )
    return resolved


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
        _safe_expanded_rect(
            rect=rect, page_rect=_page_source_rect(page), other_rects=obstacle_rects
        )
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
        page_bbox=_bbox_tuple(_page_source_rect(page)),
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
    debug_images: bool = False,
) -> tuple[Path, RenderStats]:
    extraction = _load_json(extraction_path)
    translation_payload = _load_json(translation_path)
    source_sha256, extraction_sha256 = validate_artifact_identity(
        pdf_path, extraction, translation_payload
    )
    translation_sha256 = payload_sha256(translation_payload)
    translations = _translation_map(translation_payload)
    if output_path is None:
        output_path = _default_rendered_output(pdf_path, translation_payload)
    if output_path.resolve() == pdf_path.resolve():
        raise RuntimeError("Output PDF path must differ from the source PDF path")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = RenderStats()
    strong_source_ownership = extraction.get("schema_version", 0) >= 3
    temporary_output = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    document = fitz.open(pdf_path)
    original_page_count = document.page_count
    try:
        for page_data in extraction.get("pages", []):
            page_number = page_data.get("page_number")
            if not isinstance(page_number, int) or not 1 <= page_number <= document.page_count:
                stats.warnings.append(f"Invalid page number for page entry: {page_number!r}")
                continue
            page = document[page_number - 1]
            page_bounds = _page_source_rect(page)
            page_data = {
                **page_data,
                "width": float(page_data.get("width", page.cropbox.width)),
                "height": float(page_data.get("height", page.cropbox.height)),
                "source_file": extraction.get("source_file", pdf_path.name),
                "source_sha256": source_sha256,
                "source_extraction_sha256": extraction_sha256,
                "source_translation_sha256": translation_sha256,
            }
            if strong_source_ownership:
                _validate_page_source_ownership(page_data)
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
                    document,
                    page_number - 1,
                    render_plan,
                    render_plan_debug_dir,
                    write_debug_image=debug_images,
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
                if not _supports_text_direction(unit):
                    stats.warnings.append(
                        f"Unit {unit_id}: unsupported transformed text direction; source preserved"
                    )
                    _complete_visual_source_fallback(
                        identity,
                        rect,
                        source_text,
                        fontsize,
                        [],
                        "unsupported_text_direction",
                    )
                    stats.rendered_units += 1
                    continue
                strategy = instruction.strategy

                structured_lines = None
                structured_fields = None
                structured_mismatch = False
                structured_cell_fallback = False
                fits = True
                rendered_rects: list[fitz.Rect] = []
                rendered_text = text
                rendered_font_size: float | None = None
                render_method: str | None = None
                mask_rects: list[fitz.Rect] = []
                source_covers = _owned_source_span_rects(unit, page_data)
                source_masks = [
                    _source_mask_rect(source_cover, page_bounds)
                    for source_cover in source_covers
                ]
                if strong_source_ownership and not source_masks:
                    _complete_visual_source_fallback(
                        identity,
                        rect,
                        source_text,
                        fontsize,
                        [],
                        "unsafe_mask",
                    )
                    stats.rendered_units += 1
                    continue
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
                    parent_rect = fitz.Rect(parent_region.bbox)
                    if parent_rect.contains(rect):
                        midpoint = (parent_region.bbox[0] + parent_region.bbox[2]) / 2
                        span_right_limit = (
                            midpoint - PADDING
                            if (rect.x0 + rect.x1) / 2 <= midpoint
                            else parent_region.bbox[2]
                        )
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
                    structured_spans = [
                        _StructuredSpan(
                            source_rect=span.source_rect,
                            render_rect=span.render_rect & planned_rect,
                            origin=span.origin,
                        )
                        for span in structured_spans
                        if not (span.render_rect & planned_rect).is_empty
                    ]
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
                        if not is_table_cell:
                            structured_fields = _structured_field_fragments(
                                source_text,
                                structured_text,
                                structured_spans,
                                rect,
                                visual_obstacles,
                            )
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
                        if structured_cell_fallback or structured_fields is not None:
                            visual_reason = "structured span mismatch used safe structured fallback"
                            LOGGER.debug(diagnostic)
                        else:
                            stats.warnings.append(diagnostic)
                            LOGGER.warning(diagnostic)

                if structured_lines is not None:
                    rendered_rects = [item[0].render_rect for item in structured_lines]
                    rendered_text = structured_text
                    if any(
                        _intersection_area(render_rect, obstacle) > 0
                        for render_rect in rendered_rects
                        for obstacle in visual_obstacles
                    ):
                        reason = (
                            "image_collision"
                            if any(
                                _intersection_area(render_rect, image) > 0
                                for render_rect in rendered_rects
                                for image in image_rects
                            )
                            else "graphic_collision"
                        )
                        _complete_visual_source_fallback(
                            identity, rect, source_text, fontsize, visual_obstacles, reason
                        )
                        stats.rendered_units += 1
                        continue
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
                            "font_size_below_threshold",
                        )
                        stats.rendered_units += 1
                        continue
                    mask_rects = [
                        _source_mask_rect(item[0].source_rect, page_bounds)
                        for item in structured_lines
                    ]
                    for mask in mask_rects:
                        primitives.cover(mask, background_aware=True)
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
                elif structured_fields is not None:
                    minimum_size = max(
                        STRUCTURED_MIN_FONT_SIZE, fontsize * MIN_FONT_RATIO
                    )
                    structured_fields = _resolve_structured_field_slots(
                        page,
                        structured_fields,
                        visual_obstacles,
                        fontsize,
                        flags,
                        minimum_size,
                        _unit_color(unit),
                    )
                    if structured_fields is None:
                        stats.warnings.append(
                            f"Unit {unit_id}: structured fields preserved at typography floor"
                        )
                        _complete_visual_source_fallback(
                            identity,
                            rect,
                            source_text,
                            fontsize,
                            visual_obstacles,
                            "font_size_below_threshold",
                        )
                        stats.rendered_units += 1
                        continue
                    mask_rects = [
                        _source_mask_rect(span.source_rect, page_bounds)
                        for span in structured_spans
                    ]
                    for mask in mask_rects:
                        primitives.cover(mask, background_aware=True)
                    results = [
                        primitives.render_textbox(
                            fragment.render_rect,
                            fragment.text,
                            fontsize,
                            flags,
                            minimum_size,
                            _unit_color(unit),
                        )
                        for fragment in structured_fields
                    ]
                    fits = all(result.fits for result in results)
                    rendered_font_size = min(
                        result.font_size for result in results
                        if result.font_size is not None
                    )
                    render_method = "structured_fields"
                    rendered_rects = [
                        fragment.render_rect for fragment in structured_fields
                    ]
                    rendered_text = structured_text
                elif structured_cell_fallback:
                    cover = _cover_rect(unit, rect)
                    text_rect = fitz.Rect(
                        cover.x0 + PADDING,
                        cover.y0 + PADDING,
                        cover.x1 - PADDING,
                        cover.y1 - PADDING,
                    )
                    safe_text_rect = _safe_render_area(
                        text_rect, source_covers, visual_obstacles
                    )
                    if strong_source_ownership and not source_masks:
                        _complete_visual_source_fallback(
                            identity,
                            rect,
                            source_text,
                            fontsize,
                            visual_obstacles,
                            "unsafe_mask",
                        )
                        stats.rendered_units += 1
                        continue
                    if safe_text_rect is None:
                        reason = (
                            "image_collision"
                            if any(_intersection_area(text_rect, image) > 0 for image in image_rects)
                            else "graphic_collision"
                        )
                        _complete_visual_source_fallback(
                            identity, rect, source_text, fontsize, visual_obstacles, reason
                        )
                        stats.rendered_units += 1
                        continue
                    text_rect = safe_text_rect
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
                            "font_size_below_threshold",
                        )
                        stats.rendered_units += 1
                        continue
                    mask_rects = source_masks or [cover]
                    for mask in mask_rects:
                        primitives.cover(mask, background_aware=True)
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
                        toc_anchor = instruction.page_number_anchor
                    text_rect = planned_rect
                    visual_obstacles = _protected_visual_obstacles(
                        text_rect,
                        source_covers,
                        image_rects,
                        graphic_rects,
                    )
                    if strong_source_ownership and not source_masks:
                        _complete_visual_source_fallback(
                            identity,
                            rect,
                            source_text,
                            fontsize,
                            visual_obstacles,
                            "unsafe_mask",
                        )
                        stats.rendered_units += 1
                        continue
                    if (
                        strategy == RenderingStrategy.FALLBACK_ORIGINAL_BBOX
                        and any(_intersection_area(rect, image) > 0 for image in image_rects)
                    ):
                        _complete_visual_source_fallback(
                            identity,
                            rect,
                            source_text,
                            fontsize,
                            visual_obstacles,
                            "image_collision",
                        )
                        stats.rendered_units += 1
                        continue
                    safe_text_rect = _safe_render_area(
                        text_rect, source_covers, visual_obstacles
                    )
                    if safe_text_rect is None:
                        reason = (
                            "image_collision"
                            if any(_intersection_area(text_rect, image) > 0 for image in image_rects)
                            else "graphic_collision"
                        )
                        _complete_visual_source_fallback(
                            identity, rect, source_text, fontsize, visual_obstacles, reason
                        )
                        stats.rendered_units += 1
                        continue
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
                    text_preflight = baseline_preflight is not None or _preflight_text_render(
                        page, text_rect, text, fontsize, flags, minimum_size,
                        _unit_color(unit),
                    )
                    if not text_preflight:
                        stats.warnings.append(
                            f"Unit {unit_id}: source preserved at typography floor"
                        )
                        fallback_reason = (
                            "font_size_below_threshold"
                            if instruction.source_geometry_fixed
                            else "insufficient_space"
                        )
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
                    if source_masks:
                        mask_rects = source_masks
                        for mask in mask_rects:
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
                    raise RuntimeError(
                        f"Unit {unit_id}: renderer diverged from successful text preflight"
                    )
                status = "rendered"
                fallback_reason = None
                complete_identity_record(
                    identity, rendered_rects, rendered_text, status, fallback_reason
                )
                stats.rendered_units += 1
    except BaseException:
        document.close()
        raise
    else:
        try:
            try:
                document.save(temporary_output, garbage=4, deflate=True)
            finally:
                document.close()
        except BaseException:
            temporary_output.unlink(missing_ok=True)
            raise
    try:
        with fitz.open(temporary_output) as rendered:
            if rendered.page_count != original_page_count:
                raise RuntimeError(
                    f"Page count changed: original={original_page_count}, "
                    f"rendered={rendered.page_count}"
                )
        os.replace(temporary_output, output_path)
    except BaseException as error:
        temporary_output.unlink(missing_ok=True)
        if isinstance(error, RuntimeError):
            raise
        raise RuntimeError(f"Output PDF validation or publication failed: {error}") from error
    if identity_debug_dir is not None:
        from .identity import save_identity_artifacts

        save_identity_artifacts(
            pdf_path,
            output_path,
            stats.identity_records,
            identity_debug_dir,
            extraction_path,
            translation_path,
            write_debug_images=debug_images,
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
    if stats.identity_records:
        successful = sum(
            record.get("render_status") == "rendered"
            for record in stats.identity_records
        )
        preserved = sum(
            record.get("render_status") in {"source_fallback", "source_preserved"}
            for record in stats.identity_records
        )
        errors = sum(
            record.get("render_status") in {"failed", "rendered_overflow"}
            for record in stats.identity_records
        )
        fallback_counts = Counter(
            str(record["fallback_reason"])
            for record in stats.identity_records
            if record.get("fallback_reason")
        )
        print(f"Successfully rendered translations: {successful}")
        print(f"Preserved original units: {preserved}")
        print(f"Rendering errors: {errors}")
        if fallback_counts:
            print("Fallback reasons:")
            for reason, count in sorted(fallback_counts.items()):
                print(f"- {reason}: {count}")
    print(f"Warnings: {len(stats.warnings)}")
    if stats.warnings:
        print("\nWarnings:")
        for warning in stats.warnings:
            print(f"- {warning}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render validated translations over the original PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--extraction", type=Path, default=DEFAULT_EXTRACTION_PATH)
    parser.add_argument("--translation", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source-language", default=DEFAULT_SOURCE_LANGUAGE)
    parser.add_argument("--target-language", default=DEFAULT_TARGET_LANGUAGE)
    parser.add_argument("--layout-dir", type=Path, default=None)
    parser.add_argument("--render-plan-debug-dir", type=Path, default=None)
    parser.add_argument("--identity-debug-dir", type=Path, default=None)
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help="Write diagnostic PNGs in addition to JSON artifacts",
    )
    args = parser.parse_args()
    try:
        source_language, target_language = validate_language_pair(
            args.source_language, args.target_language
        )
        translation_path = args.translation or language_scoped_translation_path(
            DEFAULT_TRANSLATION_PATH, source_language, target_language
        )
        layout_plans = load_layout_plans(args.layout_dir) if args.layout_dir else None
        debug_dir, identity_dir = _render_debug_directories(
            args.pdf,
            args.layout_dir,
            args.render_plan_debug_dir,
            args.identity_debug_dir,
            args.debug_artifacts,
            source_language,
            target_language,
        )
        output, stats = render_pdf(
            args.pdf,
            args.extraction,
            translation_path,
            args.output,
            layout_plans=layout_plans,
            render_plan_debug_dir=debug_dir,
            identity_debug_dir=identity_dir,
            debug_images=args.debug_artifacts,
        )
        with fitz.open(args.pdf) as original:
            print_summary(args.pdf, output, original.page_count, stats)
    except (RuntimeError, fitz.FileDataError, OSError) as error:
        print(f"PDF rendering failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
