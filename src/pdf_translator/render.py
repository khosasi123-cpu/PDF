from __future__ import annotations

import argparse
import html
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz

from .layout import LayoutPlan, RenderingStrategy, fallback_layout_plan, resolve_layout_plan_geometry
from .render_plan import PageGeometry, RenderPlan, RenderPlanner

LOGGER = logging.getLogger(__name__)
DEFAULT_EXTRACTION_PATH = Path("artifacts/extraction/extraction.json")
DEFAULT_TRANSLATION_PATH = Path("artifacts/translation/translation.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/rendered")
DEFAULT_LAYOUT_DIR = Path("artifacts/layout")
MIN_FONT_SIZE = 5.5
STRUCTURED_MIN_FONT_SIZE = 4.5
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
        if isinstance(unit_id, int) and unit_id not in result:
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


def _available_right_edge(page: fitz.Page, source_rect: fitz.Rect) -> float:
    right = page.rect.x1
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
    page: fitz.Page, rect: fitz.Rect, expand_last_column: bool = False
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
) -> list[_StructuredSpan]:
    source_ids = set(unit.get("source_ids", []))
    values: list[tuple[fitz.Rect, fitz.Point]] = []
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
        values.append((span_rect, point))
    if not values:
        return _structured_spans(page, rect, expand_last_column)
    values.sort(key=lambda item: (item[0].y0, item[0].x0))
    rows: list[list[tuple[fitz.Rect, fitz.Point]]] = []
    for value in values:
        if rows and abs(value[0].y0 - rows[-1][0][0].y0) <= STRUCTURED_Y_TOLERANCE:
            rows[-1].append(value)
        else:
            rows.append([value])
    result: list[_StructuredSpan] = []
    for row_index, row in enumerate(rows):
        row.sort(key=lambda item: item[0].x0)
        next_y = rows[row_index + 1][0][0].y0 if row_index + 1 < len(rows) else rect.y1
        for index, (span_rect, origin) in enumerate(row):
            if index + 1 < len(row):
                right = row[index + 1][0].x0 - PADDING
            elif expand_last_column:
                right = _available_right_edge(page, span_rect)
            else:
                right = rect.x1
            result.append(_StructuredSpan(
                source_rect=span_rect,
                render_rect=fitz.Rect(
                    span_rect.x0, span_rect.y0, max(span_rect.x1, right),
                    max(span_rect.y1, next_y - PADDING),
                ),
                origin=origin,
            ))
    return result


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
    rect: fitz.Rect, filled_rects: list[tuple[fitz.Rect, tuple[float, ...]]]
) -> tuple[float, ...]:
    center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
    candidates = [
        (filled.width * filled.height, color)
        for filled, color in filled_rects
        if filled.contains(center)
    ]
    return min(candidates, default=(0.0, (1.0, 1.0, 1.0)))[1]


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
    while current_size >= minimum_size:
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
        current_size -= 0.5
    return None


def _draw_text(page: fitz.Page, rect: fitz.Rect, text: str, fontsize: float, fontname: str) -> None:
    page.insert_textbox(
        rect,
        text,
        fontsize=fontsize,
        fontname=fontname,
        color=(0, 0, 0),
        align=0,
        overlay=True,
    )


def _measure_structured_line(
    text: str, width: float, fontsize: float, fontname: str, minimum_size: float
) -> float | None:
    current_size = max(float(fontsize), float(minimum_size), 1.0)
    minimum_size = max(float(minimum_size), 1.0)
    while current_size >= minimum_size:
        if fitz.get_text_length(text, fontname=fontname, fontsize=current_size) <= width:
            return current_size
        current_size -= 0.5
    return None


def _draw_structured_line(
    page: fitz.Page, span: _StructuredSpan, text: str, fontsize: float, fontname: str
) -> None:
    page.insert_text(
        span.origin,
        text,
        fontsize=fontsize,
        fontname=fontname,
        color=(0, 0, 0),
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
) -> bool:
    escaped_lines = [html.escape(line) for line in text.splitlines()]
    lines = "<br>".join(escaped_lines)
    weight = "bold" if flags & 16 else "normal"
    style = f"font-family: sans-serif; font-size: {max(float(fontsize), MIN_FONT_SIZE)}pt; font-weight: {weight};"
    spare_height, scale = page.insert_htmlbox(
        rect,
        f'<div style="{style}">{lines}</div>',
        scale_low=0.25,
        overlay=True,
    )
    return spare_height >= 0 and scale > 0


def _fit_text_with_fallbacks(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontsize: float,
    flags: int,
    obstacle_rects: list[fitz.Rect],
    minimum_size: float = MIN_FONT_SIZE,
    allow_expand: bool = True,
) -> bool:
    """Try, in order: (1) the original rect, (2) the rect grown into
    surrounding whitespace (never into another unit or an image), (3) an
    HTML box as a last resort. Exactly one of these ends up drawing text,
    so the unit is never rendered twice."""
    fontname = _font_name(flags)

    if _needs_unicode_fallback(text):
        return _insert_html_fallback(page, rect, text, fontsize, flags)

    size = _measure_fit(page, rect, text, fontsize, fontname, minimum_size)
    if size is not None:
        _draw_text(page, rect, text, size, fontname)
        return True

    expanded = (
        _safe_expanded_rect(rect=rect, page_rect=page.rect, other_rects=obstacle_rects)
        if allow_expand
        else rect
    )
    if allow_expand and expanded != rect:
        size = _measure_fit(page, expanded, text, fontsize, fontname, minimum_size)
        if size is not None:
            page.draw_rect(expanded, color=(1, 1, 1), fill=(1, 1, 1), width=0, overlay=True)
            _draw_text(page, expanded, text, size, fontname)
            return True

    # Last resort: HTML box with continuous scaling. Draw a white cover
    # first (in case the expanded rect differs from the original cover)
    # so this is the ONLY text object left in that area.
    fallback_rect = expanded if expanded != rect else rect
    page.draw_rect(fallback_rect, color=(1, 1, 1), fill=(1, 1, 1), width=0, overlay=True)
    return _insert_html_fallback(page, fallback_rect, text, fontsize, flags)


@dataclass
class RenderingPrimitives:
    """Small drawing toolbox used by strategy dispatch; it never selects strategy."""

    page: fitz.Page
    obstacle_rects: list[fitz.Rect]
    filled_rects: list[tuple[fitz.Rect, tuple[float, ...]]]

    def cover(self, rect: fitz.Rect, background_aware: bool = False) -> None:
        color = _background_color(rect, self.filled_rects) if background_aware else (1, 1, 1)
        self.page.draw_rect(rect, color=color, fill=color, width=0, overlay=True)

    def render_textbox(
        self,
        rect: fitz.Rect,
        text: str,
        fontsize: float,
        flags: int,
        minimum_size: float = MIN_FONT_SIZE,
    ) -> bool:
        return _fit_text_with_fallbacks(
            self.page,
            rect,
            text,
            fontsize,
            flags,
            self.obstacle_rects,
            minimum_size,
            allow_expand=False,
        )

    def render_source_spans(
        self,
        spans: list[_StructuredSpan],
        translated_lines: list[str],
        fontsize: float,
        flags: int,
    ) -> bool:
        fits = True
        fontname = _font_name(flags)
        for span, translated_line in zip(spans, translated_lines):
            source_cover = fitz.Rect(
                span.source_rect.x0 - 0.5,
                span.source_rect.y0,
                span.source_rect.x1 + 0.5,
                span.source_rect.y1,
            )
            self.cover(source_cover, background_aware=True)
            if _needs_unicode_fallback(translated_line):
                fits = _insert_html_fallback(
                    self.page, span.render_rect, translated_line, fontsize, flags
                ) and fits
                continue
            size = _measure_structured_line(
                translated_line,
                span.render_rect.width,
                fontsize,
                fontname,
                STRUCTURED_MIN_FONT_SIZE,
            )
            if size is not None:
                _draw_structured_line(self.page, span, translated_line, size, fontname)
            else:
                fits = _insert_html_fallback(
                    self.page, span.render_rect, translated_line, fontsize, flags
                ) and fits
        return fits


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
    extraction_source = extraction.get("source_file")
    translation_source = translation_payload.get("source_file")
    if isinstance(extraction_source, str) and extraction_source != pdf_path.name:
        raise RuntimeError(
            f"Extraction source mismatch: expected {pdf_path.name!r}, got {extraction_source!r}"
        )
    if (
        isinstance(extraction_source, str)
        and isinstance(translation_source, str)
        and translation_source != extraction_source
    ):
        raise RuntimeError(
            f"Translation source mismatch: extraction={extraction_source!r}, "
            f"translation={translation_source!r}"
        )
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
            layout_plan = (
                layout_plans.get(page_number)
                if layout_plans is not None
                else None
            ) or fallback_layout_plan(
                page_data, (_bbox_tuple(rect) for rect in image_rects)
            )
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
                if unit.get("translate") is not True:
                    stats.skipped_units += 1
                    continue
                unit_id = unit.get("id")
                translation_item = translations.get(unit_id)
                if translation_item is None:
                    stats.missing_translations += 1
                    stats.warnings.append(f"Unit {unit_id}: missing translation")
                    continue
                if translation_item.get("source") != unit.get("source"):
                    stats.warnings.append(f"Unit {unit_id}: translation source mismatch")
                    continue
                stats.translated_units += 1
                rect = _unit_rect(unit)
                if rect is None:
                    stats.warnings.append(f"Unit {unit_id}: invalid bounding box")
                    continue
                text = _plain_text(str(translation_item.get("translation", "")))
                structured_text = text
                if _is_compact_structured_unit(unit, rect):
                    text = " ".join(text.splitlines())
                if not text.strip():
                    stats.warnings.append(f"Unit {unit_id}: empty translation")
                    continue
                fontsize = float(unit.get("fontsize", 10.0))
                flags = int(unit.get("flags", 0))
                instruction = render_plan.instruction_for_unit(unit_id)
                if instruction is None:
                    stats.warnings.append(f"Unit {unit_id}: no executable RenderPlan instruction")
                    continue
                from .identity import identity_record, validate_source_ownership

                validate_source_ownership(unit, instruction)
                identity = identity_record(
                    page_number,
                    unit,
                    text,
                    instruction,
                    region_by_id.get(instruction.region_id),
                )
                stats.identity_records.append(identity)
                source_text = str(unit.get("source", ""))
                if text == source_text:
                    identity["render_status"] = "source_preserved"
                    identity["final_rendered_text"] = source_text
                    identity["fallback_reason"] = "translation equals source"
                    stats.rendered_units += 1
                    continue
                if _has_private_use(source_text) or _has_private_use(text):
                    stats.warnings.append(
                        f"Unit {unit_id}: private-use glyph preserved with original visual content"
                    )
                    identity["render_status"] = "source_preserved"
                    identity["final_rendered_text"] = source_text
                    identity["fallback_reason"] = "private-use glyph preserved"
                    stats.rendered_units += 1
                    continue
                strategy = instruction.strategy

                structured_lines = None
                structured_mismatch = False
                structured_cell_fallback = False
                fits = True
                if strategy in {
                    RenderingStrategy.STRUCTURED_REGION,
                    RenderingStrategy.SOURCE_SPAN_MAPPING,
                } and unit.get("line_count", 0) > 1:
                    structured_spans = _extracted_structured_spans(
                        page, unit, page_data, rect,
                        expand_last_column=unit.get("unit_type") != "table_cell"
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
                    fits = primitives.render_source_spans(
                        [item[0] for item in structured_lines],
                        [item[1] for item in structured_lines],
                        fontsize,
                        flags,
                    )
                elif structured_cell_fallback:
                    cover = _cover_rect(unit, rect)
                    primitives.cover(cover, background_aware=True)
                    text_rect = fitz.Rect(
                        cover.x0 + PADDING,
                        cover.y0 + PADDING,
                        cover.x1 - PADDING,
                        cover.y1 - PADDING,
                    )
                    fontname = _font_name(flags)
                    size = _measure_fit(page, text_rect, text, fontsize, fontname, STRUCTURED_MIN_FONT_SIZE)
                    if size is not None:
                        _draw_text(page, text_rect, text, size, fontname)
                    else:
                        fits = _insert_html_fallback(page, text_rect, text, fontsize, flags)
                elif structured_mismatch:
                    cover = _cover_rect(unit, rect)
                    if cover.width <= 0 or cover.height <= 0:
                        stats.warnings.append(f"Unit {unit_id}: invalid structured fallback rectangle")
                        identity["render_status"] = "failed"
                        identity["fallback_reason"] = "invalid structured fallback rectangle"
                        continue
                    primitives.cover(cover, background_aware=True)
                    text_rect = fitz.Rect(
                        cover.x0 + PADDING,
                        cover.y0 + PADDING,
                        cover.x1 - PADDING,
                        cover.y1 - PADDING,
                    )
                    fits = primitives.render_textbox(
                        text_rect, text, fontsize, flags, STRUCTURED_MIN_FONT_SIZE
                    )
                else:
                    cover = _cover_rect(unit, rect)
                    if cover.width <= 0 or cover.height <= 0:
                        stats.warnings.append(f"Unit {unit_id}: invalid cover rectangle")
                        continue
                    primitives.cover(cover)
                    toc_anchor: tuple[float, float, float, float] | None = None
                    if strategy == RenderingStrategy.TOC_REGION:
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
                            primitives.cover(text_rect, background_aware=True)
                    elif strategy == RenderingStrategy.EXPAND_REGION:
                        text_rect = fitz.Rect(instruction.bbox)
                        primitives.cover(text_rect)
                    else:
                        text_rect = fitz.Rect(
                            cover.x0 + PADDING,
                            cover.y0 + PADDING,
                            cover.x1 - PADDING,
                            cover.y1 - PADDING,
                        )
                    fits = primitives.render_textbox(
                        text_rect, text, fontsize, flags, MIN_FONT_SIZE
                    )
                    if toc_anchor is not None:
                        leader_end = toc_anchor[0] - PADDING
                        leader_start = max(
                            rect.x1 + PADDING,
                            text_rect.x0 + text_rect.width * 0.72,
                        )
                        dot_size = max(MIN_FONT_SIZE, min(fontsize, 8.0))
                        dot_width = max(fitz.get_text_length(".", fontname="helv", fontsize=dot_size), 1.0)
                        count = int(max(0.0, leader_end - leader_start) / dot_width)
                        if count >= 2:
                            baseline = min(toc_anchor[3] - 1.0, text_rect.y1 - 1.0)
                            page.insert_text(
                                (leader_start, baseline), "." * count,
                                fontsize=dot_size, fontname="helv", overlay=True,
                            )
                if not fits:
                    stats.warnings.append(f"Unit {unit_id}: text overflow at minimum font size")
                    identity["render_status"] = "rendered_overflow"
                    identity["fallback_reason"] = "text overflow at minimum font size"
                else:
                    identity["render_status"] = "rendered"
                identity["final_rendered_text"] = text
                if strategy == RenderingStrategy.FALLBACK_ORIGINAL_BBOX:
                    identity["fallback_reason"] = instruction.reason
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

        save_identity_artifacts(pdf_path, output_path, stats.identity_records, identity_debug_dir)
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
