from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz

LOGGER = logging.getLogger(__name__)
DEFAULT_EXTRACTION_PATH = Path("artifacts/extraction/extraction.json")
DEFAULT_TRANSLATION_PATH = Path("artifacts/translation/translation.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/rendered")
MIN_FONT_SIZE = 5.5
PADDING = 0.75
CELL_BORDER_INSET = 1.5


@dataclass
class RenderStats:
    total_units: int = 0
    translated_units: int = 0
    skipped_units: int = 0
    rendered_units: int = 0
    missing_translations: int = 0
    warnings: list[str] = field(default_factory=list)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read JSON '{path}': {error}") from error


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
    return value


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


def fit_text_to_rect(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontsize: float,
    flags: int = 0,
) -> tuple[float, bool]:
    """Insert text at the largest size that fits, returning size and fit status."""
    safe_size = max(float(fontsize), MIN_FONT_SIZE)
    fontname = _font_name(flags)
    current_size = safe_size
    while current_size >= MIN_FONT_SIZE:
        result = page.insert_textbox(
            rect,
            text,
            fontsize=current_size,
            fontname=fontname,
            color=(0, 0, 0),
            align=0,
            overlay=True,
        )
        if result >= 0:
            return current_size, True
        current_size -= 0.5
    page.insert_textbox(
        rect,
        text,
        fontsize=MIN_FONT_SIZE,
        fontname=fontname,
        color=(0, 0, 0),
        align=0,
        overlay=True,
    )
    return MIN_FONT_SIZE, False


def render_pdf(
    pdf_path: Path,
    extraction_path: Path = DEFAULT_EXTRACTION_PATH,
    translation_path: Path = DEFAULT_TRANSLATION_PATH,
    output_path: Path | None = None,
) -> tuple[Path, RenderStats]:
    extraction = _load_json(extraction_path)
    translations = _translation_map(_load_json(translation_path))
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
                if not text.strip():
                    stats.warnings.append(f"Unit {unit_id}: empty translation")
                    continue
                cover = _cover_rect(unit, rect)
                if cover.width <= 0 or cover.height <= 0:
                    stats.warnings.append(f"Unit {unit_id}: invalid cover rectangle")
                    continue
                page.draw_rect(cover, color=(1, 1, 1), fill=(1, 1, 1), width=0, overlay=True)
                text_rect = fitz.Rect(cover.x0 + PADDING, cover.y0 + PADDING, cover.x1 - PADDING, cover.y1 - PADDING)
                _, fits = fit_text_to_rect(
                    page,
                    text_rect,
                    text,
                    float(unit.get("fontsize", 10.0)),
                    int(unit.get("flags", 0)),
                )
                if not fits:
                    stats.warnings.append(f"Unit {unit_id}: text overflow at minimum font size")
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
    parser.add_argument("pdf", type=Path, nargs="?", default=Path("data/input/test.pdf"))
    parser.add_argument("--extraction", type=Path, default=DEFAULT_EXTRACTION_PATH)
    parser.add_argument("--translation", type=Path, default=DEFAULT_TRANSLATION_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        output, stats = render_pdf(args.pdf, args.extraction, args.translation, args.output)
        with fitz.open(args.pdf) as original:
            print_summary(args.pdf, output, original.page_count, stats)
    except (RuntimeError, fitz.FileDataError, OSError) as error:
        print(f"PDF rendering failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
