from __future__ import annotations

from pathlib import Path

import fitz

from .layout import LayoutPlan
from .render_plan import RenderPlan


_COLORS = (
    (0.85, 0.15, 0.15),
    (0.1, 0.45, 0.85),
    (0.05, 0.65, 0.3),
    (0.75, 0.4, 0.05),
    (0.55, 0.2, 0.75),
)


def _debug_page(source: fitz.Document, page_index: int) -> tuple[fitz.Document, fitz.Page]:
    source_page = source[page_index]
    debug = fitz.open()
    page = debug.new_page(width=source_page.rect.width, height=source_page.rect.height)
    page.show_pdf_page(page.rect, source, page_index)
    return debug, page


def _write_png(document: fitz.Document, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(path)
    document.close()


def save_layout_debug(
    source: fitz.Document, page_index: int, plan: LayoutPlan, output_path: Path
) -> None:
    debug, page = _debug_page(source, page_index)
    for index, region in enumerate(plan.regions):
        color = _COLORS[index % len(_COLORS)]
        rect = fitz.Rect(region.bbox)
        page.draw_rect(rect, color=color, width=1.2, overlay=True)
        label = (
            f"{region.id} | {region.type.value} | {region.recommended_strategy.value} | "
            f"{region.geometry_source.value} | {region.confidence:.2f} | #{region.reading_order}"
        )
        if region.source_ids:
            label += f" | src={','.join(region.source_ids[:3])}"
        if region.parent_id:
            label += f" | parent={region.parent_id}"
        if region.template_group_id:
            label += f" | template={region.template_group_id}x{region.recurrence_count}"
        if region.type.value == "toc_entry":
            label += f" | level={region.hierarchy_level} col={region.column}"
        label_rect = fitz.Rect(rect.x0, max(0, rect.y0 - 9), min(page.rect.x1, rect.x0 + 260), rect.y0)
        page.draw_rect(label_rect, color=color, fill=(1, 1, 1), width=0.5, overlay=True)
        page.insert_textbox(label_rect, label, fontsize=5.5, color=color, overlay=True)
    _write_png(debug, output_path)


def save_render_plan_debug(
    source: fitz.Document, page_index: int, plan: RenderPlan, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"page_{plan.page_number:03d}_render_plan"
    (output_dir / f"{stem}.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    debug, page = _debug_page(source, page_index)
    for index, instruction in enumerate(plan.regions):
        color = _COLORS[index % len(_COLORS)]
        rect = fitz.Rect(instruction.bbox)
        page.draw_rect(rect, color=color, width=1.2, overlay=True)
        label = (
            f"{instruction.region_id} | {instruction.strategy.value} | "
            f"{instruction.geometry_source.value}"
        )
        if instruction.unit_id is not None:
            label += f" | unit {instruction.unit_id}"
        if instruction.source_ids:
            label += f" | src={','.join(instruction.source_ids[:2])}"
        if instruction.template_group_id:
            label += f" | template={instruction.template_group_id}x{instruction.recurrence_count}"
        if instruction.hierarchy_level is not None:
            label += f" | level={instruction.hierarchy_level} col={instruction.column}"
        label_rect = fitz.Rect(rect.x0, max(0, rect.y0 - 9), min(page.rect.x1, rect.x0 + 220), rect.y0)
        page.draw_rect(label_rect, color=color, fill=(1, 1, 1), width=0.5, overlay=True)
        page.insert_textbox(label_rect, label, fontsize=5.5, color=color, overlay=True)
    _write_png(debug, output_dir / f"{stem}_debug.png")
