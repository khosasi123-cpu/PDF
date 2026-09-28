from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz

from .render_plan import RenderInstruction


def validate_source_ownership(unit: dict[str, Any], instruction: RenderInstruction) -> None:
    unit_ids = tuple(sorted(set(unit.get("source_ids", []))))
    instruction_ids = tuple(sorted(set(instruction.source_ids)))
    if unit_ids and unit_ids != instruction_ids:
        raise RuntimeError(
            f"Unit {unit.get('id')}: source ownership mismatch; "
            f"unit_source_ids={unit_ids!r}; instruction_source_ids={instruction_ids!r}"
        )


def identity_record(
    page_number: int,
    unit: dict[str, Any],
    translation: str,
    instruction: RenderInstruction,
    region: Any,
) -> dict[str, Any]:
    return {
        "page_number": page_number,
        "translation_unit": unit.get("id"),
        "source": unit.get("source", ""),
        "translation": translation,
        "source_ids": list(unit.get("source_ids", [])),
        "source_bbox": list(instruction.original_bbox),
        "layout_region": instruction.region_id,
        "semantic_type": getattr(getattr(region, "type", None), "value", "unknown"),
        "template_group": getattr(region, "template_group_id", None),
        "recommended_strategy": getattr(
            getattr(region, "recommended_strategy", None), "value", None
        ),
        "render_instruction": instruction.id,
        "final_strategy": instruction.strategy.value,
        "semantic_source_ids": list(instruction.semantic_source_ids),
        "final_source_ids": list(instruction.source_ids),
        "final_bbox": list(instruction.bbox),
        "render_status": "planned",
        "final_rendered_text": None,
        "fallback_reason": None,
    }


def _overlay_page(
    source: fitz.Document,
    page_index: int,
    records: list[dict[str, Any]],
    bbox_key: str,
    output_path: Path,
) -> None:
    original = source[page_index]
    debug = fitz.open()
    page = debug.new_page(width=original.rect.width, height=original.rect.height)
    page.show_pdf_page(page.rect, source, page_index)
    for record in records:
        rect = fitz.Rect(record[bbox_key])
        page.draw_rect(rect, color=(0.9, 0.1, 0.1), width=1.0, overlay=True)
        label = f"U{record['translation_unit']} | {record['render_instruction']}"
        label_rect = fitz.Rect(
            rect.x0, max(0, rect.y0 - 8), min(page.rect.x1, rect.x0 + 180), rect.y0
        )
        page.draw_rect(label_rect, color=(0.9, 0.1, 0.1), fill=(1, 1, 1), width=0.4)
        page.insert_textbox(label_rect, label, fontsize=5.5, color=(0.8, 0.0, 0.0))
    debug[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(output_path)
    debug.close()


def save_identity_artifacts(
    source_pdf: Path,
    output_pdf: Path,
    records: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "render_identity.json").write_text(
        json.dumps({"units": records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    by_page: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_page.setdefault(int(record["page_number"]), []).append(record)
    with fitz.open(source_pdf) as source, fitz.open(output_pdf) as output:
        for page_index in range(source.page_count):
            page_number = page_index + 1
            source[page_index].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(
                output_dir / f"source_page_{page_number:03d}.png"
            )
            output[page_index].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(
                output_dir / f"output_page_{page_number:03d}.png"
            )
            page_records = by_page.get(page_number, [])
            _overlay_page(
                source, page_index, page_records, "source_bbox",
                output_dir / f"source_page_{page_number:03d}_overlay.png",
            )
            _overlay_page(
                output, page_index, page_records, "final_bbox",
                output_dir / f"output_page_{page_number:03d}_overlay.png",
            )
