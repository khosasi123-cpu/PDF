from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fitz

if TYPE_CHECKING:
    from .render_plan import RenderInstruction


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_artifact_identity(
    pdf_path: Path,
    extraction: dict[str, Any],
    translation: dict[str, Any],
) -> tuple[str, str]:
    source_file = extraction.get("source_file")
    if isinstance(source_file, str) and source_file != pdf_path.name:
        raise RuntimeError(
            f"Extraction source mismatch: expected {pdf_path.name!r}, got {source_file!r}"
        )

    source_sha256 = extraction.get("source_sha256")
    if extraction.get("schema_version", 0) >= 3 and not isinstance(source_sha256, str):
        raise RuntimeError("Extraction artifact is missing source_sha256")
    actual_source_sha256 = sha256_file(pdf_path)
    if isinstance(source_sha256, str) and source_sha256 != actual_source_sha256:
        raise RuntimeError(
            "Extraction source hash mismatch: artifact belongs to a different PDF"
        )

    translation_source = translation.get("source_file")
    if (
        isinstance(source_file, str)
        and isinstance(translation_source, str)
        and translation_source != source_file
    ):
        raise RuntimeError(
            f"Translation source mismatch: extraction={source_file!r}, "
            f"translation={translation_source!r}"
        )

    extraction_sha256 = payload_sha256(extraction)
    if translation.get("schema_version", 0) >= 2:
        if translation.get("source_sha256") != actual_source_sha256:
            raise RuntimeError(
                "Translation source hash mismatch: artifact belongs to a different PDF"
            )
        if translation.get("source_extraction_sha256") != extraction_sha256:
            raise RuntimeError(
                "Translation extraction hash mismatch: artifact was produced from a different extraction"
            )
    return actual_source_sha256, extraction_sha256


def validate_layout_identity(
    plan: Any,
    source_sha256: str,
    extraction_sha256: str,
) -> None:
    if getattr(plan, "source_sha256", None) != source_sha256:
        raise RuntimeError(
            f"LayoutPlan page {plan.page_number} source hash mismatch"
        )
    if getattr(plan, "source_extraction_sha256", None) != extraction_sha256:
        raise RuntimeError(
            f"LayoutPlan page {plan.page_number} extraction hash mismatch"
        )


def validate_source_ownership(unit: dict[str, Any], instruction: RenderInstruction) -> None:
    unit_ids = tuple(sorted(set(unit.get("source_ids", []))))
    instruction_ids = tuple(sorted(set(instruction.source_ids)))
    if unit_ids != instruction_ids:
        raise RuntimeError(
            f"Unit {unit.get('id')}: source ownership mismatch; "
            f"unit_source_ids={unit_ids!r}; instruction_source_ids={instruction_ids!r}"
        )


def identity_record(
    page_number: int,
    unit: dict[str, Any],
    translation: str | None,
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
        "planned_bbox": list(instruction.bbox),
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
        "final_bbox": None,
        "final_bboxes": [],
        "render_status": "planned",
        "final_rendered_text": None,
        "fallback_reason": None,
    }


def complete_identity_record(
    record: dict[str, Any],
    rects: list[fitz.Rect],
    rendered_text: str,
    status: str,
    fallback_reason: str | None = None,
) -> None:
    boxes = [[rect.x0, rect.y0, rect.x1, rect.y1] for rect in rects]
    record["final_bboxes"] = boxes
    if boxes:
        record["final_bbox"] = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
    record["final_rendered_text"] = rendered_text
    record["render_status"] = status
    record["fallback_reason"] = fallback_reason


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
        bbox = record.get(bbox_key)
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        rect = fitz.Rect(bbox)
        page.draw_rect(rect, color=(0.9, 0.1, 0.1), width=1.0, overlay=True)
        source_ids = ",".join(record.get("final_source_ids", [])[:2]) or "none"
        label = (
            f"U{record['translation_unit']} src={source_ids}\n"
            f"{record['layout_region']}:{record['semantic_type']} | "
            f"{record['render_instruction']} | {record['final_strategy']}"
        )
        label_rect = fitz.Rect(
            rect.x0, max(0, rect.y0 - 16), min(page.rect.x1, rect.x0 + 280), rect.y0
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
    extraction_path: Path | None = None,
    translation_path: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage: list[dict[str, Any]] = []
    coverage_summary = {
        "translated": 0,
        "protected": 0,
        "fallback": 0,
        "non_translatable": 0,
        "duplicate": 0,
        "unaccounted": 0,
    }
    if extraction_path is not None:
        extraction_payload = json.loads(extraction_path.read_text(encoding="utf-8"))
        records_by_source: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            for source_id in record.get("source_ids", []):
                records_by_source.setdefault(source_id, []).append(record)
        protected_toc_ids = {
            source_id
            for page in extraction_payload.get("pages", [])
            for entry in page.get("toc_entries", [])
            for source_id in (
                entry.get("page_number_source_ids", [])
                + entry.get("leader_source_ids", [])
            )
        }
        for page in extraction_payload.get("pages", []):
            for source in page.get("source_objects", []):
                if source.get("kind") != "span" or not str(source.get("text", "")).strip():
                    continue
                source_id = source.get("id")
                owners = records_by_source.get(source_id, [])
                reason = None
                if len(owners) > 1:
                    status = "duplicate"
                    reason = "source object is owned by multiple rendered units"
                elif len(owners) == 1:
                    owner = owners[0]
                    render_status = owner.get("render_status")
                    if render_status == "skipped":
                        status = "non_translatable"
                        reason = owner.get("fallback_reason")
                    elif render_status == "rendered":
                        status = "translated"
                    elif render_status == "source_preserved":
                        status = "protected"
                        reason = owner.get("fallback_reason")
                    else:
                        status = "fallback"
                        reason = owner.get("fallback_reason") or str(render_status)
                elif source_id in protected_toc_ids:
                    status = "protected"
                    reason = "TOC page number or leader retains source geometry"
                else:
                    status = "unaccounted"
                    reason = "no translation unit or protected structural owner"
                coverage_summary[status] += 1
                coverage.append({
                    "source_id": source_id,
                    "page_number": page.get("page_number"),
                    "text": source.get("text", ""),
                    "status": status,
                    "reason": reason,
                    "translation_units": [
                        owner.get("translation_unit") for owner in owners
                    ],
                })

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": "render_identity",
        "source_file": source_pdf.name,
        "source_sha256": sha256_file(source_pdf),
        "output_file": output_pdf.name,
        "output_sha256": sha256_file(output_pdf),
        "source_coverage_summary": coverage_summary,
        "source_coverage": coverage,
        "units": records,
    }
    if extraction_path is not None:
        manifest["source_extraction"] = str(extraction_path)
        manifest["source_extraction_sha256"] = payload_sha256(extraction_payload)
    if translation_path is not None:
        manifest["source_translation"] = str(translation_path)
        manifest["source_translation_sha256"] = payload_sha256(
            json.loads(translation_path.read_text(encoding="utf-8"))
        )
    (output_dir / "render_identity.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
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
