from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import fitz
from dotenv import load_dotenv
from pydantic import ValidationError

from .layout import BBox, LayoutPlan, fallback_layout_plan, resolve_layout_plan_geometry


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

load_dotenv()

VISION_SYSTEM_PROMPT = """Analyze the visual and semantic layout of one PDF page.
Return only a JSON LayoutPlan. Identify regions, reading order, semantic types,
relationships, embedded image text, and one recommended rendering strategy per region.
Recommendations are advisory. Do not emit Python, PyMuPDF calls, drawing commands,
executable expressions, or final rendering coordinates. Use the supplied PDF geometry
for region bboxes and do not invent text. Allowed region types: heading, paragraph,
bullet_list, structured, table, form, image, screenshot, diagram, warning, header,
footer, running_header, running_footer, toc, toc_entry, multi-column, unknown.
Allowed strategies: preserve_region, reflow_region,
structured_region, multicolumn_region, preserve_image, source_span_mapping,
expand_region, fallback_original_bbox, toc_region.
For normal PDF-backed text and images, reference the supplied source object IDs in
source_ids. Their PyMuPDF geometry is authoritative, so bbox may be null. Use
geometry_source "pdf" for referenced objects. Only use a bbox with geometry_source
"vision_estimate" for visual regions that have no corresponding source object.
Use exactly this top-level shape:
{"page_number":1,"width":612,"height":792,"confidence":0.9,"source":"vision",
"regions":[{"id":"r1","type":"paragraph","bbox":[10,10,100,50],
"reading_order":0,"confidence":0.9,"recommended_strategy":"reflow_region",
"unit_ids":[1],"source_ids":["p0001/b0001"],"geometry_source":"pdf",
"parent_id":null,"child_ids":[],"relationships":[],
"contains_embedded_text":false}],"warnings":[]}
Relationships, when present, use {"type":"contains","target_id":"r2"}."""


class VisionLayoutClient(Protocol):
    def analyze(self, page_png: bytes, geometry: dict[str, Any]) -> str:
        ...


def _extract_json(response: str) -> dict[str, Any]:
    candidate = response.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start < 0:
            raise ValueError("vision response does not contain a JSON object")
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate[start:])
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid vision JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("vision response must be a JSON object")
    return value


def compact_geometry(page_data: dict[str, Any], image_bboxes: list[BBox]) -> dict[str, Any]:
    source_index = {
        source.get("id"): source for source in page_data.get("source_objects", [])
        if isinstance(source.get("id"), str)
    }
    selected_ids: set[str] = set()
    detailed_ids: set[str] = set()
    selected_by_unit: dict[int, list[str]] = {}
    for unit in page_data.get("units", []):
        source_ids = [source_id for source_id in unit.get("source_ids", []) if source_id in source_index]
        detailed = unit.get("unit_type") in {"table_cell", "borderless_structured", "toc_entry"}
        preferred_kinds = (
            {"cell", "span"}
            if detailed
            else {"block"}
        )
        preferred = [
            source_id for source_id in source_ids
            if source_index[source_id].get("kind") in preferred_kinds
        ]
        selected = preferred or [
            source_id for source_id in source_ids
            if source_index[source_id].get("kind") == "span"
        ]
        selected_ids.update(selected)
        if detailed:
            detailed_ids.update(selected)
        if isinstance(unit.get("id"), int):
            selected_by_unit[unit["id"]] = selected
    image_keys = {
        tuple(round(float(value), 2) for value in bbox) for bbox in image_bboxes
    }
    seen_image_keys: set[tuple[float, ...]] = set()
    for source_id, source in source_index.items():
        if source.get("kind") != "image":
            continue
        bbox = source.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        key = tuple(round(float(value), 2) for value in bbox)
        if key in image_keys and key not in seen_image_keys:
            seen_image_keys.add(key)
            selected_ids.add(source_id)
            detailed_ids.add(source_id)
    selected_sources = [source_index[source_id] for source_id in sorted(detailed_ids)]
    return {
        "page_number": page_data.get("page_number"),
        "width": page_data.get("width"),
        "height": page_data.get("height"),
        "units": [
            {
                "id": unit.get("id"),
                "unit_type": unit.get("unit_type"),
                "bbox": unit.get("bbox"),
                "line_count": unit.get("line_count"),
                "text": unit.get("source", "")[:160],
                "source_ids": selected_by_unit.get(unit.get("id"), []),
                "semantic_role": unit.get("semantic_role"),
                "template_group_id": unit.get("template_group_id"),
                "recurrence_count": unit.get("recurrence_count", 0),
                "metadata": {
                    key: value for key, value in unit.get("metadata", {}).items()
                    if key in {
                        "toc_page_number_bbox", "toc_hierarchy_level", "toc_column",
                        "structured_column", "geometry_source",
                    }
                },
            }
            for unit in page_data.get("units", [])
        ],
        "source_objects": [
            {
                "id": source.get("id"),
                "kind": source.get("kind"),
                "bbox": source.get("bbox"),
                "text": (
                    str(source.get("text", ""))[:120]
                    if source.get("kind") in {"span", "cell"} else ""
                ),
                "parent_id": source.get("parent_id"),
            }
            for source in selected_sources
        ],
        "toc_entries": page_data.get("toc_entries", []),
        "images": [{"bbox": list(bbox)} for bbox in image_bboxes],
    }


def analyze_page_layout(
    page_data: dict[str, Any],
    page_png: bytes,
    image_bboxes: list[BBox],
    client: VisionLayoutClient | None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> LayoutPlan:
    fallback = fallback_layout_plan(page_data, image_bboxes)
    if client is None:
        fallback.warnings.append("Vision layout analysis unavailable; deterministic fallback used")
        return fallback
    try:
        response = client.analyze(page_png, compact_geometry(page_data, image_bboxes))
        plan = LayoutPlan.model_validate(_extract_json(response))
        if plan.page_number != page_data.get("page_number"):
            raise ValueError("vision LayoutPlan page number does not match extraction")
        if plan.confidence < confidence_threshold:
            fallback.warnings.append(
                f"Vision LayoutPlan confidence {plan.confidence:.2f} is below {confidence_threshold:.2f}; "
                "deterministic fallback used"
            )
            return fallback
        return resolve_layout_plan_geometry(plan, page_data)
    except Exception as error:  # Model/API failures must never disable deterministic rendering.
        LOGGER.warning("Vision layout analysis failed: %s", error)
        fallback.warnings.append(f"Vision layout analysis failed; deterministic fallback used: {error}")
        return fallback


class OpenAIMinistralVisionClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    @classmethod
    def from_environment(cls) -> OpenAIMinistralVisionClient | None:
        model = os.getenv("VISION_MODEL", "").strip()
        base_url = (os.getenv("VISION_BASE_URL") or os.getenv("LLM_BASE_URL") or "").strip()
        if not model or not base_url:
            return None
        return cls(base_url, os.getenv("OPENAI_API_KEY", "local-key"), model)

    def analyze(self, page_png: bytes, geometry: dict[str, Any]) -> str:
        image_url = "data:image/png;base64," + base64.b64encode(page_png).decode("ascii")
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=4000,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "layout_plan",
                    "schema": LayoutPlan.model_json_schema(),
                    "strict": True,
                },
            },
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(geometry, ensure_ascii=False)},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("vision model returned no text")
        return content


def _image_bboxes(page: fitz.Page) -> list[BBox]:
    result: list[BBox] = []
    seen: set[tuple[float, float, float, float]] = set()
    for image in page.get_image_info():
        bbox = image.get("bbox")
        if bbox:
            rect = fitz.Rect(bbox)
            if not rect.is_empty:
                value = (rect.x0, rect.y0, rect.x1, rect.y1)
                key = tuple(round(coordinate, 3) for coordinate in value)
                if key not in seen:
                    seen.add(key)
                    result.append(value)
    return result


def analyze_document_layout(
    pdf_path: Path,
    extraction: dict[str, Any],
    output_dir: Path,
    client: VisionLayoutClient | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[int, LayoutPlan]:
    from .render_debug import save_layout_debug

    output_dir.mkdir(parents=True, exist_ok=True)
    page_data_by_number = {
        page.get("page_number"): page for page in extraction.get("pages", [])
        if isinstance(page.get("page_number"), int)
    }
    plans: dict[int, LayoutPlan] = {}
    with fitz.open(pdf_path) as document:
        for page in document:
            page_number = page.number + 1
            page_data = page_data_by_number.get(page_number, {
                "page_number": page_number,
                "width": page.rect.width,
                "height": page.rect.height,
                "units": [],
            })
            images = _image_bboxes(page)
            page_png = page.get_pixmap(matrix=fitz.Matrix(0.8, 0.8), alpha=False).tobytes("png")
            plan = analyze_page_layout(
                page_data, page_png, images, client, confidence_threshold
            )
            plans[page_number] = plan
            stem = f"page_{page_number:03d}"
            (output_dir / f"{stem}.json").write_text(
                plan.model_dump_json(indent=2), encoding="utf-8"
            )
            save_layout_debug(document, page.number, plan, output_dir / f"{stem}_debug.png")
    return plans
