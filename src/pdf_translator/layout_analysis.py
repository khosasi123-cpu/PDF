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
from .identity import payload_sha256, sha256_file


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
VISION_MAX_OUTPUT_TOKENS = 3000
VISION_TEXT_HINT_LENGTH = 80
DEFAULT_VISION_TIMEOUT_SECONDS = 300

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
    detailed_ids: set[str] = set()
    compact_units: list[dict[str, Any]] = []
    for unit in page_data.get("units", []):
        source_ids = [source_id for source_id in unit.get("source_ids", []) if source_id in source_index]
        unit_type = unit.get("unit_type")
        detailed = unit_type in {"table_cell", "borderless_structured"}
        semantic_type = detailed or unit_type == "toc_entry"
        preferred_kinds = {"cell", "span"} if semantic_type else {"block"}
        preferred = [
            source_id for source_id in source_ids
            if source_index[source_id].get("kind") in preferred_kinds
        ]
        selected = preferred
        if not selected:
            selected = source_ids[:1]
        if detailed:
            detailed_ids.update(selected)
        if not isinstance(unit.get("id"), int):
            continue
        compact_unit: dict[str, Any] = {
            "id": unit["id"],
            "bbox": unit.get("bbox"),
            "text": str(unit.get("source", ""))[:VISION_TEXT_HINT_LENGTH],
            "source_ids": selected,
        }
        if semantic_type:
            compact_unit["type"] = unit_type
        compact_units.append(compact_unit)

    geometry: dict[str, Any] = {
        "page_number": page_data.get("page_number"),
        "width": page_data.get("width"),
        "height": page_data.get("height"),
        "units": compact_units,
    }
    if detailed_ids:
        detailed_sources: list[dict[str, Any]] = []
        for source_id in sorted(detailed_ids):
            source = source_index[source_id]
            item: dict[str, Any] = {
                "id": source_id,
                "bbox": source.get("bbox"),
            }
            text = str(source.get("text", ""))[:VISION_TEXT_HINT_LENGTH]
            if text:
                item["text"] = text
            detailed_sources.append(item)
        geometry["source_objects"] = detailed_sources

    toc_entries: list[dict[str, Any]] = []
    for entry in page_data.get("toc_entries", []):
        compact_entry = {
            key: entry[key]
            for key in (
                "unit_id",
                "title_source_ids",
                "page_number_source_ids",
                "title_bbox",
                "page_number_bbox",
                "hierarchy_level",
                "column",
            )
            if entry.get(key) not in (None, [], {})
        }
        if compact_entry:
            toc_entries.append(compact_entry)
    if toc_entries:
        geometry["toc_entries"] = toc_entries

    if image_bboxes:
        image_sources = [
            source for source in source_index.values()
            if source.get("kind") == "image"
        ]
        compact_images: list[dict[str, Any]] = []
        for bbox in image_bboxes:
            item: dict[str, Any] = {"bbox": list(bbox)}
            key = tuple(round(float(value), 2) for value in bbox)
            matching_source = next((
                source for source in image_sources
                if isinstance(source.get("bbox"), (list, tuple))
                and len(source["bbox"]) == 4
                and tuple(round(float(value), 2) for value in source["bbox"]) == key
            ), None)
            if matching_source is not None:
                item["source_id"] = matching_source["id"]
            compact_images.append(item)
        geometry["images"] = compact_images
    return geometry


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
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = DEFAULT_VISION_TIMEOUT_SECONDS,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    @classmethod
    def from_environment(cls) -> OpenAIMinistralVisionClient | None:
        model = os.getenv("VISION_MODEL", "").strip()
        base_url = (os.getenv("VISION_BASE_URL") or os.getenv("LLM_BASE_URL") or "").strip()
        if not model or not base_url:
            return None
        try:
            timeout = float(os.getenv("VISION_TIMEOUT_SECONDS", DEFAULT_VISION_TIMEOUT_SECONDS))
        except ValueError as error:
            raise ValueError("VISION_TIMEOUT_SECONDS must be a number") from error
        if timeout <= 0:
            raise ValueError("VISION_TIMEOUT_SECONDS must be positive")
        return cls(base_url, os.getenv("OPENAI_API_KEY", "local-key"), model, timeout)

    def analyze(self, page_png: bytes, geometry: dict[str, Any]) -> str:
        image_url = "data:image/png;base64," + base64.b64encode(page_png).decode("ascii")
        geometry_json = json.dumps(geometry, ensure_ascii=False, separators=(",", ":"))
        image_width = image_height = 0
        if len(page_png) >= 24 and page_png[:8] == b"\x89PNG\r\n\x1a\n":
            image_width = int.from_bytes(page_png[16:20], "big")
            image_height = int.from_bytes(page_png[20:24], "big")
        LOGGER.info(
            "Vision request: page=%s geometry_chars=%s units=%s "
            "detailed_source_objects=%s image_size=%sx%s requested_output_tokens=%s",
            geometry.get("page_number"),
            len(geometry_json),
            len(geometry.get("units", [])),
            len(geometry.get("source_objects", [])),
            image_width,
            image_height,
            VISION_MAX_OUTPUT_TOKENS,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=VISION_MAX_OUTPUT_TOKENS,
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
                        {"type": "text", "text": geometry_json},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError(
                f"vision response exceeded the {VISION_MAX_OUTPUT_TOKENS}-token output limit"
            )
        content = choice.message.content
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

    actual_source_sha256 = sha256_file(pdf_path)
    if extraction.get("source_file") not in (None, pdf_path.name):
        raise RuntimeError(
            f"Extraction source mismatch: expected {pdf_path.name!r}, "
            f"got {extraction.get('source_file')!r}"
        )
    if extraction.get("schema_version", 0) >= 3:
        if extraction.get("source_sha256") != actual_source_sha256:
            raise RuntimeError(
                "Extraction source hash mismatch: artifact belongs to a different PDF"
            )
    extraction_sha256 = payload_sha256(extraction)
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
            plan.source_file = pdf_path.name
            plan.source_sha256 = actual_source_sha256
            plan.source_extraction_sha256 = extraction_sha256
            plans[page_number] = plan
            stem = f"page_{page_number:03d}"
            (output_dir / f"{stem}.json").write_text(
                plan.model_dump_json(indent=2), encoding="utf-8"
            )
            save_layout_debug(document, page.number, plan, output_dir / f"{stem}_debug.png")
    return plans
