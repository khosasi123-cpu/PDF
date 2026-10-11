from __future__ import annotations

import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import fitz
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .layout import (
    BBox,
    GeometrySource,
    LayoutPlan,
    LayoutRegion,
    RegionType,
    RenderingStrategy,
    fallback_layout_plan,
    resolve_layout_plan_geometry,
)
from .identity import payload_sha256, sha256_file


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
VISION_MAX_OUTPUT_TOKENS = 3000
VISION_TEXT_HINT_LENGTH = 40
DEFAULT_VISION_TIMEOUT_SECONDS = 300
DEFAULT_VISION_WORKERS = 2

load_dotenv()

VISION_SYSTEM_PROMPT = """Analyze one PDF page and return only the compact JSON response.
The input contains page size, extracted units, and numbered images. Group units into
semantic regions, preserve reading order, and recommend one rendering strategy.
Use only unit IDs and image indexes present in the input. Never invent references.
Do not return bboxes, source IDs, text, Python, drawing commands, or executable data;
Python resolves all final geometry and ownership from the supplied references.
Parent and child references are optional and must reference region IDs in the same
response. Emit empty arrays and null for unused required fields. Return one region per
toc_entry unit so each page-number anchor remains independent. Recommendations are
advisory and never directly control PDF drawing."""


class VisionRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: RegionType
    reading_order: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_strategy: RenderingStrategy
    unit_ids: list[int]
    image_indexes: list[int]
    parent_id: str | None
    child_ids: list[str]
    contains_embedded_text: bool

    @field_validator("unit_ids", "image_indexes")
    @classmethod
    def validate_indexes(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("references must be positive integers")
        if len(values) != len(set(values)):
            raise ValueError("references must be unique within a region")
        return values


class VisionLayoutResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    regions: list[VisionRegion]

    @model_validator(mode="after")
    def validate_region_ids(self) -> VisionLayoutResponse:
        region_ids = [region.id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region IDs must be unique")
        return self


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


def _compact_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [round(float(coordinate), 1) for coordinate in value]
    except (TypeError, ValueError):
        return None


def _image_geometry(
    page_data: dict[str, Any], image_bboxes: list[BBox]
) -> list[dict[str, Any]]:
    image_sources = [
        source for source in page_data.get("source_objects", [])
        if source.get("kind") == "image" and _compact_bbox(source.get("bbox")) is not None
    ]
    images: list[dict[str, Any]] = []
    for index, bbox in enumerate(image_bboxes, start=1):
        compact_bbox = _compact_bbox(bbox)
        if compact_bbox is None:
            continue
        exact_bbox = tuple(float(coordinate) for coordinate in bbox)
        key = tuple(round(coordinate, 3) for coordinate in exact_bbox)
        source = next((
            candidate for candidate in image_sources
            if tuple(
                round(float(coordinate), 3)
                for coordinate in candidate.get("bbox", ())
            ) == key
        ), None)
        images.append({
            "index": index,
            "box": compact_bbox,
            "bbox": exact_bbox,
            "source_id": source.get("id") if source is not None else None,
        })
    return images


def compact_geometry(page_data: dict[str, Any], image_bboxes: list[BBox]) -> dict[str, Any]:
    compact_units: list[dict[str, Any]] = []
    for unit in page_data.get("units", []):
        if not isinstance(unit.get("id"), int) or _compact_bbox(unit.get("bbox")) is None:
            continue
        compact_unit: dict[str, Any] = {
            "id": unit["id"],
            "box": _compact_bbox(unit["bbox"]),
            "text": str(unit.get("source", ""))[:VISION_TEXT_HINT_LENGTH],
        }
        unit_type = str(unit.get("unit_type", "text"))
        if unit_type != "text":
            compact_unit["type"] = unit_type
        if unit.get("semantic_role"):
            compact_unit["role"] = unit["semantic_role"]
        compact_units.append(compact_unit)

    geometry: dict[str, Any] = {
        "page": page_data.get("page_number"),
        "size": [
            round(float(page_data.get("width", 0.0)), 1),
            round(float(page_data.get("height", 0.0)), 1),
        ],
        "units": compact_units,
    }
    images = _image_geometry(page_data, image_bboxes)
    if images:
        geometry["images"] = [
            {"index": image["index"], "box": image["box"]} for image in images
        ]
    return geometry


def _union_bboxes(boxes: list[list[float]]) -> BBox:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _layout_plan_from_vision(
    response: VisionLayoutResponse,
    page_data: dict[str, Any],
    image_bboxes: list[BBox],
) -> LayoutPlan:
    unit_index = {
        unit.get("id"): unit for unit in page_data.get("units", [])
        if isinstance(unit.get("id"), int) and _compact_bbox(unit.get("bbox")) is not None
    }
    source_index = {
        source.get("id"): source for source in page_data.get("source_objects", [])
        if isinstance(source.get("id"), str)
    }
    image_index = {
        image["index"]: image for image in _image_geometry(page_data, image_bboxes)
    }
    fallback_regions = {
        unit_id: region
        for region in fallback_layout_plan(page_data, image_bboxes).regions
        for unit_id in region.unit_ids
    }
    known_region_ids = {region.id for region in response.regions}
    allocated_region_ids = set(known_region_ids)
    warnings: list[str] = []
    regions: list[LayoutRegion] = []

    def generated_region_id(base: str) -> str:
        candidate = base
        suffix = 2
        while candidate in allocated_region_ids:
            candidate = f"{base}_{suffix}"
            suffix += 1
        allocated_region_ids.add(candidate)
        return candidate

    def image_region(
        source_region: VisionRegion, image_number: int, parent_id: str | None = None
    ) -> LayoutRegion:
        image = image_index[image_number]
        source_id = image.get("source_id")
        return LayoutRegion(
            id=generated_region_id(f"{source_region.id}__image_{image_number}"),
            type=(
                source_region.type
                if source_region.type in {
                    RegionType.IMAGE, RegionType.SCREENSHOT, RegionType.DIAGRAM
                }
                else RegionType.IMAGE
            ),
            bbox=image["bbox"],
            reading_order=source_region.reading_order,
            confidence=source_region.confidence,
            recommended_strategy=RenderingStrategy.PRESERVE_IMAGE,
            source_ids=[source_id] if isinstance(source_id, str) else [],
            geometry_source=GeometrySource.PDF,
            parent_id=parent_id,
            contains_embedded_text=source_region.contains_embedded_text,
        )

    for region in response.regions:
        unit_ids = [unit_id for unit_id in region.unit_ids if unit_id in unit_index]
        unknown_units = sorted(set(region.unit_ids) - set(unit_ids))
        image_indexes = [index for index in region.image_indexes if index in image_index]
        unknown_images = sorted(set(region.image_indexes) - set(image_indexes))
        child_ids = [
            child_id for child_id in region.child_ids
            if child_id in known_region_ids and child_id != region.id
        ]
        parent_id = (
            region.parent_id
            if region.parent_id in known_region_ids and region.parent_id != region.id
            else None
        )
        if unknown_units:
            warnings.append(f"Region {region.id}: ignored unknown unit IDs {unknown_units}")
        if unknown_images:
            warnings.append(f"Region {region.id}: ignored unknown image indexes {unknown_images}")
        if region.parent_id is not None and parent_id is None:
            warnings.append(f"Region {region.id}: ignored unknown parent {region.parent_id!r}")
        unknown_children = sorted(set(region.child_ids) - set(child_ids))
        if unknown_children:
            warnings.append(f"Region {region.id}: ignored unknown child regions {unknown_children}")

        if len(unit_ids) > 1 and any(
            unit_index[unit_id].get("unit_type") == "toc_entry" for unit_id in unit_ids
        ):
            warnings.append(
                f"Region {region.id}: split grouped TOC units to preserve individual anchors"
            )
            split_regions: list[LayoutRegion] = []
            for unit_id in unit_ids:
                fallback_region = fallback_regions.get(unit_id)
                if fallback_region is None:
                    continue
                split_regions.append(fallback_region.model_copy(update={
                    "id": generated_region_id(f"{region.id}__unit_{unit_id}"),
                    "reading_order": region.reading_order,
                    "confidence": region.confidence,
                    "parent_id": region.id,
                    "child_ids": [],
                }))
            split_regions.extend(
                image_region(region, image_number, region.id)
                for image_number in image_indexes
            )
            regions.extend(split_regions)
            regions.append(LayoutRegion(
                id=region.id,
                type=RegionType.TOC,
                bbox=None,
                reading_order=region.reading_order,
                confidence=region.confidence,
                recommended_strategy=RenderingStrategy.TOC_REGION,
                geometry_source=GeometrySource.PDF,
                parent_id=parent_id,
                child_ids=[
                    item.id for item in split_regions
                ] + child_ids,
            ))
            continue

        mixed_image_indexes = image_indexes if unit_ids else []
        direct_image_indexes = [] if unit_ids else image_indexes
        boxes = [
            _compact_bbox(unit_index[unit_id]["bbox"]) for unit_id in unit_ids
        ] + [list(image_index[index]["bbox"]) for index in direct_image_indexes]
        valid_boxes = [box for box in boxes if box is not None]
        source_ids = list(dict.fromkeys([
            source_id for unit_id in unit_ids
            for source_id in unit_index[unit_id].get("source_ids", [])
            if source_id in source_index
        ] + [
            image_index[index]["source_id"] for index in direct_image_indexes
            if isinstance(image_index[index].get("source_id"), str)
        ]))
        if not valid_boxes and not child_ids:
            warnings.append(f"Region {region.id}: omitted because it has no valid references")
            continue

        first_unit = unit_index[unit_ids[0]] if unit_ids else {}
        metadata = first_unit.get("metadata", {})
        regions.append(LayoutRegion(
            id=region.id,
            type=region.type,
            bbox=_union_bboxes(valid_boxes) if valid_boxes else None,
            reading_order=region.reading_order,
            confidence=region.confidence,
            recommended_strategy=region.recommended_strategy,
            unit_ids=unit_ids,
            source_ids=source_ids,
            geometry_source=GeometrySource.PDF,
            parent_id=parent_id,
            child_ids=child_ids,
            contains_embedded_text=region.contains_embedded_text,
            template_group_id=first_unit.get("template_group_id"),
            recurrence_count=int(first_unit.get("recurrence_count", 0)),
            hierarchy_level=metadata.get("toc_hierarchy_level"),
            column=metadata.get("toc_column"),
            page_number_anchor=metadata.get("toc_page_number_bbox"),
        ))
        regions.extend(
            image_region(region, image_number) for image_number in mixed_image_indexes
        )

    if not regions:
        raise ValueError("vision response contains no regions with valid references")
    covered_unit_ids = {
        unit_id for region in regions for unit_id in region.unit_ids
    }
    missing_unit_ids = sorted(set(unit_index) - covered_unit_ids)
    if missing_unit_ids:
        warnings.append(
            f"Vision omitted {len(missing_unit_ids)} extracted units; "
            "deterministic regions added"
        )
        for unit_id in missing_unit_ids:
            fallback_region = fallback_regions.get(unit_id)
            if fallback_region is None:
                continue
            regions.append(fallback_region.model_copy(update={
                "id": generated_region_id(f"fallback-unit-{unit_id}"),
                "parent_id": None,
                "child_ids": [],
            }))
    while True:
        retained_ids = {region.id for region in regions}
        for region in regions:
            region.child_ids = [
                child_id for child_id in region.child_ids if child_id in retained_ids
            ]
            if region.parent_id not in retained_ids:
                region.parent_id = None
        retained = [
            region for region in regions
            if region.bbox is not None or region.source_ids or region.unit_ids or region.child_ids
        ]
        if len(retained) == len(regions):
            break
        regions = retained
    if not regions:
        raise ValueError("vision response contains no resolvable regions")
    plan = LayoutPlan(
        page_number=int(page_data.get("page_number", response.page_number)),
        width=float(page_data.get("width", 1.0)),
        height=float(page_data.get("height", 1.0)),
        confidence=response.confidence,
        source="vision",
        regions=regions,
        warnings=warnings,
    )
    return resolve_layout_plan_geometry(plan, page_data)


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
        compact_plan = VisionLayoutResponse.model_validate(_extract_json(response))
        if compact_plan.page_number != page_data.get("page_number"):
            raise ValueError("vision LayoutPlan page number does not match extraction")
        if compact_plan.confidence < confidence_threshold:
            fallback.warnings.append(
                f"Vision LayoutPlan confidence {compact_plan.confidence:.2f} is below "
                f"{confidence_threshold:.2f}; "
                "deterministic fallback used"
            )
            return fallback
        return _layout_plan_from_vision(compact_plan, page_data, image_bboxes)
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
            "Vision request: page=%s geometry_chars=%s units=%s images=%s "
            "image_size=%sx%s requested_output_tokens=%s",
            geometry.get("page"),
            len(geometry_json),
            len(geometry.get("units", [])),
            len(geometry.get("images", [])),
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
                    "name": "compact_layout_plan",
                    "schema": VisionLayoutResponse.model_json_schema(),
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
    max_workers: int = DEFAULT_VISION_WORKERS,
    write_debug_images: bool = False,
) -> dict[int, LayoutPlan]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
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

    def save_plan(document: fitz.Document, page_number: int, plan: LayoutPlan) -> None:
        plan.source_file = pdf_path.name
        plan.source_sha256 = actual_source_sha256
        plan.source_extraction_sha256 = extraction_sha256
        plans[page_number] = plan
        stem = f"page_{page_number:03d}"
        (output_dir / f"{stem}.json").write_text(
            plan.model_dump_json(indent=2), encoding="utf-8"
        )
        if write_debug_images:
            from .render_debug import save_layout_debug

            save_layout_debug(
                document, page_number - 1, plan, output_dir / f"{stem}_debug.png"
            )

    with fitz.open(pdf_path) as document:
        if client is not None and max_workers > 1 and document.page_count > 1:
            worker_count = min(max_workers, document.page_count)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                pending: dict[Any, int] = {}

                def save_completed() -> None:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        page_number = pending.pop(future)
                        save_plan(document, page_number, future.result())

                for page in document:
                    if len(pending) >= worker_count:
                        save_completed()
                    page_number = page.number + 1
                    page_data = page_data_by_number.get(page_number, {
                        "page_number": page_number,
                        "width": page.rect.width,
                        "height": page.rect.height,
                        "units": [],
                    })
                    images = _image_bboxes(page)
                    page_png = page.get_pixmap(
                        matrix=fitz.Matrix(0.8, 0.8), alpha=False
                    ).tobytes("png")
                    future = executor.submit(
                        analyze_page_layout,
                        page_data,
                        page_png,
                        images,
                        client,
                        confidence_threshold,
                    )
                    pending[future] = page_number
                while pending:
                    save_completed()
        else:
            for page in document:
                page_number = page.number + 1
                page_data = page_data_by_number.get(page_number, {
                    "page_number": page_number,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "units": [],
                })
                images = _image_bboxes(page)
                page_png = (
                    page.get_pixmap(matrix=fitz.Matrix(0.8, 0.8), alpha=False).tobytes("png")
                    if client is not None else b""
                )
                plan = analyze_page_layout(
                    page_data, page_png, images, client, confidence_threshold
                )
                save_plan(document, page_number, plan)
    return dict(sorted(plans.items()))
