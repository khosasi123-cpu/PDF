from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import fitz
from pydantic import BaseModel, ConfigDict, Field

from .layout import BBox, GeometrySource, LayoutPlan, LayoutRegion, RegionType, RenderingStrategy


MIN_FONT_SIZE = 5.5
STRUCTURED_MIN_FONT_SIZE = 4.5
GAP = 0.75
MODEL_CONFIDENCE_THRESHOLD = 0.6


class RenderInstruction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    region_id: str
    unit_id: int | None = None
    strategy: RenderingStrategy
    original_bbox: BBox
    bbox: BBox
    allow_expand: bool = False
    source_geometry_fixed: bool = False
    contains_embedded_text: bool = False
    source_ids: list[str] = Field(default_factory=list)
    semantic_source_ids: list[str] = Field(default_factory=list)
    geometry_source: GeometrySource = GeometrySource.LEGACY
    parent_region_id: str | None = None
    template_group_id: str | None = None
    recurrence_count: int = 0
    hierarchy_level: int | None = None
    column: int | None = None
    page_number_anchor: BBox | None = None
    reason: str


class RenderPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 2
    artifact_type: str = "render_plan"
    source_file: str | None = None
    source_sha256: str | None = None
    source_extraction_sha256: str | None = None
    source_translation_sha256: str | None = None
    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    layout_source: str
    regions: list[RenderInstruction]
    warnings: list[str] = Field(default_factory=list)

    def instruction_for_unit(self, unit_id: int) -> RenderInstruction | None:
        return next((item for item in self.regions if item.unit_id == unit_id), None)


FitChecker = Callable[[dict[str, Any], BBox, str, float], bool]


@dataclass(frozen=True)
class PageGeometry:
    page_bbox: BBox
    text_obstacles: tuple[BBox, ...] = ()
    image_obstacles: tuple[BBox, ...] = ()
    graphic_obstacles: tuple[BBox, ...] = ()
    source_spans: dict[int, tuple[BBox, ...]] = field(default_factory=dict)
    fit_checker: FitChecker | None = None

    @property
    def obstacles(self) -> tuple[BBox, ...]:
        return self.text_obstacles + self.image_obstacles + self.graphic_obstacles


def _bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        result = tuple(float(coordinate) for coordinate in value)
    except (TypeError, ValueError):
        return None
    if result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result  # type: ignore[return-value]


def _overlaps(a: BBox, b: BBox, gap: float = 0.0) -> bool:
    return (
        a[2] > b[0] + gap
        and a[0] < b[2] - gap
        and a[3] > b[1] + gap
        and a[1] < b[3] - gap
    )


def _contains(container: BBox, inner: BBox, tolerance: float = 1.0) -> bool:
    return (
        inner[0] >= container[0] - tolerance
        and inner[1] >= container[1] - tolerance
        and inner[2] <= container[2] + tolerance
        and inner[3] <= container[3] + tolerance
    )


def _area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _center_in(bbox: BBox, region: BBox) -> bool:
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]


def _estimate_fit(unit: dict[str, Any], bbox: BBox, text: str, fontsize: float) -> bool:
    width = max(bbox[2] - bbox[0], 1.0)
    height = max(bbox[3] - bbox[1], 1.0)
    fontname = "hebo" if int(unit.get("flags", 0)) & 16 else "helv"
    lines = 0
    for paragraph in text.splitlines() or [text]:
        words = paragraph.split() or [""]
        line_width = 0.0
        lines += 1
        for word in words:
            word_width = fitz.get_text_length(word + " ", fontname=fontname, fontsize=fontsize)
            if line_width and line_width + word_width > width:
                lines += 1
                line_width = word_width
            else:
                line_width += word_width
    return lines * fontsize * 1.2 <= height


def _fits(geometry: PageGeometry, unit: dict[str, Any], bbox: BBox, text: str, size: float) -> bool:
    if geometry.fit_checker is not None:
        return geometry.fit_checker(unit, bbox, text, size)
    return _estimate_fit(unit, bbox, text, size)


def _reliable_spans(spans: tuple[BBox, ...], unit_bbox: BBox, translated_text: str) -> bool:
    lines = translated_text.splitlines()
    if len(spans) <= 1 or len(spans) != len(lines):
        return False
    if not all(_contains(unit_bbox, span, tolerance=2.0) for span in spans):
        return False
    return all(_area(span) > 0 for span in spans)


def _expansion_limit(region: LayoutRegion, layout_plan: LayoutPlan, page_bbox: BBox) -> BBox:
    if region.parent_id is None or region.bbox is None:
        return page_bbox
    parent = next((item for item in layout_plan.regions if item.id == region.parent_id), None)
    if parent is not None and parent.type == RegionType.MULTI_COLUMN:
        if parent.bbox is None:
            return page_bbox
        midpoint = (parent.bbox[0] + parent.bbox[2]) / 2
        center = (region.bbox[0] + region.bbox[2]) / 2
        if center <= midpoint:
            return (page_bbox[0], page_bbox[1], midpoint - GAP, page_bbox[3])
        return (midpoint + GAP, page_bbox[1], page_bbox[2], page_bbox[3])
    return page_bbox


def safe_expansion_candidates(
    bbox: BBox,
    bounds: BBox,
    obstacles: Iterable[BBox],
) -> list[BBox]:
    """Return collision-free rightward/downward candidates within bounds."""
    relevant = [other for other in obstacles if other != bbox and not _contains(other, bbox)]
    right = bounds[2]
    for other in relevant:
        if other[0] >= bbox[2] and other[3] > bbox[1] and other[1] < bbox[3]:
            right = min(right, other[0] - GAP)
    downward = bounds[3]
    for other in relevant:
        if other[1] >= bbox[3] and other[2] > bbox[0] and other[0] < bbox[2]:
            downward = min(downward, other[1] - GAP)

    candidates: list[BBox] = []
    if right > bbox[2] + GAP:
        candidates.append((bbox[0], bbox[1], right, bbox[3]))
    if downward > bbox[3] + GAP:
        candidates.append((bbox[0], bbox[1], bbox[2], downward))
    return [
        candidate for candidate in candidates
        if _contains(bounds, candidate, tolerance=0.0)
        and not any(_overlaps(candidate, other) and not _overlaps(bbox, other) for other in relevant)
    ]


_IMAGE_TYPES = {RegionType.IMAGE, RegionType.SCREENSHOT, RegionType.DIAGRAM}
_STRUCTURED_TYPES = {RegionType.STRUCTURED, RegionType.TABLE, RegionType.FORM}
_FIXED_TYPES = {
    RegionType.HEADING, RegionType.WARNING, RegionType.HEADER, RegionType.FOOTER,
    RegionType.RUNNING_HEADER, RegionType.RUNNING_FOOTER, RegionType.TOC_ENTRY,
}


class RenderPlanner:
    """Select safe executable strategies; model recommendations are advisory."""

    def __init__(self, confidence_threshold: float = MODEL_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def plan(
        self,
        layout_plan: LayoutPlan,
        extraction: dict[str, Any],
        translations: dict[int, dict[str, Any]],
        page_geometry: PageGeometry,
    ) -> RenderPlan:
        units = {
            unit["id"]: unit
            for unit in extraction.get("units", [])
            if isinstance(unit.get("id"), int) and _bbox(unit.get("bbox")) is not None
        }
        leaf_regions = [
            region for region in layout_plan.regions
            if not region.child_ids and region.type not in _IMAGE_TYPES
        ]
        assigned: set[int] = set()
        instructions: list[RenderInstruction] = []
        warnings = list(layout_plan.warnings)

        for region in sorted(layout_plan.regions, key=lambda item: item.reading_order):
            if region.type in _IMAGE_TYPES:
                if region.bbox is None:
                    continue
                instructions.append(RenderInstruction(
                    id=region.id,
                    region_id=region.id,
                    strategy=RenderingStrategy.PRESERVE_IMAGE,
                    original_bbox=region.bbox,
                    bbox=region.bbox,
                    contains_embedded_text=region.contains_embedded_text,
                    source_ids=region.source_ids,
                    geometry_source=region.geometry_source,
                    parent_region_id=region.parent_id,
                    reason="image pixels are preserved; embedded text replacement is not supported",
                ))
                if region.contains_embedded_text:
                    warnings.append(f"Region {region.id}: embedded image text was preserved but not translated")
            elif region.type == RegionType.MULTI_COLUMN:
                if region.bbox is None:
                    continue
                instructions.append(RenderInstruction(
                    id=region.id,
                    region_id=region.id,
                    strategy=RenderingStrategy.MULTICOLUMN_REGION,
                    original_bbox=region.bbox,
                    bbox=region.bbox,
                    source_geometry_fixed=True,
                    source_ids=region.source_ids,
                    geometry_source=region.geometry_source,
                    parent_region_id=region.parent_id,
                    reason="column boundary constrains child reading order and expansion",
                ))

        for unit_id, unit in units.items():
            unit_bbox = _bbox(unit.get("bbox"))
            if unit_bbox is None:
                continue
            explicit = [region for region in leaf_regions if unit_id in region.unit_ids]
            containing = [region for region in leaf_regions if _center_in(unit_bbox, region.bbox)]
            candidates = explicit or containing
            region = min(candidates, key=lambda item: _area(item.bbox), default=None)
            if region is None:
                region = LayoutRegion(
                    id=f"fallback-unit-{unit_id}",
                    type=RegionType.UNKNOWN,
                    bbox=unit_bbox,
                    reading_order=len(instructions),
                    confidence=0.0,
                    recommended_strategy=RenderingStrategy.FALLBACK_ORIGINAL_BBOX,
                    unit_ids=[unit_id],
                )
                warnings.append(f"Unit {unit_id}: no LayoutPlan region; using original bbox")
            instruction = self._plan_unit(
                layout_plan, region, unit, translations.get(unit_id), page_geometry
            )
            instructions.append(instruction)
            assigned.add(unit_id)

        missing = set(units) - assigned
        if missing:
            warnings.append(f"Units without render instructions: {sorted(missing)}")
        return RenderPlan(
            source_file=extraction.get("source_file"),
            source_sha256=extraction.get("source_sha256"),
            source_extraction_sha256=extraction.get("source_extraction_sha256"),
            source_translation_sha256=extraction.get("source_translation_sha256"),
            page_number=layout_plan.page_number,
            width=layout_plan.width,
            height=layout_plan.height,
            layout_source=layout_plan.source,
            regions=instructions,
            warnings=warnings,
        )

    def _plan_unit(
        self,
        layout_plan: LayoutPlan,
        region: LayoutRegion,
        unit: dict[str, Any],
        translation_item: dict[str, Any] | None,
        geometry: PageGeometry,
    ) -> RenderInstruction:
        unit_id = int(unit["id"])
        original = _bbox(unit["bbox"])
        assert original is not None
        text = str((translation_item or {}).get("translation", unit.get("source", "")))
        fontsize = max(float(unit.get("fontsize", 10.0)), 1.0)
        minimum = STRUCTURED_MIN_FONT_SIZE if unit.get("unit_type") == "table_cell" else MIN_FONT_SIZE
        recommendation_trusted = region.confidence >= self.confidence_threshold

        if unit.get("unit_type") in {"table_cell", "borderless_structured"} or region.type in _STRUCTURED_TYPES:
            return self._instruction(
                region, unit, RenderingStrategy.STRUCTURED_REGION, original, original,
                "structured PDF geometry takes priority over the model recommendation", fixed=True,
            )

        if unit.get("unit_type") == "toc_entry" or region.type == RegionType.TOC_ENTRY:
            return self._instruction(
                region, unit, RenderingStrategy.TOC_REGION, original, original,
                "TOC title is rendered independently from its fixed page-number anchor", fixed=True,
            )

        spans = geometry.source_spans.get(unit_id, ())
        if _reliable_spans(spans, original, text):
            return self._instruction(
                region, unit, RenderingStrategy.SOURCE_SPAN_MAPPING, original, original,
                "translated lines match reliable source-span geometry", fixed=True,
            )

        if region.type == RegionType.MULTI_COLUMN:
            return self._instruction(
                region, unit, RenderingStrategy.MULTICOLUMN_REGION, original, original,
                "region is an explicit multi-column leaf", fixed=True,
            )

        base = RenderingStrategy.PRESERVE_REGION if region.type in _FIXED_TYPES else RenderingStrategy.REFLOW_REGION
        if recommendation_trusted:
            if region.recommended_strategy == RenderingStrategy.PRESERVE_REGION:
                base = RenderingStrategy.PRESERVE_REGION
            elif region.recommended_strategy == RenderingStrategy.REFLOW_REGION:
                base = RenderingStrategy.REFLOW_REGION

        if _fits(geometry, unit, original, text, fontsize) or _fits(
            geometry, unit, original, text, minimum
        ):
            reason = "translation fits the source region with deterministic font fitting"
            if not recommendation_trusted:
                reason += "; low-confidence recommendation ignored"
            return self._instruction(region, unit, base, original, original, reason, fixed=base == RenderingStrategy.PRESERVE_REGION)

        if region.type in _FIXED_TYPES:
            return self._instruction(
                region, unit, RenderingStrategy.FALLBACK_ORIGINAL_BBOX, original, original,
                "fixed header, footer, or TOC geometry cannot expand outside its source band",
                fixed=True,
            )

        if any(_overlaps(original, image) for image in geometry.image_obstacles):
            return self._instruction(
                region, unit, RenderingStrategy.FALLBACK_ORIGINAL_BBOX, original, original,
                "source text overlaps an image, so covering or expansion outside its bbox is unsafe",
                fixed=True,
            )

        bounds = _expansion_limit(region, layout_plan, geometry.page_bbox)
        candidates = sorted(
            safe_expansion_candidates(original, bounds, geometry.obstacles),
            key=_area,
        )
        for candidate in candidates:
            if _fits(geometry, unit, candidate, text, minimum):
                return self._instruction(
                    region, unit, RenderingStrategy.EXPAND_REGION, original, candidate,
                    "translation overflows and a collision-free expansion fits", allow_expand=True,
                )

        return self._instruction(
            region, unit, RenderingStrategy.FALLBACK_ORIGINAL_BBOX, original, original,
            "no collision-free expansion fits; renderer must use its safe original-bbox fallback",
            fixed=True,
        )

    @staticmethod
    def _instruction(
        region: LayoutRegion,
        unit: dict[str, Any],
        strategy: RenderingStrategy,
        original: BBox,
        bbox: BBox,
        reason: str,
        allow_expand: bool = False,
        fixed: bool = False,
    ) -> RenderInstruction:
        unit_id = int(unit["id"])
        return RenderInstruction(
            id=f"{region.id}:unit-{unit_id}",
            region_id=region.id,
            unit_id=unit_id,
            strategy=strategy,
            original_bbox=original,
            bbox=bbox,
            allow_expand=allow_expand,
            source_geometry_fixed=fixed,
            source_ids=list(unit.get("source_ids", [])),
            semantic_source_ids=list(region.source_ids),
            geometry_source=region.geometry_source,
            parent_region_id=region.parent_id,
            template_group_id=region.template_group_id,
            recurrence_count=region.recurrence_count,
            hierarchy_level=region.hierarchy_level,
            column=region.column,
            page_number_anchor=region.page_number_anchor,
            reason=reason,
        )
