from __future__ import annotations

from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BBox = tuple[float, float, float, float]


class RegionType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    BULLET_LIST = "bullet_list"
    STRUCTURED = "structured"
    TABLE = "table"
    FORM = "form"
    IMAGE = "image"
    SCREENSHOT = "screenshot"
    DIAGRAM = "diagram"
    WARNING = "warning"
    HEADER = "header"
    FOOTER = "footer"
    RUNNING_HEADER = "running_header"
    RUNNING_FOOTER = "running_footer"
    TOC = "toc"
    TOC_ENTRY = "toc_entry"
    MULTI_COLUMN = "multi-column"
    UNKNOWN = "unknown"


class RenderingStrategy(str, Enum):
    PRESERVE_REGION = "preserve_region"
    REFLOW_REGION = "reflow_region"
    STRUCTURED_REGION = "structured_region"
    MULTICOLUMN_REGION = "multicolumn_region"
    PRESERVE_IMAGE = "preserve_image"
    SOURCE_SPAN_MAPPING = "source_span_mapping"
    EXPAND_REGION = "expand_region"
    FALLBACK_ORIGINAL_BBOX = "fallback_original_bbox"
    TOC_REGION = "toc_region"


class GeometrySource(str, Enum):
    PDF = "pdf"
    VISION_ESTIMATE = "vision_estimate"
    LEGACY = "legacy"


class RegionRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    target_id: str


class LayoutRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: RegionType
    bbox: BBox | None = None
    reading_order: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_strategy: RenderingStrategy
    unit_ids: list[int] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    geometry_source: GeometrySource = GeometrySource.VISION_ESTIMATE
    model_bbox: BBox | None = None
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    relationships: list[RegionRelationship] = Field(default_factory=list)
    contains_embedded_text: bool = False
    template_group_id: str | None = None
    recurrence_count: int = Field(default=0, ge=0)
    hierarchy_level: int | None = Field(default=None, ge=0)
    column: int | None = Field(default=None, ge=0)
    page_number_anchor: BBox | None = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, bbox: BBox | None) -> BBox | None:
        return bbox

    @field_validator("page_number_anchor")
    @classmethod
    def validate_optional_bbox(cls, bbox: BBox | None) -> BBox | None:
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("bbox must have non-negative coordinates and positive area")
        return bbox

    @field_validator("unit_ids")
    @classmethod
    def validate_unit_ids(cls, unit_ids: list[int]) -> list[int]:
        if any(unit_id < 1 for unit_id in unit_ids):
            raise ValueError("unit IDs must be positive")
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("unit IDs must be unique within a region")
        return unit_ids


class LayoutPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "vision"
    regions: list[LayoutRegion]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_regions(self) -> LayoutPlan:
        ids = [region.id for region in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("region IDs must be unique")
        known_ids = set(ids)
        for region in self.regions:
            if region.bbox is None and not region.child_ids and not region.source_ids and not region.unit_ids:
                raise ValueError(f"region {region.id!r} requires bbox or source references")
            if region.bbox is not None and not region.source_ids:
                x0, y0, x1, y1 = region.bbox
                if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
                    raise ValueError(f"region {region.id!r} bbox must have positive area")
                if x1 > self.width or y1 > self.height:
                    raise ValueError(f"region {region.id!r} bbox is outside the page")
            references = [*region.child_ids]
            if region.parent_id is not None:
                references.append(region.parent_id)
            references.extend(relationship.target_id for relationship in region.relationships)
            missing = set(references) - known_ids
            if missing:
                raise ValueError(f"region {region.id!r} references unknown regions: {sorted(missing)}")
            if region.id in references:
                raise ValueError(f"region {region.id!r} cannot reference itself")
        return self


def _valid_bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = tuple(float(coordinate) for coordinate in value)
    except (TypeError, ValueError):
        return None
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox  # type: ignore[return-value]


def _union(boxes: Iterable[BBox]) -> BBox:
    values = list(boxes)
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )


def resolve_layout_plan_geometry(
    plan: LayoutPlan, page_data: dict[str, Any]
) -> LayoutPlan:
    """Replace advisory model geometry with immutable PDF source geometry."""
    source_index = {
        source.get("id"): source for source in page_data.get("source_objects", [])
        if isinstance(source.get("id"), str) and _valid_bbox(source.get("bbox")) is not None
    }
    unit_index = {
        unit.get("id"): unit for unit in page_data.get("units", [])
        if isinstance(unit.get("id"), int) and _valid_bbox(unit.get("bbox")) is not None
    }
    resolved = plan.model_copy(deep=True)
    for region in resolved.regions:
        boxes: list[BBox] = []
        derived_unit_ids: set[int] = set()
        if region.source_ids:
            missing = [source_id for source_id in region.source_ids if source_id not in source_index]
            if missing:
                raise ValueError(f"region {region.id!r} references unknown source IDs: {missing}")
            for source_id in region.source_ids:
                source = source_index[source_id]
                bbox = _valid_bbox(source.get("bbox"))
                if bbox is not None:
                    boxes.append(bbox)
                derived_unit_ids.update(
                    unit_id for unit_id in source.get("unit_ids", []) if unit_id in unit_index
                )
        elif region.unit_ids:
            missing_units = [unit_id for unit_id in region.unit_ids if unit_id not in unit_index]
            if missing_units:
                raise ValueError(f"region {region.id!r} references unknown unit IDs: {missing_units}")
            boxes.extend(_valid_bbox(unit_index[unit_id]["bbox"]) for unit_id in region.unit_ids)
            boxes = [bbox for bbox in boxes if bbox is not None]
            region.source_ids = sorted({
                source_id for unit_id in region.unit_ids
                for source_id in unit_index[unit_id].get("source_ids", [])
                if source_id in source_index
            })
        if boxes:
            region.model_bbox = region.bbox
            region.bbox = _union(boxes)
            region.geometry_source = GeometrySource.PDF
            if derived_unit_ids:
                region.unit_ids = sorted(derived_unit_ids)

    by_id = {region.id: region for region in resolved.regions}
    for region in reversed(resolved.regions):
        child_boxes = [
            by_id[child_id].bbox for child_id in region.child_ids
            if child_id in by_id and by_id[child_id].bbox is not None
        ]
        if child_boxes and not region.source_ids and not region.unit_ids:
            region.model_bbox = region.bbox
            region.bbox = _union(child_boxes)
            region.geometry_source = GeometrySource.PDF
    return LayoutPlan.model_validate(resolved.model_dump())


def _column_unit_ids(units: list[dict[str, Any]], width: float) -> set[int]:
    midpoint = width / 2
    left = [unit for unit in units if unit["bbox"][2] <= midpoint]
    right = [unit for unit in units if unit["bbox"][0] >= midpoint]
    if not left or not right:
        return set()
    left_top = min(unit["bbox"][1] for unit in left)
    left_bottom = max(unit["bbox"][3] for unit in left)
    right_top = min(unit["bbox"][1] for unit in right)
    right_bottom = max(unit["bbox"][3] for unit in right)
    if min(left_bottom, right_bottom) <= max(left_top, right_top):
        return set()
    return {
        unit["id"] for unit in left + right
        if isinstance(unit.get("id"), int)
    }


def fallback_layout_plan(
    page_data: dict[str, Any], image_bboxes: Iterable[BBox] = ()
) -> LayoutPlan:
    """Create a conservative LayoutPlan entirely from extracted PDF geometry."""
    page_number = int(page_data.get("page_number", 1))
    width = float(page_data.get("width", 1.0))
    height = float(page_data.get("height", 1.0))
    valid_units: list[dict[str, Any]] = []
    for unit in page_data.get("units", []):
        bbox = _valid_bbox(unit.get("bbox"))
        if bbox is not None and bbox[2] <= width and bbox[3] <= height:
            valid_units.append({**unit, "bbox": bbox})

    regions: list[LayoutRegion] = []
    column_unit_ids = _column_unit_ids(valid_units, width)
    parent_id = "columns" if column_unit_ids else None
    for order, unit in enumerate(valid_units):
        structured = unit.get("unit_type") in {"table_cell", "borderless_structured"}
        role = unit.get("semantic_role")
        if role == "running_header":
            region_type = RegionType.RUNNING_HEADER
            strategy = RenderingStrategy.SOURCE_SPAN_MAPPING
        elif role == "running_footer":
            region_type = RegionType.RUNNING_FOOTER
            strategy = RenderingStrategy.SOURCE_SPAN_MAPPING
        elif unit.get("unit_type") == "toc_entry":
            region_type = RegionType.TOC_ENTRY
            strategy = RenderingStrategy.TOC_REGION
        else:
            region_type = RegionType.STRUCTURED if structured else RegionType.PARAGRAPH
            strategy = RenderingStrategy.STRUCTURED_REGION if structured else RenderingStrategy.REFLOW_REGION
        regions.append(LayoutRegion(
            id=f"unit-{unit.get('id', order + 1)}",
            type=region_type,
            bbox=unit["bbox"],
            reading_order=order,
            confidence=1.0,
            recommended_strategy=strategy,
            unit_ids=[unit["id"]] if isinstance(unit.get("id"), int) else [],
            source_ids=[
                source_id for source_id in unit.get("source_ids", [])
                if isinstance(source_id, str)
            ],
            geometry_source=GeometrySource.PDF,
            parent_id=parent_id if unit.get("id") in column_unit_ids else None,
            template_group_id=unit.get("template_group_id"),
            recurrence_count=int(unit.get("recurrence_count", 0)),
            hierarchy_level=unit.get("metadata", {}).get("toc_hierarchy_level"),
            column=unit.get("metadata", {}).get("toc_column"),
            page_number_anchor=unit.get("metadata", {}).get("toc_page_number_bbox"),
        ))

    if parent_id is not None:
        child_ids = [region.id for region in regions if region.parent_id == parent_id]
        regions.append(LayoutRegion(
            id=parent_id,
            type=RegionType.MULTI_COLUMN,
            bbox=_union(region.bbox for region in regions if region.id in child_ids),
            reading_order=0,
            confidence=1.0,
            recommended_strategy=RenderingStrategy.MULTICOLUMN_REGION,
            geometry_source=GeometrySource.PDF,
            child_ids=child_ids,
        ))

    source_images = [
        source for source in page_data.get("source_objects", [])
        if source.get("kind") == "image" and _valid_bbox(source.get("bbox")) is not None
    ]
    image_values = [(_valid_bbox(source["bbox"]), source.get("id")) for source in source_images]
    image_values.extend((bbox, None) for bbox in image_bboxes if not source_images)
    for index, (bbox, source_id) in enumerate(image_values, start=1):
        if bbox is None:
            continue
        if bbox[2] <= width and bbox[3] <= height:
            regions.append(LayoutRegion(
                id=f"image-{index}",
                type=RegionType.IMAGE,
                bbox=bbox,
                reading_order=len(regions),
                confidence=1.0,
                recommended_strategy=RenderingStrategy.PRESERVE_IMAGE,
                source_ids=[source_id] if isinstance(source_id, str) else [],
                geometry_source=GeometrySource.PDF if source_id else GeometrySource.LEGACY,
            ))

    return LayoutPlan(
        page_number=page_number,
        width=width,
        height=height,
        confidence=1.0,
        source="deterministic_fallback",
        regions=regions,
    )
