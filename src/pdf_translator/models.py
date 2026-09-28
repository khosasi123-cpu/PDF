from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TranslationUnit:
    id: int
    page_number: int
    unit_type: str
    source: str
    bbox: tuple[float, float, float, float]
    fontsize: float
    fontname: str
    flags: int
    color: int | None
    direction: tuple[float, float] | None
    line_count: int
    translate: bool
    source_ids: list[str] = field(default_factory=list)
    semantic_role: str | None = None
    template_group_id: str | None = None
    recurrence_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageExtraction:
    page_number: int
    width: float
    height: float
    rotation: int
    units: list[TranslationUnit]
    source_objects: list[SourceObject] = field(default_factory=list)
    toc_entries: list[TocEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "units": [unit.to_dict() for unit in self.units],
            "source_objects": [source.to_dict() for source in self.source_objects],
            "toc_entries": [entry.to_dict() for entry in self.toc_entries],
        }


@dataclass
class SourceObject:
    id: str
    kind: str
    bbox: tuple[float, float, float, float]
    text: str = ""
    parent_id: str | None = None
    unit_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TocEntry:
    id: str
    unit_id: int
    title_source_ids: list[str]
    page_number_source_ids: list[str]
    leader_source_ids: list[str]
    title_bbox: tuple[float, float, float, float]
    page_number_bbox: tuple[float, float, float, float]
    page_label: str
    hierarchy_level: int
    indentation: float
    column: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    source_file: str
    pymupdf_version: str
    pages: list[PageExtraction]
    detected_tables: int = 0
    ambiguous_table_assignments: int = 0

    @property
    def units(self) -> list[TranslationUnit]:
        return [unit for page in self.pages for unit in page.units]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "source_id_scheme": "page-object-v1",
            "source_file": self.source_file,
            "pymupdf_version": self.pymupdf_version,
            "text_extraction_options": {"sort": True},
            "detected_tables": self.detected_tables,
            "ambiguous_table_assignments": self.ambiguous_table_assignments,
            "pages": [page.to_dict() for page in self.pages],
        }
