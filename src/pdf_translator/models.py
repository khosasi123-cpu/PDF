from dataclasses import asdict, dataclass
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageExtraction:
    page_number: int
    width: float
    height: float
    rotation: int
    units: list[TranslationUnit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "units": [unit.to_dict() for unit in self.units],
        }


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
            "source_file": self.source_file,
            "pymupdf_version": self.pymupdf_version,
            "text_extraction_options": {"sort": True},
            "detected_tables": self.detected_tables,
            "ambiguous_table_assignments": self.ambiguous_table_assignments,
            "pages": [page.to_dict() for page in self.pages],
        }
