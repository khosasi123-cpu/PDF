from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Iterable

from .filters import should_translate
from .models import ExtractionResult, PageExtraction, SourceObject, TocEntry, TranslationUnit


_PAGE_LABEL = re.compile(r"^(?:\d+|[ivxlcdm]+)$", re.IGNORECASE)
_LEADER = re.compile(r"^[.·•…\s]+$")


def _union(objects: Iterable[SourceObject]) -> tuple[float, float, float, float]:
    values = list(objects)
    return (
        min(item.bbox[0] for item in values),
        min(item.bbox[1] for item in values),
        max(item.bbox[2] for item in values),
        max(item.bbox[3] for item in values),
    )


def _cluster(values: list[float], tolerance: float) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if not groups or abs(value - sum(groups[-1]) / len(groups[-1])) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [sum(group) / len(group) for group in groups]


def _normalize_template_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = re.sub(r"\d+", "#", normalized)
    return " ".join(normalized.split())


def _source_index(page: PageExtraction) -> dict[str, SourceObject]:
    return {source.id: source for source in page.source_objects}


def _span_objects(unit: TranslationUnit, index: dict[str, SourceObject]) -> list[SourceObject]:
    return sorted(
        [index[source_id] for source_id in unit.source_ids if source_id in index and index[source_id].kind == "span"],
        key=lambda source: (source.bbox[1], source.bbox[0]),
    )


def _detect_toc(page: PageExtraction) -> None:
    index = _source_index(page)
    candidates: list[tuple[TranslationUnit, list[SourceObject], list[SourceObject], SourceObject]] = []
    for unit in page.units:
        spans = _span_objects(unit, index)
        if len(spans) < 2:
            continue
        page_number = spans[-1]
        if not _PAGE_LABEL.fullmatch(page_number.text.strip()):
            continue
        leaders = [span for span in spans[:-1] if _LEADER.fullmatch(span.text)]
        titles = [span for span in spans[:-1] if span not in leaders and span.text.strip()]
        if not titles or page_number.bbox[0] <= max(span.bbox[2] for span in titles):
            continue
        candidates.append((unit, titles, leaders, page_number))
    if len(candidates) < 3:
        return

    right_anchors = _cluster(
        [candidate[3].bbox[2] for candidate in candidates],
        tolerance=max(8.0, page.width * 0.025),
    )
    if len(right_anchors) > 3:
        return
    right_anchors.sort()
    columns = [
        min(range(len(right_anchors)), key=lambda index_: abs(right_anchors[index_] - candidate[3].bbox[2]))
        for candidate in candidates
    ]
    title_anchors_by_column = {
        column: sorted(_cluster([
            min(span.bbox[0] for span in candidate[1])
            for candidate, candidate_column in zip(candidates, columns)
            if candidate_column == column
        ], tolerance=max(4.0, page.width * 0.01)))
        for column in set(columns)
    }
    candidates.sort(key=lambda item: (item[0].bbox[1], item[0].bbox[0]))
    page.toc_entries = []
    for entry_index, (unit, titles, leaders, page_number) in enumerate(candidates, start=1):
        title_bbox = _union(titles)
        title_x = title_bbox[0]
        column = min(
            range(len(right_anchors)),
            key=lambda index_: abs(right_anchors[index_] - page_number.bbox[2]),
        )
        title_anchors = title_anchors_by_column[column]
        hierarchy = min(range(len(title_anchors)), key=lambda index_: abs(title_anchors[index_] - title_x))
        entry_id = f"p{page.page_number:04d}/toc{entry_index:04d}"
        title = " ".join(span.text.strip() for span in titles).strip()
        unit.source = title
        unit.bbox = title_bbox
        unit.source_ids = [span.id for span in titles]
        unit.unit_type = "toc_entry"
        unit.semantic_role = "toc_entry"
        unit.translate = should_translate(title)
        unit.metadata.update({
            "toc_entry_id": entry_id,
            "toc_page_label": page_number.text.strip(),
            "toc_page_number_bbox": list(page_number.bbox),
            "toc_page_number_source_ids": [page_number.id],
            "toc_leader_source_ids": [span.id for span in leaders],
            "toc_hierarchy_level": hierarchy,
            "toc_indentation": title_x - title_anchors[0],
            "toc_column": column,
        })
        page.toc_entries.append(TocEntry(
            id=entry_id,
            unit_id=unit.id,
            title_source_ids=[span.id for span in titles],
            page_number_source_ids=[page_number.id],
            leader_source_ids=[span.id for span in leaders],
            title_bbox=title_bbox,
            page_number_bbox=page_number.bbox,
            page_label=page_number.text.strip(),
            hierarchy_level=hierarchy,
            indentation=title_x - title_anchors[0],
            column=column,
        ))


def _detect_borderless_structured(page: PageExtraction) -> None:
    index = _source_index(page)
    toc_ids = {entry.unit_id for entry in page.toc_entries}
    rows: list[tuple[TranslationUnit, list[SourceObject]]] = []
    for unit in page.units:
        if unit.id in toc_ids or unit.unit_type == "table_cell":
            continue
        spans = _span_objects(unit, index)
        if len(spans) < 2:
            continue
        line_height = max(span.bbox[3] - span.bbox[1] for span in spans)
        separated = [spans[0]]
        for span in spans[1:]:
            if span.bbox[0] - separated[-1].bbox[2] >= line_height * 1.5:
                separated.append(span)
        if len(separated) >= 2:
            rows.append((unit, separated))
    if len(rows) < 2:
        return

    anchors = _cluster(
        [span.bbox[0] for _, spans in rows for span in spans],
        tolerance=max(4.0, page.width * 0.01),
    )
    repeated = [
        anchor for anchor in anchors
        if sum(any(abs(span.bbox[0] - anchor) <= max(4.0, page.width * 0.01) for span in spans) for _, spans in rows) >= 2
    ]
    if len(repeated) < 2:
        return
    structured_rows: dict[int, tuple[list[SourceObject], list[float]]] = {}
    for unit, spans in rows:
        aligned = sum(
            any(abs(span.bbox[0] - anchor) <= max(4.0, page.width * 0.01) for span in spans)
            for anchor in repeated
        )
        if aligned >= 2:
            structured_rows[unit.id] = (spans, repeated)
    if not structured_rows:
        return

    replaced: list[TranslationUnit] = []
    for unit in page.units:
        structured = structured_rows.get(unit.id)
        if structured is None:
            replaced.append(unit)
            continue
        spans, repeated_anchors = structured
        for column, span in enumerate(spans):
            metadata = span.metadata
            replaced.append(TranslationUnit(
                id=unit.id,
                page_number=unit.page_number,
                unit_type="borderless_structured",
                source=span.text.strip(),
                bbox=span.bbox,
                fontsize=float(metadata.get("fontsize", unit.fontsize)),
                fontname=str(metadata.get("fontname", unit.fontname)),
                flags=int(metadata.get("flags", unit.flags)),
                color=unit.color,
                direction=unit.direction,
                line_count=1,
                translate=should_translate(span.text),
                source_ids=[span.id],
                semantic_role="structured",
                metadata={
                    "structured_x_anchors": repeated_anchors,
                    "structured_column": column,
                    "geometry_source": "inferred_borderless_grid",
                },
            ))
    page.units = replaced


def _detect_running_templates(result: ExtractionResult) -> None:
    pages = result.pages
    if len(pages) < 3:
        return
    groups: dict[tuple[str, str, int], list[TranslationUnit]] = defaultdict(list)
    for page in pages:
        for unit in page.units:
            center_y = (unit.bbox[1] + unit.bbox[3]) / 2
            if center_y <= page.height * 0.15:
                role = "running_header"
            elif center_y >= page.height * 0.85:
                role = "running_footer"
            else:
                continue
            normalized = _normalize_template_text(unit.source)
            if not normalized:
                continue
            horizontal_bucket = round(((unit.bbox[0] + unit.bbox[2]) / 2) / page.width * 20)
            groups[(role, normalized, horizontal_bucket)].append(unit)

    minimum = max(3, (len(pages) + 1) // 2)
    accepted = [item for item in groups.items() if len({unit.page_number for unit in item[1]}) >= minimum]
    accepted.sort(key=lambda item: item[0])
    for group_index, ((role, _, _), units) in enumerate(accepted, start=1):
        group_id = f"{role}-{group_index:04d}"
        recurrence = len({unit.page_number for unit in units})
        for unit in units:
            unit.semantic_role = role
            unit.template_group_id = group_id
            unit.recurrence_count = recurrence


def analyze_document_semantics(result: ExtractionResult) -> ExtractionResult:
    for page in result.pages:
        _detect_toc(page)
        _detect_borderless_structured(page)
    _detect_running_templates(result)
    next_id = 1
    for page in result.pages:
        for unit in page.units:
            unit.id = next_id
            next_id += 1
        source_to_units: dict[str, list[int]] = defaultdict(list)
        for unit in page.units:
            for source_id in unit.source_ids:
                source_to_units[source_id].append(unit.id)
        for source in page.source_objects:
            source.unit_ids = source_to_units.get(source.id, [])
        for entry in page.toc_entries:
            matching = next(
                (unit for unit in page.units if unit.metadata.get("toc_entry_id") == entry.id),
                None,
            )
            if matching is not None:
                entry.unit_id = matching.id
    return result
