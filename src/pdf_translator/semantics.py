from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

import fitz

from .filters import should_translate
from .models import ExtractionResult, PageExtraction, SourceObject, TocEntry, TranslationUnit


_PAGE_LABEL = re.compile(r"^(?:\d+|[ivxlcdm]+)$", re.IGNORECASE)
_LEADER = re.compile(r"^[.·•…\s]+$")
_TOC_SUFFIX = re.compile(
    r"(?P<leader>[.·•…][.·•…\s]{2,})(?P<page>\d+|[ivxlcdm]+)\s*$",
    re.IGNORECASE,
)


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


def _toc_rows(spans: list[SourceObject]) -> list[list[SourceObject]]:
    rows: list[list[SourceObject]] = []
    for span in spans:
        if rows and span.parent_id == rows[-1][0].parent_id:
            rows[-1].append(span)
        else:
            rows.append([span])
    for row in rows:
        row.sort(key=lambda source: source.bbox[0])
    return rows


def _fragment_bbox(
    source: SourceObject, start: int, end: int
) -> tuple[float, float, float, float]:
    text = source.text
    fontsize = max(float(source.metadata.get("fontsize", 8.0)), 1.0)
    full_width = fitz.get_text_length(text, fontname="helv", fontsize=fontsize)
    if full_width <= 0:
        start_ratio = start / max(len(text), 1)
        end_ratio = end / max(len(text), 1)
    else:
        start_ratio = fitz.get_text_length(
            text[:start], fontname="helv", fontsize=fontsize
        ) / full_width
        end_ratio = fitz.get_text_length(
            text[:end], fontname="helv", fontsize=fontsize
        ) / full_width
    width = source.bbox[2] - source.bbox[0]
    return (
        source.bbox[0] + width * start_ratio,
        source.bbox[1],
        source.bbox[0] + width * end_ratio,
        source.bbox[3],
    )


def _merge_inline_continuations(page: PageExtraction) -> None:
    """Keep punctuation-linked inline references in their parent translation unit."""
    index = _source_index(page)
    pending = list(page.units)
    merged: list[TranslationUnit] = []
    position = 0
    while position < len(pending):
        unit = pending[position]
        depth = unit.source.count("(") - unit.source.count(")")
        if depth <= 0 or position + 1 >= len(pending):
            merged.append(unit)
            position += 1
            continue
        following = pending[position + 1]
        following_spans = _span_objects(following, index)
        if not following_spans:
            merged.append(unit)
            position += 1
            continue
        first = following_spans[0]
        line_height = max(first.bbox[3] - first.bbox[1], 1.0)
        geometrically_adjacent = first.bbox[1] <= unit.bbox[3] + line_height * 1.5
        if not geometrically_adjacent:
            merged.append(unit)
            position += 1
            continue
        consumed: list[SourceObject] = []
        for span in following_spans:
            consumed.append(span)
            depth += span.text.count("(") - span.text.count(")")
            if depth <= 0:
                break
        if depth > 0:
            merged.append(unit)
            position += 1
            continue

        consumed_text = " ".join(span.text.strip() for span in consumed).strip()
        separator = "" if unit.source.rstrip().endswith("(") else " "
        merged.append(replace(
            unit,
            source=f"{unit.source.rstrip()}{separator}{consumed_text}",
            bbox=_union([
                SourceObject("unit", "unit", unit.bbox),
                *consumed,
            ]),
            line_count=unit.line_count + len(consumed),
            translate=should_translate(f"{unit.source.rstrip()}{separator}{consumed_text}"),
            source_ids=[*unit.source_ids, *(span.id for span in consumed)],
            metadata={
                **unit.metadata,
                "inline_child_source_ids": [span.id for span in consumed],
            },
        ))
        consumed_ids = {span.id for span in consumed}
        residual = [span for span in following_spans if span.id not in consumed_ids]
        if residual:
            representative = residual[0]
            pending[position + 1] = replace(
                following,
                source="\n".join(span.text.strip() for span in residual),
                bbox=_union(residual),
                fontsize=float(representative.metadata.get("fontsize", following.fontsize)),
                fontname=str(representative.metadata.get("fontname", following.fontname)),
                flags=int(representative.metadata.get("flags", following.flags)),
                line_count=len(residual),
                translate=should_translate("\n".join(span.text.strip() for span in residual)),
                source_ids=[span.id for span in residual],
            )
            position += 1
        else:
            position += 2
    page.units = merged


def _toc_fragments(
    row: list[SourceObject],
    title_end: int,
    leader_start: int,
    leader_end: int,
    page_start: int,
    page_end: int,
) -> tuple[list[SourceObject], list[SourceObject], list[SourceObject]]:
    ranges = (
        ("toc-title", 0, title_end),
        ("toc-leader", leader_start, leader_end),
        ("toc-page", page_start, page_end),
    )
    result: dict[str, list[SourceObject]] = {role: [] for role, _, _ in ranges}
    offset = 0
    for source in row:
        source_start = offset
        source_end = offset + len(source.text)
        split = False
        for role, range_start, range_end in ranges:
            start = max(source_start, range_start)
            end = min(source_end, range_end)
            if end <= start:
                continue
            local_start = start - source_start
            local_end = end - source_start
            text = source.text[local_start:local_end].strip()
            if not text:
                continue
            if local_start == 0 and local_end == len(source.text) and role == "toc-title":
                result[role].append(source)
                continue
            split = True
            result[role].append(SourceObject(
                id=f"{source.id}/{role}",
                kind="span",
                bbox=_fragment_bbox(source, local_start, local_end),
                text=text,
                parent_id=source.id,
                metadata={
                    **source.metadata,
                    "source_role": role,
                    "character_range": [local_start, local_end],
                },
            ))
        if split:
            source.kind = "toc_source"
        offset = source_end
    return result["toc-title"], result["toc-leader"], result["toc-page"]


def _detect_toc(page: PageExtraction) -> None:
    index = _source_index(page)
    raw_candidates: list[dict[str, object]] = []
    for unit in page.units:
        pending: list[SourceObject] | None = None
        for row in _toc_rows(_span_objects(unit, index)):
            combined = "".join(source.text for source in row)
            suffix = _TOC_SUFFIX.search(combined)
            if suffix is None or not combined[:suffix.start()].strip():
                pending = row
                continue
            continuation: list[SourceObject] = []
            if pending:
                gap = row[0].bbox[1] - pending[-1].bbox[3]
                line_height = max(source.bbox[3] - source.bbox[1] for source in row)
                same_indent = abs(pending[0].bbox[0] - row[0].bbox[0]) <= page.width * 0.03
                same_style = int(pending[0].metadata.get("flags", 0)) == int(
                    row[0].metadata.get("flags", 0)
                )
                if -1.0 <= gap <= line_height * 1.2 and same_indent and same_style:
                    continuation = pending
            raw_candidates.append({
                "unit": unit,
                "row": row,
                "continuation": continuation,
                "suffix": suffix,
                "right_anchor": row[-1].bbox[2],
            })
            pending = None
    if len(raw_candidates) < 3:
        return

    right_anchors = sorted(_cluster(
        [float(candidate["right_anchor"]) for candidate in raw_candidates],
        tolerance=max(8.0, page.width * 0.025),
    ))
    if len(right_anchors) > 3:
        return

    derived_sources: list[SourceObject] = []
    prepared: list[dict[str, object]] = []
    for candidate in raw_candidates:
        row = candidate["row"]
        continuation = candidate["continuation"]
        suffix = candidate["suffix"]
        assert isinstance(row, list) and isinstance(continuation, list)
        assert isinstance(suffix, re.Match)
        titles, leaders, page_numbers = _toc_fragments(
            row,
            suffix.start(),
            suffix.start("leader"),
            suffix.end("leader"),
            suffix.start("page"),
            suffix.end("page"),
        )
        titles = [*continuation, *titles]
        if not titles or not leaders or len(page_numbers) != 1:
            continue
        derived_sources.extend(
            source
            for source in [*titles, *leaders, *page_numbers]
            if source.id not in index
        )
        title_bbox = _union(titles)
        page_number = page_numbers[0]
        column = min(
            range(len(right_anchors)),
            key=lambda value: abs(right_anchors[value] - page_number.bbox[2]),
        )
        prepared.append({
            **candidate,
            "titles": titles,
            "leaders": leaders,
            "page_number": page_number,
            "title_bbox": title_bbox,
            "column": column,
        })
    if len(prepared) < 3:
        return

    title_anchors_by_column = {
        column: sorted(_cluster([
            candidate["title_bbox"][0]
            for candidate in prepared
            if candidate["column"] == column
        ], tolerance=max(4.0, page.width * 0.01)))
        for column in {int(candidate["column"]) for candidate in prepared}
    }
    replacements: dict[int, list[TranslationUnit]] = defaultdict(list)
    consumed_ids: dict[int, set[str]] = defaultdict(set)
    page.toc_entries = []
    for entry_index, candidate in enumerate(prepared, start=1):
        unit = candidate["unit"]
        titles = candidate["titles"]
        leaders = candidate["leaders"]
        page_number = candidate["page_number"]
        title_bbox = candidate["title_bbox"]
        column = int(candidate["column"])
        assert isinstance(unit, TranslationUnit)
        assert isinstance(titles, list) and isinstance(leaders, list)
        assert isinstance(page_number, SourceObject) and isinstance(title_bbox, tuple)
        title_x = title_bbox[0]
        title_anchors = title_anchors_by_column[column]
        hierarchy = min(
            range(len(title_anchors)),
            key=lambda value: abs(title_anchors[value] - title_x),
        )
        entry_id = f"p{page.page_number:04d}/toc{entry_index:04d}"
        title = " ".join(source.text.strip() for source in titles).strip()
        representative = titles[0]
        replacement = TranslationUnit(
            id=unit.id,
            page_number=unit.page_number,
            unit_type="toc_entry",
            source=title,
            bbox=title_bbox,
            fontsize=float(representative.metadata.get("fontsize", unit.fontsize)),
            fontname=str(representative.metadata.get("fontname", unit.fontname)),
            flags=int(representative.metadata.get("flags", unit.flags)),
            color=unit.color,
            direction=unit.direction,
            line_count=len(titles),
            translate=should_translate(title),
            source_ids=[source.id for source in titles],
            semantic_role="toc_entry",
            metadata={
                "toc_entry_id": entry_id,
                "toc_page_label": page_number.text.strip(),
                "toc_page_number_bbox": list(page_number.bbox),
                "toc_page_number_source_ids": [page_number.id],
                "toc_leader_source_ids": [source.id for source in leaders],
                "toc_hierarchy_level": hierarchy,
                "toc_indentation": title_x - title_anchors[0],
                "toc_column": column,
            },
        )
        replacements[unit.id].append(replacement)
        consumed_ids[unit.id].update(
            source.id
            for source in [*candidate["row"], *candidate["continuation"]]
        )
        page.toc_entries.append(TocEntry(
            id=entry_id,
            unit_id=unit.id,
            title_source_ids=replacement.source_ids,
            page_number_source_ids=[page_number.id],
            leader_source_ids=[source.id for source in leaders],
            title_bbox=title_bbox,
            page_number_bbox=page_number.bbox,
            page_label=page_number.text.strip(),
            hierarchy_level=hierarchy,
            indentation=title_x - title_anchors[0],
            column=column,
        ))

    page.source_objects.extend(derived_sources)
    replaced: list[TranslationUnit] = []
    for unit in page.units:
        if unit.id not in replacements:
            replaced.append(unit)
            continue
        replaced.extend(replacements[unit.id])
        residual = [
            index[source_id] for source_id in unit.source_ids
            if source_id in index
            and index[source_id].kind == "span"
            and source_id not in consumed_ids[unit.id]
        ]
        for source in residual:
            replaced.append(TranslationUnit(
                id=unit.id,
                page_number=unit.page_number,
                unit_type="text",
                source=source.text.strip(),
                bbox=source.bbox,
                fontsize=float(source.metadata.get("fontsize", unit.fontsize)),
                fontname=str(source.metadata.get("fontname", unit.fontname)),
                flags=int(source.metadata.get("flags", unit.flags)),
                color=unit.color,
                direction=unit.direction,
                line_count=1,
                translate=should_translate(source.text),
                source_ids=[source.id],
            ))
    page.units = replaced


def _detect_borderless_structured(page: PageExtraction) -> None:
    index = _source_index(page)
    toc_ids = {entry.unit_id for entry in page.toc_entries}
    rows: list[tuple[TranslationUnit, list[SourceObject], list[SourceObject]]] = []
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
            rows.append((unit, spans, separated))
    if len(rows) < 2:
        return

    anchors = _cluster(
        [span.bbox[0] for _, _, separated in rows for span in separated],
        tolerance=max(4.0, page.width * 0.01),
    )
    repeated = [
        anchor for anchor in anchors
        if sum(
            any(
                abs(span.bbox[0] - anchor) <= max(4.0, page.width * 0.01)
                for span in separated
            )
            for _, _, separated in rows
        ) >= 2
    ]
    if len(repeated) < 2:
        return
    structured_rows: dict[int, tuple[list[SourceObject], list[float]]] = {}
    for unit, spans, separated in rows:
        aligned = sum(
            any(abs(span.bbox[0] - anchor) <= max(4.0, page.width * 0.01) for span in separated)
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
        for span in spans:
            metadata = span.metadata
            column = min(
                range(len(repeated_anchors)),
                key=lambda index_: abs(span.bbox[0] - repeated_anchors[index_]),
            )
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
        _merge_inline_continuations(page)
        _detect_toc(page)
        _detect_borderless_structured(page)
    _detect_running_templates(result)
    next_id = 1
    for page in result.pages:
        for unit in page.units:
            unit.id = next_id
            next_id += 1
        accounted_source_ids = {
            source_id for unit in page.units for source_id in unit.source_ids
        }
        accounted_source_ids.update(
            source_id
            for entry in page.toc_entries
            for source_id in (
                entry.title_source_ids
                + entry.page_number_source_ids
                + entry.leader_source_ids
            )
        )
        unaccounted = [
            source.id for source in page.source_objects
            if source.kind == "span"
            and source.text.strip()
            and source.id not in accounted_source_ids
        ]
        if unaccounted:
            raise RuntimeError(
                f"Page {page.page_number}: semantic analysis discarded source spans: {unaccounted}"
            )
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
