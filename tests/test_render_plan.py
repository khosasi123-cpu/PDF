from pdf_translator.layout import LayoutPlan, LayoutRegion, RegionType, RenderingStrategy
from pdf_translator.render_plan import PageGeometry, RenderPlanner, safe_expansion_candidates


def region(
    region_type="paragraph", strategy="reflow_region", bbox=(10, 10, 80, 30),
    unit_ids=None, confidence=0.95, region_id="r1",
):
    return LayoutRegion(
        id=region_id,
        type=region_type,
        bbox=bbox,
        reading_order=0,
        confidence=confidence,
        recommended_strategy=strategy,
        unit_ids=unit_ids or [],
    )


def plan(*regions):
    return LayoutPlan(
        page_number=1, width=200, height=100, confidence=0.9,
        source="vision", regions=list(regions),
    )


def unit(unit_id=1, unit_type="text", bbox=(10, 10, 80, 30), line_count=1):
    return {
        "id": unit_id, "unit_type": unit_type, "source": "Source text",
        "bbox": list(bbox), "fontsize": 10, "flags": 0, "line_count": line_count,
    }


def translation(unit_id=1, text="Teks terjemahan"):
    return {unit_id: {"id": unit_id, "translation": text}}


def geometry(*, obstacles=(), spans=None, fit_checker=None):
    return PageGeometry(
        page_bbox=(0, 0, 200, 100), text_obstacles=tuple(obstacles),
        source_spans=spans or {}, fit_checker=fit_checker or (lambda *_: True),
    )


def strategy(render_plan, unit_id=1):
    return render_plan.instruction_for_unit(unit_id).strategy


def test_paragraph_selects_reflow():
    result = RenderPlanner().plan(
        plan(region(unit_ids=[1])), {"units": [unit()]}, translation(), geometry()
    )
    assert strategy(result) == RenderingStrategy.REFLOW_REGION


def test_structured_selects_structured_strategy():
    result = RenderPlanner().plan(
        plan(region("table", "reflow_region", unit_ids=[1])),
        {"units": [unit(unit_type="table_cell")]}, translation(), geometry(),
    )
    assert strategy(result) == RenderingStrategy.STRUCTURED_REGION


def test_multicolumn_leaf_selects_multicolumn_strategy():
    result = RenderPlanner().plan(
        plan(region("multi-column", "preserve_region", unit_ids=[1])),
        {"units": [unit()]}, translation(), geometry(),
    )
    assert strategy(result) == RenderingStrategy.MULTICOLUMN_REGION


def test_image_region_is_preserved_and_embedded_text_is_reported():
    image = region("image", "reflow_region", unit_ids=[], region_id="image")
    image.contains_embedded_text = True
    result = RenderPlanner().plan(plan(image), {"units": []}, {}, geometry())
    assert result.regions[0].strategy == RenderingStrategy.PRESERVE_IMAGE
    assert "not translated" in result.warnings[0]


def test_matching_source_spans_select_source_span_mapping():
    result = RenderPlanner().plan(
        plan(region(unit_ids=[1], bbox=(10, 10, 100, 40))),
        {"units": [unit(bbox=(10, 10, 100, 40), line_count=2)]},
        translation(text="Baris satu\nBaris dua"),
        geometry(spans={1: ((10, 10, 40, 20), (10, 25, 45, 35))}),
    )
    assert strategy(result) == RenderingStrategy.SOURCE_SPAN_MAPPING


def test_overflow_selects_safe_expansion():
    def fits(_unit, bbox, _text, _size):
        return bbox[2] - bbox[0] >= 120

    result = RenderPlanner().plan(
        plan(region(unit_ids=[1])), {"units": [unit()]}, translation(),
        geometry(obstacles=((10, 10, 80, 30),), fit_checker=fits),
    )
    instruction = result.instruction_for_unit(1)
    assert instruction.strategy == RenderingStrategy.EXPAND_REGION
    assert instruction.bbox[2] == 200


def test_unsafe_expansion_uses_original_bbox_fallback():
    obstacles = ((10, 10, 80, 30), (81, 10, 150, 30), (10, 31, 80, 90))
    result = RenderPlanner().plan(
        plan(region(unit_ids=[1])), {"units": [unit()]}, translation(),
        geometry(obstacles=obstacles, fit_checker=lambda *_: False),
    )
    instruction = result.instruction_for_unit(1)
    assert instruction.strategy == RenderingStrategy.FALLBACK_ORIGINAL_BBOX
    assert instruction.bbox == instruction.original_bbox


def test_collision_detection_stops_before_neighbor():
    candidates = safe_expansion_candidates(
        (10, 10, 40, 20), (0, 0, 200, 100), [(60, 10, 100, 20), (10, 40, 40, 60)]
    )
    assert (10, 10, 59.25, 20) in candidates
    assert (10, 10, 40, 39.25) in candidates
    assert all(candidate[2] <= 60 or candidate[3] <= 40 for candidate in candidates)


def test_multiple_strategies_coexist_on_one_page():
    regions = (
        region("heading", "preserve_region", (10, 5, 190, 15), [1], region_id="heading"),
        region("paragraph", "reflow_region", (10, 20, 90, 60), [2], region_id="body"),
        region("image", "preserve_image", (110, 20, 190, 70), [], region_id="image"),
        region("table", "structured_region", (10, 75, 190, 95), [3], region_id="table"),
    )
    units = [
        unit(1, bbox=(10, 5, 190, 15)),
        unit(2, bbox=(10, 20, 90, 60)),
        unit(3, "table_cell", (10, 75, 190, 95)),
    ]
    translations = {item["id"]: {"translation": "Terjemahan"} for item in units}
    result = RenderPlanner().plan(plan(*regions), {"units": units}, translations, geometry())
    assert {instruction.strategy for instruction in result.regions} == {
        RenderingStrategy.PRESERVE_REGION,
        RenderingStrategy.REFLOW_REGION,
        RenderingStrategy.PRESERVE_IMAGE,
        RenderingStrategy.STRUCTURED_REGION,
    }


def test_unsafe_model_expansion_recommendation_is_not_a_command():
    recommended = region("paragraph", "expand_region", unit_ids=[1])
    result = RenderPlanner().plan(
        plan(recommended), {"units": [unit()]}, translation(), geometry()
    )
    assert strategy(result) == RenderingStrategy.REFLOW_REGION


def test_running_header_never_expands_outside_source_band():
    fixed = region("running_header", "reflow_region", unit_ids=[1])
    fixed.template_group_id = "running_header-0001"
    fixed.recurrence_count = 5
    result = RenderPlanner().plan(
        plan(fixed), {"units": [unit()]}, translation(),
        geometry(fit_checker=lambda *_: False),
    )
    instruction = result.instruction_for_unit(1)
    assert instruction.strategy == RenderingStrategy.FALLBACK_ORIGINAL_BBOX
    assert instruction.bbox == instruction.original_bbox
    assert instruction.template_group_id == "running_header-0001"


def test_toc_title_has_fixed_page_number_anchor_and_no_expansion():
    toc = region("toc_entry", "reflow_region", unit_ids=[1])
    toc.page_number_anchor = (170, 10, 180, 20)
    toc.hierarchy_level = 1
    toc.column = 0
    toc_unit = unit()
    toc_unit["unit_type"] = "toc_entry"
    result = RenderPlanner().plan(
        plan(toc), {"units": [toc_unit]},
        translation(text="Judul terjemahan yang sangat panjang"),
        geometry(fit_checker=lambda *_: False),
    )
    instruction = result.instruction_for_unit(1)
    assert instruction.strategy == RenderingStrategy.TOC_REGION
    assert instruction.page_number_anchor == (170, 10, 180, 20)
    assert instruction.allow_expand is False


def test_instruction_owns_unit_sources_not_grouped_semantic_sources():
    grouped = region(unit_ids=[1], region_id="semantic-group")
    grouped.source_ids = ["p0001/b0001", "p0001/b0002"]
    source_unit = unit()
    source_unit["source_ids"] = ["p0001/b0001"]

    result = RenderPlanner().plan(
        plan(grouped), {"units": [source_unit]}, translation(), geometry()
    )
    instruction = result.instruction_for_unit(1)

    assert instruction.source_ids == ["p0001/b0001"]
    assert instruction.semantic_source_ids == ["p0001/b0001", "p0001/b0002"]
