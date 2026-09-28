import pytest
from pydantic import ValidationError

from pdf_translator.layout import GeometrySource, LayoutPlan, resolve_layout_plan_geometry


def valid_plan():
    return {
        "page_number": 1,
        "width": 200,
        "height": 100,
        "confidence": 0.9,
        "source": "vision",
        "regions": [{
            "id": "r1",
            "type": "paragraph",
            "bbox": [10, 10, 100, 40],
            "reading_order": 0,
            "confidence": 0.94,
            "recommended_strategy": "reflow_region",
            "unit_ids": [1],
        }],
    }


def test_layout_plan_schema_validation():
    plan = LayoutPlan.model_validate(valid_plan())
    assert plan.regions[0].recommended_strategy.value == "reflow_region"


@pytest.mark.parametrize("bbox", ([10, 10, 10, 20], [-1, 0, 10, 10], [0, 0, 250, 10]))
def test_layout_plan_rejects_invalid_bbox(bbox):
    payload = valid_plan()
    payload["regions"][0]["bbox"] = bbox
    with pytest.raises(ValidationError, match="bbox"):
        LayoutPlan.model_validate(payload)


def test_layout_plan_rejects_invalid_confidence():
    payload = valid_plan()
    payload["regions"][0]["confidence"] = 1.1
    with pytest.raises(ValidationError, match="confidence"):
        LayoutPlan.model_validate(payload)


def test_layout_plan_rejects_duplicate_region_id():
    payload = valid_plan()
    payload["regions"].append({**payload["regions"][0]})
    with pytest.raises(ValidationError, match="unique"):
        LayoutPlan.model_validate(payload)


def test_layout_plan_rejects_executable_model_fields():
    payload = valid_plan()
    payload["regions"][0]["code"] = "page.insert_textbox(...)"
    with pytest.raises(ValidationError, match="Extra inputs"):
        LayoutPlan.model_validate(payload)


def test_pdf_source_ids_override_invalid_model_bbox_and_union_geometry():
    payload = valid_plan()
    payload["regions"][0].update({
        "bbox": [150, 80, 10, 5],
        "source_ids": ["p0001/b0001/l0001/s0001", "p0001/b0002/l0001/s0001"],
        "geometry_source": "pdf",
        "unit_ids": [],
    })
    page_data = {
        "page_number": 1, "width": 200, "height": 100,
        "units": [{"id": 1, "bbox": [10, 10, 100, 40], "source_ids": []}],
        "source_objects": [
            {"id": "p0001/b0001/l0001/s0001", "kind": "span", "bbox": [10, 10, 40, 20], "unit_ids": [1]},
            {"id": "p0001/b0002/l0001/s0001", "kind": "span", "bbox": [60, 25, 100, 40], "unit_ids": [1]},
        ],
    }

    resolved = resolve_layout_plan_geometry(LayoutPlan.model_validate(payload), page_data)

    assert resolved.regions[0].bbox == (10, 10, 100, 40)
    assert resolved.regions[0].model_bbox == (150, 80, 10, 5)
    assert resolved.regions[0].geometry_source == GeometrySource.PDF
    assert resolved.regions[0].unit_ids == [1]


def test_unknown_source_id_is_rejected_during_resolution():
    payload = valid_plan()
    payload["regions"][0].update({
        "bbox": None, "source_ids": ["p0001/missing"], "unit_ids": [],
        "geometry_source": "pdf",
    })
    with pytest.raises(ValueError, match="unknown source IDs"):
        resolve_layout_plan_geometry(LayoutPlan.model_validate(payload), {
            "page_number": 1, "width": 200, "height": 100,
            "units": [], "source_objects": [],
        })
