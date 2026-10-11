import json
from pathlib import Path
import threading
import time

import fitz

from pdf_translator.layout_analysis import (
    analyze_document_layout,
    analyze_page_layout,
    compact_geometry,
    VisionLayoutResponse,
)


PAGE = {
    "page_number": 1,
    "width": 200,
    "height": 100,
    "units": [{
        "id": 1, "unit_type": "text", "source": "Source text",
        "bbox": [10, 10, 100, 30], "line_count": 1,
    }],
}


class Client:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def analyze(self, page_png, geometry):
        if self.error:
            raise self.error
        return self.response


def response(confidence=0.9):
    return json.dumps({
        "page_number": 1,
        "confidence": confidence,
        "regions": [{
            "id": "r1", "type": "paragraph",
            "reading_order": 0, "confidence": confidence,
            "recommended_strategy": "reflow_region", "unit_ids": [1],
            "image_indexes": [], "parent_id": None, "child_ids": [],
            "contains_embedded_text": False,
        }],
    })


def test_vision_unavailable_uses_deterministic_fallback():
    plan = analyze_page_layout(PAGE, b"png", [], None)
    assert plan.source == "deterministic_fallback"
    assert "unavailable" in plan.warnings[0]


def test_malformed_model_response_uses_deterministic_fallback():
    plan = analyze_page_layout(PAGE, b"png", [], Client("not json"))
    assert plan.source == "deterministic_fallback"
    assert "failed" in plan.warnings[0]


def test_invalid_model_schema_uses_deterministic_fallback():
    plan = analyze_page_layout(PAGE, b"png", [], Client('{"regions": "wrong"}'))
    assert plan.source == "deterministic_fallback"


def test_low_confidence_model_response_uses_deterministic_fallback():
    plan = analyze_page_layout(PAGE, b"png", [], Client(response(0.2)))
    assert plan.source == "deterministic_fallback"
    assert "below" in plan.warnings[0]


def test_valid_model_response_is_used():
    plan = analyze_page_layout(PAGE, b"png", [], Client(response()))
    assert plan.source == "vision"
    assert plan.regions[0].id == "r1"
    assert plan.regions[0].bbox == (10.0, 10.0, 100.0, 30.0)


def test_compact_geometry_omits_source_objects_and_rounds_coordinates():
    page = {
        **PAGE,
        "units": [{
            **PAGE["units"][0],
            "bbox": [10.123, 10.456, 100.789, 30.987],
            "source_ids": ["p0001/b0001"],
        }],
        "source_objects": [{
            "id": "p0001/b0001", "kind": "block", "bbox": [10, 10, 100, 30],
        }],
    }

    geometry = compact_geometry(page, [])

    assert geometry["page"] == 1
    assert geometry["units"][0]["box"] == [10.1, 10.5, 100.8, 31.0]
    assert "source_ids" not in geometry["units"][0]
    assert "source_objects" not in geometry


def test_unknown_model_references_are_ignored_without_discarding_page():
    payload = json.loads(response())
    payload["regions"][0]["unit_ids"] = [1, 999]

    plan = analyze_page_layout(PAGE, b"png", [], Client(json.dumps(payload)))

    assert plan.source == "vision"
    assert plan.regions[0].unit_ids == [1]
    assert "unknown unit IDs" in plan.warnings[0]


def test_vision_omitted_units_receive_deterministic_regions():
    page = {
        **PAGE,
        "units": [
            PAGE["units"][0],
            {
                "id": 2, "unit_type": "table_cell", "source": "Second unit",
                "bbox": [110, 10, 190, 30], "line_count": 1,
            },
        ],
    }

    plan = analyze_page_layout(page, b"png", [], Client(response()))

    assert {unit_id for region in plan.regions for unit_id in region.unit_ids} == {1, 2}
    fallback = next(region for region in plan.regions if region.unit_ids == [2])
    assert fallback.recommended_strategy.value == "structured_region"
    assert "deterministic regions added" in plan.warnings[0]


def test_provider_schema_requires_every_compact_region_field():
    schema = VisionLayoutResponse.model_json_schema()
    region_schema = schema["$defs"]["VisionRegion"]
    assert set(region_schema["required"]) == set(region_schema["properties"])


def test_mixed_text_and_image_region_preserves_both_source_geometries():
    page = {
        **PAGE,
        "units": [{**PAGE["units"][0], "source_ids": ["p0001/b0001"]}],
        "source_objects": [
            {
                "id": "p0001/b0001", "kind": "block", "bbox": [10, 10, 100, 30],
                "unit_ids": [1],
            },
            {
                "id": "p0001/i0001", "kind": "image", "bbox": [120, 20, 180, 80],
                "unit_ids": [],
            },
        ],
    }
    payload = json.loads(response())
    payload["regions"][0]["image_indexes"] = [1]

    plan = analyze_page_layout(
        page, b"png", [(120, 20, 180, 80)], Client(json.dumps(payload))
    )

    text_region = next(region for region in plan.regions if region.unit_ids)
    image_region = next(region for region in plan.regions if region.type.value == "image")
    assert text_region.bbox == (10.0, 10.0, 100.0, 30.0)
    assert text_region.source_ids == ["p0001/b0001"]
    assert image_region.bbox == (120.0, 20.0, 180.0, 80.0)
    assert image_region.source_ids == ["p0001/i0001"]


def test_mixed_region_keeps_pdf_image_bbox_without_extracted_image_source():
    page = {
        **PAGE,
        "units": [{**PAGE["units"][0], "source_ids": ["p0001/b0001"]}],
        "source_objects": [{
            "id": "p0001/b0001", "kind": "block", "bbox": [10, 10, 100, 30],
            "unit_ids": [1],
        }],
    }
    payload = json.loads(response())
    payload["regions"][0]["image_indexes"] = [1]

    plan = analyze_page_layout(
        page, b"png", [(120, 20, 180, 80)], Client(json.dumps(payload))
    )

    text_region = next(region for region in plan.regions if region.unit_ids)
    image_region = next(region for region in plan.regions if region.type.value == "image")
    assert text_region.bbox == (10.0, 10.0, 100.0, 30.0)
    assert text_region.source_ids == ["p0001/b0001"]
    assert image_region.bbox == (120.0, 20.0, 180.0, 80.0)
    assert image_region.source_ids == []


def test_grouped_toc_units_are_split_to_preserve_individual_anchors():
    page = {
        "page_number": 1,
        "width": 200,
        "height": 100,
        "units": [
            {
                "id": 1, "unit_type": "toc_entry", "source": "First", "bbox": [10, 10, 100, 20],
                "metadata": {"toc_page_number_bbox": [180, 10, 190, 20]},
            },
            {
                "id": 2, "unit_type": "toc_entry", "source": "Second", "bbox": [10, 30, 100, 40],
                "metadata": {"toc_page_number_bbox": [180, 30, 190, 40]},
            },
        ],
    }
    payload = json.loads(response())
    payload["regions"][0]["type"] = "toc_entry"
    payload["regions"][0]["recommended_strategy"] = "toc_region"
    payload["regions"][0]["unit_ids"] = [1, 2]

    plan = analyze_page_layout(page, b"png", [], Client(json.dumps(payload)))

    toc_entries = [region for region in plan.regions if region.unit_ids]
    toc_group = next(region for region in plan.regions if region.type.value == "toc")
    assert len(toc_entries) == 2
    assert [region.page_number_anchor for region in toc_entries] == [
        (180.0, 10.0, 190.0, 20.0),
        (180.0, 30.0, 190.0, 40.0),
    ]
    assert toc_group.child_ids == [region.id for region in toc_entries]


class ConcurrentClient:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def analyze(self, page_png, geometry):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        page_number = geometry["page"]
        return json.dumps({
            "page_number": page_number,
            "confidence": 0.9,
            "regions": [{
                "id": f"r{page_number}",
                "type": "paragraph",
                "reading_order": 0,
                "confidence": 0.9,
                "recommended_strategy": "reflow_region",
                "unit_ids": [page_number],
                "image_indexes": [],
                "parent_id": None,
                "child_ids": [],
                "contains_embedded_text": False,
            }],
        })


def test_document_layout_runs_vision_concurrently_without_default_debug_pngs(tmp_path: Path):
    pdf_path = tmp_path / "input.pdf"
    document = fitz.open()
    for page_number in range(1, 3):
        page = document.new_page(width=200, height=100)
        page.insert_text((10, 20), f"Page {page_number}")
    document.save(pdf_path)
    document.close()
    extraction = {
        "source_file": pdf_path.name,
        "pages": [
            {
                "page_number": page_number,
                "width": 200,
                "height": 100,
                "units": [{
                    "id": page_number,
                    "unit_type": "text",
                    "source": f"Page {page_number}",
                    "bbox": [10, 10, 100, 30],
                }],
            }
            for page_number in range(1, 3)
        ],
    }
    client = ConcurrentClient()
    output_dir = tmp_path / "layout"

    plans = analyze_document_layout(
        pdf_path, extraction, output_dir, client=client, max_workers=2
    )

    assert len(plans) == 2
    assert client.max_active == 2
    assert (output_dir / "page_001.json").exists()
    assert not (output_dir / "page_001_debug.png").exists()
