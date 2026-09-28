import json

from pdf_translator.layout_analysis import analyze_page_layout


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
        "width": 200,
        "height": 100,
        "confidence": confidence,
        "source": "vision",
        "regions": [{
            "id": "r1", "type": "paragraph", "bbox": [10, 10, 100, 30],
            "reading_order": 0, "confidence": confidence,
            "recommended_strategy": "reflow_region", "unit_ids": [1],
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
