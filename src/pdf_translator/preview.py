from pathlib import Path

import fitz

from .models import ExtractionResult


def create_preview(input_path: Path, output_path: Path, result: ExtractionResult) -> None:
    document = fitz.open(input_path)
    try:
        for page_data in result.pages:
            page = document[page_data.page_number - 1]
            for unit in page_data.units:
                if not unit.translate:
                    color = (0.8, 0.1, 0.1)
                elif unit.unit_type == "table_cell":
                    color = (0.1, 0.25, 0.85)
                else:
                    color = (0.1, 0.55, 0.2)
                rectangle = fitz.Rect(unit.bbox)
                width = 1.2 if unit.unit_type == "table_cell" else 0.8
                page.draw_rect(rectangle, color=color, width=width, overlay=True)
                label_point = fitz.Point(rectangle.x0, max(8, rectangle.y0 - 2))
                page.insert_text(label_point, str(unit.id), fontsize=7, color=color, overlay=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        document.save(output_path, garbage=4, deflate=True)
    finally:
        document.close()
