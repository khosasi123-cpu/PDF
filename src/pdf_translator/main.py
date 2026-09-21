import argparse
import logging
from pathlib import Path
import sys

from .extractor import extract_pdf, save_extraction
from .preview import create_preview


def _print_summary(pdf_path: Path, result) -> None:
    units = result.units
    translatable = sum(unit.translate for unit in units)
    skipped = len(units) - translatable
    normal = sum(unit.unit_type == "text" for unit in units)
    table_cells = sum(unit.unit_type == "table_cell" for unit in units)
    empty_pages = [str(page.page_number) for page in result.pages if not page.units]
    print("Extraction summary")
    print("------------------")
    print(f"PDF: {pdf_path}")
    print(f"Pages: {len(result.pages)}")
    print(f"Total text units: {len(units)}")
    print(f"Normal text units: {normal}")
    print(f"Table cell units: {table_cells}")
    print(f"Translatable units: {translatable}")
    print(f"Skipped units: {skipped}")
    print(f"Detected tables: {result.detected_tables}")
    print(f"Ambiguous table assignments: {result.ambiguous_table_assignments}")
    if empty_pages:
        print(f"Warning: pages without text detected: {', '.join(empty_pages)}")
        for page_number in empty_pages:
            print(f"Page {page_number} contains no extractable text. OCR is not implemented.")


def _print_units(result) -> None:
    for unit in result.units[:20]:
        print(f"\nUNIT {unit.id}\n------")
        print(f"Page: {unit.page_number}")
        print(f"Type: {unit.unit_type}")
        print(f"Translate: {str(unit.translate).lower()}")
        print(f"BBox: {unit.bbox}")
        print(f"Font: {unit.fontname}")
        print(f"Size: {unit.fontsize}")
        print(f"Lines: {unit.line_count}")
        print("Source:")
        print(unit.source)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Extract block-level text units from a PDF.")
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument("--debug-assignments", action="store_true", help="Log text-to-table-cell assignments")
    args = parser.parse_args()
    pdf_path = args.pdf.resolve()
    if args.debug_assignments:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    result = extract_pdf(pdf_path, debug_assignments=args.debug_assignments)
    extraction_path = Path("artifacts/extraction/extraction.json")
    preview_path = Path("artifacts/extraction/preview.pdf")
    save_extraction(result, extraction_path)
    create_preview(pdf_path, preview_path, result)
    _print_summary(pdf_path, result)
    _print_units(result)
    print(f"\nExtraction JSON: {extraction_path}")
    print(f"Preview PDF: {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
