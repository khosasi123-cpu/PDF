import argparse
import logging
from pathlib import Path
import sys

import fitz

from .extractor import extract_pdf, save_extraction
from .preview import create_preview
from .render import print_summary, render_pdf
from .translate import (
    DEFAULT_DICTIONARY_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_TRANSLATION_PATH,
    TranslationError,
    run_translation,
)

DEFAULT_EXTRACTION_PATH = Path("artifacts/extraction/extraction.json")
DEFAULT_PREVIEW_PATH = Path("artifacts/extraction/preview.pdf")


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
    parser = argparse.ArgumentParser(description="Run extraction, translation, and PDF rendering.")
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument("--debug-assignments", action="store_true", help="Log text-to-table-cell assignments")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY_PATH)
    parser.add_argument("--extraction-output", type=Path, default=DEFAULT_EXTRACTION_PATH)
    parser.add_argument("--translation-output", type=Path, default=DEFAULT_TRANSLATION_PATH)
    parser.add_argument("--rendered-output", type=Path, default=None)
    args = parser.parse_args()
    pdf_path = args.pdf.resolve()
    if args.debug_assignments:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    try:
        print("PHASE 1: Extraction")
        result = extract_pdf(pdf_path, debug_assignments=args.debug_assignments)
        save_extraction(result, args.extraction_output)
        create_preview(pdf_path, DEFAULT_PREVIEW_PATH, result)
        _print_summary(pdf_path, result)
        print(f"Extraction JSON: {args.extraction_output}")
        print(f"Preview PDF: {DEFAULT_PREVIEW_PATH}")

        print("\nPHASE 2: Translation")
        run_translation(
            extraction_path=args.extraction_output,
            output_path=args.translation_output,
            dictionary_path=args.dictionary,
        )

        print("\nPHASE 3: Rendering")
        rendered_path, render_stats = render_pdf(
            pdf_path,
            extraction_path=args.extraction_output,
            translation_path=args.translation_output,
            output_path=args.rendered_output,
        )
        with fitz.open(pdf_path) as original:
            print_summary(pdf_path, rendered_path, original.page_count, render_stats)
        return 0
    except (RuntimeError, TranslationError, OSError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
