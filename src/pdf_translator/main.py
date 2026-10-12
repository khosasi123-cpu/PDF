import argparse
import logging
from pathlib import Path
import sys
import time


import fitz

from .extractor import extract_pdf, save_extraction
from .identity import sha256_file
from .layout_analysis import (
    DEFAULT_VISION_WORKERS,
    OpenAIMinistralVisionClient,
    analyze_document_layout,
)
from .preview import create_preview
from .render import print_summary, render_pdf
from .translate import (
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    TranslationError,
    language_pair_key,
    language_scoped_translation_path,
    run_translation,
    validate_language_pair,
)

DEFAULT_EXTRACTION_PATH = Path("artifacts/extraction/extraction.json")
DEFAULT_PREVIEW_PATH = Path("artifacts/extraction/preview.pdf")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


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
    parser.add_argument("--dictionary", type=Path, default=None)
    parser.add_argument("--source-language", default=DEFAULT_SOURCE_LANGUAGE)
    parser.add_argument("--target-language", default=DEFAULT_TARGET_LANGUAGE)
    parser.add_argument("--extraction-output", type=Path, default=None)
    parser.add_argument("--translation-output", type=Path, default=None)
    parser.add_argument("--rendered-output", type=Path, default=None)
    parser.add_argument("--layout-output-dir", type=Path, default=None)
    parser.add_argument(
        "--vision-workers",
        type=_positive_int,
        default=DEFAULT_VISION_WORKERS,
        help="Maximum concurrent vision page requests (default: 2)",
    )
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help="Write diagnostic PNGs in addition to JSON artifacts",
    )
    args = parser.parse_args()
    try:
        source_language, target_language = validate_language_pair(
            args.source_language, args.target_language
        )
    except TranslationError as error:
        parser.error(str(error))
    pdf_path = args.pdf.resolve()
    document_dir = (
        Path("artifacts/documents")
        / f"{pdf_path.stem}-{sha256_file(pdf_path)[:12]}"
    )
    default_language_pair = (
        source_language.casefold() == DEFAULT_SOURCE_LANGUAGE.casefold()
        and target_language.casefold() == DEFAULT_TARGET_LANGUAGE.casefold()
    )
    language_pair = language_pair_key(source_language, target_language)
    extraction_output = args.extraction_output or document_dir / "extraction.json"
    translation_output = args.translation_output or language_scoped_translation_path(
        document_dir / "translation.json", source_language, target_language
    )
    preview_output = document_dir / "preview.pdf"
    rendered_output = args.rendered_output or document_dir / (
        f"{pdf_path.stem}_id.pdf" if default_language_pair
        else f"{pdf_path.stem}_{language_pair}.pdf"
    )
    if args.debug_assignments:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    pipeline_started_at = time.perf_counter()

    try:
        print("PHASE 1: Extraction")
        result = extract_pdf(pdf_path, debug_assignments=args.debug_assignments)
        save_extraction(result, extraction_output)
        create_preview(pdf_path, preview_output, result)
        _print_summary(pdf_path, result)
        print(f"Extraction JSON: {extraction_output}")
        print(f"Preview PDF: {preview_output}")

        print("\nPHASE 1.5: Adaptive layout analysis")
        layout_output_dir = args.layout_output_dir or document_dir / "layout"
        layout_plans = analyze_document_layout(
            pdf_path,
            result.to_dict(),
            layout_output_dir,
            client=OpenAIMinistralVisionClient.from_environment(),
            max_workers=args.vision_workers,
            write_debug_images=args.debug_artifacts,
        )
        vision_pages = sum(plan.source == "vision" for plan in layout_plans.values())
        print(f"Layout plans: {layout_output_dir}")
        print(f"Vision analyzed pages: {vision_pages}/{len(layout_plans)}")

        print("\nPHASE 2: Translation")

        translation_started_at = time.perf_counter()

        run_translation(
            extraction_path=extraction_output,
            output_path=translation_output,
            dictionary_path=args.dictionary,
            source_language=source_language,
            target_language=target_language,
        )

        translation_duration = time.perf_counter() - translation_started_at
        translation_minutes, translation_seconds = divmod(translation_duration, 60)
        print(
            f"Translation duration: "
            f"{int(translation_minutes)}m {translation_seconds:.2f}s "
            f"({translation_duration:.2f} seconds)"
        )

        print("\nPHASE 3: Rendering")
        render_artifact_dir = (
            layout_output_dir
            if default_language_pair else layout_output_dir / f"render_{language_pair}"
        )
        rendered_path, render_stats = render_pdf(
            pdf_path,
            extraction_path=extraction_output,
            translation_path=translation_output,
            output_path=rendered_output,
            layout_plans=layout_plans,
            render_plan_debug_dir=render_artifact_dir,
            identity_debug_dir=render_artifact_dir / "identity",
            debug_images=args.debug_artifacts,
        )
        with fitz.open(pdf_path) as original:
            print_summary(pdf_path, rendered_path, original.page_count, render_stats)

        pipeline_duration = time.perf_counter() - pipeline_started_at
        pipeline_minutes, pipeline_seconds = divmod(pipeline_duration, 60)
        print(
            f"\nTotal pipeline duration: "
            f"{int(pipeline_minutes)}m {pipeline_seconds:.2f}s "
            f"({pipeline_duration:.2f} seconds)"
        )

        return 0
    except (RuntimeError, TranslationError, OSError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 2



def translate_pdf(
    input_pdf: str | Path,
    output_dir: str | Path,
    source_language: str = DEFAULT_SOURCE_LANGUAGE,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    dictionary_path: str | Path | None = None,
    vision_workers: int = DEFAULT_VISION_WORKERS,
    debug_assignments: bool = False,
    debug_artifacts: bool = False,
) -> Path:

    pdf_path = Path(input_pdf).resolve()
    document_dir = Path(output_dir).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not isinstance(vision_workers, int) or isinstance(vision_workers, bool) or vision_workers < 1:
        raise ValueError("vision_workers must be a positive integer")

    source_language, target_language = validate_language_pair(
        source_language,
        target_language,
    )

    document_dir.mkdir(parents=True, exist_ok=True)

    extraction_output = document_dir / "extraction.json"
    translation_output = language_scoped_translation_path(
        document_dir / "translation.json",
        source_language,
        target_language,
    )
    preview_output = document_dir / "preview.pdf"
    rendered_output = document_dir / "output.pdf"
    layout_output_dir = document_dir / "layout"

    default_language_pair = (
        source_language.casefold() == DEFAULT_SOURCE_LANGUAGE.casefold()
        and target_language.casefold() == DEFAULT_TARGET_LANGUAGE.casefold()
    )

    language_pair = language_pair_key(
        source_language,
        target_language,
    )

    if debug_assignments:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s %(message)s",
        )

    pipeline_started_at = time.perf_counter()

    # --------------------------------------------------
    # PHASE 1: Extraction
    # --------------------------------------------------

    print("PHASE 1: Extraction")

    result = extract_pdf(
        pdf_path,
        debug_assignments=debug_assignments,
    )

    save_extraction(result, extraction_output)

    create_preview(
        pdf_path,
        preview_output,
        result,
    )

    _print_summary(pdf_path, result)

    print(f"Extraction JSON: {extraction_output}")
    print(f"Preview PDF: {preview_output}")

    # --------------------------------------------------
    # PHASE 1.5: Layout Analysis
    # --------------------------------------------------

    print("\nPHASE 1.5: Adaptive layout analysis")

    layout_plans = analyze_document_layout(
        pdf_path,
        result.to_dict(),
        layout_output_dir,
        client=OpenAIMinistralVisionClient.from_environment(),
        max_workers=vision_workers,
        write_debug_images=debug_artifacts,
    )

    vision_pages = sum(
        plan.source == "vision"
        for plan in layout_plans.values()
    )

    print(f"Layout plans: {layout_output_dir}")
    print(
        f"Vision analyzed pages: "
        f"{vision_pages}/{len(layout_plans)}"
    )

    # --------------------------------------------------
    # PHASE 2: Translation
    # --------------------------------------------------

    print("\nPHASE 2: Translation")

    translation_started_at = time.perf_counter()

    run_translation(
        extraction_path=extraction_output,
        output_path=translation_output,
        dictionary_path=(
            Path(dictionary_path).resolve()
            if dictionary_path is not None
            else None
        ),
        source_language=source_language,
        target_language=target_language,
    )

    translation_duration = (
        time.perf_counter() - translation_started_at
    )

    translation_minutes, translation_seconds = divmod(
        translation_duration, 60
    )

    print(
        f"Translation duration: "
        f"{int(translation_minutes)}m "
        f"{translation_seconds:.2f}s "
        f"({translation_duration:.2f} seconds)"
    )

    # --------------------------------------------------
    # PHASE 3: Rendering
    # --------------------------------------------------

    print("\nPHASE 3: Rendering")

    render_artifact_dir = (
        layout_output_dir
        if default_language_pair
        else layout_output_dir / f"render_{language_pair}"
    )

    rendered_path, render_stats = render_pdf(
        pdf_path,
        extraction_path=extraction_output,
        translation_path=translation_output,
        output_path=rendered_output,
        layout_plans=layout_plans,
        render_plan_debug_dir=render_artifact_dir,
        identity_debug_dir=render_artifact_dir / "identity",
        debug_images=debug_artifacts,
    )

    with fitz.open(pdf_path) as original:
        print_summary(
            pdf_path,
            rendered_path,
            original.page_count,
            render_stats,
        )

    pipeline_duration = (
        time.perf_counter() - pipeline_started_at
    )

    pipeline_minutes, pipeline_seconds = divmod(
        pipeline_duration, 60
    )

    print(
        f"\nTotal pipeline duration: "
        f"{int(pipeline_minutes)}m "
        f"{pipeline_seconds:.2f}s "
        f"({pipeline_duration:.2f} seconds)"
    )

    return Path(rendered_path)
if __name__ == "__main__":
    raise SystemExit(main())
