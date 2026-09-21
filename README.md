# pdf-translator

Phase 1.5 of an offline PDF translation pipeline. This phase inspects what PyMuPDF extracts from a PDF and produces block-level text units plus geometry-aware table-cell units and a visual inspection preview. It does not call an LLM.

## Installation

Python 3.11 or newer is required. From the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

On Linux, use `.venv/bin/python` instead.

## Usage

```powershell
.\.venv\Scripts\python.exe -m pdf_translator.main data/input/test.pdf
```

Add `--debug-assignments` to log each text-to-cell assignment, including table/cell coordinates, bboxes, and whether containment or overlap selected the cell.

The command writes `artifacts/extraction/extraction.json` and `artifacts/extraction/preview.pdf` without modifying the input PDF. The preview keeps the original content and overlays green rectangles for normal text units, blue rectangles for table-cell units, and red rectangles for skipped units, with unit IDs beside them.

A smoke-test PDF can be generated with:

```powershell
.\.venv\Scripts\python.exe tests/create_smoke_pdf.py
.\.venv\Scripts\python.exe -m pdf_translator.main data/input/smoke_test.pdf
```

## Phase 1 design

A normal text unit is one PyMuPDF text block (`block["type"] == 0`) containing at least one non-whitespace span. A table-cell unit is the text assigned to one cell in a confidently detected bordered table. Spans are used only as extraction and style components. Normal unit bounding boxes are the union of meaningful span boxes; table-cell bounding boxes are the detected cell interior with a small border inset. Representative font metadata comes from the span with the most non-whitespace characters. Pages and application-facing unit page numbers are 1-based; PyMuPDF indexes are 0-based internally.

Text extraction uses `page.get_text("dict", sort=True)`. Sorting is deliberate: it asks PyMuPDF for stable reading order while retaining the native page -> block -> line -> span hierarchy. Table detection uses only `page.get_drawings()` and repeated thin horizontal/vertical geometry. It requires multiple complete adjacent cells and otherwise falls back to normal block extraction. Lines from different PyMuPDF blocks are merged when they belong to the same cell; spans crossing a cell boundary remain normal text and increment the ambiguous-assignment count. Text is reconstructed line by line, preserving line boundaries and adding a space only when span geometry indicates a gap. The JSON records PyMuPDF version, page dimensions, rotation, unit type, direction, color, line count, and the `translate` decision.

The filter is deterministic and conservative. It skips empty text, all-identifier content, and blocks where at least 80% of whitespace-delimited tokens contain identifier-like characters such as digits, underscores, slashes, backslashes, dashes, dots, or colons. Uppercase words are not filtered by themselves. Skipped units remain in the JSON with `translate: false`.

Pages without extractable text are reported and processing continues. OCR is not attempted.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Known limitations

- No LLM translation yet.
- No OCR.
- No semantic sentence detection.
- No semantic block merging outside detected table cells.
- No PDF text replacement or final PDF rendering/translation yet.
- Line-end hyphenation such as `config-` / `uration` is not normalized.
- Table boundaries come from visible PDF geometry; ordinary block boundaries come directly from PyMuPDF.
- Geometry detection favors false negatives and can miss tables with broken, unusually thick, or non-rectangular borders.
- A PDF does not explicitly encode sentence boundaries.
- The smoke-test PDF is only a validation fixture and is not representative of production PDFs.
