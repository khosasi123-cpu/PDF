# pdf-translator

Offline PDF translation pipeline with PyMuPDF extraction, local model translation, advisory vision layout analysis, deterministic render planning, and PyMuPDF rendering.

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

## Full pipeline

Run the entire pipeline from the project root with one command:

```powershell
.\.venv\Scripts\python.exe -m pdf_translator.main data/input/test.pdf
```

This runs extraction, dictionary-aware translation, and rendering in order. It writes `extraction.json`, `preview.pdf`, `translation.json`, and the final `artifacts/rendered/test_id.pdf`. The source PDF is not overwritten. The local vLLM server must already be running before this command.

Use the separate commands when rerunning only one stage:

```powershell
.\.venv\Scripts\python.exe -m pdf_translator.translate
.\.venv\Scripts\python.exe -m pdf_translator.render data/input/test.pdf
```

## Local translation

Install the OpenAI-compatible client with the project dependencies, then configure the local server in the environment:

```powershell
$env:LLM_BASE_URL = "http://localhost:1234/v1"
$env:OPENAI_API_KEY = "local-key"
$env:MODEL = "your-local-mistral-model"
$env:TRANSLATION_BATCH_SIZE = "20"
$env:TRANSLATION_MAX_RETRIES = "2"
```

Run translation independently from extraction:

```powershell
.\.venv\Scripts\python.exe -m pdf_translator.translate
```

It reads `artifacts/extraction/extraction.json`, sends only units with `translate: true`, validates every batch ID and exact source string, and writes `artifacts/translation/translation.json`. Extraction JSON remains read-only. No PDF rendering is performed in Phase 2.

## Translation dictionary

Edit [config/translation_dictionary.json](config/translation_dictionary.json) to control translation behavior without changing Python code:

- `skip_translation`: exact source units copied unchanged and never sent to the LLM.
- `keep_english`: phrases preserved in English while the surrounding sentence is translated.
- `fixed_translation`: source phrases replaced deterministically with the configured Indonesian term after LLM output.

The dictionary is validated for duplicate category conflicts. Longer phrases are matched before shorter phrases with word-boundary-aware matching. Run the same translation command again after editing the dictionary, then render as usual.

## Rendering

Render the validated translation without running extraction or translation again:

```powershell
.\.venv\Scripts\python.exe -m pdf_translator.render data/input/test.pdf
```

The output is `artifacts/rendered/test_id.pdf`. The renderer uses the original PDF as its base and does not call the LLM.

## Adaptive layout planning

The full pipeline writes validated LayoutPlans and RenderPlans to `artifacts/layout/<document>/`. Set `VISION_MODEL` and either `VISION_BASE_URL` or `LLM_BASE_URL` to enable local Ministral Vision analysis. If vision is unavailable, malformed, or low-confidence, extraction-based layout heuristics are used automatically.

LayoutPlan recommendations are not drawing commands. `RenderPlanner` validates them against source spans, font fit, page and column bounds, neighboring text, images, and vector graphics. It then selects one of the small deterministic strategies: preserve, reflow, structured, multicolumn, preserve image, source-span mapping, safe expansion, or original-bbox fallback. PyMuPDF remains the only drawing implementation.

Each page has a LayoutPlan JSON/debug PNG and a RenderPlan JSON/debug PNG. Image regions with embedded text are preserved and reported; text inside images is not translated yet.

Extraction assigns deterministic page-local IDs to PDF blocks, lines, spans, images, and detected cells. Vision regions reference these IDs, and Python resolves their union from PyMuPDF geometry. Model bboxes are retained only for diagnostics when PDF-backed geometry exists; genuinely visual-only regions are marked `vision_estimate`.

A document-level deterministic pass identifies repeated running headers/footers, source-separated TOC entries, and conservative borderless key/value grids. Headers and footers remain fixed to their source bands. TOC titles are translated independently while page-number anchors remain untouched and leader dots are regenerated deterministically.

Translation validation separates structural failures from content sanitation. Empty output falls back to source, unchanged technical text is accepted, and obvious commentary, repeated sentences, extreme growth, or altered private-use glyphs receive one targeted retry before source fallback. Identical normalized source units of the same type reuse one translation within the current document run.

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

- No OCR.
- No image inpainting or replacement of text embedded in images.
- No semantic sentence detection.
- No semantic block merging outside detected table cells.
- Overlay rendering leaves the original text in the PDF content stream, while covering it visually with white rectangles.
- Line-end hyphenation such as `config-` / `uration` is not normalized.
- Table boundaries come from visible PDF geometry; ordinary block boundaries come directly from PyMuPDF.
- Geometry detection favors false negatives and can miss tables with broken, unusually thick, or non-rectangular borders.
- Vision recommendations remain advisory and fall back to deterministic extraction geometry when unavailable or invalid.
- A PDF does not explicitly encode sentence boundaries.
- The smoke-test PDF is only a validation fixture and is not representative of production PDFs.
