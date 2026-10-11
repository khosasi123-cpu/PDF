# Codebase Review dan Session Handoff

Tanggal review: 2026-10-11
Project root: `C:\Users\USER\Documents\KERJAAN\PDF\PDF`  
Package: `pdf-translator` versi `0.1.0`  
Runtime: Python `>=3.11`  
Branch saat review: `main`  
Commit terakhir saat review: `1abeb7a baseline 2`

Dokumen ini menjelaskan kondisi **worktree saat ini**, bukan hanya commit terakhir. Nilai rahasia dari `.env` tidak disalin ke dokumen ini.

## Ringkasan

Project ini adalah pipeline translasi PDF Inggris ke Indonesia dengan tiga tanggung jawab utama:

1. PyMuPDF mengekstrak teks, struktur, style, gambar, tabel, dan geometri sumber.
2. Model lokal/OpenAI-compatible menerjemahkan teks dan, jika dikonfigurasi, memberi saran layout berbasis vision.
3. Python tetap menentukan strategi rendering akhir dan PyMuPDF menggambar hasilnya.

Prinsip arsitektur terpenting adalah model tidak boleh menghasilkan perintah drawing. Layout dari vision hanya advisory. Geometri PDF, source ownership, collision check, font fitting, dan validasi hash tetap dikendalikan secara deterministik oleh Python.

Status verifikasi saat review:

- `126` test lulus.
- `pip check` melaporkan tidak ada dependency yang rusak.
- Tidak ada linting atau type checking yang dikonfigurasi.
- Test integrasi dengan server LLM/vision nyata tidak dijalankan.

## Temuan Code Review

### High 1: Rendering yang gagal tetap menyimpan PDF parsial

Referensi: `src/pdf_translator/render.py:1413-1415`, `src/pdf_translator/render.py:2126-2128`

`document.save(output_path)` dijalankan di dalam `finally`. Jika exception terjadi di tengah loop rendering, file yang baru diproses sebagian tetap ditulis dan dapat terlihat seperti output sukses. Jika proses save ikut gagal, exception awal juga dapat tertutup dan `document.close()` tidak terjamin dijalankan.

Dampak:

- Konsumen dapat memakai PDF parsial atau tidak konsisten tanpa menyadarinya.
- Output lama dapat tertimpa oleh hasil gagal.
- Root cause exception dapat tertutup oleh error saat save.

Rekomendasi:

- Simpan hanya setelah seluruh rendering berhasil.
- Gunakan temporary sibling file, validasi page count dan kemampuan membuka file, lalu lakukan atomic replace ke output final.
- Sisakan hanya `document.close()` di `finally`.
- Tambahkan test yang memaksa exception saat unit kedua diproses dan memastikan output final tidak dibuat/diubah.

### High 2: `.env` berisi credential dan dilacak Git

Referensi: `.gitignore:1-9`; hasil `git ls-files --error-unmatch .env` menunjukkan `.env` tracked.

File `.env` ada, sedang modified, dan tidak dikecualikan oleh `.gitignore`. Nilainya tidak dicantumkan di dokumen ini.

Dampak:

- API key atau endpoint internal dapat ikut terkirim saat repository dibagikan.
- Menghapus file hanya pada commit berikutnya tidak menghapus secret dari history.

Rekomendasi:

- Anggap key yang pernah masuk repository sebagai compromised dan rotate/revoke jika key tersebut nyata.
- Hapus `.env` dari tracking dan, bila perlu, bersihkan history dengan prosedur yang disepakati tim.
- Tambahkan `.env` dan `.env.*` ke `.gitignore`, dengan pengecualian `!.env.example`.
- Simpan hanya placeholder aman di `.env.example`.

### Medium 1: Klaim offline/local tidak dipaksakan oleh kode

Referensi: `src/pdf_translator/translate.py:85-87`, `src/pdf_translator/translate.py:358-367`, `src/pdf_translator/translate.py:521-535`, `src/pdf_translator/layout_analysis.py:214-267`

`LLM_BASE_URL` dan `VISION_BASE_URL` menerima URL bebas. Teks hasil ekstraksi dikirim ke endpoint translasi, sedangkan vision juga mengirim PNG satu halaman dan geometri. Tidak ada validasi loopback, HTTPS, atau persetujuan eksplisit untuk endpoint remote.

Dampak:

- Dokumen rahasia dapat keluar dari mesin walaupun README menyebut pipeline offline/local.
- Endpoint HTTP remote dapat menerima konten tanpa transport encryption.

Rekomendasi:

- Izinkan loopback secara default.
- Wajibkan flag seperti `--allow-remote-model` untuk host non-loopback.
- Tolak HTTP non-loopback atau wajibkan HTTPS.
- Tampilkan peringatan jelas tentang data yang dikirim sebelum request pertama.

### Medium 2: Retry translasi tidak mencakup kegagalan transport utama

Referensi: `src/pdf_translator/translate.py:376-480`, `src/pdf_translator/translate.py:521-540`

Loop `TRANSLATION_MAX_RETRIES` menangani validation/content error, tetapi call utama `client.chat_completion()` tidak menangkap `OSError` atau exception SDK OpenAI pada `translate.py:462`. Timeout translasi juga tidak dikonfigurasi secara eksplisit.

Dampak:

- Network error atau server restart dapat langsung menghentikan pipeline pada percobaan pertama.
- Error SDK dapat keluar sebagai traceback yang tidak konsisten dengan kontrak `TranslationError`.

Rekomendasi:

- Tetapkan timeout yang bounded.
- Bungkus exception transport/API menjadi `TranslationError`.
- Masukkan error transport yang memang retryable ke retry loop dengan backoff terbatas.
- Jangan retry error autentikasi atau konfigurasi permanen.

### Medium 3: Artefak legacy dapat melewati validasi identity yang kuat

Referensi: `src/pdf_translator/identity.py:29-70`, `src/pdf_translator/render.py:1401-1407`

Hash source wajib hanya untuk extraction schema `>=3`, dan relasi source/extraction hash wajib hanya untuk translation schema `>=2`. Artefak lama tanpa hash masih dapat diterima berdasarkan metadata yang lebih lemah.

Dampak:

- Extraction atau translation dari PDF lain dapat dipasang ke koordinat dokumen yang salah.
- Masking dan overlay dapat menutupi area yang tidak semestinya.

Rekomendasi:

- Wajibkan schema modern secara default.
- Jika kompatibilitas legacy dibutuhkan, letakkan di belakang `--allow-legacy-artifacts` dengan warning keras.
- Tambahkan migrator artefak daripada menerima payload tanpa identity secara diam-diam.

### Medium 4: Fallback structured span mengabaikan batas kolom kanan

Referensi: `src/pdf_translator/render.py:359-407`, terutama `src/pdf_translator/render.py:397-399`; fallback dipanggil di `src/pdf_translator/render.py:438-439`

`_structured_spans()` menerima `right_limit`, tetapi saat `expand_last_column=True`, fungsi memanggil `_available_right_edge(page, span_rect)` tanpa meneruskan `right_limit`.

Dampak:

- Saat source-object metadata tidak tersedia, teks structured dapat meluas melewati batas kolom dan masuk ke kolom sebelah.

Rekomendasi:

- Panggil `_available_right_edge(page, span_rect, right_limit)`.
- Tambahkan regression test untuk `_extracted_structured_spans()` ketika source object tidak ditemukan.

### Medium 5: Default dictionary tidak portable setelah package di-install

Referensi: `src/pdf_translator/dictionary.py:10`, `src/pdf_translator/dictionary.py:23-28`, `pyproject.toml:16-17`

Default `config/translation_dictionary.json` dihitung relatif terhadap current working directory dan tidak dikemas sebagai package data.

Dampak:

- Instalasi package yang dijalankan dari luar checkout dapat gagal menemukan dictionary.

Rekomendasi:

- Pindahkan default dictionary ke dalam package dan akses dengan `importlib.resources`, atau wajibkan path eksplisit.
- Tambahkan test terhadap wheel yang di-install ke environment sementara dan dijalankan dari directory lain.

### Medium 6: Default path full pipeline dan CLI per tahap tidak konsisten

Referensi: `src/pdf_translator/main.py:77-84`, `src/pdf_translator/translate.py:24-25`, `src/pdf_translator/render.py:33-36`, `README.md:31-38`, `README.md:58`, `README.md:78`, `README.md:94`

Full pipeline menulis ke `artifacts/documents/<stem>-<hash>/`, sedangkan CLI translation/render standalone memakai path global lama di `artifacts/extraction`, `artifacts/translation`, dan `artifacts/rendered`.

Dampak:

- Perintah rerun per tahap dari README dapat gagal atau memakai artefak stale milik dokumen lain.

Rekomendasi:

- Buat satu workspace/path resolver yang dipakai semua CLI.
- Alternatif minimal: wajibkan path eksplisit pada CLI standalone dan perbarui README.

### Medium 7: Beberapa error input/config lolos dari kontrak CLI

Referensi: `src/pdf_translator/main.py:76-90`, `src/pdf_translator/main.py:151-153`, `src/pdf_translator/translate.py:125-138`, `src/pdf_translator/translate.py:604-614`, `src/pdf_translator/dictionary.py:13-46`, `src/pdf_translator/layout_analysis.py:220-225`

Hash input dihitung sebelum `try` utama. `DictionaryError` dan invalid `VISION_TIMEOUT_SECONDS` adalah `ValueError`, tetapi tidak ditangkap oleh full pipeline. Field extraction yang hilang juga dapat menimbulkan `KeyError` karena akses langsung pada `unit["id"]`, `unit["unit_type"]`, dan `unit["source"]`.

Dampak:

- Missing file, dictionary invalid, timeout invalid, atau JSON parsial menghasilkan traceback dan exit behavior yang tidak konsisten.

Rekomendasi:

- Pindahkan path resolution/hash ke dalam error boundary.
- Validasi artifact dengan model schema yang ketat sebelum dipakai.
- Normalisasi config/schema error menjadi domain error dan return exit code `2`.

### Low 1: Layout page 1000 ke atas tidak dimuat

Referensi: `src/pdf_translator/render.py:88-100`, `src/pdf_translator/layout_analysis.py:341-343`

Writer dapat menghasilkan `page_1000.json`, tetapi loader hanya glob `page_[0-9][0-9][0-9].json`.

Rekomendasi:

- Glob `page_*.json`, validasi suffix numerik, dan cocokkan suffix dengan `plan.page_number`.

### Low 2: Batas kompatibilitas dependency belum presisi

Referensi: `pyproject.toml:10-14`, `src/pdf_translator/translate.py:530-535`, import `fitz` seperti `src/pdf_translator/main.py:8`

`openai>=1.0` terlalu longgar dibanding penggunaan API `responses.parse`. Versi PyMuPDF saat ini juga memberi arah migrasi dari alias `fitz` ke `pymupdf`.

Rekomendasi:

- Tetapkan minimum OpenAI client berdasarkan versi terendah yang benar-benar dites dengan `responses.parse`.
- Pertimbangkan upper bound atau lock-file policy untuk aplikasi.
- Migrasikan import `fitz` ke `pymupdf` sebelum alias deprecated dihapus.

### Low 3: Dokumentasi dan generated metadata tertinggal

Referensi: `README.md:31`, `README.md:58`, `README.md:78`, `README.md:82`, `README.md:94`; `src/pdf_translator.egg-info/`

README mencampur lokasi artefak document-scoped dan path legacy. Metadata `egg-info` di source tree juga berisiko stale terhadap modul terbaru.

Rekomendasi:

- Jadikan README mengikuti path resolver aktual.
- Jangan version-control generated `*.egg-info` kecuali ada alasan khusus.

## Kekuatan Codebase

- Pemisahan antara rekomendasi model dan keputusan drawing sudah kuat.
- `LayoutPlan` dan `RenderPlan` memakai Pydantic dengan `extra="forbid"`.
- Compact vision references dipetakan kembali ke authoritative source geometry oleh Python.
- Hash PDF, extraction, translation, dan layout mencegah sebagian besar cross-document misuse pada schema modern.
- Translation response divalidasi berdasarkan ID dan source asli.
- Terminology enforcement dilakukan lagi secara deterministik setelah output model.
- Suspicious translation dapat di-retry dan akhirnya fallback ke source.
- Source span ownership diperiksa sebelum render.
- Renderer mempertimbangkan text, image, dan vector obstacle.
- Test suite luas untuk tabel, structured text, multi-column, TOC, Unicode, identity, fallback, dan collision.
- Vision failure tidak memblokir pipeline karena ada deterministic layout fallback.

## Arsitektur

```text
Input PDF
   |
   v
extractor.py -- PyMuPDF text/drawing/image extraction
   |
   v
semantics.py -- inline continuation, TOC, borderless grid, header/footer
   |
   +--> extraction.json
   +--> preview.pdf
   |
   v
layout_analysis.py
   |-- vision tersedia --> validated LayoutPlan
   |-- vision gagal -----> deterministic fallback LayoutPlan
   |
   v
translate.py + dictionary.py
   |
   +--> translation.json
   |
   v
render_plan.py -- deterministic strategy selection
   |
   +--> RenderPlan JSON/debug PNG
   |
   v
render.py -- mask, fit, expand, draw, preserve/fallback
   |
   +--> translated PDF
   +--> identity manifest/debug images
```

### Boundary tanggung jawab

| Komponen | Boleh menentukan | Tidak boleh menentukan |
|---|---|---|
| Vision model | Region type, reading order, relationship, rekomendasi strategi | Perintah PyMuPDF, final drawing coordinates, executable code |
| RenderPlanner | Strategi final berdasarkan geometri, fit, obstacle, confidence | Menggambar langsung ke PDF |
| Renderer | Masking dan drawing sesuai instruction tervalidasi | Mempercayai geometri model tanpa rekonsiliasi |

## Struktur Directory

```text
PDF/
|-- src/pdf_translator/              package runtime
|   |-- main.py                      orchestrator full pipeline
|   |-- models.py                    dataclass extraction
|   |-- extractor.py                 ekstraksi PDF dan ruled-table detection
|   |-- semantics.py                 semantic post-processing
|   |-- filters.py                   keputusan translate/skip
|   |-- dictionary.py                konfigurasi terminologi
|   |-- translate.py                 batching, client, validasi, retry
|   |-- layout.py                    LayoutPlan dan fallback layout
|   |-- layout_analysis.py           vision adapter dan geometry reconciliation
|   |-- render_plan.py               pemilihan strategi render
|   |-- render.py                    rendering primitives dan eksekusi
|   |-- identity.py                  hash, ownership, provenance
|   |-- preview.py                   preview extraction
|   |-- render_debug.py              JSON/PNG diagnostics
|   `-- __init__.py                  package version
|-- tests/                           9 module test dan smoke generator
|-- config/translation_dictionary.json
|-- data/input/                      fixture/input PDF lokal
|-- data/output/                     output data placeholder
|-- artifacts/                       extraction/layout/render/regression outputs
|-- gradio_files/                    belum ada implementasi Gradio runtime
|-- pyproject.toml                   package dan pytest config
|-- uv.lock                          dependency lock
|-- README.md                        dokumentasi penggunaan
|-- adaptive-pdf-renderplanner-architecture.json
|                                    archive session/desain, bukan runtime input
`-- .env                             config lokal, saat ini tracked dan harus diamankan
```

## Fungsi Setiap Modul

### `main.py`

Referensi utama: `src/pdf_translator/main.py:64-157`

- Entry point full pipeline: `python -m pdf_translator.main INPUT.pdf`.
- Membuat workspace `artifacts/documents/<nama>-<12-char-sha256>/`.
- Menjalankan extraction, preview, layout analysis, translation, dan rendering.
- Worktree saat ini juga mencetak durasi translation dan total pipeline.

### `models.py`

Referensi: `src/pdf_translator/models.py:7-110`

| Model | Fungsi |
|---|---|
| `TranslationUnit` | Unit teks yang diterjemahkan beserta bbox, font, source IDs, dan semantic metadata |
| `PageExtraction` | Ukuran/rotasi halaman, unit, source objects, dan TOC entries |
| `SourceObject` | Objek PDF stabil seperti block, line, span, image, dan cell |
| `TocEntry` | Pemisahan title, leader, dan page-number anchor pada TOC |
| `ExtractionResult` | Dokumen extraction lengkap, schema v3, dan source hash |

### `extractor.py`

Referensi: `src/pdf_translator/extractor.py:51-83`, `src/pdf_translator/extractor.py:173-313`, `src/pdf_translator/extractor.py:387-424`, `src/pdf_translator/extractor.py:545-690`

- Mengambil `page.get_text("dict", sort=True)`.
- Memberi ID stabil pada block, line, span, image, table, dan cell.
- Mendeteksi ruled table dari horizontal/vertical drawing segments.
- Menetapkan span ke cell melalui containment atau overlap konservatif.
- Menata reading order dua kolom jika tidak ada table.
- Menghasilkan source objects dan unit IDs global lintas halaman.
- Menolak encrypted PDF yang membutuhkan password.
- Menyerahkan hasil ke semantic pass sebelum disimpan.

### `semantics.py`

Referensi: `src/pdf_translator/semantics.py:97-169`, `src/pdf_translator/semantics.py:223-407`, `src/pdf_translator/semantics.py:410-491`, `src/pdf_translator/semantics.py:494-572`

- Menggabungkan inline continuation berbasis tanda baca.
- Mendeteksi TOC dan mempertahankan title/page-number ownership terpisah.
- Mendeteksi borderless key/value grid secara konservatif.
- Mengelompokkan running header/footer berulang.
- Memastikan tidak ada meaningful source span yang hilang setelah transformasi.

### `filters.py`

Referensi: `src/pdf_translator/filters.py:3-20`

- `should_translate()` melewati teks kosong, symbol-only, dan unit yang sekurangnya 80% tokennya identifier-like.
- Uppercase text tidak otomatis dilewati.

### `dictionary.py`

Referensi: `src/pdf_translator/dictionary.py:10-122`

- Memvalidasi tiga key wajib: `skip_translation`, `keep_english`, dan `fixed_translation`.
- Menolak conflict lintas category secara case-insensitive.
- Mencocokkan phrase dengan boundary dan memprioritaskan phrase lebih panjang.
- Menghasilkan context prompt dan stable context hash.

### `translate.py`

Referensi: `src/pdf_translator/translate.py:49-121`, `src/pdf_translator/translate.py:125-219`, `src/pdf_translator/translate.py:259-331`, `src/pdf_translator/translate.py:334-601`

- Membaca config model dari environment.
- Memuat hanya unit yang harus diterjemahkan atau exact dictionary skip.
- Deduplicate source setara berdasarkan language, dictionary hash, unit type, dan normalized source.
- Mengirim batch dengan kontrak JSON berbasis ID.
- Memvalidasi missing, duplicate, unexpected ID, dan source association.
- Mendeteksi commentary, repetition, extreme growth, dan perubahan private-use glyph.
- Melakukan targeted retry untuk output suspicious.
- Fallback ke source bila sanitation tetap gagal.
- Menerapkan `keep_english` dan `fixed_translation` secara deterministik.
- Menulis translation schema v2 untuk extraction modern.

### `layout.py`

Referensi: `src/pdf_translator/layout.py:12-146`, `src/pdf_translator/layout.py:171-227`, `src/pdf_translator/layout.py:248-341`

- Mendefinisikan region type, rendering strategy, geometry source, relationship, dan `LayoutPlan` schema v2.
- Menolak extra field, duplicate region, invalid reference, confidence, dan bbox.
- Mengganti bbox advisory dari model dengan union source-object geometry.
- Menghasilkan fallback layout dari extraction, semantic role, column, dan image.

### `layout_analysis.py`

Referensi: `src/pdf_translator/layout_analysis.py:51-198`, `src/pdf_translator/layout_analysis.py:201-346`

- Membuat compact geometry untuk mengurangi prompt vision.
- Mengirim PNG halaman dan geometry JSON ke endpoint vision.
- Memvalidasi compact semantic response, lalu Python membangun `LayoutPlan` dari unit/image references.
- Menjalankan maksimal dua request halaman secara concurrent secara default.
- Fallback jika client tidak tersedia, response invalid, confidence rendah, atau API gagal.
- Menulis `page_NNN.json`; debug PNG hanya ditulis dengan `--debug-artifacts`.

### `render_plan.py`

Referensi: `src/pdf_translator/render_plan.py:18-76`, `src/pdf_translator/render_plan.py:168-193`, `src/pdf_translator/render_plan.py:204-424`

- Mendefinisikan executable-safe `RenderInstruction` dan `RenderPlan` schema v2.
- Menghitung collision-free expansion candidates.
- Memilih strategi final per unit.
- Prioritas khusus diberikan kepada structured/table, TOC, reliable source spans, fixed regions, dan image overlap safety.
- Rekomendasi model hanya dipakai jika confidence cukup dan tetap tunduk pada constraint deterministik.

### `render.py`

Referensi utama: `src/pdf_translator/render.py:48-112`, `src/pdf_translator/render.py:923-1230`, `src/pdf_translator/render.py:1233-1336`, `src/pdf_translator/render.py:1392-2201`

- Memuat dan memvalidasi extraction, translation, dan layout plans.
- Membangun page geometry serta obstacle map.
- Menjalankan `RenderPlanner`.
- Memvalidasi source ownership dan exact source association.
- Membuat mask dengan proteksi terhadap text/image/vector neighbor.
- Mencoba source-baseline/span mapping, textbox fit, safe expansion, dan Unicode HTML fallback.
- Mempertahankan source visual jika typography floor tidak dapat dipenuhi dengan aman.
- Mempertahankan TOC page-number anchor dan membuat ulang leader dots.
- Menyimpan final PDF dan diagnostic identity artifacts.

Catatan penting: rendering adalah visual overlay. Original text tetap berada di content stream dan ditutup secara visual, bukan dihapus secara semantik.

### `identity.py`

Referensi: `src/pdf_translator/identity.py:14-96`, `src/pdf_translator/identity.py:98-310`

- Menghitung SHA-256 file dan canonical JSON payload.
- Memastikan extraction/translation/layout berasal dari PDF dan extraction yang sesuai.
- Membandingkan source ownership unit dengan render instruction.
- Menulis manifest status translated, protected, fallback, non-translatable, duplicate, atau unaccounted.

### `preview.py` dan `render_debug.py`

Referensi: `src/pdf_translator/preview.py:8-28`, `src/pdf_translator/render_debug.py:20-86`

- `preview.py` memberi rectangle dan unit ID pada salinan PDF hasil extraction.
- `render_debug.py` membuat JSON/PNG untuk LayoutPlan dan RenderPlan.

## Alur Data End-to-End

### Fase 1: Extraction

1. Input PDF di-hash untuk membuat document-scoped workspace.
2. PyMuPDF mengekstrak blocks, lines, spans, image, dan drawings.
3. Ruled tables dan kemungkinan dua kolom dianalisis.
4. Unit extraction dibentuk beserta source ownership.
5. Semantic pass mendeteksi continuation, TOC, borderless grid, dan running template.
6. `extraction.json` schema v3 dan `preview.pdf` ditulis.

### Fase 1.5: Layout analysis

1. Setiap halaman diraster menjadi PNG skala `0.8`.
2. Compact geometry dan PNG dikirim ke vision jika dikonfigurasi.
3. Response divalidasi terhadap `LayoutPlan`.
4. Python membangun bbox dan source ownership dari unit IDs serta image indexes hasil vision.
5. Jika vision gagal atau confidence `<0.6`, deterministic fallback dipakai.

### Fase 2: Translation

1. Dictionary dimuat dan divalidasi.
2. Exact skip terms disalin tanpa dikirim ke model.
3. Unit setara di-deduplicate untuk satu document run.
4. Batch dikirim dan response divalidasi.
5. Suspicious output mendapat satu targeted content retry.
6. Terminology enforcement diterapkan.
7. `translation.json` ditulis dengan source/extraction identity.

### Fase 3: Planning dan rendering

1. Hash source dan artefak diverifikasi.
2. Layout geometry direkonsiliasi dengan PDF.
3. RenderPlanner memilih satu strategi aman per unit.
4. Renderer memeriksa ownership dan translation source.
5. Masking dan drawing dilakukan dengan obstacle protection.
6. Unit yang tidak aman untuk dirender dipertahankan sebagai source visual.
7. Final PDF, RenderPlan debug, dan identity diagnostics ditulis.

## Artefak dan Schema

| Artefak | Schema | Producer | Isi utama |
|---|---:|---|---|
| `extraction.json` | 3 | `extractor.py` | pages, units, source objects, TOC, source SHA-256 |
| `translation.json` | 2 | `translate.py` | translations, model, source dan extraction hashes |
| `page_NNN.json` | 2 | `layout_analysis.py` | validated LayoutPlan |
| `page_NNN_render_plan.json` | 2 | `render_debug.py` | executable-safe RenderPlan |
| `render_identity.json` | internal diagnostic | `identity.py` | accounting setiap source span/unit |
| `preview.pdf` | PDF | `preview.py` | extraction boxes dan unit IDs |
| `<stem>_id.pdf` | PDF | `render.py` | hasil translasi visual |

Default full-pipeline workspace:

```text
artifacts/documents/<pdf-stem>-<first-12-source-sha256>/
|-- extraction.json
|-- preview.pdf
|-- translation.json
|-- <pdf-stem>_id.pdf
`-- layout/
    |-- page_NNN.json
    |-- page_NNN_debug.png
    |-- page_NNN_render_plan.json
    |-- page_NNN_render_plan_debug.png
    `-- identity/
```

## Konfigurasi

### Environment variables

| Variable | Wajib | Default | Fungsi |
|---|---|---|---|
| `LLM_BASE_URL` | Ya untuk translasi | tidak ada | OpenAI-compatible translation endpoint |
| `OPENAI_API_KEY` | Ya untuk translasi | vision memakai `local-key` bila kosong | Credential client |
| `MODEL` | Ya untuk translasi | tidak ada | Model translation |
| `TRANSLATION_BATCH_SIZE` | Tidak | `30` | Batas unit representative per batch |
| `TRANSLATION_MAX_BATCH_CHARACTERS` | Tidak | `6000` | Batas karakter source per batch |
| `TRANSLATION_MAX_RETRIES` | Tidak | `2` | Retry validation batch |
| `VISION_MODEL` | Tidak | kosong | Mengaktifkan vision jika endpoint juga ada |
| `VISION_BASE_URL` | Tidak | fallback ke `LLM_BASE_URL` | Vision endpoint |
| `VISION_TIMEOUT_SECONDS` | Tidak | `300` | Timeout vision request |

Bahasa source dan target saat ini hard-coded sebagai English ke Indonesian di `src/pdf_translator/translate.py:339-340` dan `src/pdf_translator/translate.py:511-512`.

### Translation dictionary

Path repository: `config/translation_dictionary.json`

| Key | Perilaku |
|---|---|
| `skip_translation` | Exact source unit disalin apa adanya dan tidak dikirim ke model |
| `keep_english` | Phrase dipertahankan dalam bahasa Inggris |
| `fixed_translation` | Phrase source diganti dengan istilah Indonesia yang ditentukan |

## Cara Menjalankan

### Setup

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

### Full pipeline

```powershell
.\.venv\Scripts\python.exe -m pdf_translator.main data/input/test.pdf
```

Opsi:

```text
--debug-assignments
--dictionary PATH
--extraction-output PATH
--translation-output PATH
--rendered-output PATH
--layout-output-dir PATH
```

### Translation standalone yang aman

Karena default path standalone masih legacy, gunakan path eksplisit:

```powershell
.\.venv\Scripts\python.exe -m pdf_translator.translate `
  --input artifacts/documents/<document-id>/extraction.json `
  --output artifacts/documents/<document-id>/translation.json `
  --dictionary config/translation_dictionary.json
```

### Rendering standalone yang aman

```powershell
.\.venv\Scripts\python.exe -m pdf_translator.render data/input/test.pdf `
  --extraction artifacts/documents/<document-id>/extraction.json `
  --translation artifacts/documents/<document-id>/translation.json `
  --output artifacts/documents/<document-id>/test_id.pdf `
  --layout-dir artifacts/documents/<document-id>/layout `
  --render-plan-debug-dir artifacts/documents/<document-id>/layout `
  --identity-debug-dir artifacts/documents/<document-id>/layout/identity
```

### Test

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
```

Hasil saat review:

```text
126 passed in 1.14s
No broken requirements found.
```

## Cakupan Test

| File | Cakupan utama |
|---|---|
| `tests/test_dictionary.py` | validation, conflict, phrase matching, terminology enforcement |
| `tests/test_filters.py` | natural-language vs identifier filtering |
| `tests/test_extractor.py` | blocks, reading order, IDs, table assignment, real fixture |
| `tests/test_semantics.py` | headers/footer, TOC, borderless grid, inline continuation |
| `tests/test_layout.py` | schema dan source-backed geometry resolution |
| `tests/test_layout_analysis.py` | vision valid/invalid/low-confidence fallback |
| `tests/test_render_plan.py` | strategy selection, expansion, collision, ownership |
| `tests/test_render.py` | Unicode, structured layout, obstacles, identity, TOC, debug artifacts |
| `tests/test_translate.py` | response validation, retry, sanitation, deduplication, environment |

Test yang masih kurang:

- Atomicity output saat rendering exception.
- CLI exit behavior untuk missing file, malformed JSON, invalid dictionary, dan invalid environment.
- Translation network timeout dan transport retry.
- Installed-wheel execution di luar repository root.
- Rejection/default policy untuk legacy artifact.
- Privacy policy untuk remote endpoint.
- Structured fallback tanpa source-object metadata.
- Layout loading untuk dokumen lebih dari 999 halaman.
- Integrasi dengan server OpenAI-compatible translation dan vision nyata.
- Fuzzing PDF malformed, sangat besar, encrypted, atau adversarial.

## Optimisasi Performa yang Sudah Diterapkan

- Vision output limit diturunkan menjadi `3000` token.
- Vision memakai compact semantic response; bbox dan source ownership diselesaikan Python.
- Vision page request berjalan dengan bounded concurrency default `2`, dapat diubah melalui `--vision-workers`.
- Translation default batch menjadi maksimal `30` unit dan target `6000` source characters.
- Translation output limit dihitung per batch, dengan ruang lebih besar hanya untuk atomic unit yang sangat panjang.
- LayoutPlan, RenderPlan, dan identity JSON tetap ditulis pada full pipeline.
- Diagnostic PNG dan identity overlays hanya ditulis dengan `--debug-artifacts`.
- Cache lintas eksekusi belum diterapkan sesuai keputusan user.
- Perilaku model loading/switching server belum diubah dan perlu diperiksa dari log server.

## Known Limitations

- Tidak ada OCR untuk scanned PDF.
- Teks yang tertanam di image tidak diterjemahkan.
- Tidak ada image inpainting.
- Tidak ada general sentence reconstruction lintas block.
- Line-end hyphenation belum dinormalisasi.
- Table detection bergantung pada visible PDF geometry.
- Deteksi layout sengaja konservatif dan lebih memilih false negative.
- Original text tetap ada di content stream di bawah visual mask.
- Page rasterization dapat memakai memori besar; identity/debug PNG bersifat opt-in.
- Correctness visual belum dapat dijamin hanya dari unit test berbasis text/geometry.

## Kondisi Worktree Saat Review

`git status --short` sebelum dokumen ini dibuat menunjukkan:

```text
 M .env
 M src/pdf_translator/layout_analysis.py
 M src/pdf_translator/main.py
```

Perubahan code yang sudah ada dan bukan dibuat oleh review ini:

- `layout_analysis.py`: output limit vision saat ini `3000`; compact response dan bounded concurrency sudah diterapkan setelah review awal.
- `main.py`: penambahan pengukuran durasi translation dan total pipeline.
- `.env`: modified; jangan tampilkan atau salin nilainya ke session/chat.

Session berikutnya harus mempertahankan perubahan tersebut kecuali user secara eksplisit meminta perubahan atau rollback.

## Prioritas Perbaikan yang Disarankan

1. Amankan `.env`, rotate credential bila nyata, dan hentikan tracking secret.
2. Buat rendering output atomic agar kegagalan tidak menghasilkan PDF parsial.
3. Tambahkan policy local/remote endpoint agar klaim offline benar-benar enforced.
4. Perbaiki transport timeout/retry dan normalisasi CLI error.
5. Wajibkan artifact identity schema modern secara default.
6. Perbaiki `right_limit` pada structured fallback beserta regression test.
7. Satukan document workspace resolver dan perbarui README.
8. Package default dictionary dengan benar.
9. Tambahkan linting/type checking dan installed-wheel smoke test.

## Handoff Singkat untuk Session Berikutnya

Gunakan blok berikut sebagai konteks awal:

```text
Project: Python PDF translator di C:\Users\USER\Documents\KERJAAN\PDF\PDF.

Baca CODEBASE_REVIEW.md sebagai peta arsitektur dan hasil review terkini. Pipeline-nya adalah extraction -> semantic analysis -> optional vision LayoutPlan -> validated translation -> deterministic RenderPlan -> PyMuPDF overlay rendering. Vision hanya advisory; jangan izinkan model membuat code atau drawing command.

Baseline terverifikasi: 126 pytest passed dan pip check bersih pada 2026-10-11.

Worktree sebelum CODEBASE_REVIEW.md dibuat sudah memiliki perubahan user pada .env, src/pdf_translator/layout_analysis.py, dan src/pdf_translator/main.py. Jangan revert perubahan tersebut. Jangan tampilkan isi .env.

Risiko prioritas tertinggi:
1. render.py menyimpan output dari finally sehingga exception dapat menghasilkan PDF parsial.
2. .env tracked dan tidak di-ignore.
3. endpoint remote tidak dibatasi walaupun project diklaim offline/local.
4. retry translation tidak menangani transport failure utama.
5. artifact legacy dapat melewati hash identity.

Sebelum mengubah code, jalankan pytest. Setelah perubahan, jalankan full pytest dan pip check. Pertahankan source ownership, artifact hash validation, deterministic fallback, dan larangan model mengontrol drawing.
```

## File yang Paling Relevan untuk Dibaca Lebih Dulu

1. `CODEBASE_REVIEW.md`
2. `README.md`
3. `src/pdf_translator/main.py`
4. `src/pdf_translator/models.py`
5. `src/pdf_translator/extractor.py`
6. `src/pdf_translator/semantics.py`
7. `src/pdf_translator/translate.py`
8. `src/pdf_translator/layout.py`
9. `src/pdf_translator/layout_analysis.py`
10. `src/pdf_translator/render_plan.py`
11. `src/pdf_translator/render.py`
12. `src/pdf_translator/identity.py`
13. Test yang terkait langsung dengan perubahan yang akan dibuat
