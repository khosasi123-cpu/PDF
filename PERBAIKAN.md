Waktu **3 menit untuk 4 halaman** dan **5 menit untuk 10 halaman** masih cukup wajar untuk pipeline offline berkualitas tinggi yang melakukan vision analysis, translation, dan rendering. Namun implementasi sekarang belum optimal. Ada beberapa pekerjaan yang terbuang dan dapat dipercepat tanpa mengurangi kualitas.

## Analisis Waktu

Dari dua pengukuran:

```text
4 halaman  = 180 detik
10 halaman = 300 detik
```

Estimasi kasarnya:

```text
Tambahan per halaman = (300 - 180) / (10 - 4)
                     = sekitar 20 detik/halaman

Fixed overhead       = 180 - (4 x 20)
                     = sekitar 100 detik
```

Ini mengindikasikan dua komponen:

- Sekitar `100 detik` fixed overhead, kemungkinan model loading, model switching, atau request pertama.
- Sekitar `20 detik` tambahan per halaman.

Dokumennya memang tidak identik, jadi angka ini hanya indikasi. Tetapi scaling-nya cukup baik: rata-rata turun dari `45 detik/halaman` menjadi `30 detik/halaman`.

## Data Artefak Saat Ini

| Dokumen | Halaman | Unit | Unit translasi | Karakter | Unique unit | Estimasi batch | Vision berhasil |
|---|---:|---:|---:|---:|---:|---:|---:|
| `test` | 4 | 135 | 113 | 5.419 | 79 | 4 | 2/4 |
| `test5` | 10 | 137 | 124 | 16.772 | 117 | 6 | 7/10 |

Dokumen 10 halaman memiliki lebih dari 3 kali karakter yang harus diterjemahkan, walaupun jumlah unit hampir sama. Jadi waktu translation lebih dipengaruhi panjang teks dan jumlah batch, bukan hanya halaman.

## Masalah Efisiensi Terbesar

### 1. Vision output token terlalu tinggi

Saat ini:

```python
VISION_MAX_OUTPUT_TOKENS = 10000
```

Referensi: `src/pdf_translator/layout_analysis.py:20`

Ini sudah menyebabkan request gagal:

```text
test halaman 2:
input 22651 + output 10000 = 32651
context maksimum = 26000

test5 halaman 7:
input 19895 + output 10000 = 29895
context maksimum = 26000
```

Jadi nilai `10000` bukan hanya boros, tetapi membuat beberapa halaman langsung gagal dan memakai fallback.

Perlu dicatat bahwa `max_tokens` adalah batas maksimum, bukan berarti model selalu menghasilkan 10.000 token. Tetapi server tetap menghitungnya untuk context admission.

Rekomendasi awal:

```python
VISION_MAX_OUTPUT_TOKENS = 3000
```

Nilai sebelumnya memang `3000`. Lebih aman lagi menggunakan output schema yang lebih ringkas sehingga `2000-3000` cukup.

### 2. Sekitar 36% vision analysis dibuang

Dari total 14 halaman sample:

```text
Vision berhasil : 9 halaman
Fallback         : 5 halaman
Fallback rate    : 35,7%
```

Penyebab yang ditemukan:

- Output token melewati context window.
- Model menghasilkan source ID yang tidak dikenal.
- Model menghasilkan relationship ke region yang tidak ada.
- Bbox berada di luar halaman.

Artinya model sudah menghabiskan waktu menganalisis halaman, tetapi seluruh hasil halaman kemudian dibuang.

Solusi terbaik bukan memperpanjang prompt, melainkan memperkecil tanggung jawab vision model.

Vision sebenarnya tidak perlu mengembalikan:

- Bbox yang sudah diketahui Python.
- Width dan height.
- Source IDs lengkap.
- Semua field internal `LayoutPlan`.
- Geometri detail setiap span.

Cukup minta respons ringkas seperti:

```json
{
  "page_number": 1,
  "confidence": 0.9,
  "regions": [
    {
      "type": "paragraph",
      "unit_ids": [1, 2, 3],
      "reading_order": 0,
      "strategy": "reflow_region"
    }
  ]
}
```

Python kemudian mengisi bbox dan source IDs dari extraction. Ini akan:

- Mengurangi input/output token.
- Mengurangi hallucinated source IDs.
- Menghilangkan bbox invalid.
- Mempercepat generation.
- Menurunkan fallback rate.

### 3. Semua request vision berjalan serial

Referensi: `src/pdf_translator/layout_analysis.py:323-345`

Saat ini:

```python
for page in document:
    plan = analyze_page_layout(...)
```

Setiap halaman menunggu halaman sebelumnya selesai.

Jika model server mendukung concurrent request atau continuous batching, gunakan bounded concurrency, misalnya dua halaman bersamaan.

Namun ini perlu benchmark:

- Jika satu GPU dengan server yang hanya memproses satu request, concurrency mungkin tidak membantu.
- Jika menggunakan vLLM dengan continuous batching, concurrency `2-3` kemungkinan meningkatkan throughput.
- Jangan langsung memakai concurrency tinggi karena dapat menyebabkan VRAM penuh.

### 4. Translation batch juga serial

Referensi: `src/pdf_translator/translate.py:369-480`

Sample menggunakan:

```text
4 halaman  -> sekitar 4 batch
10 halaman -> sekitar 6 batch
```

`TRANSLATION_BATCH_SIZE=20` cukup konservatif. Berdasarkan jumlah karakter sample, batch `30` layak diuji:

```powershell
$env:TRANSLATION_BATCH_SIZE = "30"
```

Perkiraan jumlah request:

```text
4 halaman: 4 batch -> sekitar 3 batch
10 halaman: 6 batch -> sekitar 4 batch
```

Jangan langsung menaikkan terlalu besar. Batch besar dapat menyebabkan:

- Structured response lebih sering invalid.
- Output terpotong.
- Retry seluruh batch menjadi lebih mahal.

Solusi yang lebih baik adalah batching berdasarkan estimasi token/karakter, bukan sekadar jumlah unit.

### 5. Translation output cap juga terlalu generik

Saat ini:

```python
MAX_OUTPUT_TOKENS = 10000
```

Referensi: `src/pdf_translator/translate.py:28`

Untuk 20 unit per batch, batas ini kemungkinan jauh lebih besar dari kebutuhan. Prompt translation sendiri bukan masalah besar, tetapi output cap sebaiknya dihitung berdasarkan panjang source batch.

Contoh pendekatan:

```text
estimated_source_tokens = total_characters / 4
output_limit = estimated_source_tokens * 1.5 + safety_margin
```

Dengan minimum dan maksimum, misalnya:

```text
minimum: 512
maximum: 4000
```

### 6. Terlalu banyak debug PNG selalu dibuat

Full pipeline selalu mengaktifkan:

```python
render_plan_debug_dir=layout_output_dir
identity_debug_dir=layout_output_dir / "identity"
```

Referensi: `src/pdf_translator/main.py:136-137`

Untuk setiap halaman, pipeline dapat membuat:

- Layout debug PNG.
- RenderPlan debug PNG.
- Source identity PNG.
- Output identity PNG.
- Source overlay PNG.
- Output overlay PNG.

Identity PNG dibuat dengan skala `1.5`, lihat `src/pdf_translator/identity.py:293-309`.

Untuk dokumen 10 halaman, ini bisa menghasilkan sekitar 60 debug PNG resolusi tinggi, di luar raster vision.

Rekomendasi:

```text
Default:
- Simpan LayoutPlan JSON
- Simpan RenderPlan JSON
- Jangan membuat PNG diagnostic

Dengan --debug-artifacts:
- Buat semua PNG diagnostic dan identity overlay
```

Ini terutama mengurangi CPU dan disk I/O. Kemungkinan tidak menghemat beberapa menit, tetapi dapat menghemat beberapa detik sampai puluhan detik pada PDF kompleks.

### 7. Belum ada cache lintas eksekusi

Workspace sudah memakai source hash:

```text
artifacts/documents/<nama>-<source-hash>/
```

Tetapi full pipeline tetap menjalankan ulang:

- Extraction.
- Vision per halaman.
- Translation.
- Rendering.

Padahal ini mudah dibuat reusable.

Cache yang paling berguna:

- Reuse extraction jika source SHA-256 sama.
- Reuse layout jika source/extraction hash, vision model, dan prompt version sama.
- Reuse translation jika source/extraction hash, model, dan dictionary hash sama.
- Jalankan ulang hanya rendering jika yang berubah adalah renderer.

Untuk rerun dokumen yang sama, ini bisa mengubah waktu dari beberapa menit menjadi hitungan detik.

### 8. Kemungkinan ada model-switching overhead

Vision menggunakan:

```text
VISION_BASE_URL
atau fallback ke LLM_BASE_URL
```

Translation menggunakan:

```text
LLM_BASE_URL
```

Jika `VISION_MODEL` dan `MODEL` berbeda tetapi dijalankan pada server/GPU yang sama, server mungkin:

1. Memuat vision model.
2. Menjalankan seluruh halaman.
3. Unload vision model.
4. Memuat translation model.
5. Menjalankan translation.

Estimasi fixed overhead sekitar `100 detik` sangat mungkin berasal dari model loading atau switching.

Periksa log server untuk pesan model load/unload. Opsi optimasi:

- Biarkan kedua model tetap resident jika VRAM cukup.
- Gunakan server/endpoint terpisah.
- Gunakan satu multimodal model untuk layout dan translation jika kualitasnya cukup.
- Hindari auto-unload antar-request.

## Apakah Memperpendek Prompt Akan Signifikan?

Untuk translation, kemungkinan tidak terlalu signifikan.

`SYSTEM_PROMPT` translation relatif pendek. Dictionary saat ini juga kecil. Mengurangi beberapa kalimat mungkin hanya memberi penghematan kecil.

Untuk vision, yang besar bukan hanya system prompt, tetapi:

- PNG halaman.
- Semua unit halaman.
- Text hint hingga 80 karakter per unit.
- Bbox.
- Source IDs.
- Detail source objects.
- JSON Schema penuh.
- Output LayoutPlan yang sangat verbose.

Jadi optimasi vision sebaiknya fokus pada **compact input dan compact output schema**, bukan sekadar mempersingkat instruksi bahasa Inggris.

## Prioritas Implementasi

Urutan yang paling masuk akal:

1. Tambahkan timing extraction, setiap vision page, translation batch, rendering, dan diagnostic generation.
2. Turunkan `VISION_MAX_OUTPUT_TOKENS` dari `10000` ke `3000`.
3. Buat schema respons vision yang lebih ringkas dan biarkan Python mengisi geometri.
4. Jadikan debug PNG opt-in.
5. Tambahkan cache untuk extraction, layout, dan translation.
6. Uji `TRANSLATION_BATCH_SIZE=30`.
7. Uji vision concurrency `2`.
8. Periksa model loading/switching di server.

## Kesimpulan

Untuk pipeline offline dengan visual layout analysis dan rendering-preserving, **5 menit untuk 10 halaman masih acceptable sebagai batch process**. Untuk penggunaan interaktif, waktunya masih terlalu lama.

Namun kondisi sekarang belum optimal karena:

- `VISION_MAX_OUTPUT_TOKENS=10000` menyebabkan context overflow.
- `35,7%` hasil vision sample dibuang dan diganti fallback.
- Semua vision dan translation request serial.
- Diagnostic PNG selalu dibuat.
- Tidak ada cache lintas run.

Perbaikan paling penting adalah mengecilkan kontrak vision dan mengembalikan output cap ke sekitar `3000`. Itu akan memperbaiki kecepatan sekaligus reliability, bukan hanya mengubah prompt.