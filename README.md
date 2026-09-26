# AI based Safety Data Sheet Manager Assist

A Python 3.11 + Streamlit application focused on **Safety Data Sheet (SDS)** extraction.

## Scope

The application intentionally ignores the previous generic MDS/component/compliance sections and extracts only:

### Table 1 — Product & Supplier Information
- Product Name
- Supplier Name (only when available)
- Supplier Address
- Supplier Contact details / phone

### Table 2 — Substance Composition
- Substance
- CAS Number
- Weight Fraction (%)
- Weight (ppm / mg per m3)
- Confidence

## Pipeline

**Step 1 — Document extraction**
- For born-digital PDFs, PyMuPDF/native text extraction is preferred.
- For scanned PDFs/images, Tesseract OCR is used.
- PaddleOCR remains optional for difficult layouts.
- Page boundaries are preserved.

**Step 2 — Targeted SDS NER/table extraction**
- Section 1 is used for product/supplier metadata.
- Section 3 is used for substance/CAS/composition.
- Values are associated by SDS table row, not by a large generic context window.
- Weight-% ranges are preserved exactly.
- ppm is derived from wt-% only when the SDS does not explicitly provide ppm.
- Section 8 exposure limits in mg/m3 are not incorrectly treated as composition weight.

## Important warnings

1. A composition range such as `0.3-2.5%` is not converted to a single invented concentration.
2. Derived ppm is a mathematical conversion (`1 wt-% = 10,000 ppm`), not a measured exposure concentration.
3. `mg/m3` values in Section 8 are exposure limits and are intentionally excluded from the composition table.
4. Trade-secret/proprietary values are never guessed.
5. Low-confidence or missing fields require manual review before safety/regulatory decisions.
6. The application is an extraction assistant and does not replace the supplier SDS or professional regulatory/safety review.

## Run

```bash
conda create -n sds-manager python=3.11 -y
conda activate sds-manager
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run streamlit_app.py
```

For PDF rendering with OCR fallback, install Poppler if it is not already installed.

On macOS with Homebrew:
```bash
brew install poppler tesseract
```
