# Streamlit UI — MDS Checker Step 1 & Step 2

## What this UI does

1. Upload an MDS PDF or image.
2. **Step 1 — OCR/Vision:** converts PDF pages to images, preprocesses them and extracts text with Tesseract or PaddleOCR.
3. **Step 2 — NLP/NER:** extracts:
   - Substance name
   - CAS number
   - Weight fraction (%)
   - Weight (ppm)
   - Component
   - Supplier
   - Confidence
   - Extraction warnings
4. Download the Step 2 entity table as CSV.

## Run

From the `mds_checker` folder:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Tesseract prerequisite

Tesseract OCR must be installed on the machine and available on PATH.

### PDF prerequisite

`pdf2image` requires Poppler. On macOS, install Poppler with Homebrew:

```bash
brew install poppler tesseract
```

### Optional spaCy model

The updated Step 2 automatically falls back to a rule-based spaCy pipeline if no English spaCy model is installed. For better generic NER, install:

```bash
python -m spacy download en_core_web_sm
```

For the original transformer pipeline:

```bash
python -m spacy download en_core_web_trf
```

## Expected Step 2 output

For a row such as:

`Lead | 7439-92-1 | 0.002%`

Step 2 should produce approximately:

| Substance | CAS Number | Weight Fraction (%) | Weight (ppm) |
|---|---|---:|---:|
| Lead | 7439-92-1 | 0.002 | 20 |

The extracted data is intentionally passed as an `NLPResult` so it can be connected directly to the existing Step 3 `DataStructurer`.
