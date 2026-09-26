"""
STEP 1 - Safety Data Sheet document extraction.

Hybrid strategy:
1. Try native PDF text extraction first (best for born-digital SDS PDFs).
2. Fall back to OCR for scanned/image PDFs.
3. Preserve page boundaries so Step 2 can target SDS sections reliably.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import re
import numpy as np
from PIL import Image, ImageEnhance
from pdf2image import convert_from_path
import cv2
import pytesseract

logger = logging.getLogger(__name__)

@dataclass
class PageOCRResult:
    page_number: int
    raw_text: str
    confidence: float
    words: List[dict] = field(default_factory=list)
    tables_detected: bool = False
    extraction_method: str = "ocr"

@dataclass
class OCRResult:
    file_path: str
    total_pages: int
    pages: List[PageOCRResult]
    full_text: str
    avg_confidence: float
    engine_used: str
    document_text_source: str = "ocr"

def preprocess_image(pil_image: Image.Image) -> Image.Image:
    gray = pil_image.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    arr = np.array(gray)
    arr = cv2.fastNlMeansDenoising(arr, h=8)
    return Image.fromarray(arr)

def _ocr_tesseract(image: Image.Image, lang="eng") -> PageOCRResult:
    config = r"--oem 3 --psm 6"
    data = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )
    words, confs = [], []
    for i, raw_conf in enumerate(data["conf"]):
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            continue
        word = (data["text"][i] or "").strip()
        if conf >= 0 and word:
            words.append({
                "word": word, "confidence": conf,
                "bbox": {
                    "x": data["left"][i], "y": data["top"][i],
                    "w": data["width"][i], "h": data["height"][i]
                }
            })
            confs.append(conf)
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    return PageOCRResult(
        page_number=0, raw_text=text,
        confidence=float(np.mean(confs)) if confs else 0.0,
        words=words,
        tables_detected=_looks_like_table(text),
        extraction_method="tesseract"
    )

def _looks_like_table(text: str) -> bool:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    table_headers = ("chemical name", "cas no", "cas number", "weight-%", "weight fraction")
    return any(sum(h in x.lower() for h in table_headers) >= 2 for x in lines) or text.count("|") >= 3

def _native_pdf_text(path: Path):
    """Return per-page native text if PyMuPDF is available and text is usable."""
    try:
        import fitz
    except ImportError:
        return None
    doc = fitz.open(str(path))
    pages = []
    total_chars = 0
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        total_chars += len(text.strip())
        pages.append(text)
    doc.close()
    if not pages or total_chars < 150:
        return None
    return pages

class OCRProcessor:
    def __init__(self, engine="auto", dpi=300, lang="eng"):
        # auto = native PDF text when available, OCR otherwise
        self.engine = engine
        self.dpi = dpi
        self.lang = lang

    def process(self, file_path: str) -> OCRResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(file_path)

        if path.suffix.lower() == ".pdf" and self.engine in ("auto", "native", "tesseract"):
            native = _native_pdf_text(path)
            if native and self.engine in ("auto", "native"):
                pages = [
                    PageOCRResult(i + 1, txt, 100.0, [], _looks_like_table(txt), "native_pdf")
                    for i, txt in enumerate(native)
                ]
                full = "\n\n--- PAGE BREAK ---\n\n".join(p.raw_text for p in pages)
                return OCRResult(str(path), len(pages), pages, full, 100.0, "native_pdf", "native_pdf")

        images = self._load_images(path)
        results = []
        for i, image in enumerate(images):
            processed = preprocess_image(image)
            if self.engine == "paddleocr":
                result = self._paddle_or_tesseract(processed)
            elif self.engine in ("qwen", "qwen-vl"):
                result = self._qwen_or_tesseract(processed)
            else:
                result = _ocr_tesseract(processed, self.lang)
            result.page_number = i + 1
            results.append(result)

        full = "\n\n--- PAGE BREAK ---\n\n".join(p.raw_text for p in results)
        avg = float(np.mean([p.confidence for p in results])) if results else 0.0
        return OCRResult(str(path), len(results), results, full, avg, self.engine, "ocr")

    def _paddle_or_tesseract(self, image):
        try:
            from paddleocr import PaddleOCR
            engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            result = engine.ocr(np.array(image), cls=True)
            lines, words, confs = [], [], []
            if result and result[0]:
                for item in result[0]:
                    bbox, (text, conf) = item
                    lines.append(text)
                    words.append({"word": text, "confidence": conf * 100, "bbox": bbox})
                    confs.append(conf * 100)
            return PageOCRResult(0, "\n".join(lines),
                                 float(np.mean(confs)) if confs else 0.0,
                                 words, _looks_like_table("\n".join(lines)), "paddleocr")
        except Exception:
            return _ocr_tesseract(image, self.lang)

    def _qwen_or_tesseract(self, image):
        """Qwen-VL OCR via DashScope (Alibaba Cloud Model Studio).

        Requires the DASHSCOPE_API_KEY environment variable. Falls back to
        Tesseract if the API is unavailable or the call fails.
        """
        try:
            import base64, io, os
            from dashscope import MultiModalConversation

            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                raise RuntimeError("DASHSCOPE_API_KEY environment variable not set")

            model = getattr(self, "qwen_model", "qwen-vl-max-latest")

            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

            messages = [{
                "role": "user",
                "content": [
                    {"image": data_url},
                    {"text": "Extract ALL text from this Safety Data Sheet page, "
                             "preserving layout, tables and field labels. "
                             "Return only the extracted text."},
                ],
            }]
            resp = MultiModalConversation.call(model=model, messages=messages,
                                               api_key=api_key)
            if resp.status_code != 200:
                raise RuntimeError(f"Qwen API error {resp.status_code}: {getattr(resp, 'message', '')}")

            content = resp.output.choices[0].message.content
            text = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            ) if isinstance(content, list) else str(content)
            text = text.strip()
            if not text:
                raise RuntimeError("Qwen returned empty text")

            # Qwen does not provide per-word confidence; use a high fixed value.
            return PageOCRResult(0, text, 95.0, [], _looks_like_table(text), "qwen-vl")
        except Exception as exc:
            logger.warning(f"Qwen OCR failed ({exc}); falling back to Tesseract.")
            return _ocr_tesseract(image, self.lang)

    def _load_images(self, path: Path):
        if path.suffix.lower() == ".pdf":
            return convert_from_path(str(path), dpi=self.dpi)
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}:
            return [Image.open(str(path))]
        raise ValueError(f"Unsupported file type: {path.suffix}")
