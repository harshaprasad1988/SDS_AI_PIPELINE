from modules.step1_ocr import OCRProcessor
from modules.step2_nlp import NLPExtractor

def test_sample_sds():
    result = NLPExtractor().extract(
        OCRProcessor(engine="auto").process("sample.pdf")
    )
    assert result.supplier_info.product_name
    assert len(result.substances) >= 1
    assert all(s.cas_number for s in result.substances)
