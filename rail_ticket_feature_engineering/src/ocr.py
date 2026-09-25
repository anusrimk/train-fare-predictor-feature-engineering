"""Optional OCR scaffold.

OCR is deliberately conservative because ticket layouts vary. The Streamlit app
keeps the extracted values editable instead of pretending OCR is always correct.
"""
import re
try:
    import pytesseract
except Exception:
    pytesseract=None

def ocr_text(image):
    if pytesseract is None:
        return "OCR dependency unavailable"
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"OCR failed: {e}"
