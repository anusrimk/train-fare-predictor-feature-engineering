import re

try:
    import pytesseract
except ImportError:
    pytesseract = None


def ocr_text(image):
    """Extract raw text from a ticket image."""
    if pytesseract is None:
        return ""

    try:
        return pytesseract.image_to_string(image)
    except Exception:
        return ""


def parse_ticket_text(text):
    """Parse common ticket fields from OCR text."""

    result = {
        "origin": "",
        "destination": "",
        "start_date": "",
        "end_date": "",
        "train_type": "",
        "train_class": "",
        "fare": "",
        "price": "",
    }

    if not text:
        return result

    text = text.replace("\r", "")
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Origin / Destination
    m = re.search(
        r"ORIGIN\s+DESTINATION\s*\n\s*([A-Za-z .'-]+)\s+([A-Za-z .'-]+)",
        text,
        re.IGNORECASE
    )

    if m:
        result["origin"] = m.group(1).strip()
        result["destination"] = m.group(2).strip()

    # Departure / Arrival
    m = re.search(
        r"DEPARTURE\s+ARRIVAL\s*\n\s*"
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})",
        text,
        re.IGNORECASE
    )

    if m:
        result["start_date"] = m.group(1)
        result["end_date"] = m.group(2)

    # Train type / class
    m = re.search(
        r"TRAIN TYPE\s+CLASS\s*\n\s*([A-Za-z0-9 .'-]+?)\s+([A-Za-z0-9 .'-]+)",
        text,
        re.IGNORECASE
    )

    if m:
        result["train_type"] = m.group(1).strip()
        result["train_class"] = m.group(2).strip()

    # Fare / price
    m = re.search(
        r"FARE\s+PRICE\s*\n\s*([A-Za-z]+)\s+([\d,.]+)\s*(?:EUR|€)?",
        text,
        re.IGNORECASE
    )

    if m:
        result["fare"] = m.group(1).strip()
        result["price"] = m.group(2).strip()

    return result