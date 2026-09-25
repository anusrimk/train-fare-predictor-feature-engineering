import re

try:
    import pytesseract
except ImportError:
    pytesseract = None


def ocr_text(image):
    """Extract raw text from a ticket image using Tesseract OCR."""

    if pytesseract is None:
        raise ImportError(
            "pytesseract is not installed. Run: pip install pytesseract"
        )

    return pytesseract.image_to_string(image)


def parse_ticket_text(text):
    """Convert OCR text into structured railway-ticket fields."""

    result = {
        "origin": "",
        "destination": "",
        "start_date": "",
        "end_date": "",
        "train_type": "",
        "train_class": "",
        "fare": "",
        "price": ""
    }

    if not text:
        return result

    text = text.replace("\r", "\n")

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    # ORIGIN / DESTINATION
    for i, line in enumerate(lines):

        if "ORIGIN" in line.upper() and "DESTINATION" in line.upper():

            if i + 1 < len(lines):

                parts = lines[i + 1].split()

                if len(parts) >= 2:
                    result["origin"] = parts[0]
                    result["destination"] = " ".join(parts[1:])

            break

    # DEPARTURE / ARRIVAL
    for i, line in enumerate(lines):

        if "DEPARTURE" in line.upper() and "ARRIVAL" in line.upper():

            if i + 1 < len(lines):

                next_line = lines[i + 1]

                dates = re.findall(
                    r"\d{4}-\d{2}-\d{2}"
                    r"(?:\s+\d{2}:\d{2}(?::\d{2})?)?",
                    next_line
                )

                if len(dates) >= 1:
                    result["start_date"] = dates[0]

                if len(dates) >= 2:
                    result["end_date"] = dates[1]

            break

    # TRAIN TYPE / CLASS
    for i, line in enumerate(lines):

        if "TRAIN TYPE" in line.upper() and "CLASS" in line.upper():

            if i + 1 < len(lines):

                parts = lines[i + 1].split()

                if len(parts) >= 1:
                    result["train_type"] = parts[0]

                if len(parts) >= 2:
                    result["train_class"] = " ".join(parts[1:])

            break

    # FARE / PRICE
    for i, line in enumerate(lines):

        if "FARE" in line.upper() and "PRICE" in line.upper():

            if i + 1 < len(lines):

                fare_line = lines[i + 1]

                price_match = re.search(
                    r"(\d+(?:[.,]\d+)?)\s*(?:EUR|INR|₹|\$)?",
                    fare_line,
                    re.IGNORECASE
                )

                if price_match:
                    result["price"] = price_match.group(1)

                fare = re.sub(
                    r"\d+(?:[.,]\d+)?\s*(?:EUR|INR|₹|\$)?",
                    "",
                    fare_line,
                    flags=re.IGNORECASE
                ).strip()

                result["fare"] = fare

            break

    return result