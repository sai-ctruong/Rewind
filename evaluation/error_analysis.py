ERROR_CATEGORIES = (
    "wrong video", "right video wrong frame", "object confusion", "action confusion",
    "temporal order error", "OCR failure", "metadata distraction", "Q&A hallucination",
    "missing mp4 limitation", "fallback encoder used",
)


def count_errors(rows):
    return {category: sum(row.get("category") == category for row in rows) for category in ERROR_CATEGORIES}
