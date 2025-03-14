import re
from typing import Optional

def extract_translation(text: Optional[str]) -> Optional[str]:
    """
    Processes the input text and extracts the text after "translation:" if it exists,
    returning the original text if not found. Uses regex for case-insensitive matching.
    Returns None if the input is None.

    Args:
        text: The input text string.

    Returns:
        The extracted text after "translation:", or the original text if "translation:" is not found.
        Returns None if the input is None.
    """
    if text is None:
        return None
    regex = r".*?translation:\s*(.+)"

    match = re.search(regex, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return text
