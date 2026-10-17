"""
guard/normalize.py - Text normalization and obfuscation flagging.
Strips zero-width characters and ASCII/Unicode control characters,
applies Unicode NFKC normalization, and flags (without decoding)
long base64-like strings and HTML comments.
"""

import re
import unicodedata
from typing import List, Tuple

# Zero-width spaces, joiners, direction marks, and invisible formatting characters
ZERO_WIDTH_RE = re.compile(
    r"[\u200B-\u200D\uFEFF\u200E\u200F\u2060\u202A-\u202E]"
)

# Control characters (excluding standard whitespace \t, \n, \r)
CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)

# Long base64-like string detection (20+ base64 chars, with optional padding or note prefix)
BASE64_RE = re.compile(
    r"(?:Configuration\s+note\s*\(base64\):\s*)?[A-Za-z0-9+/]{20,}={0,2}"
)

# HTML comment detection
HTML_COMMENT_RE = re.compile(
    r"<!--.*?-->",
    re.DOTALL
)


def normalize(text: str) -> Tuple[str, List[str]]:
    """
    Normalize text and detect obfuscation / injection flags.

    Operations:
      1. Detect flags: long base64 strings and HTML comments (not decoded or removed).
      2. Strip zero-width characters (e.g. U+200B, U+FEFF).
      3. Strip control characters (excluding standard whitespace).
      4. Apply Unicode NFKC normalization.

    Returns:
      (clean_text, flags): Tuple containing the cleaned string and a list of detected flag names.
    """
    if not text:
        return "", []

    flags: List[str] = []

    # 1. Flag HTML comments
    if HTML_COMMENT_RE.search(text):
        flags.append("html_comment")

    # 2. Flag long base64-like strings (do not decode)
    if BASE64_RE.search(text):
        flags.append("base64")

    # 3. Detect and strip zero-width characters
    if ZERO_WIDTH_RE.search(text):
        flags.append("zero_width")
        text = ZERO_WIDTH_RE.sub("", text)

    # 4. Detect and strip control characters
    if CONTROL_CHARS_RE.search(text):
        flags.append("control_chars")
        text = CONTROL_CHARS_RE.sub("", text)

    # Remove any additional Unicode 'Cc' category characters except \t, \n, \r
    filtered_chars = []
    has_other_control = False
    for ch in text:
        if ch in ("\t", "\n", "\r"):
            filtered_chars.append(ch)
        elif unicodedata.category(ch) == "Cc":
            has_other_control = True
        else:
            filtered_chars.append(ch)

    if has_other_control and "control_chars" not in flags:
        flags.append("control_chars")

    text = "".join(filtered_chars)

    # 5. Apply Unicode NFKC normalization
    clean_text = unicodedata.normalize("NFKC", text)

    return clean_text, sorted(list(set(flags)))
