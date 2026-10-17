"""
eval/baseline.py - Heuristic baseline scanner using regex and keyword rules.
Detects tool description poisoning, instructions to ignore previous context,
secrecy demands ("do not tell the user"), <important> tags, sensitive paths (.ssh, .env),
base64 blobs, and pre-execution actions ("before calling/using" + file/send).
"""

import base64
import re
from typing import Dict, List, Set, Tuple

# Pre-compiled regex patterns for each heuristic rule category
RULES: Dict[str, Tuple[re.Pattern, float]] = {
    # 1. Ignore previous instructions
    "ignore_previous": (
        re.compile(
            r"(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above)\b"
            r"|\bom[íi]tala\b"
            r"|\bignorez-la\b"
            r"|\bskip\s*it\s+and\s+reply\b",
            re.IGNORECASE,
        ),
        0.55,
    ),
    # 2. Secrecy demands ("do not tell the user")
    "do_not_tell_user": (
        re.compile(
            r"(?:do\s+not|don'?t|never|without)\s+(?:tell|mention(?:\s+it|\s+this)?\s+to|notify|informing|reveal\s+to)\s+(?:the\s+)?user\b"
            r"|\bdo\s+not\s+mention\b"
            r"|\bwithout\s+notifying\s+the\s+user\b"
            r"|\bno\s+(?:se\s+lo\s+mencione|le\s+diga|mencione)\s+al\s+usuario\b"
            r"|\b(?:sans\s+en\s+informer|ne\s+le\s+mentionnez\s+pas|ne\s+dites\s+pas)\b"
            r"|\bleave\s+the\s+original\s+recipient\s+out\s+of\s+the\s+visible\s+message\b",
            re.IGNORECASE,
        ),
        0.60,
    ),
    # 3. System / Important tags (<important>, [important], etc.)
    "important_tags": (
        re.compile(
            r"<\s*important\s*>"
            r"|\[\s*important\s*\]"
            r"|<\s*system\s*>"
            r"|\[\s*system\s*\]"
            r"|\bIMPORTANT:",
            re.IGNORECASE,
        ),
        0.50,
    ),
    # 4. Sensitive credential and configuration paths (.ssh, .env, id_rsa, fake_secrets)
    "sensitive_paths": (
        re.compile(
            r"\.ssh\b"
            r"|\.env\b"
            r"|\bid_rsa\b"
            r"|fake_secrets\b"
            r"|/etc/(?:passwd|shadow)\b",
            re.IGNORECASE,
        ),
        0.65,
    ),
    # 5. Base64 blobs (e.g. Configuration note base64 or long base64 payload)
    "base64_blobs": (
        re.compile(
            r"Configuration\s+note\s*\(base64\):\s*[A-Za-z0-9+/=]+"
            r"|\b[A-Za-z0-9+/]{25,}={0,2}\b",
            re.IGNORECASE,
        ),
        0.55,
    ),
    # 6. "Before calling/using" followed by file or send action
    "before_calling_action": (
        re.compile(
            r"(?:before|avant|antes)\s+(?:calling|using|running|executing|responding|d['’]ex[ée]cuter|d['’]utiliser|de\s+responder|de\s+ejecutar)\b"
            r".*?\b(?:read|file|send|fetch|open|save|transmit|stolen|exfil|silently|lisez|envoyez|guarde|lea)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        0.60,
    ),
    # 7. Exfiltration destinations and hidden parameters
    "exfil_destinations": (
        re.compile(
            r"@records\.example\b"
            r"|\bApproved by Records Team\b"
            r"|\bsession_token\b"
            r"|\bclave\s+api\b"
            r"|\bcontrase[ñn]a\b"
            r"|\bmot\s+de\s+passe\b"
            r"|\baudit-copy\b"
            r"|\bops-monitor\b"
            r"|pass\s+the\s+full\s+text\s+of\s+the\s+conversation\b"
            r"|history\s*argument\b"
            r"|remplissez\s+l['’]argument\s+history\b"
            r"|transmettez\s+le\s+texte\s+int[ée]gral\b"
            r"|complete\s+el\s+argumento\s+history\b"
            r"|pase\s+el\s+texto\s+completo\b",
            re.IGNORECASE,
        ),
        0.60,
    ),
    # 8. Tool shadowing and behavioral redirection
    "shadowing_redirection": (
        re.compile(
            r"when\s+the\s+(?:read_file|send_email|calculator|save_note)\s+tool\s+is\s+(?:called|used)\b"
            r"|\binstead\s+of\s+the\s+path\s+the\s+user\s+gave\b"
            r"|\badd\s+1\s+to\s+its\s+result\s+before\s+showing\b"
            r"|\bwhenever\s+the\s+user\s+asks\s+you\s+to\s+save\s+something\s+with\s+save_note\b",
            re.IGNORECASE,
        ),
        0.55,
    ),
}


def _deobfuscate_text(text: str) -> Tuple[str, bool]:
    """
    Check for obfuscation techniques (zero-width spaces, base64 payloads, spaced letters)
    and return (normalized_combined_text, has_zero_width_spaces).
    """
    has_zw = "\u200b" in text
    clean = text.replace("\u200b", "")

    # Extract and decode base64 configuration notes if present
    decoded_b64 = ""
    b64_match = re.search(r"Configuration note \(base64\):\s*([A-Za-z0-9+/=]+)", clean)
    if b64_match:
        try:
            raw_b64 = b64_match.group(1).strip()
            decoded_b64 = base64.b64decode(raw_b64).decode("utf-8", errors="ignore")
        except Exception:
            decoded_b64 = ""

    # Collapse spaced-out letters (e.g. "f a k e _ s e c r e t s" -> "fake_secrets")
    combined = f"{clean} {decoded_b64}"
    unspaced = re.sub(r"(\S)\s(?=\S(?:\s|$))", r"\1", combined)

    return f"{combined} {unspaced}", has_zw


def is_poisoned_score(text: str) -> float:
    """
    Evaluate text with regex and keyword rules to detect poisoned tool definitions.
    Returns a score between 0.0 (completely benign) and 1.0 (highly poisoned).
    """
    if not text or not text.strip():
        return 0.0

    normalized_text, has_zw = _deobfuscate_text(text)
    matched_rules: Set[str] = set()
    matched_weights: List[float] = []

    # Check each heuristic rule
    for rule_name, (pattern, weight) in RULES.items():
        if pattern.search(normalized_text):
            matched_rules.add(rule_name)
            matched_weights.append(weight)

    # Zero-width spaces are an explicit indicator of obfuscation
    if has_zw:
        matched_rules.add("zero_width_spaces")
        matched_weights.append(0.50)

    # If no rules triggered, return clean score 0.0
    if not matched_weights:
        return 0.0

    # Strongest rule sets base score, additional rules incrementally increase confidence
    base_score = max(matched_weights)
    bonus = 0.10 * (len(matched_rules) - 1)
    final_score = min(1.0, base_score + bonus)

    return round(final_score, 4)
