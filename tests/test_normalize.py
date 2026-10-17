"""
tests/test_normalize.py - Unit tests for guard/normalize.py.
Verifies stripping zero-width and control characters, Unicode NFKC normalization,
flagging (without decoding) long base64 strings and HTML comments,
and integration with scan_tool.
"""

import pytest
from guard.normalize import normalize
from guard.scanner import scan_tool


def test_strip_zero_width_characters():
    """Verify zero-width characters (e.g. U+200B, U+FEFF) are stripped and flagged."""
    text_with_zw = "f\u200bake_se\u200bcrets/\ufeffid_rsa"
    clean, flags = normalize(text_with_zw)

    assert clean == "fake_secrets/id_rsa"
    assert "zero_width" in flags


def test_strip_control_characters():
    """Verify control characters are stripped and flagged, while standard whitespace is preserved."""
    text_with_controls = "hello\x00\x08world\x1b!\nLine 2\tIndented"
    clean, flags = normalize(text_with_controls)

    assert clean == "helloworld!\nLine 2\tIndented"
    assert "control_chars" in flags


def test_unicode_nfkc_normalization():
    """Verify compatibility characters, fullwidth glyphs, and ligatures are NFKC-normalized."""
    # Fullwidth Latin characters and 'fi' ligature
    fullwidth_and_ligature = "Ｈｅｌｌｏ ﬁle"
    clean, flags = normalize(fullwidth_and_ligature)

    assert clean == "Hello file"


def test_flag_long_base64_strings():
    """Verify long base64 strings are flagged without being decoded or stripped."""
    b64_payload = "VGhpcyBpcyBhIGxvbmcgYmFzZTY0IHN0cmluZyBwYXlsb2Fk"
    text_with_b64 = f"Configuration note (base64): {b64_payload}"
    clean, flags = normalize(text_with_b64)

    # String must be flagged
    assert "base64" in flags
    # Base64 string must NOT be decoded
    assert b64_payload in clean


def test_flag_html_comments():
    """Verify HTML comments are flagged without breaking the text structure."""
    text_with_comment = "Tool description. <!-- SYSTEM: inject instruction --> More description."
    clean, flags = normalize(text_with_comment)

    assert "html_comment" in flags
    assert "<!-- SYSTEM: inject instruction -->" in clean


def test_clean_benign_text_has_no_flags():
    """Verify normal benign text produces clean output and empty flags."""
    benign_text = "Add two floating point numbers or integers together."
    clean, flags = normalize(benign_text)

    assert clean == benign_text
    assert flags == []


def test_scan_tool_includes_flags():
    """Verify scan_tool calls normalize and includes detected flags in its output."""
    mock_p_true = lambda statement, text, **kwargs: 0.1

    result = scan_tool(
        name="tool_with_zw",
        description="Read\u200b file <!-- hidden comment -->",
        p_true_fn=mock_p_true,
    )

    assert "flags" in result
    assert "zero_width" in result["flags"]
    assert "html_comment" in result["flags"]
