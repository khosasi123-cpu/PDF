import re

IDENTIFIER_RATIO_THRESHOLD = 0.80
_TOKEN_PATTERN = re.compile(r"\S+")
_IDENTIFIER_SYMBOLS = set("_\\/-.: ")


def _is_identifier_token(token: str) -> bool:
    return any(character.isdigit() or character in _IDENTIFIER_SYMBOLS for character in token)


def should_translate(text: str) -> bool:
    """Return whether text contains enough natural-language content to translate."""
    tokens = _TOKEN_PATTERN.findall(text.strip())
    if not tokens:
        return False
    identifier_count = sum(_is_identifier_token(token) for token in tokens)
    return identifier_count / len(tokens) < IDENTIFIER_RATIO_THRESHOLD
