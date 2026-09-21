from pdf_translator.filters import should_translate


def test_identifier_filtering():
    for text in ["123", "A2-A-RF0181-D-F6117-09001-A-001-01", "DI-AUI", "P01583A", "https://example.com"]:
        assert not should_translate(text)


def test_uppercase_words_remain_eligible():
    for text in ["WARNING", "NOTE", "SETTINGS", "USB", "OK", "Server Configuration", "MMS Interface"]:
        assert should_translate(text)


def test_identifier_ratio_threshold_is_conservative():
    assert not should_translate("alpha 123 2026-09-01 A2 555")
    assert should_translate("Note: USB configuration A2-A-RF0181")
