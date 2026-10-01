import unicodedata

from security_worker.worker import keywords_present_in, split_claims_by_evidence


def test_keywords_present_in_exact_whole_word() -> None:
    content = "Báo cáo doanh thu và chi phí của công ty."
    # 'thu' should NOT match 'thu' inside 'doanh thu' if the keyword is 'thu thập'
    assert keywords_present_in(content, ["thu thập"]) == ()
    assert keywords_present_in(content, ["doanh thu"]) == ("doanh thu",)


def test_keywords_present_in_avoids_substring_false_positives() -> None:
    # 'cat' in 'category' must NOT trigger DLP
    content = "This document falls under the category of public reports."
    assert keywords_present_in(content, ["cat"]) == ()

    # Independent 'cat' DOES trigger DLP
    content_with_cat = "This document mentions a confidential cat project."
    assert keywords_present_in(content_with_cat, ["cat"]) == ("cat",)


def test_keywords_present_in_multiple_whitespace_and_unicode_nfc() -> None:
    # Multiple spaces between words
    content = "Tài liệu này chứa thông tin mật   quốc   gia bí mật."
    assert keywords_present_in(content, ["thông tin mật quốc gia"]) == ("thông tin mật quốc gia",)

    # NFD decomposed content normalized to NFC
    decomposed_content = unicodedata.normalize("NFD", "bí mật kinh doanh")
    assert keywords_present_in(decomposed_content, ["bí mật kinh doanh"]) == ("bí mật kinh doanh",)


def test_split_claims_by_evidence_verifies_model_claims() -> None:
    content = "Tài liệu nội bộ về kế hoạch tài chính."
    claimed = ("kế hoạch tài chính", "lộ đề thi", "bảo mật")

    supported, unsupported = split_claims_by_evidence(content, claimed)
    assert supported == ("kế hoạch tài chính",)
    assert set(unsupported) == {"lộ đề thi", "bảo mật"}
