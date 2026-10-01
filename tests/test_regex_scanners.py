from security_worker.regex_scanners import scan_and_mask_regex_pii


def test_scan_and_mask_regex_pii_clean_text() -> None:
    text = "Bản báo cáo tiến độ dự án WarpTalk quý 3."
    res = scan_and_mask_regex_pii(text)
    assert not res.detected
    assert res.matches_count == 0
    assert res.masked_text == text


def test_scan_and_mask_regex_pii_email() -> None:
    text = "Vui lòng liên hệ test.user@warptalk.ai hoặc admin@company.vn để được hỗ trợ."
    res = scan_and_mask_regex_pii(text)
    assert res.detected
    assert res.matches_count == 2
    assert "test.user@warptalk.ai" not in res.masked_text
    assert "admin@company.vn" not in res.masked_text
    expected = "Vui lòng liên hệ [EMAIL_REDACTED] hoặc [EMAIL_REDACTED] để được hỗ trợ."
    assert res.masked_text == expected


def test_scan_and_mask_regex_pii_vn_phone() -> None:
    text = "Hotline: 0988123456 hoặc +84903456789."
    res = scan_and_mask_regex_pii(text)
    assert res.detected
    assert res.matches_count == 2
    assert "0988123456" not in res.masked_text
    assert "+84903456789" not in res.masked_text
    assert res.masked_text == "Hotline: [PHONE_REDACTED] hoặc [PHONE_REDACTED]."


def test_scan_and_mask_regex_pii_vn_phone_formatted() -> None:
    text = (
        "CSKH 1: 0988 123 456, CSKH 2: +84 903 456 789, CSKH 3: 0912.345.678, CSKH 4: 0977-654-321."
    )
    res = scan_and_mask_regex_pii(text)
    assert res.detected
    assert res.matches_count == 4
    expected = (
        "CSKH 1: [PHONE_REDACTED], CSKH 2: [PHONE_REDACTED], "
        "CSKH 3: [PHONE_REDACTED], CSKH 4: [PHONE_REDACTED]."
    )
    assert res.masked_text == expected


def test_scan_and_mask_regex_pii_cccd() -> None:
    text = "Số định danh cá nhân CCCD: 079198001234 của công dân."
    res = scan_and_mask_regex_pii(text)
    assert res.detected
    assert res.matches_count == 1
    assert "079198001234" not in res.masked_text
    assert res.masked_text == "Số định danh cá nhân CCCD: [ID_REDACTED] của công dân."


def test_scan_and_mask_regex_pii_cccd_formatted() -> None:
    text = "Hồ sơ gồm CCCD 1: 079 198 001234 và CCCD 2: 001-098-001234 và CCCD 3: 079 198 001 234."
    res = scan_and_mask_regex_pii(text)
    assert res.detected
    assert res.matches_count == 3
    expected = "Hồ sơ gồm CCCD 1: [ID_REDACTED] và CCCD 2: [ID_REDACTED] và CCCD 3: [ID_REDACTED]."
    assert res.masked_text == expected


def test_scan_and_mask_regex_pii_credit_card() -> None:
    # Valid Visa card sample for testing Luhn algorithm (4532 0150 0000 0000 -> 4532015000000000)
    # Checksum: 4*2(8)+5+3*2(6)+2+0+1+5*2(1)+0+0+0+0+0+0+0+0+0 = 8+5+6+2+0+1+1+0 = 23 (not valid)
    # Valid Luhn: 49927398716
    # Let's use 49927398716 or calculate valid:
    # 4532-0150-0000-0007 -> 4532015000000007 (checksum: 23+7=30 -> valid Luhn)
    text = "Số thẻ thanh toán: 4532-0150-0000-0007."
    res = scan_and_mask_regex_pii(text)
    assert res.detected
    assert res.matches_count == 1
    assert "4532-0150-0000-0007" not in res.masked_text
    assert res.masked_text == "Số thẻ thanh toán: [CARD_REDACTED]."
