"""The parts of the second-pass transcriber that do not need a network.

The decoding prompt is the cheapest accuracy this pass can buy — on the sample clips, one glossary
term turned "Edoa Conan" into "Edogawa Conan" — and it is also the thing `stt_worker/model.py`
documents production RECITING into a transcript on marginal audio. So it is bounded, and the bound
is tested.
"""

from __future__ import annotations

from types import SimpleNamespace

from retranscribe_worker.batch_transcriber import MAX_PROMPT_CHARS, _read_segments, build_prompt


def test_speaker_names_come_before_glossary_terms():
    # A mis-heard proper noun is the error people notice, and the one a glossary is least likely
    # to already contain.
    prompt = build_prompt(["Kubernetes", "Qdrant"], speakers=["Huỳnh Thái Tú"])

    assert prompt.startswith("Huỳnh Thái Tú")
    assert "Kubernetes" in prompt


def test_duplicates_are_dropped_case_insensitively():
    prompt = build_prompt(["Qdrant", "qdrant", "QDRANT"])

    assert prompt == "Qdrant"


def test_the_prompt_is_bounded_and_never_cut_mid_term():
    # Half a name is a worse hint than no name.
    terms = [f"term{index}-{'x' * 40}" for index in range(50)]

    prompt = build_prompt(terms)

    assert len(prompt) <= MAX_PROMPT_CHARS
    assert not prompt.endswith("x" * 5) or prompt.split(", ")[-1] in terms


def test_blank_terms_contribute_nothing():
    assert build_prompt(["", "   ", "Qdrant"]) == "Qdrant"
    assert build_prompt([]) == ""


def test_segments_are_read_with_their_times_and_confidence():
    response = SimpleNamespace(
        segments=[
            SimpleNamespace(start=1.5, end=3.25, text=" xin chào ", avg_logprob=-0.4),
        ]
    )

    segments = _read_segments(response)

    assert len(segments) == 1
    assert segments[0].start_ms == 1500
    assert segments[0].end_ms == 3250
    assert segments[0].text == "xin chào"
    assert segments[0].confidence == -0.4


def test_a_missing_confidence_stays_unknown():
    # WT-277's rule: never coalesced to a number, because a fabricated value is indistinguishable
    # from a real one.
    response = SimpleNamespace(segments=[SimpleNamespace(start=0, end=1, text="a")])

    assert _read_segments(response)[0].confidence is None


def test_a_model_without_timestamps_yields_one_untimed_segment():
    # gpt-transcribe rejects verbose_json outright, so this is the shape its answer arrives in.
    # Untimed rather than absent: the text is real and losing it would be the worse error.
    response = SimpleNamespace(text="Xin chào tất cả mọi người.")

    segments = _read_segments(response)

    assert len(segments) == 1
    assert segments[0].text == "Xin chào tất cả mọi người."
    assert segments[0].start_ms == 0 and segments[0].end_ms == 0


def test_an_empty_answer_is_no_segments_rather_than_an_empty_line():
    assert _read_segments(SimpleNamespace(text="   ")) == []
    assert _read_segments(SimpleNamespace(segments=[], text=None)) == []


def test_blank_segments_are_skipped():
    response = SimpleNamespace(
        segments=[
            SimpleNamespace(start=0, end=1, text="  "),
            SimpleNamespace(start=1, end=2, text="thật"),
        ]
    )

    assert [segment.text for segment in _read_segments(response)] == ["thật"]


def test_a_dict_response_reads_the_same_as_an_object():
    # The one place a provider SDK change lands; being strict here would silently re-transcribe a
    # meeting to nothing.
    response = {"segments": [{"start": 0, "end": 1, "text": "xin chào", "avg_logprob": -0.2}]}

    assert _read_segments(response)[0].text == "xin chào"


def test_unreadable_times_fall_back_to_zero_rather_than_raising():
    response = SimpleNamespace(segments=[SimpleNamespace(start="?", end=None, text="a")])

    segments = _read_segments(response)
    assert segments[0].start_ms == 0 and segments[0].end_ms == 0
