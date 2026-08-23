"""WT-529: a summary that attributed decisions to `speaker-019f0d00-…`."""

from types import SimpleNamespace

from ai_assistant_worker.speaker_names import SpeakerNamer, parse_speaker_names

IDENTITY = "speaker-019f0d00-0de0-7000-9000-000000000003"
BARE = "019f0d00-0de0-7000-9000-000000000003"


class TestParseSpeakerNames:
    def test_reads_the_published_map(self):
        assert parse_speaker_names('{"a": "Ngọc Kỳ"}') == {"a": "Ngọc Kỳ"}

    def test_reads_it_as_bytes_too(self):
        assert parse_speaker_names(b'{"a": "Kh\xe1\xbb\xb3"}') == {"a": "Khỳ"}

    def test_a_missing_key_is_no_names_rather_than_a_crash(self):
        # The whole point of the pseudonym fallback: this case is normal, not exceptional.
        assert parse_speaker_names(None) == {}
        assert parse_speaker_names("") == {}
        assert parse_speaker_names(b"") == {}

    def test_a_broken_write_is_no_names_rather_than_a_lost_summary(self):
        # Failing here would trade an ugly summary for no summary at all.
        assert parse_speaker_names("{not json") == {}
        assert parse_speaker_names('["a", "b"]') == {}
        assert parse_speaker_names("null") == {}

    def test_blank_and_non_string_names_are_dropped(self):
        # A blank name resolves to nothing useful; letting it through would print an empty
        # speaker label, which reads as a transcript line nobody spoke.
        parsed = parse_speaker_names('{"a": "  ", "b": null, "c": 7, "d": " Tú "}')
        assert parsed == {"d": "Tú"}


class TestSpeakerNamer:
    def test_a_known_speaker_is_named(self):
        namer = SpeakerNamer({IDENTITY: "Ngọc Kỳ"})
        assert namer.name_for(IDENTITY) == "Ngọc Kỳ"

    def test_the_identity_prefix_is_matched_either_way(self):
        # The STT stream carries the LiveKit identity; a roster keyed by user id carries the
        # bare uuid. Accepting both removes a whole class of "the map matched nothing".
        assert SpeakerNamer({BARE: "Ngọc Kỳ"}).name_for(IDENTITY) == "Ngọc Kỳ"
        assert SpeakerNamer({IDENTITY: "Ngọc Kỳ"}).name_for(BARE) == "Ngọc Kỳ"

    def test_an_unknown_speaker_becomes_a_readable_pseudonym_not_a_uuid(self):
        # The reported bug. A uuid in a summary is not a degraded name — it is noise that
        # makes the summary unusable.
        namer = SpeakerNamer({})
        assert namer.name_for(IDENTITY) == "Speaker 1"
        assert BARE not in namer.name_for(IDENTITY)

    def test_pseudonyms_are_stable_across_the_whole_transcript(self):
        # "Speaker 2" has to mean the same person on every line, or the summary invents a
        # conversation that did not happen.
        namer = SpeakerNamer({})
        assert namer.name_for("a") == "Speaker 1"
        assert namer.name_for("b") == "Speaker 2"
        assert namer.name_for("a") == "Speaker 1"
        assert namer.name_for("b") == "Speaker 2"

    def test_numbering_only_counts_the_unknown(self):
        # A named speaker must not consume a pseudonym number, or the first unknown person
        # in a mostly-named room is introduced as "Speaker 4".
        namer = SpeakerNamer({"known": "Tú"})
        assert namer.name_for("known") == "Tú"
        assert namer.name_for("stranger") == "Speaker 1"

    def test_a_named_and_an_unnamed_speaker_coexist(self):
        namer = SpeakerNamer({IDENTITY: "Ngọc Kỳ"})
        assert namer.name_for(IDENTITY) == "Ngọc Kỳ"
        assert namer.name_for("speaker-other") == "Speaker 1"

    def test_a_blank_id_does_not_inflate_the_speaker_count(self):
        # Two blank ids are not evidence of two people.
        namer = SpeakerNamer({})
        assert namer.name_for("") == "Unknown speaker"
        assert namer.name_for("   ") == "Unknown speaker"
        assert namer.name_for("real") == "Speaker 1"

    def test_it_is_callable_so_it_can_be_passed_as_a_formatter(self):
        namer = SpeakerNamer({IDENTITY: "Ngọc Kỳ"})
        assert namer(IDENTITY) == "Ngọc Kỳ"

    def test_whitespace_around_an_id_still_matches(self):
        assert SpeakerNamer({IDENTITY: "Ngọc Kỳ"}).name_for(f"  {IDENTITY} ") == "Ngọc Kỳ"


class TestTheSummariserActuallyUsesIt:
    """The wiring, not the resolver — the fix is worthless if the worker never calls it.

    Asserted on the transcript the summariser is HANDED, because that string is the whole
    interface between this bug and the model: whatever is in it is what the model can repeat.
    """

    @staticmethod
    def _worker(published: dict[str, str] | None):
        from unittest.mock import AsyncMock

        from ai_assistant_worker.worker import AIAssistantWorker

        worker = AIAssistantWorker.__new__(AIAssistantWorker)
        worker._transcripts = {
            "m1": [
                (IDENTITY, "we should ship on Friday", 1000),
                ("speaker-nobody-knows", "agreed", 2000),
            ]
        }

        async def _get(key: str):
            return None

        async def _hgetall(key: str):
            if key.endswith(":speaker_names") and published is not None:
                return {k.encode(): v.encode() for k, v in published.items()}
            return {}

        worker.redis = SimpleNamespace(
            get=AsyncMock(side_effect=_get),
            hgetall=AsyncMock(side_effect=_hgetall),
            hset=AsyncMock(),
        )
        worker.logger = SimpleNamespace(
            info=lambda *a, **k: None,
            debug=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
        worker.publish = AsyncMock()

        captured: dict[str, str] = {}

        async def _summarize(transcript_text: str, **kwargs):
            captured["transcript"] = transcript_text
            return "a summary"

        async def _extract(transcript_text: str, **kwargs):
            return []

        async def _structured(transcript_text: str, **kwargs):
            captured["transcript"] = transcript_text
            return {"insufficientData": False, "sections": []}

        worker._require_assistant = lambda: SimpleNamespace(
            summarize=_summarize,
            extract_action_items=_extract,
            generate_structured_summary=_structured,
        )
        return worker, captured

    async def test_a_published_name_reaches_the_model_instead_of_the_uuid(self):
        worker, captured = self._worker({IDENTITY: "Ngọc Kỳ"})

        await worker._generate_summary("m1")

        assert "Ngọc Kỳ" in captured["transcript"]
        # The reported symptom, asserted directly.
        assert IDENTITY not in captured["transcript"]
        assert BARE not in captured["transcript"]

    async def test_with_no_published_names_the_uuid_still_never_reaches_the_model(self):
        # The fallback is the half that matters in production today: nothing publishes this
        # map for older rooms, and a uuid in the prompt is the bug regardless of why.
        worker, captured = self._worker(None)

        await worker._generate_summary("m1")

        assert IDENTITY not in captured["transcript"]
        assert "Speaker 1" in captured["transcript"]
        assert "Speaker 2" in captured["transcript"]

    async def test_an_unreadable_map_does_not_lose_the_summary(self):
        worker, captured = self._worker({"": "", "x": "  "})

        await worker._generate_summary("m1")

        assert captured["transcript"]
        assert IDENTITY not in captured["transcript"]


class TestParseSpeakerNamesFromTheHash:
    """`hgetall` hands back bytes on both sides — the shape actually in production."""

    def test_reads_bytes_keys_and_values(self):
        assert parse_speaker_names({IDENTITY.encode(): "Ngọc Kỳ".encode()}) == {IDENTITY: "Ngọc Kỳ"}

    def test_an_empty_hash_is_no_names(self):
        assert parse_speaker_names({}) == {}

    def test_a_field_written_blank_is_dropped_not_shown_as_a_speaker(self):
        # A blank label renders as a transcript line nobody spoke.
        assert parse_speaker_names({b"a": b"   ", b"b": b"T\xc3\xba"}) == {"b": "Tú"}

    def test_undecodable_bytes_lose_that_field_only(self):
        parsed = parse_speaker_names({b"a": b"\xff\xfe", b"b": b"Nhi"})
        assert parsed == {"b": "Nhi"}
