"""The stand-in voice belongs to the SPEAKER, not to the listener's language.

WT-528, from production data on 18 Aug (room 01a01542). The rule was
`catalog[hash(speaker) % len(catalog)]`, evaluated against the catalog of whatever language the
listener happened to be reading in, and with no reference to anybody else in the room. Two
defects fell out of that, and the second is worse:

    Tuấn → en 30894953 | → ja 861213b7 | → vi 0e58d60a     one speaker, three voices
    Kỳ   → en 62ae83ad | → ja 498e7f37
    Tú   → en 62ae83ad | → ja 498e7f37                     two speakers, one voice

The first made a man female to anyone listening in another language. The second made two
different people indistinguishable to every English listener and every Japanese listener — in a
conversation product, losing who is speaking is worse than getting their gender wrong.
"""

from __future__ import annotations

from typing import Any

from tts_worker.worker import TTSWorker

# Shaped like the real catalogs, including their lopsided gender split: the Vietnamese library
# offers Lien/Linh/Xia (feminine) against one Minh (masculine), and Japanese five feminine
# against one Naoki. A man therefore had roughly a 5-in-6 chance of being assigned a woman.
VI: list[dict[str, Any]] = [
    {"id": "vi-lien", "name": "Lien", "gender": "feminine"},
    {"id": "vi-linh", "name": "Linh", "gender": "feminine"},
    {"id": "vi-xia", "name": "Xia", "gender": "feminine"},
    {"id": "vi-minh", "name": "Minh", "gender": "masculine"},
]
JA: list[dict[str, Any]] = [
    {"id": "ja-aiko", "name": "Aiko", "gender": "feminine"},
    {"id": "ja-ayumi", "name": "Ayumi", "gender": "feminine"},
    {"id": "ja-haruka", "name": "Haruka", "gender": "feminine"},
    {"id": "ja-keiko", "name": "Keiko", "gender": "feminine"},
    {"id": "ja-sakura", "name": "Sakura", "gender": "feminine"},
    {"id": "ja-naoki", "name": "Naoki", "gender": "masculine"},
]

SPEAKERS = ["tuan", "ky", "tu", "van", "nhi", "speaker-6", "speaker-7"]


def _gender_of(catalog: list[dict[str, Any]], voice_id: str) -> str:
    return next(str(v["gender"]) for v in catalog if v["id"] == voice_id)


def _assigned(catalog: list[dict[str, Any]], roster: list[str], speaker: str) -> str:
    return str(TTSWorker._assign_voice(catalog, sorted(roster), speaker)["id"])


class TestOneSpeakerIsOnePerson:
    def test_gender_does_not_change_with_the_listeners_language(self) -> None:
        """The reported bug: same speaker, female to one listener and male to another.

        A two-person room, which is the shape every report came from and the shape where both
        catalogs have room to honour the choice. The exhausted-pool case is its own test below.
        """
        for speaker in ["ky", "tu"]:
            in_vi = _gender_of(VI, _assigned(VI, ["ky", "tu"], speaker))
            in_ja = _gender_of(JA, _assigned(JA, ["ky", "tu"], speaker))
            assert in_vi == in_ja, f"{speaker} is {in_vi} in Vietnamese but {in_ja} in Japanese"

    def test_being_a_distinct_person_outranks_being_a_consistent_gender(self) -> None:
        """When the pool runs out, gender consistency is what gives way — deliberately.

        Vietnamese offers ONE masculine voice against three feminine, Japanese one against five.
        Put three masculine-assigned speakers in a room and two of them cannot both be masculine
        AND distinct; one has to spill. Sounding like a different person is the property worth
        keeping, because a listener who cannot tell two speakers apart has lost the conversation,
        while a listener who hears the wrong gender has only lost a detail.

        The real remedy is not in this file: these catalogs are ~5:1 feminine, so a masculine
        speaker is assigned a woman most of the time no matter how the picking works. Widening
        the library is what fixes that, and it is tracked separately.
        """
        roster = SPEAKERS[:6]
        assigned = [_assigned(VI, roster, speaker) for speaker in roster]
        assert len(set(assigned)) == len(VI), "every available voice should be in use"

    def test_the_same_room_gives_the_same_answer_every_time(self) -> None:
        first = [_assigned(VI, SPEAKERS, speaker) for speaker in SPEAKERS]
        second = [_assigned(VI, SPEAKERS, speaker) for speaker in SPEAKERS]
        assert first == second

    def test_catalog_order_does_not_change_the_answer(self) -> None:
        """Cartesia's API order and whichever worker warmed the cache must not matter."""
        shuffled = list(reversed(VI))
        assert [_assigned(VI, SPEAKERS, s) for s in SPEAKERS] == [
            _assigned(shuffled, SPEAKERS, s) for s in SPEAKERS
        ]


class TestTwoSpeakersAreTwoPeople:
    def test_nobody_shares_a_voice_while_there_are_voices_left(self) -> None:
        """Kỳ and Tú were assigned one voice in English AND one in Japanese."""
        for catalog in (VI, JA):
            roster = SPEAKERS[: len(catalog)]
            assigned = [_assigned(catalog, roster, speaker) for speaker in roster]
            assert len(set(assigned)) == len(roster)

    def test_a_full_gender_pool_spills_rather_than_duplicates(self) -> None:
        """Vietnamese has one masculine voice. A second speaker routed to it must still be
        distinguishable — being audibly a different person matters more than the gender."""
        roster = SPEAKERS[:4]
        assigned = [_assigned(VI, roster, speaker) for speaker in roster]
        assert len(set(assigned)) == 4

    def test_more_speakers_than_voices_still_answers(self) -> None:
        """Sharing becomes unavoidable; it must not raise, and must stay deterministic."""
        roster = SPEAKERS[:7]
        assigned = [_assigned(VI, roster, speaker) for speaker in roster]
        assert len(assigned) == 7
        assert assigned == [_assigned(VI, roster, speaker) for speaker in roster]


class TestDegenerateCatalogs:
    def test_a_single_gender_catalog_uses_everything(self) -> None:
        """Forcing a gender split with nothing to split would empty the pool."""
        feminine_only = [v for v in JA if v["gender"] == "feminine"]
        roster = SPEAKERS[:5]
        assigned = [_assigned(feminine_only, roster, s) for s in roster]
        assert len(set(assigned)) == 5

    def test_missing_gender_metadata_is_not_fatal(self) -> None:
        catalog: list[dict[str, Any]] = [{"id": "a"}, {"id": "b", "gender": None}]
        assert _assigned(catalog, ["x", "y"], "x") in {"a", "b"}

    def test_a_speaker_the_roster_has_not_caught_up_with_still_gets_a_voice(self) -> None:
        """The languages hash can lag a brand-new joiner. They must not be dubbed silently."""
        assert _assigned(VI, ["someone-else"], "not-in-roster") in {v["id"] for v in VI}
