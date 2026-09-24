"""The eval harness's arithmetic, over a tiny inline fixture set. WT-716.

These do not measure the prepass — `tests/test_disfluency.py` does that. They measure the
MEASUREMENT, because a scoreboard nobody has checked is worse than no scoreboard: it still
gets quoted in a review, and a decision gets made from it.

THE ONE THAT MATTERS MOST
    `test_a_wrongly_deleted_token_moves_over_deletion_and_a_missed_filler_does_not`. The ruling
    behind this ticket is faithfulness first — deleting too little beats deleting too much —
    and an F1 that trades a missed "um" against a deleted "not" at par cannot express it. So
    the two kinds of error are pinned to move DIFFERENT numbers, and the one the exit code is
    about only moves for the dangerous kind.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.disfluency_eval import (
    Fixture,
    Report,
    deleted_positions,
    load_fixtures,
    load_tier2,
    main,
    score,
    score_row,
)

# "um" is the only disfluency: 5 tokens, 4 of which must survive.
FILLER = Fixture(
    id="F1",
    language="en",
    raw="um we ship the units",
    expected_clean="We ship the units.",
    kind="filler",
)
CONTROL = Fixture(
    id="C1",
    language="en",
    raw="we ship the units",
    expected_clean="We ship the units.",
    kind="clean_control",
)
QUESTION = Fixture(
    id="Q1",
    language="en",
    raw="um is it ready",
    expected_clean="Is it ready?",
    kind="filler",
)


def _score(outputs: dict[str, str], fixtures: list[Fixture] | None = None) -> Report:
    return score(fixtures if fixtures is not None else [FILLER], outputs, tier="test")


# --- the alignment ----------------------------------------------------------------------------


class TestWhichTokensWentMissing:
    def test_a_deletion_is_found_by_position(self) -> None:
        deleted, broken = deleted_positions("um we ship", "we ship", "en")
        assert deleted == {0}
        assert not broken

    def test_capitalisation_and_terminal_punctuation_are_not_deletions(self) -> None:
        """The two edits the prepass IS allowed to make must score as no change at all."""
        deleted, broken = deleted_positions("um we ship", "We ship.", "en")
        assert deleted == {0}
        assert not broken

    def test_a_repeated_token_resolves_leftmost_on_both_sides(self) -> None:
        # "the the contract" losing one "the": whichever position the walk picks, it picks the
        # same one for the expected and the actual text, so the choice cancels out.
        deleted, _ = deleted_positions("the the contract", "the contract", "en")
        assert deleted == {1}

    def test_a_word_that_was_never_there_is_reported_not_scored(self) -> None:
        _, broken = deleted_positions("we ship", "we ship it", "en")
        assert broken

    def test_a_filler_only_line_deletes_everything(self) -> None:
        deleted, broken = deleted_positions("Ummm", "", "en")
        assert deleted == {0}
        assert not broken


# --- the metrics ------------------------------------------------------------------------------


class TestTheArithmetic:
    def test_a_perfect_answer_scores_perfectly(self) -> None:
        report = _score({"F1": "We ship the units."})

        assert report.overall.over_deletion_rate == 0.0
        assert report.overall.precision == 1.0
        assert report.overall.recall == 1.0
        assert report.overall.f1 == 1.0
        assert report.overall.exact_rate == 1.0
        assert report.overall.invariant_violations == 0

    def test_a_wrongly_deleted_token_moves_over_deletion_and_a_missed_filler_does_not(
        self,
    ) -> None:
        """THE ASYMMETRY. Both runs are imperfect; only one of them is dangerous."""
        # Deleted "the" as well as the filler: 1 of the 4 tokens that should have survived.
        over = _score({"F1": "We ship units."})
        # Left the "um" in: nothing was lost, the line is just untidy.
        under = _score({"F1": "um we ship the units"})

        assert over.overall.over_deletions == 1
        assert over.overall.over_deletion_rate == pytest.approx(0.25)
        assert under.overall.over_deletions == 0
        assert under.overall.over_deletion_rate == 0.0

        # AND F1 GETS THE ORDER BACKWARDS. It scores the harmless run (a filler left in) WORSE
        # than the harmful one (a real word deleted), because it counts a missed deletion and a
        # wrong deletion as the same kind of mistake. That is precisely why the exit code reads
        # over-deletion and not F1 — and why no recall floor belongs in this harness.
        assert over.overall.f1 < 1.0
        assert under.overall.f1 < over.overall.f1

    def test_over_deletion_is_measured_against_the_tokens_that_should_have_survived(self) -> None:
        """Not against every token: the denominator excludes the disfluency itself."""
        report = _score({"F1": "We ship units."})

        assert report.overall.tokens == 5
        assert report.overall.survivors == 4
        assert report.overall.over_deletion_rate == pytest.approx(1 / 4)

    def test_precision_and_recall_split_the_two_errors(self) -> None:
        report = _score({"F1": "We ship units."})

        assert report.overall.true_positives == 1
        assert report.overall.precision == pytest.approx(0.5)  # 1 of 2 deletions was right
        assert report.overall.recall == 1.0  # the filler did go

    def test_exact_match_is_the_whole_string(self) -> None:
        """A right set of deletions with the wrong punctuation is not an exact match."""
        report = _score({"F1": "We ship the units"})

        assert report.overall.over_deletions == 0
        assert report.overall.exact_rate == 0.0

    def test_question_punctuation_accuracy(self) -> None:
        asked = score([QUESTION], {"Q1": "Is it ready?"}, tier="t")
        flattened = score([QUESTION], {"Q1": "Is it ready."}, tier="t")

        assert asked.overall.question_accuracy == 1.0
        assert flattened.overall.question_accuracy == 0.0

    def test_an_invariant_violation_is_counted(self) -> None:
        # Deleting the negation is the failure the invariants exist to catch.
        negation = Fixture(
            id="N1",
            language="en",
            raw="um we do not ship",
            expected_clean="We do not ship.",
            kind="filler",
        )
        report = score([negation], {"N1": "We do ship."}, tier="t")

        assert report.overall.invariant_violations >= 1
        assert "I2_negation_count" in report.rows[0].invariants

    def test_a_fixture_the_tier_did_not_answer_is_named_not_scored(self) -> None:
        report = _score({})

        assert report.missing == ["F1"]
        assert report.overall.fixtures == 0

    def test_a_fixture_whose_own_label_is_not_a_deletion_is_refused(self) -> None:
        broken = Fixture(
            id="B1", language="en", raw="we ship", expected_clean="we ship it", kind="filler"
        )

        with pytest.raises(ValueError, match="not a deletion"):
            score_row(broken, "we ship")

    def test_slices_are_kept_per_language_and_per_kind(self) -> None:
        vietnamese = Fixture(
            id="V1", language="vi", raw="ờ để em xem", expected_clean="Để em xem.", kind="filler"
        )
        report = _score(
            {"F1": "We ship the units.", "V1": "Để em xem.", "C1": "We ship the units."},
            [FILLER, vietnamese, CONTROL],
        )

        assert set(report.by_language) == {"en", "vi"}
        assert report.by_language["en"].fixtures == 2
        assert set(report.by_kind) == {"filler", "clean_control"}


# --- the gates --------------------------------------------------------------------------------


class TestWhatFailsARun:
    def test_a_clean_run_fails_nothing(self) -> None:
        report = _score({"F1": "We ship the units."})

        assert report.failures(max_over_deletion=0.005) == []

    def test_over_deletion_above_the_ceiling_fails(self) -> None:
        report = _score({"F1": "We ship units."})

        assert any("over-deletion" in reason for reason in report.failures(0.005))

    def test_over_deletion_below_the_ceiling_does_not(self) -> None:
        report = _score({"F1": "We ship units."})  # 25%

        assert report.failures(max_over_deletion=0.30) == []

    def test_a_clean_control_that_lost_a_token_fails_at_any_ceiling(self) -> None:
        """A sentence with nothing wrong in it losing a word is never an acceptable trade."""
        report = _score({"C1": "We ship units."}, [CONTROL])

        assert report.clean_control_regressions == 1
        assert any("clean_control" in reason for reason in report.failures(1.0))

    def test_an_invariant_violation_fails_at_any_ceiling(self) -> None:
        negation = Fixture(
            id="N1",
            language="en",
            raw="we do not ship",
            expected_clean="We do not ship.",
            kind="clean_control",
        )
        report = score([negation], {"N1": "We do ship."}, tier="t")

        assert any("invariant" in reason for reason in report.failures(1.0))


# --- loading, and the command itself ----------------------------------------------------------


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    return path


class TestLoadingFixtures:
    def test_a_file_of_rows_becomes_fixtures(self, tmp_path: Path) -> None:
        file = _write(
            tmp_path / "en.jsonl",
            [
                {
                    "id": "E1",
                    "language": "en",
                    "raw": "um we ship",
                    "expected_clean": "We ship.",
                    "kind": "filler",
                }
            ],
        )

        fixtures = load_fixtures([file])

        assert [f.id for f in fixtures] == ["E1"]
        assert fixtures[0].standalone_turn is None  # absent means "let the prepass infer"

    def test_turn_context_is_carried_when_a_fixture_states_it(self, tmp_path: Path) -> None:
        file = _write(
            tmp_path / "en.jsonl",
            [
                {
                    "id": "E2",
                    "language": "en",
                    "raw": "Uh-huh.",
                    "expected_clean": "Uh-huh.",
                    "kind": "trap",
                    "prev_turn_is_question": True,
                    "standalone_turn": True,
                }
            ],
        )

        fixture = load_fixtures([file])[0]

        assert fixture.prev_turn_is_question is True
        assert fixture.standalone_turn is True

    def test_a_directory_loads_every_jsonl_in_it(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "en.jsonl",
            [
                {
                    "id": "E1",
                    "language": "en",
                    "raw": "um we ship",
                    "expected_clean": "We ship.",
                    "kind": "filler",
                }
            ],
        )
        _write(
            tmp_path / "vi.jsonl",
            [
                {
                    "id": "V1",
                    "language": "vi",
                    "raw": "ờ để em xem",
                    "expected_clean": "Để em xem.",
                    "kind": "filler",
                }
            ],
        )

        assert len(load_fixtures([tmp_path])) == 2

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            ({"id": "E1", "language": "en", "raw": "x", "kind": "filler"}, "expected_clean"),
            (
                {
                    "id": "E1",
                    "language": "en",
                    "raw": "x",
                    "expected_clean": "x",
                    "kind": "fillers",
                },
                "not one of",
            ),
        ],
    )
    def test_a_malformed_row_raises_rather_than_being_skipped(
        self, tmp_path: Path, row: dict[str, object], message: str
    ) -> None:
        """A silently dropped fixture is a case that stops being checked unnoticed."""
        file = _write(tmp_path / "en.jsonl", [row])

        with pytest.raises(ValueError, match=message):
            load_fixtures([file])

    def test_a_duplicate_id_raises(self, tmp_path: Path) -> None:
        row = {
            "id": "E1",
            "language": "en",
            "raw": "um we ship",
            "expected_clean": "We ship.",
            "kind": "filler",
        }
        file = _write(tmp_path / "en.jsonl", [row, row])

        with pytest.raises(ValueError, match="duplicate"):
            load_fixtures([file])

    def test_a_tier2_dump_is_read_in_either_shape(self, tmp_path: Path) -> None:
        mapping = tmp_path / "a.json"
        mapping.write_text(json.dumps({"E1": "We ship."}), encoding="utf-8")
        listed = tmp_path / "b.json"
        listed.write_text(
            json.dumps([{"id": "E1", "clean_text": "We ship."}]),
            encoding="utf-8",
        )

        assert load_tier2(mapping) == {"E1": "We ship."}
        assert load_tier2(listed) == {"E1": "We ship."}


class TestTheCommand:
    def test_the_seeded_fixtures_pass_their_own_gates(self, capsys: pytest.CaptureFixture) -> None:
        """The harness is meant to be adoptable by CI unchanged, so the seed set must be green."""
        assert main([]) == 0
        assert "OVER-DELETION" in capsys.readouterr().out

    def test_json_output_leads_with_the_over_deletion_rate(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        main(["--json"])

        payload = json.loads(capsys.readouterr().out)
        overall = payload["reports"][0]["overall"]
        assert overall["overDeletionRate"] == 0.0
        assert payload["reports"][0]["failures"] == []

    def test_a_language_filter_scores_only_that_language(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        main(["--json", "--language", "vi"])

        payload = json.loads(capsys.readouterr().out)
        assert list(payload["reports"][0]["byLanguage"]) == ["vi"]

    def test_the_exit_code_is_non_zero_when_the_ceiling_is_exceeded(self, tmp_path: Path) -> None:
        """End to end: a fixture that INSISTS the "um" is a word, so the prepass over-deletes.

        A deliberately wrong label is the only way to make the real prepass fail this gate,
        which is itself the finding — and it exercises the whole command, not just the maths.
        """
        file = _write(
            tmp_path / "en.jsonl",
            [
                {
                    "id": "X1",
                    "language": "en",
                    "raw": "um we ship",
                    "expected_clean": "um we ship",
                    "kind": "trap",
                }
            ],
        )

        assert main(["--fixtures", str(file), "--quiet"]) == 1
        # 1 of the 3 tokens the label says must survive: under a 50% ceiling the same run is
        # a pass, so the exit code is reading the threshold and not just the count.
        assert main(["--fixtures", str(file), "--max-over-deletion", "50", "--quiet"]) == 0

    def test_no_fixtures_is_an_error_not_a_pass(self) -> None:
        assert main(["--language", "de"]) == 2
