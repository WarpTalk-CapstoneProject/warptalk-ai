"""Parsing what a model returns when asked for one fact.

Every case here is a real failure mode, not a hypothetical: models answer in fenced JSON,
answer in prose, invent categories, and — for the majority of chunks in any real document —
correctly have nothing to say.
"""

from embedding_worker.facts import (
    DEFAULT_CATEGORY,
    FACT_CATEGORIES,
    MAX_FACT_CHARS,
    build_extraction_prompt,
    fact_payload_fields,
    parse_fact_response,
)


def test_a_clean_answer_is_taken_as_given() -> None:
    parsed = parse_fact_response('{"fact": "Payment terms are net 30", "category": "requirement"}')
    assert parsed == {"fact": "Payment terms are net 30", "category": "requirement"}


def test_a_fenced_answer_is_still_an_answer() -> None:
    # Models wrap JSON in ```json often enough that not handling it is a bug, not a nicety.
    parsed = parse_fact_response(
        '```json\n{"fact": "Retention is 30 days", "category": "risk"}\n```'
    )
    assert parsed is not None
    assert parsed["fact"] == "Retention is 30 days"


def test_no_fact_is_a_valid_answer() -> None:
    # Most chunks of most documents are page headers and filler. A table of invented facts
    # about boilerplate is worse than a shorter table.
    assert parse_fact_response('{"fact": "", "category": "reference"}') is None
    assert parse_fact_response('{"fact": "   "}') is None
    assert parse_fact_response("") is None
    assert parse_fact_response(None) is None


def test_prose_instead_of_json_yields_nothing_rather_than_garbage() -> None:
    assert parse_fact_response("Sure! Here is the fact you asked for.") is None
    assert parse_fact_response("[1, 2, 3]") is None


def test_an_unknown_category_does_not_cost_a_good_fact() -> None:
    parsed = parse_fact_response('{"fact": "The API rate limit is 300/min", "category": "banana"}')
    assert parsed is not None
    assert parsed["fact"] == "The API rate limit is 300/min"
    assert parsed["category"] == DEFAULT_CATEGORY


def test_a_long_fact_is_cut_to_fit_a_table_row() -> None:
    parsed = parse_fact_response('{"fact": "' + "x" * 1000 + '", "category": "definition"}')
    assert parsed is not None
    assert len(parsed["fact"]) <= MAX_FACT_CHARS


def test_a_chunk_without_a_fact_carries_no_keys() -> None:
    # Not a null the reader has to interpret — the absence is the statement.
    assert fact_payload_fields(None) == {}
    assert fact_payload_fields({"fact": "A", "category": "risk"}) == {
        "fact": "A",
        "fact_category": "risk",
    }


def test_the_prompt_names_every_category_it_will_accept() -> None:
    # A prompt that offers a category the parser rejects would silently downgrade every use
    # of it to the default, which looks like the model ignoring instructions.
    prompt = build_extraction_prompt()
    for category in FACT_CATEGORIES:
        assert category in prompt
