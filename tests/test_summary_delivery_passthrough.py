"""`delivery` is the backend's routing decision, and this worker must not have an opinion.

The field says where an answer goes: `canonical` replaces the room's summary artifact,
`variant` fills the per-language cache beside it. A Standup-in-Japanese summary is
byte-identical either way, so only the publisher knows which act a request was — which makes
"echo it back untouched" the whole contract, and re-deriving it here a bug that would either
lose a reader's rendering or destroy what the host published.
"""

from __future__ import annotations

from shared.schemas import SummaryRequestMessage, SummaryResultMessage


def test_a_request_without_delivery_means_canonical() -> None:
    # Every request published before the field existed was a rewrite of the room's summary. An
    # in-flight message at deploy time has to keep meaning that.
    message = SummaryRequestMessage.from_redis(
        {"request_id": "r", "room_id": "m", "workspace_id": "w"}
    )
    assert message.delivery == "canonical"


def test_an_empty_delivery_means_canonical_too() -> None:
    # Redis has no null: an unset field arrives as "". Falling back on falsiness rather than on
    # key-presence is what makes both spellings of "nobody said" mean the same thing.
    message = SummaryRequestMessage.from_redis(
        {"request_id": "r", "room_id": "m", "workspace_id": "w", "delivery": ""}
    )
    assert message.delivery == "canonical"


def test_a_variant_request_survives_the_round_trip() -> None:
    message = SummaryRequestMessage.from_redis(
        {"request_id": "r", "room_id": "m", "workspace_id": "w", "delivery": "variant"}
    )
    assert message.delivery == "variant"
    assert message.to_redis()["delivery"] == "variant"


def test_a_result_carries_delivery_on_the_wire() -> None:
    # The backend reads this field off the RESULT, so it has to survive to_redis — a field the
    # model knows about but the serialiser drops would route every rendering to the canonical
    # artifact and overwrite the host's summary.
    result = SummaryResultMessage(
        request_id="r", room_id="m", template_key="standup", status="completed", delivery="variant"
    )
    assert result.to_redis()["delivery"] == "variant"
    assert SummaryResultMessage.from_redis(result.to_redis()).delivery == "variant"


def test_a_result_without_delivery_reads_as_canonical() -> None:
    assert (
        SummaryResultMessage.from_redis(
            {"request_id": "r", "room_id": "m", "template_key": "general", "status": "completed"}
        ).delivery
        == "canonical"
    )
