"""In-meeting clones are deleted from the Cartesia account once nothing can reach them.

Until this existed, nothing in the repository ever called `voices.delete`. Every meeting left
one voice per cloned speaker in the account, and every upgrade left the one it replaced — with
the only pointer to either (`voice:{meeting}:{speaker}`) expiring after 12h and taking the id
with it. The account was the one place the leak was visible, and nothing looked there.

The sweep decides on a name and an age, because this service has no database. These tests pin
both halves, and in particular pin the thing that must NEVER happen: deleting an upload-made
voice, whose id lives in voice_profiles.provider_voice_id where this side cannot see it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from tts_worker.worker import TTSWorker


class _Redis:
    """Redis reduced to the one call the sweep makes: claiming this cycle."""

    def __init__(self, *, lock_free: bool = True) -> None:
        self.lock_free = lock_free
        self.claims: list[tuple[str, str, int]] = []

    async def set_if_absent(self, key: str, value: str, ttl_seconds: int) -> bool:
        self.claims.append((key, value, ttl_seconds))
        return self.lock_free


class _Cartesia:
    def __init__(self, voices: list[dict[str, Any]], *, delete_ok: bool = True) -> None:
        self._voices = voices
        self._delete_ok = delete_ok
        self.listed = 0
        self.deleted: list[str] = []

    async def list_owned_voices(self, max_scanned: int = 5000) -> list[dict[str, Any]]:
        self.listed += 1
        return self._voices

    async def delete_voice(self, voice_id: str) -> bool:
        self.deleted.append(voice_id)
        return self._delete_ok


def _worker(cartesia: _Cartesia, redis: _Redis | None = None, **overrides: Any) -> TTSWorker:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings(**overrides)
    worker.logger = MagicMock()
    worker._consumer_name = "tts-test"
    worker.worker_name = "tts"
    worker.cartesia = cartesia  # type: ignore[assignment]
    worker.redis = redis or _Redis()  # type: ignore[assignment]
    return worker


def _voice(name: str, age_hours: float, voice_id: str = "") -> dict[str, Any]:
    return {
        "id": voice_id or f"id-{name}",
        "name": name,
        "created_at": datetime.now(UTC) - timedelta(hours=age_hours),
    }


@pytest.mark.asyncio
async def test_deletes_an_in_meeting_clone_past_the_cutoff() -> None:
    cartesia = _Cartesia([_voice("speaker-abc12345-def67890", age_hours=48)])
    worker = _worker(cartesia)

    await worker._sweep_orphan_voices_once()

    assert cartesia.deleted == ["id-speaker-abc12345-def67890"]


@pytest.mark.asyncio
async def test_never_deletes_an_uploaded_profile_voice_however_old() -> None:
    """The one deletion that cannot be undone from this side.

    An uploaded voice's id is stored in voice_profiles.provider_voice_id by AuthService, and a
    person picked it in the dub-voice picker. This worker has no database and could not tell
    afterwards that it had destroyed a voice somebody chose.
    """
    cartesia = _Cartesia([_voice("profile-abc12345", age_hours=24 * 365)])
    worker = _worker(cartesia)

    await worker._sweep_orphan_voices_once()

    assert cartesia.deleted == []


@pytest.mark.asyncio
async def test_keeps_a_clone_that_is_still_young_enough_to_be_in_use() -> None:
    cartesia = _Cartesia([_voice("speaker-abc12345-def67890", age_hours=2)])
    worker = _worker(cartesia)

    await worker._sweep_orphan_voices_once()

    assert cartesia.deleted == []


@pytest.mark.asyncio
async def test_leaves_voices_it_did_not_name_alone() -> None:
    """An unrecognised name was not made by this worker, so it is not this worker's to delete."""
    cartesia = _Cartesia([_voice("Brooke - Big Sister", age_hours=24 * 365)])
    worker = _worker(cartesia)

    await worker._sweep_orphan_voices_once()

    assert cartesia.deleted == []


@pytest.mark.asyncio
async def test_keeps_a_clone_whose_age_is_unreadable() -> None:
    """No age means no proof it is unreachable, and the fail-safe direction is to keep it.

    Keeping a voice costs account storage. Deleting one a meeting is still speaking through
    costs the meeting.
    """
    cartesia = _Cartesia([{"id": "id-x", "name": "speaker-abc12345-def67890", "created_at": None}])
    worker = _worker(cartesia)

    await worker._sweep_orphan_voices_once()

    assert cartesia.deleted == []


@pytest.mark.asyncio
async def test_treats_a_naive_timestamp_as_utc_rather_than_raising() -> None:
    """Comparing a naive datetime against an aware one raises — which would abort the sweep."""
    cartesia = _Cartesia(
        [
            {
                "id": "id-naive",
                "name": "speaker-abc12345-def67890",
                "created_at": datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=48),
            }
        ]
    )
    worker = _worker(cartesia)

    await worker._sweep_orphan_voices_once()

    assert cartesia.deleted == ["id-naive"]


@pytest.mark.asyncio
async def test_sorts_a_mixed_account_in_one_pass() -> None:
    cartesia = _Cartesia(
        [
            _voice("speaker-old00001-meeting1", age_hours=48),
            _voice("speaker-old00002-meeting2", age_hours=100),
            _voice("speaker-new00003-meeting3", age_hours=1),
            _voice("profile-keep0001", age_hours=48),
            _voice("Some Library Voice", age_hours=48),
        ]
    )
    worker = _worker(cartesia)

    await worker._sweep_orphan_voices_once()

    assert cartesia.deleted == ["id-speaker-old00001-meeting1", "id-speaker-old00002-meeting2"]


@pytest.mark.asyncio
async def test_another_replica_holding_the_cycle_stops_the_pass_before_cartesia() -> None:
    """The lock is what keeps a fleet-wide restart from walking the account once per replica."""
    cartesia = _Cartesia([_voice("speaker-abc12345-def67890", age_hours=48)])
    worker = _worker(cartesia, _Redis(lock_free=False))

    await worker._sweep_orphan_voices_once()

    assert cartesia.listed == 0
    assert cartesia.deleted == []


@pytest.mark.asyncio
async def test_the_claim_is_held_for_a_whole_interval() -> None:
    redis = _Redis()
    worker = _worker(_Cartesia([]), redis, orphan_voice_sweep_interval_seconds=1234)

    await worker._sweep_orphan_voices_once()

    assert redis.claims == [("voice:orphan_sweep:lock", "tts-test", 1234)]


@pytest.mark.asyncio
async def test_a_cartesia_listing_failure_is_a_quiet_no_op() -> None:
    """list_owned_voices answers [] on any failure — the sweep must read that as nothing to do."""
    cartesia = _Cartesia([])
    worker = _worker(cartesia)

    await worker._sweep_orphan_voices_once()

    assert cartesia.deleted == []


@pytest.mark.asyncio
async def test_a_refused_delete_does_not_stop_the_rest_of_the_pass() -> None:
    cartesia = _Cartesia(
        [
            _voice("speaker-aaa00001-meeting1", age_hours=48),
            _voice("speaker-bbb00002-meeting2", age_hours=48),
        ],
        delete_ok=False,
    )
    worker = _worker(cartesia)

    await worker._sweep_orphan_voices_once()

    assert cartesia.deleted == ["id-speaker-aaa00001-meeting1", "id-speaker-bbb00002-meeting2"]
