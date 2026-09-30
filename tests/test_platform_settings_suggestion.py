"""The suggestion worker reads its five thresholds and `flags.ai_suggest` live.

Every test drives `SuggestionWorker.process` — the real gate sequence — against a published
value, then changes the value in the same test and moves the reader's clock past its TTL. The
worker is never rebuilt.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from shared import platform_settings as ps
from shared.platform_settings import PLATFORM_HASH, PlatformSettings
from suggestion_worker.suggester import SuggestionDecision
from suggestion_worker.worker import SuggestionWorker
from tests.conftest import FakeClock
from tests.test_suggestion_worker import (
    FakeRedis,
    RecordingSuggester,
    approving_suggester,
    build_worker,
    stt_message,
)

ROOM = "room-1"
WS_IN = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"  # flags.ai_suggest bucket 14
WS_OUT = "00000000-0000-0000-0000-000000000001"  # flags.ai_suggest bucket 76
FOUR_WORDS = "chúng ta chốt deadline"  # not a question, 4 words


class SettingsRedis(FakeRedis):
    """The suggestion test double, plus the platform settings hash and the room projection."""

    def __init__(self, policy: bool | None = True) -> None:
        super().__init__(policy=policy)
        self.platform: dict[str, str] = {ps.VERSION_FIELD: "1"}
        self.cooldown_ttls: list[int] = []

    def put(self, key: str, value: Any) -> None:
        self.platform[key] = json.dumps(value)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.platform) if key == PLATFORM_HASH else {}

    async def set_if_absent(self, key: str, value: str, ttl_seconds: int) -> bool:
        if key.startswith("suggest:cd:"):
            self.cooldown_ttls.append(ttl_seconds)
        return await super().set_if_absent(key, value, ttl_seconds)


def _worker(
    clock: FakeClock, suggester: Any = None, **env: Any
) -> tuple[SuggestionWorker, SettingsRedis, RecordingSuggester]:
    worker, _, recorder = build_worker(suggester=suggester, **env)
    redis = SettingsRedis()
    worker.redis = redis  # type: ignore[assignment]
    worker._platform_settings = PlatformSettings(redis, clock=clock)  # type: ignore[arg-type]
    return worker, redis, recorder


def _next_segment(redis: SettingsRedis) -> None:
    """Let the cooldown lapse, as it would between two real segments."""
    redis.values.pop(SuggestionWorker._cooldown_key(ROOM), None)


async def test_nothing_stored_means_the_env_thresholds_apply(fake_clock: FakeClock) -> None:
    worker, _, suggester = _worker(fake_clock, min_words=5)
    await worker.process(b"1-0", stt_message(FOUR_WORDS))
    assert suggester.decide_calls == []

    worker, _, suggester = _worker(fake_clock, min_words=4)
    await worker.process(b"1-0", stt_message(FOUR_WORDS))
    assert len(suggester.decide_calls) == 1


async def test_min_words_is_read_live(fake_clock: FakeClock) -> None:
    worker, redis, suggester = _worker(fake_clock, min_words=4)
    redis.put(ps.SUGGEST_MIN_WORDS, 6)
    await worker.process(b"1-0", stt_message(FOUR_WORDS))
    assert suggester.decide_calls == []

    redis.put(ps.SUGGEST_MIN_WORDS, 2)
    fake_clock.advance(11)
    await worker.process(b"1-1", stt_message(FOUR_WORDS))
    assert len(suggester.decide_calls) == 1


async def test_min_stt_confidence_is_read_live(fake_clock: FakeClock) -> None:
    worker, redis, suggester = _worker(fake_clock, min_stt_confidence=-0.5)
    redis.put(ps.SUGGEST_MIN_STT_CONFIDENCE, -0.2)
    await worker.process(b"1-0", stt_message(confidence=-0.4))
    assert suggester.decide_calls == []

    redis.put(ps.SUGGEST_MIN_STT_CONFIDENCE, -1.0)
    fake_clock.advance(11)
    await worker.process(b"1-1", stt_message(confidence=-0.4))
    assert len(suggester.decide_calls) == 1


async def test_min_confidence_is_read_live(fake_clock: FakeClock) -> None:
    worker, redis, _ = _worker(fake_clock, suggester=approving_suggester())  # model says 0.9
    redis.put(ps.SUGGEST_MIN_CONFIDENCE, 0.95)
    await worker.process(b"1-0", stt_message())
    assert redis.published == []

    redis.put(ps.SUGGEST_MIN_CONFIDENCE, 0.5)
    fake_clock.advance(11)
    await worker.process(b"1-1", stt_message())
    assert redis.published != []


async def test_cooldown_seconds_is_read_live(fake_clock: FakeClock) -> None:
    worker, redis, _ = _worker(fake_clock, suggester=approving_suggester(), cooldown_seconds=20)
    await worker.process(b"1-0", stt_message())
    assert redis.cooldown_ttls == [20]

    redis.put(ps.SUGGEST_COOLDOWN_SECONDS, 90)
    fake_clock.advance(11)
    _next_segment(redis)
    await worker.process(b"1-1", stt_message())
    assert redis.cooldown_ttls == [20, 90]

    redis.put(ps.SUGGEST_COOLDOWN_SECONDS, 0)  # "no cooldown": the shortest TTL Redis takes
    fake_clock.advance(11)
    _next_segment(redis)
    await worker.process(b"1-2", stt_message())
    assert redis.cooldown_ttls == [20, 90, 1]


async def test_max_per_meeting_is_read_live_and_zero_means_none(fake_clock: FakeClock) -> None:
    worker, redis, suggester = _worker(fake_clock, suggester=approving_suggester())
    redis.put(ps.SUGGEST_MAX_PER_MEETING, 0)
    await worker.process(b"1-0", stt_message())
    assert suggester.decide_calls == [], "a cap of 0 must not spend a token"
    assert redis.published == []

    redis.put(ps.SUGGEST_MAX_PER_MEETING, 1)
    fake_clock.advance(11)
    await worker.process(b"1-1", stt_message())
    published_once = len(redis.published)
    assert published_once > 0

    _next_segment(redis)
    await worker.process(b"1-2", stt_message())
    assert len(redis.published) == published_once, "the second suggestion exceeded a cap of 1"

    redis.put(ps.SUGGEST_MAX_PER_MEETING, 5)
    fake_clock.advance(11)
    _next_segment(redis)
    await worker.process(b"1-3", stt_message())
    assert len(redis.published) > published_once


async def test_the_flag_is_a_live_kill_switch(fake_clock: FakeClock) -> None:
    worker, redis, suggester = _worker(fake_clock)
    redis.put(ps.FLAG_AI_SUGGEST, {"enabled": False})
    await worker.process(b"1-0", stt_message())
    assert suggester.decide_calls == []

    redis.put(ps.FLAG_AI_SUGGEST, {"enabled": True})
    fake_clock.advance(11)
    await worker.process(b"1-1", stt_message())
    assert len(suggester.decide_calls) == 1


@pytest.mark.parametrize(("workspace", "expected_calls"), [(WS_IN, 1), (WS_OUT, 0), (None, 0)])
async def test_a_partial_rollout_is_decided_by_the_rooms_workspace(
    fake_clock: FakeClock, workspace: str | None, expected_calls: int
) -> None:
    worker, redis, suggester = _worker(fake_clock)
    if workspace is not None:
        redis.values[f"meeting:room:v2:{ROOM}"] = json.dumps({"WorkspaceId": workspace})
    redis.put(ps.FLAG_AI_SUGGEST, {"enabled": True, "rolloutPercent": 50})

    await worker.process(b"1-0", stt_message())

    assert len(suggester.decide_calls) == expected_calls


async def test_the_env_switch_stays_the_ceiling(fake_clock: FakeClock) -> None:
    worker, redis, suggester = _worker(fake_clock)
    worker.suggestion_settings = worker.suggestion_settings.model_copy(update={"enabled": False})
    redis.put(ps.FLAG_AI_SUGGEST, {"enabled": True, "allowWorkspaces": [WS_IN]})
    redis.values[f"meeting:room:v2:{ROOM}"] = json.dumps({"WorkspaceId": WS_IN})

    await worker.process(b"1-0", stt_message())

    assert suggester.decide_calls == []


async def test_a_settings_outage_changes_nothing(fake_clock: FakeClock) -> None:
    worker, redis, suggester = _worker(
        fake_clock,
        suggester=RecordingSuggester(decision=SuggestionDecision.decline()),
        min_words=4,
    )

    async def broken(key: str) -> dict[str, str]:
        raise ConnectionError("redis down")

    redis.hgetall = broken  # type: ignore[method-assign]
    await worker.process(b"1-0", stt_message(FOUR_WORDS))

    assert len(suggester.decide_calls) == 1
