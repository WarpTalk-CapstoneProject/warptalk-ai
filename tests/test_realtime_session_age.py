"""A Realtime socket is retired before OpenAI closes it, not after it fails.

Production, 2026-09-08 17:33Z, the only meeting held after that day's release. The first audio
chunk of the meeting came back

    1001 (going away) Your session hit the maximum duration of 60 minutes

because the STT warm pool was still handing out sockets it had opened at worker startup, four
hours and forty minutes earlier. Twenty seconds later the translation pool did the same thing
four times over and fell back to chat completions for each one.

Neither pool was broken in a way either pool could see. STT evicted on IDLENESS, translation
evicted on FAILURE, and a healthy, recently-used, 61-minute-old connection is neither. The cost
landed on whoever spoke first: ~6 of the ~10 seconds before the meeting's first caption.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.openai_options import REALTIME_SESSION_MAX_AGE_S, realtime_session_expired
from stt_worker.model import OpenAISTT
from translation_worker.translator import OpenAITranslator


class TestExpiryRule:
    def test_a_fresh_socket_is_usable(self):
        assert realtime_session_expired(time.monotonic()) is False

    def test_a_socket_past_the_cutoff_is_not(self):
        assert realtime_session_expired(time.monotonic() - REALTIME_SESSION_MAX_AGE_S - 1) is True

    def test_the_cutoff_leaves_headroom_under_the_providers_hour(self):
        # The point of the number: retiring early costs a background reconnect, arriving late
        # costs somebody's first sentence.
        assert REALTIME_SESSION_MAX_AGE_S < 60 * 60

    def test_an_unstamped_socket_is_treated_as_expired(self):
        # "Nobody wrote down when this was opened" is not evidence that it is young.
        assert realtime_session_expired(None) is True


class _Conn:
    def __init__(self) -> None:
        self.session = MagicMock()
        self.session.update = AsyncMock()


class _Manager:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn
        self.closed = False

    async def __aenter__(self) -> _Conn:
        return self._conn

    async def __aexit__(self, *exc) -> bool:
        self.closed = True
        return False


def _stt() -> OpenAISTT:
    stt = OpenAISTT.__new__(OpenAISTT)
    stt._sessions = {}
    stt._warm_target = 0
    stt._client = MagicMock()
    return stt


class TestSttWarmPool:
    @pytest.mark.asyncio
    async def test_an_aged_socket_is_discarded_rather_than_handed_over(self):
        from collections import deque

        old_conn, live_conn = _Conn(), _Conn()
        old_manager = _Manager(old_conn)
        stt = _stt()
        stt._warm_sessions = deque(
            [
                {
                    "manager": old_manager,
                    "conn": old_conn,
                    "opened_at": time.monotonic() - REALTIME_SESSION_MAX_AGE_S - 60,
                },
                {"manager": _Manager(live_conn), "conn": live_conn, "opened_at": time.monotonic()},
            ]
        )

        claimed = await stt._claim_warm_socket()

        assert claimed is not None
        assert claimed["conn"] is live_conn, "the expired socket was handed to a speaker"
        # Closed, not merely dropped — an abandoned manager leaks the connection.
        await asyncio.sleep(0)
        assert old_manager.closed is True

    @pytest.mark.asyncio
    async def test_a_pool_of_nothing_but_aged_sockets_claims_nothing(self):
        from collections import deque

        conn = _Conn()
        stt = _stt()
        stt._warm_sessions = deque(
            [
                {
                    "manager": _Manager(conn),
                    "conn": conn,
                    "opened_at": time.monotonic() - REALTIME_SESSION_MAX_AGE_S - 1,
                }
            ]
        )

        # None, so the caller opens a fresh one instead of using a dead socket.
        assert await stt._claim_warm_socket() is None


class TestSttSessionSweep:
    @pytest.mark.asyncio
    async def test_a_busy_session_is_still_retired_on_age(self):
        # The case the idle sweep alone could never catch: used seconds ago, over an hour old.
        # Async because the sweep closes through asyncio.create_task, which needs a live loop.
        stt = _stt()
        now = time.monotonic()
        stt._sessions = {
            ("m1", "s1"): {
                "manager": _Manager(_Conn()),
                "conn": _Conn(),
                "last_used": now,
                "opened_at": now - REALTIME_SESSION_MAX_AGE_S - 5,
            }
        }

        stt._sweep_idle_sessions()
        await asyncio.sleep(0)

        assert ("m1", "s1") not in stt._sessions

    @pytest.mark.asyncio
    async def test_a_young_session_in_use_is_left_alone(self):
        stt = _stt()
        now = time.monotonic()
        stt._sessions = {
            ("m1", "s1"): {
                "manager": _Manager(_Conn()),
                "conn": _Conn(),
                "last_used": now,
                "opened_at": now,
            }
        }
        stt._sweep_idle_sessions()
        assert ("m1", "s1") in stt._sessions


class _PoolConn:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class TestTranslatorPool:
    def _translator(self) -> OpenAITranslator:
        translator = OpenAITranslator.__new__(OpenAITranslator)
        translator.realtime_pool_size = 1
        translator._realtime_connections = [None]
        translator._realtime_opened_at = [None]
        translator._realtime_available = asyncio.Queue()
        translator._realtime_available.put_nowait(0)
        translator._realtime_connect_lock = asyncio.Lock()
        return translator

    @pytest.mark.asyncio
    async def test_an_aged_connection_is_replaced_before_it_is_used(self):
        aged, fresh = _PoolConn(), _PoolConn()
        translator = self._translator()
        translator._realtime_connections = [aged]
        translator._realtime_opened_at = [time.monotonic() - REALTIME_SESSION_MAX_AGE_S - 30]
        translator._connect_realtime = AsyncMock(return_value=fresh)

        index, connection = await translator._acquire_realtime_connection()

        assert connection is fresh, "a translation was handed a socket OpenAI had already closed"
        assert aged.closed is True
        assert translator._realtime_opened_at[index] is not None

    @pytest.mark.asyncio
    async def test_a_young_connection_is_reused(self):
        # The pooling this fix must not undo: a healthy connection is still worth keeping.
        live = _PoolConn()
        translator = self._translator()
        translator._realtime_connections = [live]
        translator._realtime_opened_at = [time.monotonic()]
        translator._connect_realtime = AsyncMock(side_effect=AssertionError("reconnected anyway"))

        _, connection = await translator._acquire_realtime_connection()

        assert connection is live
        assert live.closed is False


# ── A dead socket is not a verdict on the model ──────────────────────────────────────────
#
# `session.update` is the first thing sent down a freshly claimed socket, so it is also where an
# already-closed one surfaces. Read as "the model refused these fields", that walks the whole
# degrade ladder and writes the bare rung's verdict into a PROCESS-WIDE memo — one stale socket
# degrading every session the worker opens afterwards.
#
# The 2026-09-08 meeting logged `session_optional_fields_rejected` four times and
# `stt_session_capability_downgraded` — the log that means a model actually refused something —
# zero times. All four were dead sockets.


class _ClosedConnectionError(Exception):
    """Stands in for websockets' ConnectionClosedOK, matched by class-name prefix."""


# The predicate matches on the exception's class NAME, so the stand-in has to carry the
# provider's. (The class itself is named for ruff's N818, which is about source style.)
_ClosedConnectionError.__name__ = "ConnectionClosedOK"


class TestConnectionErrorIsNotACapabilityVerdict:
    def test_a_provider_close_is_recognised_by_its_wording(self):
        from stt_worker.model import _is_connection_error

        assert _is_connection_error(
            RuntimeError("received 1001 (going away) Your session hit the maximum duration")
        )

    def test_a_real_parameter_rejection_is_not(self):
        from stt_worker.model import _is_connection_error

        assert not _is_connection_error(
            RuntimeError("Unknown parameter: 'keywords' is not supported for this model.")
        )

    @pytest.mark.asyncio
    async def test_a_dead_socket_does_not_degrade_the_model_memo(self):
        from stt_worker.model import (
            OpenAISTT,
            _supports_structured_context,
            reset_capability_memo,
        )

        reset_capability_memo()
        stt = OpenAISTT.__new__(OpenAISTT)
        stt.model = "some-capable-model"
        stt.noise_reduction = "off"

        conn = MagicMock()
        conn.session.update = AsyncMock(
            side_effect=_ClosedConnectionError("received 1001 (going away) ... 60 minutes.")
        )

        with pytest.raises(_ClosedConnectionError):
            await stt._degrade_session_config(conn, "vi", "topic", {"vi"}, ["Codex"])

        # Untouched: nothing was learned, because nothing was tested.
        assert _supports_structured_context("some-capable-model") is True
