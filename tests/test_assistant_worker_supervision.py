"""One worker's death must not be its siblings' death.

Three features share the ai-assistant container: meeting summarisation, the WarpBot chat
assistant, and on-demand re-summarisation. They were started with a bare `asyncio.gather`,
which has no `return_exceptions` — so the first worker to raise brought the process down and
took the other two with it. A misconfigured summariser could remove WarpBot from the product,
and nothing on screen would say why.
"""

from __future__ import annotations

import asyncio

import pytest

import ai_assistant_worker.__main__ as entry


@pytest.mark.asyncio
async def test_a_crashing_worker_is_retried_rather_than_fatal(monkeypatch) -> None:
    monkeypatch.setattr(entry, "RESTART_DELAY_SECONDS", 0.001)
    attempts = 0

    class Flaky:
        async def start(self) -> None:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("missing key")
            # Third attempt exits cleanly, as a stopped worker does.

    await asyncio.wait_for(entry._supervise("flaky", Flaky()), timeout=5)

    assert attempts == 3, "a crashing worker was not retried"


@pytest.mark.asyncio
async def test_a_sibling_that_cannot_start_does_not_prevent_another_from_running(
    monkeypatch,
) -> None:
    monkeypatch.setattr(entry, "RESTART_DELAY_SECONDS", 0.001)
    ran = []

    class Doomed:
        attempts = 0

        async def start(self) -> None:
            Doomed.attempts += 1
            if Doomed.attempts < 2:
                raise RuntimeError("cannot start")

    class Healthy:
        async def start(self) -> None:
            ran.append("healthy")

    await asyncio.wait_for(
        asyncio.gather(
            entry._supervise("doomed", Doomed()),
            entry._supervise("healthy", Healthy()),
        ),
        timeout=5,
    )

    # Under the old bare gather the first raise would have propagated out of main() and the
    # healthy worker would never have been given the chance.
    assert ran == ["healthy"]
    assert Doomed.attempts == 2


@pytest.mark.asyncio
async def test_cancellation_still_stops_a_worker() -> None:
    """Shutdown must not be swallowed by the restart loop."""

    class Blocking:
        async def start(self) -> None:
            await asyncio.Event().wait()

    task = asyncio.create_task(entry._supervise("blocking", Blocking()))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
