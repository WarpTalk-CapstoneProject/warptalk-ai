"""The Silero download retries transient network faults, and only those.

torch.hub.load reaches github.com at image BUILD time, so one dropped connection fails the whole
release. It has done so three times in two days — and the same commit built fine on main six
minutes after failing on development, which is what identifies it as transient rather than broken.

The half that matters as much as the retry is the half that does NOT retry: a deleted tag or a
403 fails identically on attempt four, so retrying only delays the same message.
"""

from __future__ import annotations

import http.client
import urllib.error

import pytest

from livekit_ingress_worker import prefetch_model


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prefetch_model.time, "sleep", lambda _seconds: None)


def _load_failing(times: int, error: Exception):
    """A torch.hub.load that fails `times` times before succeeding."""
    calls = {"n": 0}

    def fake_load(*_args: object, **_kwargs: object) -> str:
        calls["n"] += 1
        if calls["n"] <= times:
            raise error
        return "model"

    fake_load.calls = calls  # type: ignore[attr-defined]
    return fake_load


@pytest.mark.parametrize(
    "error",
    [
        # The exact failure from the build log.
        http.client.RemoteDisconnected("Remote end closed connection without response"),
        urllib.error.URLError("temporary failure in name resolution"),
        http.client.IncompleteRead(b"half a file"),
        TimeoutError("timed out"),
        ConnectionResetError("connection reset by peer"),
        urllib.error.HTTPError("url", 503, "Service Unavailable", {}, None),  # type: ignore[arg-type]
        urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None),  # type: ignore[arg-type]
    ],
)
def test_transient_faults_are_retried_until_they_succeed(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    fake_load = _load_failing(2, error)
    monkeypatch.setattr(prefetch_model.torch.hub, "load", fake_load)

    prefetch_model.main()

    assert fake_load.calls["n"] == 3, f"{type(error).__name__} should have been retried"


@pytest.mark.parametrize(
    "error",
    [
        # A tag that does not exist does not start existing on attempt four.
        urllib.error.HTTPError("url", 404, "Not Found", {}, None),  # type: ignore[arg-type]
        urllib.error.HTTPError("url", 403, "Forbidden", {}, None),  # type: ignore[arg-type]
        ValueError("unknown model silero_vad"),
        RuntimeError("checksum mismatch"),
    ],
)
def test_permanent_faults_fail_on_the_first_attempt(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    fake_load = _load_failing(99, error)
    monkeypatch.setattr(prefetch_model.torch.hub, "load", fake_load)

    with pytest.raises(type(error)):
        prefetch_model.main()

    assert fake_load.calls["n"] == 1, "a permanent failure must not burn the retry budget"


def test_a_download_that_never_recovers_still_fails_the_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The retry must not become a way for a genuinely unreachable model to produce a green image
    # with no VAD baked in.
    error = http.client.RemoteDisconnected("nope")
    fake_load = _load_failing(99, error)
    monkeypatch.setattr(prefetch_model.torch.hub, "load", fake_load)

    with pytest.raises(http.client.RemoteDisconnected):
        prefetch_model.main()

    assert fake_load.calls["n"] == prefetch_model.MAX_ATTEMPTS


def test_the_happy_path_downloads_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_load = _load_failing(0, RuntimeError("unused"))
    monkeypatch.setattr(prefetch_model.torch.hub, "load", fake_load)

    prefetch_model.main()

    assert fake_load.calls["n"] == 1
