"""Bake the immutable Silero VAD release into the ingress image.

WHY THIS RETRIES
    torch.hub.load reaches github.com for snakers4/silero-vad at IMAGE BUILD time, and that one
    network call decides whether the whole release builds. It has failed three times in two days
    with `http.client.RemoteDisconnected: Remote end closed connection without response` — and the
    proof it is transient rather than broken is that the same commit built fine on main six
    minutes after failing on development.

    A flaky build step is worse than it looks. It does not just cost a re-run: it teaches everyone
    that a red pipeline means "try again", which is exactly the habit that lets a real failure sit
    unnoticed. So the retry lives here rather than in a maintainer's fingers.

WHAT IT DOES NOT RETRY
    Anything that is not a network fault. A repository that does not exist, a tag that was
    deleted, a checksum mismatch — those fail identically on attempt four, so retrying only turns
    a five-second error into a minute of waiting before the same message.
"""

from __future__ import annotations

import http.client
import time
import urllib.error
from typing import Final

import torch

from livekit_ingress_worker.worker import SILERO_VAD_REPOSITORY

MAX_ATTEMPTS: Final = 4

# 2s, 4s, 8s. Deliberately short: this blocks an image build, and a download that has not
# recovered inside fifteen seconds is usually not going to inside sixty either.
BASE_DELAY_SECONDS: Final = 2.0

# Faults that mean "the network misbehaved", not "you asked for the wrong thing".
# RemoteDisconnected and IncompleteRead are subclasses of these, and both are what GitHub's CDN
# produces when it drops a connection mid-transfer.
TRANSIENT_ERRORS: Final = (
    urllib.error.URLError,
    http.client.HTTPException,
    ConnectionError,
    # Covers socket.timeout, which has been an alias for this since Python 3.10.
    TimeoutError,
)


def _is_transient(error: Exception) -> bool:
    """Whether retrying this could plausibly succeed."""
    if isinstance(error, urllib.error.HTTPError):
        # 404 means the tag is wrong and 403 means we are blocked; neither improves with time.
        # 5xx and 429 are the server having a moment.
        return error.code >= 500 or error.code == 429
    return isinstance(error, TRANSIENT_ERRORS)


def main() -> None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            torch.hub.load(  # type: ignore[no-untyped-call]
                SILERO_VAD_REPOSITORY,
                "silero_vad",
                trust_repo=True,
            )
            return
        except Exception as error:  # noqa: BLE001 - re-raised below unless transient
            if attempt == MAX_ATTEMPTS or not _is_transient(error):
                raise

            delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            # print, not logging: this runs as a bare RUN step in a Dockerfile with no logging
            # configured, and the only reader is someone looking at a build log.
            print(
                f"prefetch_model: attempt {attempt}/{MAX_ATTEMPTS} failed with "
                f"{type(error).__name__}: {error}; retrying in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)


if __name__ == "__main__":
    main()
