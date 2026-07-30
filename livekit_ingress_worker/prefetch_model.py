"""Bake the immutable Silero VAD release into the ingress image."""

import torch

from livekit_ingress_worker.worker import SILERO_VAD_REPOSITORY


def main() -> None:
    torch.hub.load(  # type: ignore[no-untyped-call]
        SILERO_VAD_REPOSITORY,
        "silero_vad",
        trust_repo=True,
    )


if __name__ == "__main__":
    main()
