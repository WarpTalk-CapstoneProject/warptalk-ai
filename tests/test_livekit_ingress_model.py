from unittest.mock import MagicMock, patch

from livekit_ingress_worker.worker import LiveKitIngressWorker


async def test_silero_model_uses_an_immutable_release() -> None:
    model = object()
    worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    worker.logger = MagicMock()

    torch = MagicMock()
    torch.hub.load.return_value = (model, [])
    with patch("livekit_ingress_worker.worker.torch", torch):
        await worker.load_model()

    assert worker._vad_model is model
    torch.hub.load.assert_called_once_with(
        "snakers4/silero-vad:v6.2.1",
        "silero_vad",
        trust_repo=True,
    )
