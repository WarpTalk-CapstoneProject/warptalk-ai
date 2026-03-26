"""STT Worker — Consumes audio chunks, produces text segments."""

from __future__ import annotations

from shared.logger import get_logger
from shared.redis_client import RedisStreamClient

logger = get_logger(__name__)


class STTWorker:
    """Speech-to-Text worker using Whisper model.

    Pipeline:
        Redis Stream (audio:chunks:{meetingId})
        → Whisper STT
        → Redis Stream (stt:results:{meetingId})
    """

    def __init__(self) -> None:
        self.redis = RedisStreamClient()
        self.model = None

    async def load_model(self) -> None:
        """Load Whisper model. Override model name via STT_MODEL env var."""
        logger.info("Loading STT model...")
        # TODO: Load faster-whisper or openai-whisper model
        # from faster_whisper import WhisperModel
        # self.model = WhisperModel(os.getenv("STT_MODEL", "large-v3"))
        logger.info("STT model loaded")

    async def run(self) -> None:
        """Main worker loop."""
        await self.redis.connect()
        await self.load_model()

        logger.info("STT Worker ready, waiting for audio chunks...")

        # TODO: Consume from audio:chunks:{meetingId} streams
        # async for message_id, data in self.redis.consume(
        #     stream="audio:chunks:*",
        #     group="stt-workers",
        # ):
        #     result = await self.transcribe(data)
        #     await self.redis.publish(f"stt:results:{meeting_id}", result)

    async def transcribe(self, audio_data: dict) -> dict:
        """Transcribe an audio chunk to text."""
        # TODO: Implement transcription
        return {"text": "", "language": "en", "confidence": 0.0}
