"""Audio processing utilities for chunk-based streaming."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import numpy.typing as npt


def bytes_to_numpy(
    audio_bytes: bytes,
    sample_rate: int = 16000,
) -> npt.NDArray[np.float32]:
    """Convert raw 16-bit PCM audio bytes to float32 numpy array for Whisper."""
    # Whisper expects float32 in range [-1.0, 1.0]
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_data


def numpy_to_bytes(
    audio_array: npt.NDArray[np.floating[Any]],
    sample_rate: int = 16000,
) -> bytes:
    """Convert numpy array to WAV bytes."""
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def split_into_chunks(
    audio_array: npt.NDArray[np.float32],
    chunk_duration_ms: int = 2000,
    sample_rate: int = 16000,
    overlap_ms: int = 200,
) -> list[npt.NDArray[np.float32]]:
    """Split audio into overlapping chunks for streaming pipeline."""
    chunk_size = int(sample_rate * chunk_duration_ms / 1000)
    overlap_size = int(sample_rate * overlap_ms / 1000)
    step = chunk_size - overlap_size

    chunks = []
    for start in range(0, len(audio_array), step):
        end = min(start + chunk_size, len(audio_array))
        chunks.append(audio_array[start:end])
        if end == len(audio_array):
            break

    return chunks
