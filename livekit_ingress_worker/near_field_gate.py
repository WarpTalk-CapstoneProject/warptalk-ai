"""Near-field energy gate for livekit_ingress_worker.

Problem: Silero VAD (and the coarse RMS floor in worker.py's _publish_speech_chunk)
only answer "is this speech-shaped audio", not "is this speech close/clear enough to
trust an STT model with". A person talking across the room, or background chatter loud
enough to trip VAD, still produces plausible-looking "speech" to the VAD — but
Whisper-family models (including gpt-4o-transcribe, which this pipeline uses — see
STTSettings.model) expose no no_speech_prob/avg_logprob signal for
stt_worker/model.py's _filter_segments to actually filter on (those checks are
permanently inert placeholders for this model, see that function's comments), so
marginal/muffled audio that reaches the model tends to get hallucinated into a full,
plausible-sounding sentence instead of correctly producing nothing. The fix has to
happen before the audio ever reaches the model.

This gate tracks a rolling "how loud does THIS participant's own near-mic speech
usually peak at" baseline per track, and rejects any chunk whose peak amplitude is
much quieter than that baseline — the signature of a far-field/muffled voice bleeding
into the mic rather than the primary speaker actually talking into it.

Deliberately one-directional: a LOUDER chunk than the baseline is never rejected, and
nudges the baseline up. A quiet/distant chunk captured first (before the real speaker
has said anything) therefore self-corrects as soon as the real speaker talks, instead
of permanently anchoring on whatever was loudest first — the gate only ever requires
"at least this loud", never "close to a moving target in both directions". This also
means it does not catch a loud, sudden transient noise (a door slam, a nearby clap) —
raising vad_threshold (see WorkerSettings) and/or LiveKit's own Krisp noise
cancellation (client-side track processor, see warptalk-web's use-track-processors.ts)
are the intended defenses against that, not this gate.

Fails open when disabled.
"""

from __future__ import annotations

from shared.config import WorkerSettings
from shared.logger import get_logger

logger = get_logger(__name__)


class NearFieldGate:
    """Accepts/rejects one track's speech chunks by comparing peak amplitude against
    a running near-field baseline built from that same track's earlier accepted chunks.

    One instance per LiveKit audio track — see livekit_ingress_worker.worker.process_audio_track.
    """

    def __init__(self, settings: WorkerSettings):
        self._enabled = settings.near_field_gate_enabled
        self._relative_floor = settings.near_field_gate_relative_floor
        self._min_baseline_chunks = max(1, settings.near_field_gate_min_baseline_chunks)
        self._ema_alpha = settings.near_field_gate_baseline_ema_alpha
        self._baseline_peak: float | None = None
        self._baseline_count = 0

    def accept(self, raw_peak: float) -> bool:
        """True if this chunk's peak amplitude is consistent with this track's own
        established near-field baseline (or the baseline isn't established yet)."""
        if not self._enabled:
            return True

        if self._baseline_count < self._min_baseline_chunks:
            self._update_baseline(raw_peak)
            return True

        floor = self._baseline_peak * self._relative_floor
        if raw_peak < floor:
            logger.info(
                "near_field_gate_rejected_chunk",
                raw_peak=round(raw_peak, 4),
                baseline_peak=round(self._baseline_peak, 4),
                floor=round(floor, 4),
            )
            return False

        self._update_baseline(raw_peak)
        return True

    def _update_baseline(self, raw_peak: float) -> None:
        if self._baseline_peak is None:
            self._baseline_peak = raw_peak
        else:
            # Move toward a louder peak quickly (a new, higher "true near-field" level
            # just got confirmed) but toward a quieter one slowly (ordinary volume dip,
            # not a signal to lower the bar for what still counts as "close enough").
            alpha = self._ema_alpha if raw_peak >= self._baseline_peak else self._ema_alpha / 4
            self._baseline_peak = (1 - alpha) * self._baseline_peak + alpha * raw_peak
        self._baseline_count += 1
