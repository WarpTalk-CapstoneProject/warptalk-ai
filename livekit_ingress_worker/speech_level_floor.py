"""The ingress energy floor, judged per speaker: a quiet microphone is not noise.

THE ABSOLUTE FLOOR ALONE DROPPED A QUIET SPEAKER'S REAL SENTENCES, SILENTLY.

`_ENERGY_FLOOR_RMS` (worker.py) asks one question of every chunk: is the speech in it at least
0.02 RMS? For most microphones that is a comfortable 8-15 dB below how loud people talk, and it
is what keeps room babble and fan noise that tripped VAD from reaching an STT model that will
turn them into fluent invented sentences. But it is the same number for everyone. A speaker whose
own voice arrives at 0.027 (-31 dBFS) clears it by 2.6 dB, so every softer sentence they say
falls under it: tools/meeting_sim's incident post-mortem lost six of one engineer's chunks this
way — 14.5 s of VAD-marked speech, whole sentences ("Không phải volatile-lru, hồi trước là
noeviction giống staging…") — in a meeting where the other three speakers lost nothing. It logged
at DEBUG, so production could never have seen it.

Real microphones are that quiet. Of two production speakers on 1 Oct, the quieter one's softest
10% of PUBLISHED chunks sat at 0.019 RMS; whatever fell below the floor is not in any log.

SO THE FLOOR IS LOWERED — AND ONLY LOWERED — FOR A SPEAKER WHO HAS PROVEN THEY ARE QUIET.

Once a track has published `min_baseline_chunks` chunks that cleared the absolute floor on their
own, its baseline is the median speech level of its last `window` such chunks. A chunk under the
absolute floor is still accepted when its speech level is at least `relative_ratio` of that
baseline (0.4 is 8 dB below the speaker's own typical level).

  Measured on every chunk of five simulated meetings (levels rebuilt from the tracks, checked
  against the logged raw_rms of the 974 published chunks: median error 0.3-0.8%):
    the dropped real speech sat at 0.56-0.77 of its speaker's baseline;
    every dropped noise chunk — babble, fan, echo, a muted mic — at 0.18 or below, and at 0.23
    or below when re-run through this code with the real audio path.

Three properties keep this from being the near-field gate (near_field_gate.py), which is relative
too and is off because it dropped valid middle chunks in production:

  * It never raises the bar. Anything the absolute floor accepts is accepted, whatever the
    baseline says, so no chunk that reaches STT today can stop reaching it.
  * The baseline learns only from chunks that cleared the ABSOLUTE floor. A run of quiet
    relative admissions cannot walk it down, so the floor cannot drift toward the noise.
  * No baseline, no relaxation. A track's first chunks — before it has proven anything — are
    judged exactly as before. That is deliberate: the simulator's loudest noise chunk was 5 s of
    office babble arriving before its speaker had said a word, indistinguishable by level from a
    quiet first sentence.

What is still dropped is now logged at INFO (`ingress_low_energy_dropped`), with the speech it
carried and running per-track totals, so a production microphone this floor is failing shows up
in the logs instead of as missing captions.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass

from shared.config import WorkerSettings


@dataclass(frozen=True)
class FloorVerdict:
    accept: bool
    # RMS of the SPEECH in the chunk: raw RMS undiluted by the VAD padding around it (see
    # worker.py _ENERGY_FLOOR_RMS for why the share matters). The absolute test on it is exactly
    # the one the worker has always made: raw_rms >= floor * sqrt(speech share).
    speech_rms: float
    absolute_floor: float
    # Median speech level of this track's recent absolute-clearing chunks; None until it has
    # enough of them.
    baseline: float | None
    # The level a chunk under the absolute floor still needs; None when there is no baseline.
    relative_floor: float | None
    # True when the chunk got in only because of the speaker-relative rule.
    relative: bool


class SpeechLevelFloor:
    """One track's energy floor: the absolute floor, relaxed for a speaker proven to be quiet.

    One instance per LiveKit audio track, for the track's whole lifetime — the baseline is this
    speaker's own history (see process_audio_track).
    """

    def __init__(
        self,
        absolute_floor: float,
        *,
        relative_ratio: float = 0.0,
        min_baseline_chunks: int = 3,
        baseline_window: int = 8,
    ) -> None:
        self._absolute_floor = absolute_floor
        self._ratio = max(0.0, relative_ratio)
        self._min_baseline = max(1, min_baseline_chunks)
        self._levels: deque[float] = deque(maxlen=max(1, baseline_window))
        self.dropped_chunks = 0
        self.dropped_speech_ms = 0

    @classmethod
    def from_settings(cls, settings: WorkerSettings, absolute_floor: float) -> SpeechLevelFloor:
        return cls(
            absolute_floor,
            relative_ratio=settings.ingress_energy_relative_ratio,
            min_baseline_chunks=settings.ingress_energy_baseline_min_chunks,
            baseline_window=settings.ingress_energy_baseline_window,
        )

    def judge(self, raw_rms: float, speech_share: float | None) -> FloorVerdict:
        share = speech_share if speech_share is not None and 0.0 < speech_share < 1.0 else 1.0
        speech_rms = raw_rms / share**0.5
        baseline = (
            statistics.median(self._levels) if len(self._levels) >= self._min_baseline else None
        )
        relative_floor = baseline * self._ratio if baseline is not None and self._ratio else None

        if speech_rms >= self._absolute_floor:
            self._levels.append(speech_rms)
            return FloorVerdict(
                True, speech_rms, self._absolute_floor, baseline, relative_floor, False
            )
        relative = relative_floor is not None and speech_rms >= relative_floor
        return FloorVerdict(
            relative, speech_rms, self._absolute_floor, baseline, relative_floor, relative
        )

    def note_dropped(self, speech_ms: int) -> None:
        self.dropped_chunks += 1
        self.dropped_speech_ms += max(0, speech_ms)
