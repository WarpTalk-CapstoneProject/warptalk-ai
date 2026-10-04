"""A quiet speaker's real sentences reach STT; noise still does not; and every drop is visible.

The ingress energy floor (worker.py _ENERGY_FLOOR_RMS) is one absolute number for everyone. In
tools/meeting_sim's incident post-mortem it dropped six chunks of one quiet engineer's speech —
14.5 s, whole sentences — and logged it at DEBUG, which production never emits. The floor is now
lowered (never raised) for a speaker proven quiet; see speech_level_floor.py.

Four groups of tests:
  * the rule itself, on hand-made levels;
  * a REPLAY of every chunk of three simulated tracks (tests/fixtures/ingress_energy_levels.json):
    the quiet speaker's dropped sentences come back, and not one of the babble, fan or echo
    chunks the floor rejected gets in;
  * the worker: drops are logged at INFO with speech_ms, RMS and running totals, and a
    relative admission says so;
  * the wiring: every call site passes the track's floor — a per-track baseline that each call
    rebuilt from nothing would be this repo's "written, tested, connected to nothing" again.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from livekit_ingress_worker import worker as ingress_module
from livekit_ingress_worker.speech_level_floor import FloorVerdict, SpeechLevelFloor
from livekit_ingress_worker.worker import _ENERGY_FLOOR_RMS, LiveKitIngressWorker
from shared.config import LiveKitSettings, WorkerSettings

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "ingress_energy_levels.json").read_text()
)


def _floor(**overrides: Any) -> SpeechLevelFloor:
    return SpeechLevelFloor.from_settings(WorkerSettings(**overrides), _ENERGY_FLOOR_RMS)


def _prove_quiet(floor: SpeechLevelFloor, level: float = 0.027, n: int = 3) -> None:
    """A speaker whose ordinary sentences clear the absolute floor by ~2.6 dB."""
    for _ in range(n):
        assert floor.judge(level, None).accept


# --------------------------------------------------------------------------------- the rule


@pytest.mark.parametrize("share", [None, 0.25, 0.5, 0.83, 1.0])
@pytest.mark.parametrize("raw_rms", [0.0, 0.005, 0.0099, 0.0101, 0.0149, 0.015, 0.0199, 0.0201])
def test_without_a_baseline_it_is_exactly_the_old_floor(
    raw_rms: float, share: float | None
) -> None:
    """A track's first chunks are judged as they always were: raw_rms >= 0.02 * sqrt(share)."""
    effective = share if share is not None and 0 < share < 1 else 1.0
    expected = raw_rms >= _ENERGY_FLOOR_RMS * effective**0.5
    verdict = _floor().judge(raw_rms, share)
    assert verdict.accept is expected
    assert verdict.relative is False


def test_a_proven_quiet_speaker_keeps_a_softer_sentence() -> None:
    floor = _floor()
    _prove_quiet(floor)  # baseline 0.027
    verdict = floor.judge(0.018, None)  # 0.67 of their own level, under the absolute 0.02
    assert verdict.accept and verdict.relative
    assert verdict.relative_floor == pytest.approx(0.4 * 0.027)


def test_noise_well_under_the_speakers_own_level_is_still_dropped() -> None:
    floor = _floor()
    _prove_quiet(floor)
    # 0.35 of the speaker's level (-9 dB): the loudest noise the simulator produced sat at 0.18.
    assert not floor.judge(0.0095, None).accept


def test_it_never_raises_the_bar() -> None:
    """A loud speaker's baseline puts the relative floor ABOVE 0.02; the absolute floor still
    rules, so nothing that reaches STT today can stop reaching it."""
    floor = _floor()
    _prove_quiet(floor, level=0.2)  # relative floor would be 0.08
    assert floor.judge(0.021, None).accept
    assert not floor.judge(0.019, None).accept  # and below 0.02 it is the old answer


def test_relative_admissions_cannot_walk_the_baseline_down() -> None:
    """Only chunks that cleared the ABSOLUTE floor teach the baseline, so a long run of quiet
    admissions leaves it where the speaker's real voice put it."""
    floor = _floor()
    _prove_quiet(floor)  # baseline 0.027 -> relative floor 0.0108
    for _ in range(50):
        assert floor.judge(0.0110, None).accept
    assert not floor.judge(0.0105, None).accept


def test_no_relaxation_before_the_speaker_has_proven_anything() -> None:
    """The simulator's loudest noise was 5 s of office babble BEFORE its speaker said a word —
    by level alone, a quiet first sentence. Two proven chunks are not yet a baseline."""
    floor = _floor()
    _prove_quiet(floor, n=2)
    assert not floor.judge(0.018, None).accept


def test_ratio_zero_turns_the_rule_off() -> None:
    floor = _floor(ingress_energy_relative_ratio=0.0)
    _prove_quiet(floor)
    assert not floor.judge(0.019, None).accept


# ------------------------------------------------------------------- replay of logged tracks


def _replay(key: str) -> list[tuple[list[Any], FloorVerdict]]:
    floor = _floor()
    return [(row, floor.judge(row[0], row[1] or None)) for row in _FIXTURE[key]]


def test_replay_the_quiet_speakers_dropped_sentences_come_back() -> None:
    replay = _replay("incident_be_vi")
    recovered = [row[4] for row, v in replay if not row[3] and v.accept and v.relative]
    # u030 u045 u049 u080 u097: 14.0 s of VAD speech the absolute floor threw away.
    assert recovered == ["u030", "u045", "u049", "u080", "u097"]
    # The speaker's FIRST chunk (u004, 0.48 s) has no baseline, so the old floor alone judges
    # it — and the run dropped it. (Its rebuilt level sits 9% above the floor, the largest
    # reconstruction error in the fixture; either way the new rule plays no part in it.)
    first_row, first = replay[0]
    assert first_row[4] == "u004" and not first_row[3]
    assert first.baseline is None and not first.relative


@pytest.mark.parametrize("key", ["incident_be_vi", "launch_design_en", "launch_eng_ja"])
def test_replay_no_noise_chunk_gets_in(key: str) -> None:
    admitted_noise = [
        row for row, v in _replay(key) if row[2] == "noise" and not row[3] and v.accept
    ]
    assert admitted_noise == []


@pytest.mark.parametrize("key", ["incident_be_vi", "launch_design_en", "launch_eng_ja"])
def test_replay_everything_published_before_is_published_now(key: str) -> None:
    assert all(v.accept for row, v in _replay(key) if row[3])


# ------------------------------------------------------------------------------- the worker

_RATE = 16000


def _worker() -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")
    )
    return LiveKitIngressWorker(settings=settings)


def _chunk(amplitude: int, samples: int = _RATE) -> bytearray:
    return bytearray(amplitude.to_bytes(2, "little", signed=True) * samples)


async def _run(
    worker: LiveKitIngressWorker, chunks: list[int], floor: SpeechLevelFloor
) -> tuple[list[Any], list[tuple[str, dict[str, Any]]]]:
    sent: list[Any] = []
    logged: list[tuple[str, dict[str, Any]]] = []

    async def _capture(stream: str, room: str, payload: Any) -> None:
        sent.append(payload)

    async def _language(_room: str, _speaker: str) -> str:
        return "vi"

    class _Log:
        def info(self, event: str, **fields: Any) -> None:
            logged.append((event, fields))

        def __getattr__(self, _name: str) -> Any:
            return lambda *a, **k: None

    worker.publish = _capture  # type: ignore[method-assign]
    worker._speaker_language = _language  # type: ignore[method-assign]
    worker.logger = _Log()  # type: ignore[assignment]
    for i, amplitude in enumerate(chunks):
        await worker._publish_speech_chunk(
            "room", "be_vi", _chunk(amplitude), i, _RATE, speech_samples=_RATE, energy_floor=floor
        )
    return sent, logged


@pytest.mark.asyncio
async def test_a_drop_is_logged_at_info_with_its_speech_and_running_totals() -> None:
    worker = _worker()
    floor = SpeechLevelFloor.from_settings(worker.settings, _ENERGY_FLOOR_RMS)
    # 300/32768 = 0.0092 RMS: under the absolute floor, and no baseline to relax it.
    sent, logged = await _run(worker, [300, 300], floor)
    assert sent == []
    drops = [fields for event, fields in logged if event == "ingress_low_energy_dropped"]
    assert len(drops) == 2
    assert drops[0]["speech_ms"] == 1000
    assert drops[0]["raw_rms"] == pytest.approx(300 / 32768, abs=1e-6)
    assert drops[0]["speaker_id"] == "be_vi"
    assert [d["dropped_chunks"] for d in drops] == [1, 2]
    assert [d["dropped_speech_ms"] for d in drops] == [1000, 2000]


@pytest.mark.asyncio
async def test_a_proven_quiet_speakers_softer_chunk_is_published_and_says_why() -> None:
    worker = _worker()
    floor = SpeechLevelFloor.from_settings(worker.settings, _ENERGY_FLOOR_RMS)
    # Three ordinary chunks at 0.027 RMS, then one at 0.018: under 0.02, at 0.67 of their level.
    sent, logged = await _run(worker, [885, 885, 885, 590], floor)
    assert len(sent) == 4
    admitted = [fields for event, fields in logged if event == "ingress_low_energy_admitted"]
    assert len(admitted) == 1
    assert admitted[0]["chunk_index"] == 3
    assert admitted[0]["rule"] == "speaker_relative"
    assert not [event for event, _ in logged if event == "ingress_low_energy_dropped"]


def test_every_call_site_passes_the_tracks_energy_floor() -> None:
    """Without it each call builds a fresh floor with no history, and the speaker-relative rule
    never fires — the absolute floor alone again, with nothing failing to say so."""
    source = Path(inspect.getfile(ingress_module)).read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_publish_speech_chunk"
    ]
    assert calls
    missing = [
        node.lineno
        for node in calls
        if not any(keyword.arg == "energy_floor" for keyword in node.keywords)
    ]
    assert not missing, f"_publish_speech_chunk called without energy_floor at line(s) {missing}"
