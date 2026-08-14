"""Shared configuration loader using pydantic-settings.

Each worker has its own settings class loaded from environment variables
with a unique prefix (STT_, TRANSLATION_, TTS_, ASSISTANT_, EMBEDDING_).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


def resolve_openai_api_key(stage_api_key: str = "") -> str:
    """Prefer a stage-specific key, then fall back to shared OPENAI_API_KEY."""
    return stage_api_key or os.getenv("OPENAI_API_KEY", "")


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = {"env_prefix": "REDIS_"}

    url: str = "redis://localhost:6379"
    password: str = ""
    sentinel_urls: str = ""
    sentinel_service_name: str = "mymaster"
    # livekit_ingress alone holds 1 connection per concurrent room (pubsub listener)
    # plus one per in-flight XADD from VAD-triggered chunk publishes — 10 was only
    # enough for a single active room at a time; opening a second meeting while one
    # is still live exhausted the pool ("MaxConnectionsError: Too many connections"),
    # which read as the whole AI pipeline going unresponsive.
    max_connections: int = 50
    # XREADGROUP uses a blocking read (currently 2s for AI workers). Keep a
    # generous socket margin for Docker Desktop/Redis scheduling jitter so a
    # normal long-poll does not become a retry storm under load.
    socket_timeout: float = 15.0
    socket_connect_timeout: float = 5.0
    stream_maxlen: int = 1000  # MAXLEN ~ for XADD trimming
    retry_max_attempts: int = 5
    retry_base_delay: float = 0.5


class LiveKitSettings(BaseSettings):
    """LiveKit Server connection settings."""

    model_config = {"env_prefix": "LIVEKIT_"}

    url: str = "ws://localhost:7880"
    api_key: str = "YOUR_LIVEKIT_API_KEY"
    api_secret: str = "YOUR_LIVEKIT_API_SECRET"


class WorkerSettings(BaseSettings):
    """Base worker settings shared by all workers."""

    model_config = {"env_prefix": ""}

    log_level: str = "INFO"
    # Max only for uninterrupted speech; ordinary short turns still flush on VAD silence.
    # Six seconds gives the model enough lexical context for natural Vietnamese sentences
    # containing English technical terms without adding delay after an ordinary pause.
    chunk_duration_ms: int = 6000
    redis: RedisSettings = RedisSettings()
    livekit: LiveKitSettings = LiveKitSettings()

    # VAD gating settings (used by ingress worker).
    #
    # This number has now been moved in both directions, and the trade-off is real either
    # way: too low and distant or ambiguous sound reaches an STT model with no confidence
    # signal of its own, which hallucinates it into a fluent sentence; too high and the
    # speaker has to lean into the microphone to be heard at all.
    #
    # It was 0.3, raised to 0.5 to stop the hallucinations. In a room — a lecture hall, a
    # meeting room with the laptop on the far side of a table — 0.5 silently drops most of
    # what is said: a speaker at any distance reports having to shout to register. That is
    # the worse failure of the two for this product. A sentence transcribed imperfectly can
    # still be read and corrected; a sentence that never existed cannot, and the person
    # speaking has no way to tell it happened.
    #
    # 0.35 sits between the two previous values rather than returning to 0.3, because the
    # earlier hallucination reports came from a pipeline that was also feeding STT the same
    # audio two or three times over (the ingress reader was keyed on track sid, not on the
    # speaker). With that fixed, the model sees each utterance once, and some of what was
    # blamed on a permissive gate was volume.
    #
    # Overridable per deployment as VAD_THRESHOLD, with no rebuild: a close-mic studio can
    # raise it, a hall can lower it further, and neither needs this default to move again.
    vad_threshold: float = 0.35  # Speech detection threshold
    vad_pre_speech_ms: int = 192  # Two ~96ms windows preserve word onsets
    # Four ~96ms windows retain quiet final syllables and natural micro-pauses. A
    # two-window production replay cut "Kubernetes" to "Kuber".
    vad_silence_hangover_ms: int = 576
    vad_min_speech_ms: int = 288  # Keeps short English keywords and acknowledgements

    # Near-field energy gate (ingress worker only, see livekit_ingress_worker/near_field_gate.py).
    # Opt-in only: a relative peak threshold cannot tell a distant speaker from a quiet
    # syllable in the primary speaker's own turn. Enabling it dropped valid middle
    # chunks in production, so the default preserves audio recall and lets the
    # context-aware stage handle relevance instead.
    near_field_gate_enabled: bool = False
    near_field_gate_relative_floor: float = (
        0.35  # chunk peak must be >= 35% of this track's own established near-field peak
    )
    near_field_gate_min_baseline_chunks: int = 2
    near_field_gate_baseline_ema_alpha: float = 0.3


class STTSettings(BaseSettings):
    """Speech-to-Text worker settings."""

    model_config = {"env_prefix": "STT_"}

    provider: str = "openai"
    api_key: str = ""
    # Accuracy-first model for completed utterances. It supports expected `languages`,
    # structured `keywords`, and prompt context in Realtime transcription sessions.
    model: str = "gpt-transcribe"
    language: str = "auto"  # Auto-detect for code-switching (Vi + En)
    # Browser capture already applies WebRTC echo cancellation/noise suppression and
    # may additionally use the optional Krisp processor. A second Realtime denoising
    # pass distorted clean close-mic speech in production replay tests, so leave it off
    # by default. Deployments ingesting genuinely raw room audio can still opt in with
    # STT_NOISE_REDUCTION=far_field.
    noise_reduction: str = "off"
    # Production replay placed unrelated, unstable sentences at -0.767 to -0.8833.
    # Clear code-switched technical speech stayed above this boundary.
    min_avg_logprob: float = -0.7
    # Per-language overrides for the discard floor above, keyed by bare ISO-639-1 code
    # (e.g. STT_MIN_AVG_LOGPROB_BY_LANGUAGE='{"vi": -0.75}'). Empty by default: every
    # language uses min_avg_logprob until someone measures a better value for it.
    #
    # Why the mechanism exists at all — a single global floor is not neutral. Published
    # multilingual benchmarks put Vietnamese word error rate around 10% on FLEURS against
    # roughly 6% for Japanese and Korean, so the same acoustic quality yields a lower
    # average logprob in Vietnamese than in Japanese. One shared threshold therefore
    # discards more real speech from the languages the model finds hardest, which is the
    # opposite of what a multilingual product wants.
    #
    # HOW TO CALIBRATE (do not guess these): run a labelled set per language through
    # tools/stt_filter_audit.py, sweep the floor, and take the value where content
    # retention stops improving. ViMedCSS (Vietnamese-English code-switching, 34.6h),
    # CanVEC and the relevant FLEURS split are suitable sources.
    min_avg_logprob_by_language: dict[str, float] = {}
    # Warm WebSockets are claimed by the first active speakers so their first utterance
    # does not pay the ~1–2s Realtime connection handshake.
    realtime_pool_size: int = 4
    # Measure HOW an utterance was said and attach it to the transcript segment, so the dub can
    # be delivered the way the speaker delivered it (shared/prosody.py).
    #
    # Off is a real position, not a placeholder: the measurement is ~3ms of CPU per second of
    # audio (14ms for a full 6s chunk, benchmarked), and a deployment that finds that too
    # expensive should be able to drop it without redeploying anything else. Turning it off
    # removes the `prosody` field from stt:results; every consumer already treats that field as
    # optional, so the pipeline reverts to exactly its pre-prosody behaviour.
    prosody_enabled: bool = True
    # A speaker's rolling normal lives for this long after their last utterance. Meeting-scoped
    # on purpose — a different room means a different microphone, and a baseline built in one is
    # not a description of how they sound in the other.
    prosody_baseline_ttl_seconds: int = 21600  # 6h


class TranslationSettings(BaseSettings):
    """Translation worker settings."""

    model_config = {"env_prefix": "TRANSLATION_"}

    provider: str = "openai"  # 'openai' only — no fallback
    api_key: str = ""
    # Live vi→en production probes were semantically equivalent to gpt-4.1-mini while
    # cutting warm median translation latency from ~1.4–1.8s to ~0.8s.
    model: str = "gpt-5.4-nano"
    # Persistent Realtime text responses avoid a fresh HTTP/model-routing round trip.
    # The chat model above remains the correctness fallback when a WebSocket is unhealthy.
    # Mini was faster and more stable on the production Vietnamese/code-switching probe:
    # 660/919/655/737/715ms versus the full model's 667/1873/864/880/920ms.
    realtime_model: str = "gpt-realtime-2.1-mini"
    # How hard the realtime model thinks before answering. Stays at minimal, and the
    # sweep that says so is tools/probe_realtime_effort.py.
    #
    # The realtime path carries the FIRST sentence of every utterance, so it is the one
    # call in the pipeline a listener waits on directly. Raising effort was tried as a
    # way to make it repair ASR mishearings (see _ASR_REPAIR_INSTRUCTION); it does not
    # work, and it is actively harmful at the current 128-token ceiling because hidden
    # reasoning tokens are drawn from that same budget:
    #
    #     effort     max_tokens=128            max_tokens=512
    #     minimal    859ms, no repair          800ms, no repair
    #     low        incomplete -> fallback    1483ms, no repair
    #     medium     incomplete -> fallback    1541ms (p-max 2045ms), no repair
    #
    # Every non-minimal cell either broke outright or cost latency for the same wrong
    # answer, and medium at 512 crossed realtime_timeout_seconds. The mishearing is not
    # a thinking-budget problem.
    #
    # What DOES fix it is the model, measured on the same probe and the same sentence
    # ("cu bơ nét" -> Kubernetes, glossary supplied):
    #
    #     gpt-realtime-2.1-mini   effort=minimal   893ms   repaired 0/3
    #     gpt-realtime-2.1        effort=minimal   985ms   repaired 3/3
    #
    # and the full model is not the latency risk the older comment above feared — on
    # ordinary speech over 18 runs each it was p90 776ms vs mini's 840ms, nothing over
    # 2s on either. Treat that as "not slower", not "faster": one machine, one session.
    # Changing TRANSLATION_REALTIME_MODEL in production is a deploy decision, so the
    # evidence is recorded here rather than acted on.
    realtime_reasoning_effort: str = "minimal"
    realtime_pool_size: int = 4
    # Do not turn a rare ~1.2s provider response into a 5–19s latency cliff by
    # prematurely starting the slower HTTP fallback.
    realtime_timeout_seconds: float = 2.0
    # TranslationWorker already splits the stream into sentences. Keep this bounded,
    # but leave enough headroom for the model's hidden minimal-reasoning tokens: 64
    # intermittently ended short translations as `incomplete`, triggering a multi-second
    # HTTP fallback that was far worse for p95 latency than the larger ceiling.
    realtime_max_output_tokens: int = 128
    max_tokens: int = 512
    # Fully deterministic, not just "near" — measured via the real pipeline benchmark
    # that 0.1 let identical repeated sentences translate to different (equally valid)
    # phrasings across separate calls, breaking tts_worker's text-based synthesis
    # cache: a real repeated meeting phrase missed the cache and paid a full ~1s
    # Cartesia call instead of a ~2ms cache hit. See translation_worker/translator.py.
    temperature: float = 0.0


class TTSSettings(BaseSettings):
    """Text-to-Speech worker settings."""

    model_config = {"env_prefix": "TTS_"}

    provider: str = "cartesia"
    api_key: str = ""
    # sonic-turbo does not support Vietnamese (confirmed via a live 400 "language_not_supported"
    # response) — sonic-3.5 is Cartesia's current model with Vietnamese in its language table.
    # This product's whole premise is cross-language dubbing, so the default must support the
    # target languages it actually needs, not just English.
    model: str = "sonic-3.5"
    sample_rate: int = 44100
    # "slow" | "normal" | "fast" — the whole of Cartesia's range (cartesia.types.ModelSpeed).
    # Defaults to fast: a dub has to land inside the gap the speaker left, and at normal it
    # consistently finished after they had already moved on. Overridable as TTS_SPEED.
    speed: str = "fast"
    # Raised from 10.0. Ten seconds is Cartesia's floor, not a good reference: it is whatever
    # the speaker happened to say first, which is usually "alo alo, nghe rõ không". Twenty
    # seconds of ACCEPTED speech (see tts_worker/clone_sample_quality.py) is enough for the
    # clone to carry a person's timbre rather than their microphone check.
    voice_clone_min_seconds: float = 20.0
    # How much audio may be held while waiting for a clip that passes the quality gate. Rejected
    # audio slides out of the front of the buffer; without a cap a speaker in a noisy room would
    # accumulate the whole meeting in memory and never clone.
    voice_clone_max_buffer_seconds: float = 90.0
    # WT-371 #9: how many times a speaker's clone may be REPLACED by a better sample within one
    # meeting. Cloning used to happen exactly once, from the first clip that passed the gate, and
    # the worker then stopped listening — so the voice was locked to whatever register the speaker
    # happened to open in. Raise or crack your voice and the clone stopped being you.
    #
    # One upgrade, not unlimited: each re-clone is a paid Cartesia call, and the synthesised voice
    # audibly changes when it lands. One is enough to escape a bad opening clip; more would be a
    # voice that keeps shifting under the listener.
    voice_clone_max_upgrades: int = 1
    # How much better a later clip must score before it is worth replacing a working clone. Small
    # gains are noise in the estimator, and re-cloning for them would change the voice people are
    # listening to in exchange for nothing.
    voice_clone_upgrade_margin: float = 0.15
    min_clone_chars: int = 8
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    voice_clone_key_ttl_seconds: int = (
        43200  # 12h — unbounded before this; matches AudioRouteCacheService's own Redis TTL
    )
    # How many public Cartesia voices to offer per language — both for the per-speaker
    # hashed default (auto-diversity when nobody has cloned/chosen a voice) and for the
    # control-bar voice picker's option list.
    voice_catalog_size: int = 6
    # Cartesia's public library doesn't churn often — cache the per-language catalog
    # in Redis this long before re-fetching, to avoid a /voices call on every miss.
    voice_catalog_cache_ttl_seconds: int = 21600  # 6h
    # Deliver the dub the way the speaker delivered it, using the prosody measured upstream
    # (STT_PROSODY_ENABLED) and carried on the translation message. Independent of the STT flag
    # so the measurement and its use can be turned off separately — which is what makes an A/B
    # possible without stopping the measurement.
    prosody_enabled: bool = True


class AssistantSettings(BaseSettings):
    """AI Assistant worker settings."""

    model_config = {"env_prefix": "ASSISTANT_"}

    api_key: str = ""
    model: str = "gpt-4.1"
    max_tokens: int = 2048
    temperature: float = 0.3


class ChatAssistantSettings(BaseSettings):
    """Global AI assistant (chat-with-tools) worker settings.

    Distinct from AssistantSettings (per-meeting summarization) — this worker answers
    free-form questions in the global "Ask WarpTalk" widget and can call tools that read
    real workspace data from sibling .NET services.
    """

    model_config = {"env_prefix": "ASSISTANT_CHAT_"}

    api_key: str = ""
    model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.4
    max_tool_iterations: int = 5
    # Flush a streamed chunk to Redis every N characters rather than per-token — keeps
    # Redis Stream / SignalR traffic bounded, matching the rest of the pipeline's coarse
    # buffered-unit convention (STT/TTS/AI-assistant results are never per-token either).
    chunk_flush_chars: int = 40
    workspace_service_url: str = "http://localhost:5106"
    transcript_service_url: str = "http://localhost:5103"
    translation_room_service_url: str = "http://localhost:5102"


class SuggestionSettings(BaseSettings):
    """Inline transcript-suggestion worker settings.

    A third, distinct assistant surface: AssistantSettings summarizes a finished
    meeting, ChatAssistantSettings answers explicit questions, and this one decides
    — unprompted, mid-meeting — whether a just-transcribed segment deserves a short
    inline hint. Nothing here triggers on a user action, so every default below is
    tuned to stay silent rather than to be helpful.
    """

    model_config = {"env_prefix": "SUGGESTION_"}

    # Ships disabled. The worker consumes stt:results in its own consumer group, so it
    # can be rolled out to production and verified healthy while producing nothing at
    # all; flipping this is the only step that changes user-visible behaviour.
    enabled: bool = False
    api_key: str = ""

    # Two-stage models. Nearly every segment stops at the decide stage, so that call has
    # to be cheap enough to run ~150 times per meeting; only the few that pass reach the
    # larger generate model.
    decide_model: str = "gpt-4o-mini"
    generate_model: str = "gpt-4.1"
    decide_max_tokens: int = 64  # a {should_suggest, category, confidence, reason} object
    generate_max_tokens: int = 200
    temperature: float = 0.2
    # A hung request would stall this consumer's whole loop, and a suggestion that arrives
    # after the conversation has moved on is worse than none — fail fast and stay quiet.
    request_timeout_seconds: float = 8.0

    # Stage-1 gate. The decide prompt is instructed to default to false, and anything it
    # is not clearly confident about is dropped rather than shown.
    min_confidence: float = 0.7

    # Spam control. These are enforced in Redis, not in worker memory: the production
    # chart runs AI workers with replicas >= 2 (see deploy/k3s/chart/values.yaml), and
    # segments from one room are spread across replicas by the consumer group — so a
    # per-process counter would let each replica spend the full budget independently and
    # multiply the real cap by the replica count.
    cooldown_seconds: int = 45
    max_per_meeting: int = 15
    # Bounds the cap counter's lifetime — longer than any realistic meeting, so a room
    # can never silently regain budget mid-session, and short enough that keys expire on
    # their own for rooms that end without a cleanup signal.
    state_ttl_seconds: int = 14400  # 4h

    # Stage-0 heuristics — reject before spending a single token.
    min_words: int = 5
    # AVG LOGPROB, not a 0-1 confidence. The `confidence` field on stt:results carries
    # the model's average token logprob straight through, so it is always <= 0 — verified
    # against production data, where 1,422 stored segments span -0.699 to 0.000 and never
    # once reach a positive value.
    #
    # This was 0.6, a threshold on a 0-1 scale that does not exist here, so the stage-0
    # gate rejected EVERY segment: the worker would have run, stayed healthy, consumed the
    # stream and produced not one suggestion. That stayed invisible while
    # SUGGESTION_ENABLED was false and would have surfaced as "the feature does nothing"
    # the moment it was switched on.
    #
    # -0.35 matches the two sibling gates on the same field (stt_worker/worker.py:59 and
    # translation_worker/worker.py:99) and keeps roughly the cleaner two thirds of real
    # production speech, which suits a feature whose whole design is to stay quiet.
    min_stt_confidence: float = -0.35

    # Rolling transcript window handed to the decide stage. Enough to tell a genuine
    # open question from a mid-sentence fragment without resending the whole meeting.
    window_size: int = 8
    # A suggestion renders as a one-line strip above a transcript bubble; anything longer
    # is truncated by the UI anyway, so bound it at the source.
    max_suggestion_chars: int = 140


class EmbeddingSettings(BaseSettings):
    """Knowledge embedding settings for WarpBot RAG."""

    model_config = {"env_prefix": "EMBEDDING_"}

    provider: str = "openai"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 64
    timeout_ms: int = 30000
    # How many index-request messages (each = one document/transcript/glossary source, itself
    # made of one or more chunks) EmbeddingWorker processes at once. Unlike stt/tts/translation,
    # these jobs are independent of each other and I/O-bound (an OpenAI embed call + a Qdrant
    # upsert), so this is a real throughput win rather than a correctness risk — see
    # Keep embedding pressure bounded while sharing Redis with real-time audio.
    # Increase deliberately after measuring Redis timeout/error rates.
    concurrency: int = 2


class DatabaseSettings(BaseSettings):
    """Postgres connection settings for the billing settlement worker.

    No AI worker wrote to Postgres before billing_worker — everything else in
    warptalk-ai is Redis-only. Keep this settings class scoped to billing_worker
    only; do not reach for it from stt/translation/tts workers, which must stay
    on the real-time Redis Streams path.
    """

    model_config = {"env_prefix": "BILLING_DB_"}

    dsn: str = "postgresql://postgres:postgres@localhost:5432/warptalk"
    min_pool_size: int = 1
    max_pool_size: int = 5


class BillingSettings(BaseSettings):
    """Billing settlement worker settings."""

    model_config = {"env_prefix": "BILLING_"}

    database: DatabaseSettings = DatabaseSettings()
    # Subscription lookups (translation_room_id -> subscription_id) are cached for the
    # room's lifetime — refresh periodically in case a workspace's active subscription
    # changes mid-room (plan upgrade/downgrade).
    subscription_cache_ttl_seconds: int = 300


class VectorDbSettings(BaseSettings):
    """Vector database settings for text/RAG embeddings."""

    model_config = {"env_prefix": "VECTOR_DB_"}

    provider: str = "qdrant"
    url: str = "http://localhost:6333"
    api_key: str = ""
    distance_metric: str = "cosine"


class SecuritySettings(BaseSettings):
    """Security scanning worker settings."""

    model_config = {"env_prefix": "SECURITY_"}

    api_key: str = ""
    model: str = "gpt-4o-mini"
    max_tokens: int = 2000
    temperature: float = 0.0
    max_analyze_length: int = 20000
    result_ttl_seconds: int = 300
