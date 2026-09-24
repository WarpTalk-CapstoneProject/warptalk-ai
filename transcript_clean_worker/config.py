"""Settings for the clean transcript worker (WT-716).

Deliberately its own settings class rather than another entry in shared/config.py: nothing
else in the pipeline reads these, and every one of them is a knob on a stage that must be
able to be turned off in production without touching a worker that carries audio.

Environment prefix is ``TRANSCRIPT_CLEAN_``, so the field ``merge_gap_ms`` is
``TRANSCRIPT_CLEAN_MERGE_GAP_MS``.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class TranscriptCleanSettings(BaseSettings):
    """Clean transcript worker settings."""

    model_config = {"env_prefix": "TRANSCRIPT_CLEAN_"}

    # Ships ON, unlike suggestion_worker's flag, because the stage is additive: it publishes a
    # NEW stream (transcript:clean) that no existing consumer reads, and it never edits, delays
    # or drops a raw segment. Turning it off stops the clean view from filling; it cannot
    # damage the live path either way.
    enabled: bool = True

    # Falls back to the shared OPENAI_API_KEY (see resolve_openai_api_key). An empty key is not
    # a startup failure here: the deterministic prepass tier still produces every line, so the
    # worker degrades to rule-only cleaning rather than to silence.
    api_key: str = ""

    # HOW LONG A PAUSE ENDS A SENTENCE.
    #
    # STT cuts on silence, not on grammar, so "えーと、あのー、来週の会議ですが" and
    # "火曜日に変更になりました" arrive as two segments 500 ms apart and are one sentence. The
    # threshold is the gap above which a continuation stops being a continuation: measured
    # conversational within-sentence pauses cluster well under a second, while the gap between
    # two sentences by the same speaker is typically longer. 1200 ms sits above the first and
    # below most of the second, and being wrong is cheap in one direction only — a line split
    # in two is readable, a line that swallowed the next sentence is not.
    merge_gap_ms: int = 1200

    # A sentence nobody ever finished still has to be published. Someone who talks for thirty
    # seconds without a terminal punctuation mark gets a line cut at that point, as is: no
    # words are added to round it off (see the segmenter — deletion-only means exactly that).
    max_sentence_ms: int = 30000

    # The speaker stopped and nothing else arrived. Without this, the last sentence of a turn
    # would wait for the NEXT segment — which, at the end of a meeting, never comes.
    idle_flush_ms: int = 4000

    # The LLM tier's model. NOT the translation model (gpt-5.4-nano): a reasoning model
    # over-deletes on this task — it reasons its way to a tidier sentence than the one that was
    # said, which is precisely what the deletion-index protocol exists to prevent — and it
    # spends hidden reasoning tokens on a job whose whole output is a list of integers. This is
    # the non-reasoning chat model the text stages used before that switch; `completion_options`
    # still governs which parameters it is sent.
    model: str = "gpt-4.1-mini"

    # The clean tier is post-hoc: revision 0 is already published and readable before this call
    # starts, so a slow model costs a polish, never a line. Fail fast and keep the prepass line.
    llm_timeout_s: float = 8.0

    # How much of a sentence the LLM may delete. A hesitant sentence really can be a third
    # filler; a model that removes more than this is summarising, not cleaning, and its answer
    # is discarded in favour of the prepass line. Lower than the guardian's 0.5 because this
    # tier is given a sentence the prepass has already had a pass at.
    max_delete_ratio: float = 0.4

    # Bound on concurrent LLM calls in flight for this worker. The calls are independent and
    # nothing waits on them, so this is a spend/rate-limit bound rather than an ordering one.
    llm_concurrency: int = 4
