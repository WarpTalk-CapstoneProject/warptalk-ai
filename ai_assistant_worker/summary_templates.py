"""What shape a meeting summary takes, as data rather than a hardcoded prompt.

There used to be one summary: a "concise overview paragraph", plus decisions and action
items. Every meeting got that shape whether it was a standup, an interview or a customer
demo — and the word *concise* in the prompt is what produced the thin, characterless
paragraphs the owner complained about.

A template names the sections a kind of meeting actually has, and the prompt and the output
schema are both generated from it. Adding a new kind of meeting is adding a record here, not
editing a prompt string.

CITATIONS
    Every item a template produces must carry `atMs` — the moment in the meeting it came
    from. This is not decoration. A claim that cannot point at a moment in the transcript is
    a claim the model invented, so requiring the citation is the anti-fabrication mechanism;
    it does far more than the sentence "do not invent content" ever did. The reader gets to
    check, which is the point: a summary nobody can verify is a summary nobody should trust.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SummarySection:
    """One section of a summary: what it is called and what belongs in it."""

    key: str
    title: str
    guidance: str
    #  "paragraph" renders as prose; "items" as a cited list; "sentences" as prose that has
    #  been broken along its sentence boundaries so each unit can carry its own moments.
    kind: str = "items"


@dataclass(frozen=True)
class SummaryTemplate:
    key: str
    label: str
    description: str
    sections: tuple[SummarySection, ...] = field(default_factory=tuple)


_OVERVIEW = SummarySection(
    key="summary",
    title="Overview",
    kind="paragraph",
    guidance=(
        "What the meeting was for and what actually happened in it, in 3–6 sentences. "
        "Name the people, the systems and the numbers that were discussed rather than "
        "describing the conversation in the abstract. Never write that a meeting was "
        "'informal' or 'had no clear agenda' — say what was talked about instead."
    ),
)

_NARRATIVE = SummarySection(
    key="narrative",
    title="What happened",
    kind="sentences",
    guidance=(
        "The overview, broken into units a reader can check: each element is ONE complete "
        "sentence, and read in order the sentences must form a single continuous paragraph. "
        "This is prose split along its sentence boundaries, not a bullet list. Write 3–6 "
        "sentences. Every sentence carries `atMs`, and `alsoAtMs` holds the OTHER moments the "
        "same sentence rests on — at most two more, so no sentence stands on more than three "
        "moments. Only join moments that are far apart when the later one directly answers, "
        "confirms or reverses the earlier one; everything else becomes two sentences. "
        "Gathering scattered remarks into one general observation is the General template's "
        "job, not this one — a sentence that cannot point at where it was said is exactly "
        "what this template exists to prevent. A merged sentence's `atMs` is the EARLIEST of "
        "its moments, and the sentences are ordered by that moment, so the summary runs in "
        "meeting order. Stay close to what was said: condense the words, do not interpret "
        "them."
    ),
)

_DECISIONS = SummarySection(
    key="decisions",
    title="Decisions",
    guidance=(
        "Each decision that was actually settled, in the words of the meeting. A decision "
        "is something that changed as a result of the conversation — not a topic that was "
        "raised. If a question was left open, it belongs in open questions, not here."
    ),
)

_ACTION_ITEMS = SummarySection(
    key="actionItems",
    title="Action items",
    guidance=(
        "Each commitment somebody made, with the owner's name if it was said and an empty "
        "owner if it was not. Include any deadline that was spoken. Do not invent an owner "
        "to make the item look complete."
    ),
)

_OPEN_QUESTIONS = SummarySection(
    key="openQuestions",
    title="Open questions",
    guidance=(
        "Questions raised and not answered by the end of the meeting. These are the most "
        "useful part of a summary for somebody who missed it, and the easiest to lose."
    ),
)

GENERAL = SummaryTemplate(
    key="general",
    label="General meeting",
    description="Overview, decisions, action items and anything left unresolved.",
    sections=(_OVERVIEW, _DECISIONS, _ACTION_ITEMS, _OPEN_QUESTIONS),
)

STANDUP = SummaryTemplate(
    key="standup",
    label="Standup",
    description="Per-person progress, plans and blockers.",
    sections=(
        _OVERVIEW,
        SummarySection(
            key="progress",
            title="Progress",
            guidance="What each person reported finishing, attributed by name.",
        ),
        SummarySection(
            key="plans",
            title="Plans",
            guidance="What each person said they would do next, attributed by name.",
        ),
        SummarySection(
            key="blockers",
            title="Blockers",
            guidance=(
                "Anything stated as blocking someone, and who they said they needed it "
                "from. A blocker with no owner is still a blocker — record it."
            ),
        ),
        _ACTION_ITEMS,
    ),
)

INTERVIEW = SummaryTemplate(
    key="interview",
    label="Candidate interview",
    description="Background, answers, strengths, concerns and next steps.",
    sections=(
        _OVERVIEW,
        SummarySection(
            key="background",
            title="Background",
            guidance="The candidate's stated experience, in their own framing.",
        ),
        SummarySection(
            key="strengths",
            title="Strengths",
            guidance=(
                "Evidence of capability the candidate actually gave. Quote what they "
                "described doing rather than grading them."
            ),
        ),
        SummarySection(
            key="concerns",
            title="Concerns",
            guidance=(
                "Gaps or hesitations that came up in the conversation. Report what was "
                "said; do not infer a verdict the interviewers did not state."
            ),
        ),
        _OPEN_QUESTIONS,
        _ACTION_ITEMS,
    ),
)

DEMO = SummaryTemplate(
    key="demo",
    label="Product demo",
    description="What was shown, how it landed, objections and follow-ups.",
    sections=(
        _OVERVIEW,
        SummarySection(
            key="shown",
            title="What was shown",
            guidance="Each capability demonstrated, in the order it was shown.",
        ),
        SummarySection(
            key="reactions",
            title="Reactions",
            guidance="What the audience said about it — praise and doubt alike.",
        ),
        SummarySection(
            key="objections",
            title="Objections",
            guidance=(
                "Concerns raised that were not resolved during the call. Losing these is "
                "how a demo gets remembered as going better than it did."
            ),
        ),
        _ACTION_ITEMS,
    ),
)

TECHNICAL = SummaryTemplate(
    key="technical",
    label="Technical discussion",
    description="Problems, options weighed, what was chosen and why.",
    sections=(
        _OVERVIEW,
        SummarySection(
            key="problems",
            title="Problems raised",
            guidance="Each concrete problem or symptom described, with the detail given.",
        ),
        SummarySection(
            key="options",
            title="Options considered",
            guidance=(
                "Each approach weighed, with the trade-off that was stated for it. An "
                "option recorded without its trade-off is not worth recording."
            ),
        ),
        _DECISIONS,
        _OPEN_QUESTIONS,
        _ACTION_ITEMS,
    ),
)

#: The one template with no overview paragraph, and that absence is the whole design. An
#: overview is the part of a summary nobody can check — it draws on the meeting in general,
#: so a fabricated clause hides in it perfectly. Replacing it with sentences that each name
#: their moments costs the reader nothing (read end to end they are still a paragraph) and
#: leaves the model nowhere to put an uncitable claim.
TRACEABLE = SummaryTemplate(
    key="traceable",
    label="Traceable summary",
    description="Every sentence carries the moments it came from.",
    sections=(_NARRATIVE, _DECISIONS, _ACTION_ITEMS, _OPEN_QUESTIONS),
)

TEMPLATES: dict[str, SummaryTemplate] = {
    template.key: template
    for template in (GENERAL, STANDUP, INTERVIEW, DEMO, TECHNICAL, TRACEABLE)
}

DEFAULT_TEMPLATE_KEY = GENERAL.key


def resolve_template(key: str | None) -> SummaryTemplate:
    """An unknown or missing key falls back to General rather than failing.

    A summary that comes out in the wrong shape is recoverable; a meeting that ends with no
    summary at all because somebody sent a typo is not.
    """
    if not key:
        return GENERAL
    return TEMPLATES.get(key.strip().lower(), GENERAL)


def build_system_prompt(template: SummaryTemplate) -> str:
    """Generate the system prompt and the JSON shape from the template."""
    lines = [
        "You are a meeting analyst. Read the transcript and return a single JSON object "
        "only — no markdown, no commentary.",
        "",
        "Every line of the transcript is prefixed with the moment it was spoken, as "
        "[t=<milliseconds>]. You MUST cite that number in `atMs` for every item you "
        "produce, using the moment the point was actually made.",
        "",
        "If you cannot point to a moment in the transcript for a statement, do not make "
        "the statement. An uncitable claim is a fabricated claim, and a short honest "
        "summary is worth more than a full invented one.",
        "",
        f"This is a {template.label.lower()}. Return exactly this shape:",
        "{",
    ]

    shape: list[str] = []
    for section in template.sections:
        if section.kind == "paragraph":
            shape.append(f'  "{section.key}": "<text>",')
        elif section.kind == "sentences":
            # `alsoAtMs` lives on this kind alone. A cited list is one claim per moment and
            # stays that way; only a sentence carved out of a paragraph can legitimately
            # rest on more than one.
            shape.append(
                f'  "{section.key}": [{{"text": "<text>", "atMs": <number>, '
                '"alsoAtMs": [<number>, ...]}],'
            )
        elif section.key == "actionItems":
            shape.append(
                f'  "{section.key}": [{{"task": "<text>", "owner": "<name or empty>", '
                '"atMs": <number>}],'
            )
        else:
            shape.append(f'  "{section.key}": [{{"text": "<text>", "atMs": <number>}}],')

    # "citations" is the overview paragraph's evidence and nothing else's. A template built
    # without an overview (traceable) has no field for it to point at, so asking for it
    # invites the model to invent one — and a JSON key nobody declared is the fabrication
    # this whole file is arranged to make impossible.
    has_paragraph = any(section.kind == "paragraph" for section in template.sections)
    if has_paragraph:
        shape.append('  "citations": [{"key": "summary", "atMs": <number>}]')
    else:
        # The last entry of an object carries no comma, and the citations line was it.
        shape[-1] = shape[-1].rstrip(",")

    lines.extend(shape)
    lines.append("}")
    lines.append("")
    if has_paragraph:
        lines.append(
            'The "citations" array carries the moments the overview paragraph draws on — at '
            "least one, at most five."
        )
        lines.append("")
    lines.append("What belongs in each section:")
    for section in template.sections:
        lines.append(f"- {section.key} ({section.title}): {section.guidance}")

    lines.append("")
    lines.append(
        "Write in the language the meeting was held in. Summarise what the transcript "
        "actually contains, however short that is — two sentences of real content is a "
        "valid summary. Never claim the transcript is empty or has no substantive "
        "content: whether there is enough to summarise is decided before you are called, "
        "so you are only ever given a transcript that has something in it."
    )
    return "\n".join(lines)


def format_transcript_line(at_ms: int, speaker: str, text: str) -> str:
    """One transcript line, carrying the moment the model must cite.

    The offset is relative to the first segment, matching how the saved transcript is
    rendered (a base time plus a per-segment offset), so a cited `atMs` can be resolved
    back to a segment on the meeting page without any id correspondence between the live
    STT stream and the stored transcript.
    """
    return f"[t={max(at_ms, 0)}] [{speaker}] {text}"


#: The `[t=<ms>] [<speaker>] ` that `format_transcript_line` puts in front of every line.
#: The moment is captured so `transcript_offsets` can read it back; one pattern, so the set
#: we check citations against can never drift from the prefix we actually strip.
_TRANSCRIPT_LINE_PREFIX = re.compile(r"^\[t=(\d+)\]\s*\[[^\]]*\]\s*")


def spoken_text_only(transcript: str) -> str:
    """Just the words people said, with every timestamp and speaker label removed.

    WT-478: `format_transcript_line` emits a NON-EMPTY line even for a segment whose text is
    empty — `"[t=0] [Nhi] "` is 12 characters of pure scaffolding. A transcript made of those
    survives a `.strip()` check while containing nothing anybody said, so an emptiness test
    against the formatted string passes and the model is then asked to summarise punctuation.
    It answered the only way it could: by reporting the transcript was empty — and that
    sentence came back as a normal summary, which is what the user saw on screen.

    Emptiness is therefore tested against what this returns, never against the formatted
    transcript. Deliberately NOT a length threshold: the ticket asks for a summary of short
    meetings too, so the question is "did anyone say anything", not "did they say enough".
    """
    return "\n".join(
        stripped
        for line in transcript.splitlines()
        if (stripped := _TRANSCRIPT_LINE_PREFIX.sub("", line).strip())
    )


def transcript_offsets(transcript: str) -> set[int]:
    """Every moment we handed the model, as the set its `atMs` must be a member of.

    Requiring a citation only means something if the citation is checkable, and the check is
    exact membership with zero tolerance: not "near a real line", not "within a second of
    one". A nearest-match fallback would silently repair the single failure the citation
    exists to expose — a number the model produced from nothing. An `atMs` outside this set
    points at a moment that was never in the meeting, so the claim resting on it is not a
    claim about the meeting.

    Only the `[t=<ms>]` that opens a line is a moment. Somebody who says "[t=500]" out loud
    has said words, not marked a timestamp, and must not hand a fabricated citation a place
    to land.
    """
    return {
        int(match.group(1))
        for line in transcript.splitlines()
        if (match := _TRANSCRIPT_LINE_PREFIX.match(line))
    }
