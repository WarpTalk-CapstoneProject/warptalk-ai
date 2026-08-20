"""Which context the chat assistant reaches for, as data rather than one hardcoded prompt.

There used to be a single system prompt covering every question the assistant would ever be
asked. It named "members, terminology, recent meetings" as the tool-shaped things worth
looking up — transcripts and documents appeared nowhere in it — and it never said that the
entity id the assistant is handed on a meeting page IS the `meeting_id` argument to
get_transcript. So questions about what was said in a meeting got answered out of the chat
history the model already had in front of it. That is the whole of the "assistant doesn't
read the transcript" report: not a broken pipeline, a prompt that never asked.

A template names the grounding sources a *situation* has and the order to reach for them,
and the prompt is generated from it. A new situation is a new record here, not an edit to a
prompt string.

SITUATION, NOT PREFERENCE
    Templates are resolved from the request itself — its origin (@WarpBot inside a meeting's
    chat vs. the global "Ask WarpTalk" widget) and the page the user has open. Nobody picks
    one from a menu. A question asked inside a live meeting is a question about that meeting
    and the template says so; the same words typed on the documents page are not.

THE ID PROBLEM
    Ambient page context arrives as `entity_id=<uuid>` and nothing more. A bare uuid is not
    an instruction, and the model treated it as trivia. Every template that sits on a page
    with an entity therefore states what that id is in the vocabulary of the tools — which
    argument, of which tool. That single line is what turns transcript retrieval on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The shared persona every template embeds, and the one place the retrieve-before-answering
# rule is stated. Re-exported by chat_worker as SYSTEM_PROMPT because it is still literally
# the start of every prompt the worker sends.
PERSONA = (
    "You are WarpTalk AI, the assistant embedded in the WarpTalk real-time speech "
    "translation platform. Answer clearly and concisely, in the language the user wrote in.\n"
    "\n"
    "You do not know what was said in a meeting, what a document contains, or how this "
    "workspace translates a term until you look it up. Retrieve first, then answer. A "
    "question about meeting or document content is never answered from the conversation "
    "history alone — that history is what you and the user have said to each other, not the "
    "source material. If a lookup returns nothing, say so plainly instead of filling the gap."
)


@dataclass(frozen=True)
class ContextSource:
    """One place the assistant can get grounding from, and when it is the right one.

    `caveat` is the honest limit of the source. Naming it in the prompt is what stops the
    model from treating an empty result as an answer — semantic_search returning nothing
    means the workspace has nothing indexed, not that the meeting never happened.
    """

    tool: str
    use_when: str
    caveat: str = ""


@dataclass(frozen=True)
class EntityBinding:
    """What the ambient `entity_id` on this page actually is, per tool that consumes it.

    Tools disagree on the argument name for the same thing — get_room_detail takes
    `room_id`, get_transcript and get_meeting_summary take `meeting_id` — so the binding
    carries the pairs rather than one name.
    """

    noun: str
    arguments: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ChatTemplate:
    key: str
    label: str
    situation: str
    sources: tuple[ContextSource, ...] = field(default_factory=tuple)
    binding: EntityBinding | None = None
    style: str = ""


_TRANSCRIPT = ContextSource(
    tool="get_transcript",
    use_when=(
        "the question is about what was said, who said it, whether something came up, or "
        "asks for a quote. This is the source of truth for the content of a meeting"
    ),
    caveat=(
        "segments appear as they are spoken, so a live meeting's transcript is partial and "
        "its last segments are the most recent thing said. Each segment carries its own "
        "language — quote the one the speaker actually used and translate it yourself if the "
        "user wrote in another language"
    ),
)

_MEETING_SUMMARY = ContextSource(
    tool="get_meeting_summary",
    use_when=(
        "the user wants the gist, the decisions or the action items of a meeting that has "
        "already ended, rather than the words themselves"
    ),
    caveat=(
        "only exists once a meeting's transcript pipeline has finished; there is no way to "
        "generate one on demand. Fall back to get_transcript when there is no summary yet"
    ),
)

_DOCUMENT = ContextSource(
    tool="get_document",
    use_when="you have a document id and the question is about that document's content",
    caveat=(
        "the excerpt is the beginning of the extracted text, not the whole file, and is "
        "absent while a document is still ingesting"
    ),
)

_SEARCH_DOCUMENTS = ContextSource(
    tool="search_documents",
    use_when=(
        "the user names a document, refers to 'the spec'/'the contract'/'the deck', or asks "
        "what documents exist. Use it to turn a name into an id, then call get_document"
    ),
    caveat=(
        "matches on the document's name — punctuation, case and diacritics are ignored — and "
        "falls back to content matches when no name matches, so an empty answer from it means "
        "the workspace genuinely has nothing, not that the title was worded differently"
    ),
)

_TERMINOLOGY = ContextSource(
    tool="search_terminology",
    use_when=(
        "the question is about a term, an acronym, a product name, or how this workspace "
        "wants something translated. Check the glossary before translating any domain term "
        "yourself — the workspace's preferred wording outranks yours"
    ),
)

_SEMANTIC_SEARCH = ContextSource(
    tool="semantic_search",
    use_when=(
        "the user asks where something was discussed, or the answer could be anywhere across "
        "meetings and documents rather than in one you can name"
    ),
    caveat=(
        "it searches only what has been indexed, and returns passages to follow up on rather "
        "than whole sources. An empty result means nothing is indexed, not that nothing "
        "happened — say which meeting or document you could not find and offer to look "
        "directly if the user can name it"
    ),
)

_RECENT_MEETINGS = ContextSource(
    tool="list_recent_meetings",
    use_when=(
        "the user means a meeting you have no id for — 'yesterday's standup', 'the last call "
        "with the client'. Use it to find the id, then get_transcript or get_meeting_summary"
    ),
)

_MEMBERS = ContextSource(
    tool="search_workspace_members",
    use_when="the question is about a person in this workspace — their role, email, or status",
)

_MEETING_BINDING = EntityBinding(
    noun="translation room / meeting",
    arguments=(
        ("get_transcript", "meeting_id"),
        ("get_meeting_summary", "meeting_id"),
        ("get_room_detail", "room_id"),
    ),
)

GENERAL = ChatTemplate(
    key="general",
    label="General assistant",
    situation=(
        "The user opened the assistant from anywhere in the app. The question could be about "
        "anything in this workspace, so work out what kind of thing is being asked about "
        "before answering."
    ),
    sources=(
        _RECENT_MEETINGS,
        _TRANSCRIPT,
        _MEETING_SUMMARY,
        _SEARCH_DOCUMENTS,
        _DOCUMENT,
        _TERMINOLOGY,
        _SEMANTIC_SEARCH,
        _MEMBERS,
    ),
)

MEETING_CHAT = ChatTemplate(
    key="meeting_chat",
    label="In-meeting @WarpBot",
    situation=(
        "You were mentioned as @WarpBot in the chat of a meeting that is happening right now. "
        "Everyone in the meeting can read your reply. Assume every question is about THIS "
        "meeting unless the user clearly points somewhere else, and read the transcript before "
        "answering — the meeting chat messages you were given are the side conversation, not "
        "what was spoken."
    ),
    sources=(_TRANSCRIPT, _TERMINOLOGY, _SEARCH_DOCUMENTS, _DOCUMENT, _SEMANTIC_SEARCH, _MEMBERS),
    binding=_MEETING_BINDING,
    style=(
        "Keep it to a few sentences — this is a live chat panel beside a call, not a report. "
        "Lead with the answer. Attribute anything you quote to the speaker who said it."
    ),
)

MEETING = ChatTemplate(
    key="meeting",
    label="Meeting page",
    situation=(
        "The user has a specific meeting open. Treat questions as being about that meeting "
        "first, and read its transcript rather than describing what the page shows."
    ),
    sources=(
        _TRANSCRIPT,
        _MEETING_SUMMARY,
        _TERMINOLOGY,
        _SEARCH_DOCUMENTS,
        _DOCUMENT,
        _SEMANTIC_SEARCH,
        _MEMBERS,
    ),
    binding=_MEETING_BINDING,
)

DOCUMENT = ChatTemplate(
    key="document",
    label="Document page",
    situation=(
        "The user has a specific document open. Treat questions as being about that document "
        "first, and read it before answering."
    ),
    sources=(_DOCUMENT, _SEARCH_DOCUMENTS, _TERMINOLOGY, _SEMANTIC_SEARCH, _RECENT_MEETINGS),
    binding=EntityBinding(
        noun="workspace document",
        arguments=(("get_document", "document_id"),),
    ),
)

DOCUMENTS = ChatTemplate(
    key="documents",
    label="Document library",
    situation=(
        "The user is looking at the workspace's documents. Questions are most likely about "
        "which documents exist or what one of them says."
    ),
    sources=(_SEARCH_DOCUMENTS, _DOCUMENT, _SEMANTIC_SEARCH, _TERMINOLOGY),
)

HISTORY = ChatTemplate(
    key="history",
    label="Meeting history",
    situation=(
        "The user is looking at their past meetings. Questions are most likely about one of "
        "them — find which one before answering."
    ),
    sources=(
        _RECENT_MEETINGS,
        _MEETING_SUMMARY,
        _TRANSCRIPT,
        _SEMANTIC_SEARCH,
        _SEARCH_DOCUMENTS,
        _MEMBERS,
    ),
)

TEMPLATES: dict[str, ChatTemplate] = {
    template.key: template
    for template in (GENERAL, MEETING_CHAT, MEETING, DOCUMENT, DOCUMENTS, HISTORY)
}

DEFAULT_TEMPLATE_KEY = GENERAL.key

# The pageType values the web app registers (see assistant-context-store) plus the one
# MeetingChatService stamps on the @WarpBot path. An unlisted page falls back to GENERAL,
# which is a worse prompt but never a broken one.
_PAGE_TEMPLATES: dict[str, ChatTemplate] = {
    "meeting_chat": MEETING_CHAT,
    "in_meeting": MEETING,
    "room_detail": MEETING,
    "document_detail": DOCUMENT,
    "documents": DOCUMENTS,
    "history": HISTORY,
}

# Origin comes from the service that owns the stream, so it outranks page context: a
# @WarpBot mention is a meeting-chat turn whatever the browser last registered.
_ORIGIN_TEMPLATES: dict[str, ChatTemplate] = {
    "meeting_chat": MEETING_CHAT,
}


def resolve_template(origin: str | None = None, page_type: str | None = None) -> ChatTemplate:
    """Pick the template for this turn. Unknown values fall back to General.

    An assistant that answers with a slightly generic prompt is fine; one that raises because
    a new pageType shipped in the web app before this dict learned about it is not.
    """
    if origin:
        matched = _ORIGIN_TEMPLATES.get(origin.strip().lower())
        if matched is not None:
            return matched
    if page_type:
        matched = _PAGE_TEMPLATES.get(page_type.strip().lower())
        if matched is not None:
            return matched
    return GENERAL


def build_system_prompt(template: ChatTemplate) -> str:
    """Generate the system prompt from the template."""
    lines = [PERSONA, "", f"SITUATION — {template.label}", template.situation, ""]

    if template.binding is not None and template.binding.arguments:
        pairs = ", ".join(f"{argument} of {tool}" for tool, argument in template.binding.arguments)
        lines.extend(
            [
                "THE ID YOU WERE GIVEN",
                f"The entity_id in your page context is the id of a {template.binding.noun}. "
                f"Pass it as the {pairs}. You already have it — do not go looking for it and "
                "do not ask the user for it.",
                "",
            ]
        )

    lines.append("WHERE TO GET CONTEXT — in this order of preference for this situation:")
    for source in template.sources:
        entry = f"- {source.tool}: use when {source.use_when}."
        if source.caveat:
            entry += f" Limit: {source.caveat}."
        lines.append(entry)

    lines.extend(
        [
            "",
            "GROUND RULES",
            "- Look it up, then answer. Guessing from the conversation so far is the one "
            "failure mode that matters here.",
            "- One lookup is rarely enough. If the transcript answers half the question and "
            "the glossary the other half, call both before you reply.",
            "- Cite what you used: name the meeting, document or term the answer came from so "
            "the user can check you.",
            '- Never invent a quote, a speaker, a decision or a document. An answer of "the '
            "transcript doesn't cover that\" is a correct answer.",
            "- If a tool errors or returns nothing, say which one and what you were looking "
            "for. Do not silently fall back to your own knowledge.",
        ]
    )

    if template.style:
        lines.extend(["", "STYLE", template.style])

    return "\n".join(lines)
