"""WT-474 — pasted screenshots reaching the model.

The behaviours worth pinning are the ones that decide whether a screenshot is USABLE as the
question: it has to land on the last user turn, keep the text it was pasted with, and be dropped
rather than fail the turn when it is malformed.
"""

from __future__ import annotations

import json
from typing import Any

from ai_assistant_worker.chat_worker import (
    MAX_ATTACHMENTS_PER_TURN,
    _attach_attachments,
)

PNG = "data:image/png;base64,iVBORw0KGgo="
JPEG = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
PDF = "data:application/pdf;base64,JVBERi0xLjQK"


def image(data_url: str = PNG) -> dict[str, str]:
    return {"name": "screenshot.png", "mimeType": "image/png", "dataUrl": data_url}


def document(data_url: str = PDF, name: str = "contract.pdf") -> dict[str, str]:
    return {"name": name, "mimeType": "application/pdf", "dataUrl": data_url}


class RecordingLogger:
    """Captures structlog-style calls so the tests can assert a drop was ANNOUNCED.

    A silently ignored screenshot looks like a model that cannot see, which is the one failure mode
    that would send somebody debugging the wrong layer.
    """

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, Any]]] = []
        self.infos: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **kwargs: Any) -> None:
        self.warnings.append((event, kwargs))

    def info(self, event: str, **kwargs: Any) -> None:
        self.infos.append((event, kwargs))

    @property
    def warned(self) -> list[str]:
        return [event for event, _ in self.warnings]


def _conversation(*turns: tuple[str, str]) -> list[dict[str, Any]]:
    return [{"role": role, "content": content} for role, content in turns]


def test_image_lands_on_the_last_user_turn_beside_its_text() -> None:
    conversation = _conversation(
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "what is wrong with this screen?"),
    )
    logger = RecordingLogger()

    _attach_attachments(conversation, json.dumps([image()]), logger)

    # The text survives: a content array holding only the image would arrive as a turn with no
    # question in it, and the model answers the text it was given.
    assert conversation[2]["content"] == [
        {"type": "input_text", "text": "what is wrong with this screen?"},
        {"type": "input_image", "image_url": PNG},
    ]
    # Earlier turns are untouched.
    assert conversation[0]["content"] == "hello"
    assert conversation[1]["content"] == "hi"


def test_image_skips_a_trailing_assistant_turn() -> None:
    """The LAST USER turn, not the last turn.

    A tool result or an assistant reply can be last, and hanging the image off either puts it where
    the model does not read it as the question.
    """
    conversation = _conversation(
        ("user", "look at this"),
        ("assistant", "sure"),
    )
    logger = RecordingLogger()

    _attach_attachments(conversation, json.dumps([image()]), logger)

    assert conversation[0]["content"] == [
        {"type": "input_text", "text": "look at this"},
        {"type": "input_image", "image_url": PNG},
    ]
    assert conversation[1]["content"] == "sure"


def test_multiple_images_keep_their_order() -> None:
    conversation = _conversation(("user", "before and after"))
    _attach_attachments(conversation, json.dumps([image(), image(JPEG)]), RecordingLogger())

    assert conversation[0]["content"][1:] == [
        {"type": "input_image", "image_url": PNG},
        {"type": "input_image", "image_url": JPEG},
    ]


def test_images_beyond_the_cap_are_dropped_and_announced() -> None:
    conversation = _conversation(("user", "lots"))
    logger = RecordingLogger()

    _attach_attachments(
        conversation, json.dumps([image()] * (MAX_ATTACHMENTS_PER_TURN + 3)), logger
    )

    images = [part for part in conversation[0]["content"] if part["type"] == "input_image"]
    assert len(images) == MAX_ATTACHMENTS_PER_TURN
    assert "chat_attachments_truncated" in logger.warned


def test_a_non_image_data_url_is_dropped_without_failing_the_turn() -> None:
    """The question is still a question.

    Refusing to answer because one attachment was a PDF serves nobody — but the drop is logged.
    """
    conversation = _conversation(("user", "and this?"))
    logger = RecordingLogger()

    _attach_attachments(
        conversation,
        json.dumps(
            [
                {
                    "name": "x.bin",
                    "mimeType": "application/octet-stream",
                    "dataUrl": "data:application/octet-stream;base64,AAAA",
                },
                image(),
            ]
        ),
        logger,
    )

    assert conversation[0]["content"] == [
        {"type": "input_text", "text": "and this?"},
        {"type": "input_image", "image_url": PNG},
    ]
    assert "chat_attachment_rejected" in logger.warned


def test_an_oversized_image_is_dropped() -> None:
    conversation = _conversation(("user", "huge"))
    logger = RecordingLogger()
    oversized = "data:image/png;base64," + ("A" * 7_000_001)

    _attach_attachments(conversation, json.dumps([image(oversized)]), logger)

    # Nothing attached, so the turn is left exactly as it was — a plain string, not an array of one
    # text part, because rewriting it would change the request for no reason.
    assert conversation[0]["content"] == "huge"
    assert any(
        event == "chat_attachment_rejected" and kwargs.get("reason") == "too_large"
        for event, kwargs in logger.warnings
    )


def test_no_images_leaves_the_conversation_untouched() -> None:
    conversation = _conversation(("user", "just text"))
    _attach_attachments(conversation, "", RecordingLogger())
    assert conversation[0]["content"] == "just text"


def test_unparseable_json_is_logged_and_ignored() -> None:
    conversation = _conversation(("user", "hm"))
    logger = RecordingLogger()

    _attach_attachments(conversation, "{not json", logger)

    assert conversation[0]["content"] == "hm"
    assert "chat_attachments_unparseable" in logger.warned


def test_images_with_no_user_turn_are_dropped() -> None:
    """A history with no user turn is a caller bug, not a reason to crash the turn."""
    conversation = _conversation(("assistant", "I spoke first"))
    logger = RecordingLogger()

    _attach_attachments(conversation, json.dumps([image()]), logger)

    assert conversation[0]["content"] == "I spoke first"
    assert "chat_attachments_dropped" in logger.warned


def test_an_already_multimodal_turn_is_appended_to() -> None:
    conversation: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "existing"}],
        }
    ]

    _attach_attachments(conversation, json.dumps([image()]), RecordingLogger())

    assert conversation[0]["content"] == [
        {"type": "input_text", "text": "existing"},
        {"type": "input_image", "image_url": PNG},
    ]


def test_a_pdf_becomes_an_input_file_with_its_filename() -> None:
    """A document is a different content type from an image, and it needs its name.

    `filename` is required by the Responses API, and it is also the only handle the model has for
    referring to one document among several — "the contract" is not resolvable from bytes alone.
    """
    conversation = _conversation(("user", "does this allow subletting?"))

    _attach_attachments(conversation, json.dumps([document()]), RecordingLogger())

    assert conversation[0]["content"] == [
        {"type": "input_text", "text": "does this allow subletting?"},
        {"type": "input_file", "filename": "contract.pdf", "file_data": PDF},
    ]


def test_a_document_with_no_name_still_gets_a_filename() -> None:
    """The API rejects input_file without one, so a nameless upload must not 400 the turn."""
    conversation = _conversation(("user", "read this"))

    _attach_attachments(
        conversation,
        json.dumps([{"name": "", "mimeType": "application/pdf", "dataUrl": PDF}]),
        RecordingLogger(),
    )

    part = conversation[0]["content"][1]
    assert part["type"] == "input_file"
    assert part["filename"] == "attachment"


def test_images_and_documents_mix_in_one_turn() -> None:
    conversation = _conversation(("user", "compare these"))

    _attach_attachments(conversation, json.dumps([image(), document()]), RecordingLogger())

    types = [part["type"] for part in conversation[0]["content"]]
    assert types == ["input_text", "input_image", "input_file"]


def test_the_mime_type_comes_from_the_bytes_not_the_caller() -> None:
    """A caller-supplied mimeType that disagrees with the data URL must not decide the shape.

    Trusting the field would let an image be submitted as `input_file` (or worse), and the bytes are
    the only side that decides how OpenAI reads them.
    """
    conversation = _conversation(("user", "what is this?"))

    _attach_attachments(
        conversation,
        # Claims to be a PDF; the data URL says PNG.
        json.dumps([{"name": "lying.pdf", "mimeType": "application/pdf", "dataUrl": PNG}]),
        RecordingLogger(),
    )

    assert conversation[0]["content"][1] == {"type": "input_image", "image_url": PNG}


def test_a_bare_data_url_string_is_still_accepted() -> None:
    """Wire compatibility with the first cut of this feature.

    It published plain strings. A message already sitting on the Redis Stream when the worker
    restarts must not be dropped for using the older shape.
    """
    conversation = _conversation(("user", "legacy shape"))

    _attach_attachments(conversation, json.dumps([PNG]), RecordingLogger())

    assert conversation[0]["content"][1] == {"type": "input_image", "image_url": PNG}
