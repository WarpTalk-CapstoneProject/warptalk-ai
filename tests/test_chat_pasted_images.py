"""WT-474 — pasted screenshots reaching the model.

The behaviours worth pinning are the ones that decide whether a screenshot is USABLE as the
question: it has to land on the last user turn, keep the text it was pasted with, and be dropped
rather than fail the turn when it is malformed.
"""

from __future__ import annotations

import json
from typing import Any

from ai_assistant_worker.chat_worker import (
    MAX_IMAGES_PER_TURN,
    _attach_images,
)

PNG = "data:image/png;base64,iVBORw0KGgo="
JPEG = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="


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

    _attach_images(conversation, json.dumps([PNG]), logger)

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

    _attach_images(conversation, json.dumps([PNG]), logger)

    assert conversation[0]["content"] == [
        {"type": "input_text", "text": "look at this"},
        {"type": "input_image", "image_url": PNG},
    ]
    assert conversation[1]["content"] == "sure"


def test_multiple_images_keep_their_order() -> None:
    conversation = _conversation(("user", "before and after"))
    _attach_images(conversation, json.dumps([PNG, JPEG]), RecordingLogger())

    assert conversation[0]["content"][1:] == [
        {"type": "input_image", "image_url": PNG},
        {"type": "input_image", "image_url": JPEG},
    ]


def test_images_beyond_the_cap_are_dropped_and_announced() -> None:
    conversation = _conversation(("user", "lots"))
    logger = RecordingLogger()

    _attach_images(conversation, json.dumps([PNG] * (MAX_IMAGES_PER_TURN + 3)), logger)

    images = [part for part in conversation[0]["content"] if part["type"] == "input_image"]
    assert len(images) == MAX_IMAGES_PER_TURN
    assert "chat_images_truncated" in logger.warned


def test_a_non_image_data_url_is_dropped_without_failing_the_turn() -> None:
    """The question is still a question.

    Refusing to answer because one attachment was a PDF serves nobody — but the drop is logged.
    """
    conversation = _conversation(("user", "and this?"))
    logger = RecordingLogger()

    _attach_images(
        conversation,
        json.dumps(["data:application/pdf;base64,JVBER", PNG]),
        logger,
    )

    assert conversation[0]["content"] == [
        {"type": "input_text", "text": "and this?"},
        {"type": "input_image", "image_url": PNG},
    ]
    assert "chat_image_rejected" in logger.warned


def test_an_oversized_image_is_dropped() -> None:
    conversation = _conversation(("user", "huge"))
    logger = RecordingLogger()
    oversized = "data:image/png;base64," + ("A" * 7_000_001)

    _attach_images(conversation, json.dumps([oversized]), logger)

    # Nothing attached, so the turn is left exactly as it was — a plain string, not an array of one
    # text part, because rewriting it would change the request for no reason.
    assert conversation[0]["content"] == "huge"
    assert any(
        event == "chat_image_rejected" and kwargs.get("reason") == "too_large"
        for event, kwargs in logger.warnings
    )


def test_no_images_leaves_the_conversation_untouched() -> None:
    conversation = _conversation(("user", "just text"))
    _attach_images(conversation, "", RecordingLogger())
    assert conversation[0]["content"] == "just text"


def test_unparseable_json_is_logged_and_ignored() -> None:
    conversation = _conversation(("user", "hm"))
    logger = RecordingLogger()

    _attach_images(conversation, "{not json", logger)

    assert conversation[0]["content"] == "hm"
    assert "chat_images_unparseable" in logger.warned


def test_images_with_no_user_turn_are_dropped() -> None:
    """A history with no user turn is a caller bug, not a reason to crash the turn."""
    conversation = _conversation(("assistant", "I spoke first"))
    logger = RecordingLogger()

    _attach_images(conversation, json.dumps([PNG]), logger)

    assert conversation[0]["content"] == "I spoke first"
    assert "chat_images_dropped" in logger.warned


def test_an_already_multimodal_turn_is_appended_to() -> None:
    conversation: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "existing"}],
        }
    ]

    _attach_images(conversation, json.dumps([PNG]), RecordingLogger())

    assert conversation[0]["content"] == [
        {"type": "input_text", "text": "existing"},
        {"type": "input_image", "image_url": PNG},
    ]
