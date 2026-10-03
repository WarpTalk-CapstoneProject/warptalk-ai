"""WarpBot answers in the asker's language — or English — never in Vietnamese by default.

WHAT WAS REPORTED
    A reader on an English interface pressed "Draft this task" under a meeting suggestion. The
    hint quoted the meeting (Vietnamese, with a Japanese word in it), the request around it was
    English, and the ask card came back in Vietnamese: "NGƯỜI PHỤ TRÁCH — Ai sẽ là người gửi...".
    The old rule was "in the language the user wrote in", and the quote was most of what was
    written.

WHAT THESE PIN
    - An explicit "Reply in X" wins; otherwise the user's own words, not quoted material.
    - English is the fallback, and Vietnamese is named as NOT the default.
    - ask_user is bound to the same rule, because the card is where it showed.
"""

from __future__ import annotations

from ai_assistant_worker.chat_templates import PERSONA
from ai_assistant_worker.chat_tools import TOOLS_BY_NAME


def test_persona_states_the_reply_language_order() -> None:
    assert "if the message says which language to reply in, use that" in PERSONA
    assert "user's OWN words" in PERSONA
    assert "meeting quote" in PERSONA
    assert "reply in English" in PERSONA


def test_persona_refuses_a_vietnamese_default() -> None:
    assert "Never default to Vietnamese" in PERSONA


def test_ask_user_cards_follow_the_reply_language() -> None:
    description = TOOLS_BY_NAME["ask_user"].description
    assert "language you are replying in" in description
    assert "not in the language of the meeting text" in description
