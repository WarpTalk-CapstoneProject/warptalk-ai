"""What "today" means to WarpBot.

The reported failure was a deadlock, reproduced verbatim:

    bot   Vui lòng chọn ngày và ghi rõ giờ bắt đầu, ví dụ: ngày mai lúc 09:30
    user  Hôm nay
    bot   Vui lòng chọn giờ bắt đầu cuộc họp hôm nay.
    user  09:00
    bot   Tôi cần ngày cụ thể theo định dạng YYYY-MM-DD...

Every field `create_meeting` needs was on the table. The model could not turn "hôm nay" into a
date because nothing in the worker had ever told it what day it was — not the persona, not the
templates, not a tool schema. Its own example, "ngày mai lúc 09:30", was advice it could not
follow.

Guessing is worse than looping: a guessed year books a meeting eleven months away and looks like
it worked. So the date is supplied, and these tests pin that it is supplied correctly.
"""

from __future__ import annotations

import inspect
from datetime import datetime

from ai_assistant_worker import chat_worker
from ai_assistant_worker.chat_worker import WORKSPACE_TIMEZONE, _now_message


def _at(text: str) -> str:
    return _now_message(datetime.fromisoformat(text).replace(tzinfo=WORKSPACE_TIMEZONE))


def test_the_message_carries_an_iso_date_the_model_can_copy_into_a_tool_call() -> None:
    # create_meeting takes YYYY-MM-DD. A prose-only date ("Friday the 14th") would leave the model
    # doing the format conversion, which is the step it was already failing.
    assert "2026-08-14" in _at("2026-08-14T09:00")


def test_it_also_spells_the_weekday_out() -> None:
    # "next Friday" cannot be resolved from a number alone.
    assert "Friday" in _at("2026-08-14T09:00")


def test_the_clock_is_vietnam_time() -> None:
    # Meetings are scheduled by people in Vietnam and "9 giờ sáng" means 9am there. UTC+7 has no
    # DST, so the offset is exact.
    assert WORKSPACE_TIMEZONE.utcoffset(None).total_seconds() == 7 * 3600
    assert "UTC+7" in _at("2026-08-14T09:00")


def test_it_tells_the_model_to_do_the_arithmetic_rather_than_ask() -> None:
    # The instruction is the fix, not the date on its own: with a date but no instruction the
    # model may still ask the user to convert, which is the exact loop being closed.
    message = _at("2026-08-14T09:00")
    assert "Never ask the user to convert a date" in message


def test_it_names_the_relative_forms_the_user_actually_used() -> None:
    message = _at("2026-08-14T09:00").lower()
    for phrase in ["today", "tomorrow", "cuối tuần này"]:
        assert phrase in message, f"{phrase!r} is not covered"


def test_a_time_with_no_timezone_is_read_as_vietnam_time() -> None:
    # Otherwise "09:00" is ambiguous and the model may resolve it as UTC — a meeting seven hours
    # off, which nobody notices until they miss it.
    assert "Vietnam time" in _at("2026-08-14T09:00")


def test_it_reads_the_real_clock_when_given_none() -> None:
    # The default path is the one production uses; a helper that only works with an injected
    # clock would be pinned by every test here and still broken in the worker.
    assert "UTC+7" in _now_message()


def test_the_message_is_actually_put_in_front_of_the_model() -> None:
    """The one that matters.

    Everything above passes on a `_now_message` that is never called — which is precisely how a
    fix ships, reads well, and changes nothing. This asserts the wiring: the worker builds its
    instructions from it, so removing the call from `instructions_parts` fails here rather than
    silently restoring the deadlock.
    """
    source = inspect.getsource(chat_worker.ChatAssistantWorker._run_agent_loop)

    assert "_now_message()" in source, (
        "_now_message is defined but never reaches the model's instructions"
    )
    assert "instructions_parts" in source
