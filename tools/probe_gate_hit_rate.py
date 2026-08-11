"""How often does the evidence gate actually stay shut in a real meeting?

_asr_repair_clause only saves tokens on calls with no suspected mishearing and no meeting
context. Whether that is most calls or almost none depends entirely on
TranslationWorker._select_relevance_context, which DROPS context whenever the utterance
shares a content token with recent speech — i.e. for ordinary on-topic talk.

So the saving is not a fixed 458 characters per call, and it is not "only the first few
utterances" either. This replays a plausible Vietnamese stand-up through the worker's own
context bookkeeping (a deque of recent source utterances, sliced [-3:], filtered by
_select_relevance_context) and counts it. No API calls: every function involved is local.
"""

from __future__ import annotations

import sys
from collections import deque

sys.path.insert(0, ".")

from translation_worker.translator import (  # noqa: E402
    _ASR_REPAIR_INSTRUCTION,
    _SYSTEM_PROMPT,
    _asr_repair_clause,
    _select_relevant_glossary_terms,
)
from translation_worker.worker import _select_relevance_context  # noqa: E402

# A stand-up as STT would hand it over: mostly clean, a few mishearings, some short turns.
MEETING = [
    "Chào mọi người, mình bắt đầu nhé",
    "Hôm qua tôi làm xong phần đăng nhập rồi",
    "Ừ đúng rồi",
    "Cái phần thanh toán còn một chỗ chưa chạy được",
    "Con cu bơ nét tự restart cái pod đó rồi",  # Kubernetes, fuzzy-matchable
    "Tôi sẽ kiểm tra lại log của service đó",
    "Ok",
    "Bên frontend đã merge nhánh của bạn chưa",
    "Cái gờ ra pha na đang không hiện số liệu",  # Grafana, NOT in glossary
    "Rồi tôi xem sau",
    "Sprint này còn ba ticket chưa xong",
    "Tôi nghĩ nên dời một cái sang tuần sau",
    "Cái cô đích nó báo lỗi khi tôi chạy lại",  # Codex, skeleton too short
    "Vâng",
    "Chiều nay mình deploy lên staging nhé",
    "Để tôi hỏi lại anh Tú về cái đó",
    "Phần báo cáo tôi làm gần xong rồi",
    "Không, cái đó để tuần sau đi",
    "Mọi người có câu hỏi gì trước khi kết thúc không",
    "Cảm ơn mọi người nhé",
]
GLOSSARY = [
    {"source": "staging", "target": "staging"},
    {"source": "Kubernetes", "target": "Kubernetes"},
    {"source": "sprint", "target": "sprint"},
]
STATIC_CONTEXT: list[str] = []  # No meeting topic configured — the common case.
CONTEXT_SEGMENTS = 6  # TranslationWorker._CONTEXT_SEGMENTS


def main() -> None:
    window: deque[str] = deque(maxlen=CONTEXT_SEGMENTS)
    open_shut = {"OPEN": 0, "shut": 0}
    reasons: dict[str, int] = {}
    saved = 0
    print(f"\n  {'utterance':<46}{'gate':>6}{'why':>12}{'system':>9}")
    print(f"  {'-' * 73}")

    for text in MEETING:
        # Exactly what TranslationWorker.process assembles per utterance.
        meeting_context = list(window)[-3:] + STATIC_CONTEXT
        selected = _select_relevance_context(text, meeting_context)
        terms = _select_relevant_glossary_terms(text, GLOSSARY)
        clause = _asr_repair_clause(terms, selected)

        suspected = any(t.get("match") == "possible" for t in terms)
        why = "mishearing" if suspected else ("context" if selected else "-")
        state = "OPEN" if clause else "shut"
        open_shut[state] += 1
        reasons[why] = reasons.get(why, 0) + 1
        if not clause:
            saved += len(_ASR_REPAIR_INSTRUCTION)

        system = len(_SYSTEM_PROMPT) + len(clause)
        print(f"  {text[:44]:<46}{state:>6}{why:>12}{system:>8}c")

        window.append(" ".join(text.split()))

    total = len(MEETING)
    possible = total * len(_ASR_REPAIR_INSTRUCTION)
    mean = (total * len(_SYSTEM_PROMPT) + (possible - saved)) / total
    print(f"\n  gate shut on {open_shut['shut']}/{total}, open on {open_shut['OPEN']}/{total}")
    print(f"  opened by: {', '.join(f'{k} x{v}' for k, v in sorted(reasons.items()) if k != '-')}")
    print(f"  characters not sent: {saved} of {possible} possible")
    print(
        f"  mean system prompt: {mean:.0f}c"
        f"  vs {len(_SYSTEM_PROMPT) + len(_ASR_REPAIR_INSTRUCTION)}c ungated"
        f"  vs {len(_SYSTEM_PROMPT)}c production"
    )


if __name__ == "__main__":
    main()
