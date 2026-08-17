"""Establish how the Responses API behaves for the chat assistant's agent loop.

WHY
---
ASSISTANT_CHAT_MODEL was moved to gpt-5.6-luna in prod-20260810-...-v47 and every
"Ask WarpTalk" message failed:

    400 Function tools with reasoning_effort are not supported for gpt-5.6-luna in
    /v1/chat/completions. To use function tools, use /v1/responses or set
    reasoning_effort to 'none'.

Migrating ai_assistant_worker/chat_worker.py:_run_agent_loop to /v1/responses is the
only way to keep a reasoning model AND tool calling. That loop depends on the Chat
Completions wire shape in six places — streaming deltas, tool-call accumulation by
index, `finish_reason == "tool_calls"`, the assistant/tool message pair, and the nested
tool schema — so the migration cannot be written from documentation alone without
guessing. This probe replaces the guesses with observed behaviour.

WHAT IT ANSWERS
    1. Does Luna accept function tools on /v1/responses at all?
    2. Nested tool schema ({"type":"function","function":{...}}) or flat?
    3. Which streaming events carry assistant text, and which carry tool arguments?
    4. How is a tool RESULT fed back for the next turn?
    5. What signals "the model is done" versus "it wants a tool"?
    6. WT-474: does Luna accept `input_image` and `input_file` content parts at all, and in what
       shape? This one cannot be answered from the code: the worker builds those parts, but nothing
       in the test suite talks to the live API, so "Luna is multimodal" is an assumption until this
       probe says otherwise. gpt-4o-mini takes both; a reasoning model is a different question.

Prints event TYPES and shapes, never message content beyond short excerpts, and never
the API key (read via the same load_dotenv() the workers use).

USAGE
    uv run python -m tools.responses_api_probe
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import AsyncOpenAI

from shared.config import resolve_openai_api_key

MODEL = "gpt-5.6-luna"
FALLBACK_MODEL = "gpt-4o-mini"

# Deliberately trivial, and phrased so the model has to call it rather than answer.
TOOL_NAME = "get_active_meeting_count"
TOOL_DESCRIPTION = "Return how many meetings are currently active in this workspace."
TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "workspace_id": {"type": "string", "description": "Workspace to count meetings in."}
    },
    "required": ["workspace_id"],
    "additionalProperties": False,
}

FLAT_TOOL = {
    "type": "function",
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "parameters": TOOL_PARAMETERS,
}
NESTED_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": TOOL_PARAMETERS,
    },
}

QUESTION = "How many meetings are active in workspace ws-123? Use the tool."


async def probe_schema_shape(client: AsyncOpenAI, model: str) -> str | None:
    """Which tool schema the endpoint accepts. Returns 'flat', 'nested', or None."""
    for label, tool in (("flat", FLAT_TOOL), ("nested", NESTED_TOOL)):
        try:
            await client.responses.create(
                model=model,
                input=[{"role": "user", "content": QUESTION}],
                tools=[tool],  # type: ignore[list-item]
                max_output_tokens=64,
            )
            print(f"    {label:7} schema: ACCEPTED")
            return label
        except Exception as exc:
            print(f"    {label:7} schema: rejected — {str(exc)[:130]}")
    return None


async def probe_streaming(client: AsyncOpenAI, model: str, tool: Any) -> dict[str, Any]:
    """Record which streaming events appear, and what a tool call looks like."""
    seen: dict[str, int] = {}
    text_events: set[str] = set()
    args_events: set[str] = set()
    call_id: str | None = None
    fn_name: str | None = None
    arguments = ""
    output_items: list[Any] = []

    stream = await client.responses.create(
        model=model,
        input=[{"role": "user", "content": QUESTION}],
        tools=[tool],
        max_output_tokens=256,
        stream=True,
    )
    async for event in stream:
        etype = getattr(event, "type", "?")
        seen[etype] = seen.get(etype, 0) + 1

        if etype.endswith("output_text.delta"):
            text_events.add(etype)
        if "function_call_arguments" in etype and etype.endswith("delta"):
            args_events.add(etype)
            arguments += getattr(event, "delta", "") or ""
        if etype == "response.output_item.added":
            item = getattr(event, "item", None)
            if getattr(item, "type", "") == "function_call":
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                fn_name = getattr(item, "name", None)
        if etype == "response.completed":
            response = getattr(event, "response", None)
            output_items = list(getattr(response, "output", []) or [])

    return {
        "events": seen,
        "text_events": sorted(text_events),
        "args_events": sorted(args_events),
        "call_id": call_id,
        "name": fn_name,
        "arguments": arguments,
        "output_items": [getattr(i, "type", "?") for i in output_items],
        "raw_output": output_items,
    }


async def probe_tool_result_roundtrip(
    client: AsyncOpenAI,
    model: str,
    tool: Any,
    call_id: str,
    fn_name: str,
    arguments: str,
) -> None:
    """Feed a tool result back and confirm the model finishes with text."""
    conversation: list[Any] = [
        {"role": "user", "content": QUESTION},
        {"type": "function_call", "call_id": call_id, "name": fn_name, "arguments": arguments},
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps({"active_meetings": 3}),
        },
    ]
    try:
        response = await client.responses.create(
            model=model,
            input=conversation,
            tools=[tool],
            max_output_tokens=200,
        )
        text = getattr(response, "output_text", "") or ""
        print(f"    round-trip: OK — model replied {text[:90]!r}")
    except Exception as exc:
        print(f"    round-trip: FAILED — {str(exc)[:200]}")


async def run() -> None:
    api_key = resolve_openai_api_key()
    if not api_key:
        raise SystemExit("No OPENAI_API_KEY. It is read from .env and never printed.")
    client = AsyncOpenAI(api_key=api_key)

    for model in (MODEL, FALLBACK_MODEL):
        print(f"\n=== {model} ===")

        print("  [1] tool schema shape")
        shape = await probe_schema_shape(client, model)
        if shape is None:
            print("    -> this model cannot take function tools here; skipping the rest")
            continue
        tool = FLAT_TOOL if shape == "flat" else NESTED_TOOL

        print("  [2] streaming events")
        try:
            result = await probe_streaming(client, model, tool)
        except Exception as exc:
            print(f"    streaming FAILED — {str(exc)[:200]}")
            continue

        for etype, count in sorted(result["events"].items(), key=lambda kv: -kv[1]):
            print(f"    {count:3d}  {etype}")
        print(f"    text deltas carried by : {result['text_events'] or '(none seen)'}")
        print(f"    tool args carried by   : {result['args_events'] or '(none seen)'}")
        print(f"    final output items     : {result['output_items']}")
        print(
            f"    tool call              : name={result['name']!r} "
            f"call_id={'set' if result['call_id'] else 'MISSING'} "
            f"arguments={result['arguments'][:70]!r}"
        )

        print("  [3] tool result round-trip")
        if result["call_id"] and result["name"]:
            await probe_tool_result_roundtrip(
                client,
                model,
                tool,
                result["call_id"],
                result["name"],
                result["arguments"] or "{}",
            )
        else:
            print("    skipped: the model did not request the tool")

        # WT-474. Asked LAST so a failure here cannot mask the tool-calling answers above, which
        # are what the agent loop depends on.
        print("  [4] attachment content parts (WT-474)")
        await probe_attachment(
            client,
            model,
            "input_image",
            {"type": "input_image", "image_url": TINY_PNG},
        )
        await probe_attachment(
            client,
            model,
            "input_file",
            {"type": "input_file", "filename": "probe.pdf", "file_data": TINY_PDF},
        )


#: A 1x1 transparent PNG and a two-line PDF, small enough to inline and real enough to be parsed.
#: Nothing here is user content, so the probe can be run against prod credentials safely.
#: Generated programmatically rather than pasted: a hand-copied base64 blob that does not
#: decode makes the API answer "does not represent a valid image", which reads exactly like
#: "this model cannot take images" and is not the same statement at all. Both fixtures below
#: are real files — a 1x1 truecolour PNG and a one-page PDF with a text object.
TINY_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSu"
    "QmCC"
)
TINY_PDF = (
    "data:application/pdf;base64,"
    "JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5kb2JqCjIgMCBvYmoK"
    "PDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4KZW5kb2JqCjMgMCBvYmoKPDwgL1R5cGUg"
    "L1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVkaWFCb3ggWzAgMCAyMDAgNTBdIC9SZXNvdXJjZXMgPDwgL0ZvbnQgPDwg"
    "L0YxIDUgMCBSID4+ID4+IC9Db250ZW50cyA0IDAgUiA+PgplbmRvYmoKNCAwIG9iago8PCAvTGVuZ3RoIDQ0ID4+"
    "CnN0cmVhbQpCVCAvRjEgMTIgVGYgMTAgMjAgVGQgKFdhcnBUYWxrIHByb2JlKSBUaiBFVAplbmRzdHJlYW0KZW5k"
    "b2JqCjUgMCBvYmoKPDwgL1R5cGUgL0ZvbnQgL1N1YnR5cGUgL1R5cGUxIC9CYXNlRm9udCAvSGVsdmV0aWNhID4+"
    "CmVuZG9iagp4cmVmCjAgNgowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwMDkgMDAwMDAgbiAKMDAwMDAwMDA1"
    "OCAwMDAwMCBuIAowMDAwMDAwMTE1IDAwMDAwIG4gCjAwMDAwMDAyNDAgMDAwMDAgbiAKMDAwMDAwMDMzNCAwMDAw"
    "MCBuIAp0cmFpbGVyCjw8IC9TaXplIDYgL1Jvb3QgMSAwIFIgPj4Kc3RhcnR4cmVmCjQwNAolJUVPRgo="
)


async def probe_attachment(
    client: AsyncOpenAI,
    model: str,
    label: str,
    part: dict[str, Any],
) -> None:
    """WT-474: ask whether ONE content-part shape is accepted, and print the verdict.

    Non-streaming and unbounded on purpose — the question is whether the request is accepted, not
    what the answer says, and a 400 arrives faster and more legibly without a stream to unwind.
    """
    try:
        response = await client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Reply with one word: ACCEPTED."},
                        part,
                    ],
                }
            ],
            max_output_tokens=32,
        )
    except Exception as exc:  # noqa: BLE001 - the message IS the result
        print(f"    {label:14s} REJECTED — {str(exc)[:220]}")
        return

    text = getattr(response, "output_text", "") or ""
    print(f"    {label:14s} accepted — replied {text.strip()[:40]!r}")


if __name__ == "__main__":
    asyncio.run(run())
