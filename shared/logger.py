"""Structured logging configuration."""

import logging
from typing import cast

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured JSON logging."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            # WITHOUT THIS, EVERY TRACEBACK IN EVERY WORKER IS DISCARDED.
            #
            # `logger.exception(...)` and `exc_info=True` only mark the event; something has to
            # turn that mark into text. With no such processor the JSON renderer serialised the
            # flag itself — production logs read `"exc_info": true` and the exception, its type
            # and its stack were gone.
            #
            # That is not a cosmetic loss. tts_worker logged `prosody_context_failed_falling_back`
            # with exc_info on every sentence for two releases while the reason stayed invisible;
            # finding it needed a probe against the live vendor API, and the answer turned out to
            # be one line of the traceback that had been thrown away each time.
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
            if log_level == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
