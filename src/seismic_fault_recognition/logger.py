"""Centralized logging configuration for the project."""

from __future__ import annotations

import logging
import sys
from typing import Any


class DynamicStreamHandler(logging.StreamHandler):
    """Handler that always uses the current sys.stderr/stdout."""

    def __init__(self, stream_name: str = "stderr") -> None:
        super().__init__()
        self._stream_name = stream_name

    @property
    def stream(self) -> Any:
        return getattr(sys, self._stream_name)

    @stream.setter
    def stream(self, value: Any) -> None:
        # Ignore attempts to set the stream directly
        pass


def setup_logger(
    name: str = "sfr",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a project-specific logger.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logger instance.
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = DynamicStreamHandler("stderr")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str = "sfr") -> logging.Logger:
    """Return a logger by name, creating it if necessary."""
    return setup_logger(name)
