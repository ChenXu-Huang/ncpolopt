"""Logging helpers shared by all ncpolopt modules.

The package never prints to stdout directly; every module obtains a named
logger through :func:`module_logger` and exposes a user-facing ``verbose``
level that maps to standard logging severities.
"""

from __future__ import annotations

import logging

_LOG_LEVELS: dict[int, int] = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}


def module_logger(name: str) -> logging.Logger:
    """Return the logger for a module, quiet by default.

    The returned logger inherits the root handler configuration; call
    :func:`setup_logging` once from an entry point to enable output.

    Args:
        name: The ``__name__`` of the calling module.

    Returns:
        A logger that only emits warnings unless the package is configured
        for verbose output.
    """
    return logging.getLogger(name)


def verbosity_to_level(verbose: int) -> int:
    """Map the package's ``verbose`` integer to a logging severity.

    Args:
        verbose: 0 (warnings only), 1 (info) or 2 (debug). Higher values
            are clamped to debug.

    Returns:
        The matching ``logging`` level constant.
    """
    return _LOG_LEVELS.get(verbose, logging.DEBUG)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a console handler.

    Idempotent: calling it repeatedly does not stack duplicate handlers.
    The handler carries the ``"ncpolopt"`` name so tests (and the function
    itself) can recognize and detach it.

    Args:
        level: The logging severity to emit, e.g. ``logging.INFO``.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        if getattr(handler, "name", None) == "ncpolopt":
            return
    handler = logging.StreamHandler()
    handler.name = "ncpolopt"
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    root.addHandler(handler)
