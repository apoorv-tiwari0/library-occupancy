"""
logger.py — Structured logger for the Library Occupancy Detection System.

Features:
  - One log file per section (e.g. logs/section_A.log)
  - A shared system-wide log file (logs/system.log)
  - Rotating file handler (max 5 MB, keeps last 3 files)
  - Console output with colour-coded levels (Windows-safe)
  - Single get_logger() call used everywhere in the codebase

Usage:
    from utils.logger import get_logger
    log = get_logger("section_A")
    log.info("Frame received")
    log.warning("Low confidence detection")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.config_loader import cfg


# ── Colour codes (ANSI) ────────────────────────────────────────────────────────
# On Windows these work in Windows Terminal & VS Code; plain cmd may ignore them.

_COLOURS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET":    "\033[0m",
}


class _ColouredFormatter(logging.Formatter):
    """Console formatter that prepends a colour code to the level name."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        reset  = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname:<8}{reset}"
        return super().format(record)


# ── Internal cache — one Logger object per name ────────────────────────────────
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for the given name (typically a section_id or 'system').

    First call creates the logger and attaches handlers.
    Subsequent calls return the cached instance.

    Args:
        name: Identifier string, e.g. "section_A", "section_B", "system".

    Returns:
        Configured logging.Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    log_dir = Path(cfg.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(cfg.logging.log_level)
    logger.propagate = False   # Prevent double-printing to root logger

    fmt_str   = "[%(asctime)s] [%(name)s] %(levelname)s %(message)s"
    date_fmt  = "%Y-%m-%d %H:%M:%S"

    # ── Rotating file handler ──────────────────────────────────────────────────
    log_file = log_dir / f"{name}.log"
    file_handler = RotatingFileHandler(
        filename    = log_file,
        maxBytes    = cfg.logging.max_bytes,
        backupCount = cfg.logging.backup_count,
        encoding    = "utf-8",
    )
    file_handler.setLevel(cfg.logging.log_level)
    file_handler.setFormatter(logging.Formatter(fmt_str, datefmt=date_fmt))

    # ── Console handler ────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(cfg.logging.log_level)
    console_handler.setFormatter(_ColouredFormatter(fmt_str, datefmt=date_fmt))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    _loggers[name] = logger
    return logger