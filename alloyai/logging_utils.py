from __future__ import annotations

import logging
from typing import Optional


def configure_logging(
    level: Optional[str] = None,
    prefix: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    resolved_level = (level or "info").upper()
    if log_format:
        fmt = log_format
    else:
        prefix_text = f"{prefix} " if prefix else ""
        fmt = f"%(asctime)s {prefix_text}%(levelname)s %(name)s: %(message)s"

    logging.basicConfig(level=resolved_level, format=fmt)
