"""
TripoSR worker (legacy / single-view fallback).

Retained for backward compatibility. The primary pipeline is now the
canonical 3-view multi-view reconstruction (canonical_mv_worker.py).

TripoSR can still be used for quick single-view reconstruction when
only one image is available.

Usage:
  python -m workers.triposr_worker
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("triposr_worker")


def main() -> None:
    """
    Placeholder entry point.

    TripoSR single-view reconstruction is retained as a fallback.
    The primary pipeline is the canonical 3-view multi-view reconstruction.
    """
    logger.info(
        "TripoSR worker: placeholder. Primary pipeline is canonical_mv_worker."
    )


if __name__ == "__main__":
    main()

