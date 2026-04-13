"""
Dedicated Trellis.2 worker (placeholder).

In the current architecture, Trellis.2 is invoked as a subprocess by the
canonical_mv_worker (via pipelines/trellis_completion.py) because it requires
a separate Python environment (micromamba + Python 3.10 + cu124).

This file exists as a placeholder for a future architecture where Trellis.2
runs as an independent worker process consuming from its own Redis queue.

Usage:
  python -m workers.trellis2_worker
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
logger = logging.getLogger("trellis2_worker")


def main() -> None:
    """
    Placeholder entry point.

    Trellis.2 completion is currently handled by the canonical MV
    worker via subprocess invocation into the trellis2 micromamba environment.

    A future version may split this into a separate worker that:
    1. Listens on a dedicated Redis queue
    2. Runs natively inside the trellis2 micromamba env
    3. Returns results via Redis pub/sub or shared storage
    """
    logger.info(
        "Trellis.2 worker: currently handled by canonical_mv_worker subprocess. "
        "This standalone worker is a placeholder for future architecture."
    )


if __name__ == "__main__":
    main()

