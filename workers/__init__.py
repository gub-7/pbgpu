"""
Workers module for GPU cluster 3D reconstruction service.

Workers are imported lazily to avoid pulling in heavy dependencies
(e.g. torch) when only a specific worker is needed.
"""


def __getattr__(name: str):
    if name == "TripoSRWorker":
        from .triposr_worker import TripoSRWorker
        return TripoSRWorker
    if name == "Trellis2Worker":
        from .trellis2_worker import Trellis2Worker
        return Trellis2Worker
    if name == "CanonicalMVWorker":
        from .canonical_mv_worker import CanonicalMVWorker
        return CanonicalMVWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TripoSRWorker", "Trellis2Worker", "CanonicalMVWorker"]
