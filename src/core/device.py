"""Pick the inference device (CUDA, Apple MPS, or CPU) and log what is available."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

VALID_DEVICES: tuple[str, ...] = ("auto", "cuda", "mps", "cpu")


@dataclass(frozen=True)
class DeviceInfo:
    """Result of device selection."""

    device: str
    cuda_available: bool
    mps_available: bool


def probe_backends() -> tuple[bool, bool]:
    """Return (cuda_available, mps_available) from torch."""
    import torch

    cuda = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend is not None and mps_backend.is_available())
    return cuda, mps


def select_device(
    requested: str = "auto",
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
) -> DeviceInfo:
    """Resolve `requested` to a usable device; 'auto' prefers cuda > mps > cpu.

    An explicitly requested accelerator that is unavailable falls back to CPU with a warning.
    Availability can be injected for testing; otherwise it is probed from torch.
    """
    req = requested.lower().strip()
    if req not in VALID_DEVICES:
        raise ValueError(f"Unknown device '{requested}', expected one of {VALID_DEVICES}")

    if cuda_available is None or mps_available is None:
        probed_cuda, probed_mps = probe_backends()
        cuda_available = probed_cuda if cuda_available is None else cuda_available
        mps_available = probed_mps if mps_available is None else mps_available

    logger.info("torch.cuda.is_available()=%s, mps_available=%s", cuda_available, mps_available)

    if req == "auto":
        device = "cuda" if cuda_available else ("mps" if mps_available else "cpu")
    elif req == "cuda" and not cuda_available:
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    elif req == "mps" and not mps_available:
        logger.warning("MPS requested but not available, falling back to CPU")
        device = "cpu"
    else:
        device = req

    return DeviceInfo(device=device, cuda_available=cuda_available, mps_available=mps_available)
