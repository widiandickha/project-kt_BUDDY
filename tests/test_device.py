"""Tests for src.core.device."""

from __future__ import annotations

import pytest

from src.core.device import select_device


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_prefers_cuda_then_mps_then_cpu(cuda: bool, mps: bool, expected: str) -> None:
    assert select_device("auto", cuda_available=cuda, mps_available=mps).device == expected


def test_unavailable_accelerator_falls_back_to_cpu() -> None:
    assert select_device("cuda", cuda_available=False, mps_available=True).device == "cpu"
    assert select_device("mps", cuda_available=False, mps_available=False).device == "cpu"


def test_explicit_cpu_kept() -> None:
    assert select_device("cpu", cuda_available=True, mps_available=True).device == "cpu"


def test_unknown_device_rejected() -> None:
    with pytest.raises(ValueError):
        select_device("tpu", cuda_available=False, mps_available=False)


def test_probe_runs_against_real_torch() -> None:
    info = select_device("auto")
    assert info.device in {"cuda", "mps", "cpu"}
