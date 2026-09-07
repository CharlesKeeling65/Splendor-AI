"""Reproducibility and provenance helpers for policy-imitation runs."""

import platform
import random
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every RNG used by the repository's engine and neural agents."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_seed(seed: int) -> Iterator[None]:
    """Run a game or audit probe without consuming the caller's RNG stream."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        seed_everything(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Machine and library information that can affect an experiment."""

    python_version: str
    platform: str
    numpy_version: str
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_devices: tuple[str, ...]
    requested_device: str
    resolved_device: str

    @classmethod
    def collect(cls, requested_device: str = "cpu") -> "RuntimeSnapshot":
        """Collect a serializable snapshot and resolve unavailable accelerators."""
        if requested_device not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"unsupported device {requested_device!r}")
        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        cuda_devices = tuple(
            torch.cuda.get_device_name(index) for index in range(cuda_device_count)
        )
        if requested_device == "cuda" and cuda_available:
            resolved_device = "cuda"
        elif requested_device == "mps" and torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
        return cls(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            numpy_version=np.__version__,
            torch_version=torch.__version__,
            torch_cuda_version=torch.version.cuda,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_devices=cuda_devices,
            requested_device=requested_device,
            resolved_device=resolved_device,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible fields."""
        return asdict(self)


def _git_output(repo: Path, *arguments: str) -> str | None:
    """Read one Git value without invoking a shell."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def capture_code_revision(repo: Path | None = None) -> dict[str, Any]:
    """Capture the commit and exact dirty paths used to create a manifest."""
    root = (repo or Path.cwd()).resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    tracked = _git_output(root, "diff", "--name-only") or ""
    untracked = _git_output(root, "ls-files", "--others", "--exclude-standard") or ""
    dirty_files = sorted(
        {line for line in (*tracked.splitlines(), *untracked.splitlines()) if line}
    )
    return {
        "repository": str(root),
        "commit": commit,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
    }
