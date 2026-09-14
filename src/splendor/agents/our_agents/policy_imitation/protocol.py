"""Reproducibility and provenance helpers for policy-imitation runs."""

import base64
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Iterator, Mapping
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
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
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
    python_executable: str
    platform: str
    numpy_version: str
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_devices: tuple[str, ...]
    requested_device: str
    resolved_device: str
    cpu_count: int | None
    torch_num_threads: int
    torch_num_interop_threads: int
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cudnn_allow_tf32: bool
    matmul_allow_tf32: bool
    cublas_workspace_config: str | None
    python_hash_seed: str | None
    cuda_driver_version: str | None

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
        deterministic_warn_only = False
        if torch.are_deterministic_algorithms_enabled():
            deterministic_warn_only = bool(
                torch.is_deterministic_algorithms_warn_only_enabled()
            )
        return cls(
            python_version=sys.version.split()[0],
            python_executable=sys.executable,
            platform=platform.platform(),
            numpy_version=np.__version__,
            torch_version=torch.__version__,
            torch_cuda_version=torch.version.cuda,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_devices=cuda_devices,
            requested_device=requested_device,
            resolved_device=resolved_device,
            cpu_count=os.cpu_count(),
            torch_num_threads=torch.get_num_threads(),
            torch_num_interop_threads=torch.get_num_interop_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=deterministic_warn_only,
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
            cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
            cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
            matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            python_hash_seed=os.environ.get("PYTHONHASHSEED"),
            cuda_driver_version=_cuda_driver_version(),
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


def _command_output(*arguments: str) -> str | None:
    """Read a short, optional command value without invoking a shell."""
    try:
        result = subprocess.run(
            list(arguments),
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


def _cuda_driver_version() -> str | None:
    value = _command_output(
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    )
    return value.splitlines()[0].strip() if value else None


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Encode JSON deterministically for provenance digests."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_canonical_json(value: object) -> str:
    """Hash a JSON-compatible value using the repository canonical form."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _git_bytes(repo: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return b""
    return result.stdout if result.returncode == 0 else b""


def capture_dirty_patch(repo: Path) -> bytes:
    """Return a replayable tracked/untracked worktree snapshot."""
    tracked = _git_bytes(repo, "diff", "HEAD", "--binary")
    untracked_raw = _git_bytes(repo, "ls-files", "--others", "--exclude-standard")
    chunks = [b"# splendor uncommitted patch snapshot v1\n", tracked]
    for raw_relative in sorted(untracked_raw.splitlines()):
        relative = raw_relative.decode("utf-8")
        path = repo / relative
        if not path.is_file():
            continue
        chunks.extend(
            (
                b"\n# UNTRACKED_FILE_BEGIN " + raw_relative + b"\n",
                base64.b64encode(path.read_bytes()),
                b"\n# UNTRACKED_FILE_END\n",
            )
        )
    return b"".join(chunks)


def capture_code_revision(repo: Path | None = None) -> dict[str, Any]:
    """Capture commit, dirty paths, and the replayable patch digest."""
    root = (repo or Path.cwd()).resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    tracked = _git_output(root, "diff", "--name-only") or ""
    untracked = _git_output(root, "ls-files", "--others", "--exclude-standard") or ""
    dirty_files = sorted(
        {line for line in (*tracked.splitlines(), *untracked.splitlines()) if line}
    )
    patch = capture_dirty_patch(root)
    return {
        "repository": str(root),
        "commit": commit,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "patch_format": "tracked git diff plus base64 untracked files",
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
    }


def capture_dependency_provenance(repo: Path | None = None) -> dict[str, Any]:
    """Hash dependency declarations that can change a formal run."""
    root = (repo or Path.cwd()).resolve()
    candidates = [root / "pyproject.toml", root / "uv.lock", root / "requirements.txt"]
    requirements = root / "requirements"
    if requirements.is_dir():
        candidates.extend(sorted(requirements.glob("*.txt")))
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in candidates
        if path.is_file()
    }
    return {
        "files": files,
        "aggregate_sha256": sha256_canonical_json(files),
    }


def require_reproducible_code(code: Mapping[str, Any]) -> None:
    """Reject provenance that is neither clean nor backed by a patch digest."""
    if code.get("commit") is None:
        raise RuntimeError("formal execution requires a Git commit")
    if code.get("dirty") and not code.get("patch_sha256"):
        raise RuntimeError("dirty formal execution requires a replayable patch SHA-256")
