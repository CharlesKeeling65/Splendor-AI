"""Formal paired evaluation over immutable :class:`ScenarioV1` openings.

The legacy evaluation helpers predate Task-1 and intentionally keep their
integer-seed behavior.  This module is the opt-in formal path: it enumerates
both seats, couples opponent randomness by semantic event key, retains failed
games in the scheduled denominator, and appends canonical episode rows that
can be resumed but never overwritten.
"""

from __future__ import annotations

import dis
import fcntl
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import IO, Any, Final, Literal, cast

import numpy as np
import torch
from torch import nn

from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.splendor_model import SplendorState

from .policies import CandidateSpec
from .protocol import RngKey, SeedLineage, derive_seed, isolated_seed, sha256_file
from .runner import TeacherDecisionError, select_action
from .scenario import (
    ScenarioV1,
    rule_from_scenario,
    scenario_collection_sha256,
    scenario_state_set_sha256,
    validate_scenario,
)
from .scenario_bank import ScenarioBank, inspect_scenario_bank

EPISODE_SCHEMA_VERSION: Final = "splendor-episode/1"
PAIRED_EVALUATION_PROTOCOL: Final = "paired-evaluation-v1"
POLICY_IDENTITY_PROTOCOL: Final = "evaluation-policy-identity-v1"
SHA256_HEX_LENGTH: Final = 64
ACTION_SPACE_SIZE: Final = 3510
CARD_POSITION_SIZE: Final = 2
PolicyRole = Literal[
    "control",
    "treatment",
    "official",
    "frozen-baseline",
    "opponent",
    "ci-fixture",
]


class PairedEvaluationError(ValueError):
    """Raised when a formal evaluation artifact violates its declaration."""


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, field: str) -> str:
    if not _is_sha256(value):
        raise PairedEvaluationError(f"{field} must be a lowercase SHA-256")
    return cast(str, value)


def _require_string(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise PairedEvaluationError(f"{field} must be a non-empty string")
    return value


def _require_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise PairedEvaluationError(f"{field} must be a non-negative integer")
    return value


def _require_finite_number(value: object, field: str) -> float:
    if type(value) not in {int, float}:
        raise PairedEvaluationError(f"{field} must be a finite number")
    numeric = float(cast(int | float, value))
    if not math.isfinite(numeric):
        raise PairedEvaluationError(f"{field} must be a finite number")
    return numeric


def _factory_source_descriptor(factory: object) -> dict[str, object]:
    target = factory.func if isinstance(factory, partial) else factory
    if not (inspect.isfunction(target) or inspect.isclass(target)):
        raise PairedEvaluationError(
            "evaluation policy factory must be a class, function, or partial thereof"
        )
    module = inspect.getmodule(target)
    module_name = getattr(target, "__module__", None)
    qualified_name = getattr(target, "__qualname__", None)
    try:
        source_path_raw = inspect.getsourcefile(cast(Any, target))
    except (OSError, TypeError):
        source_path_raw = None
    if (
        module is None
        or type(module_name) is not str
        or not module_name
        or type(qualified_name) is not str
        or not qualified_name
        or source_path_raw is None
    ):
        raise PairedEvaluationError(
            "evaluation policy factory must have an inspectable source module"
        )
    source_path = Path(source_path_raw)
    if not source_path.is_file():
        raise PairedEvaluationError(
            f"evaluation policy source does not exist: {source_path}"
        )
    return {
        "module": module_name,
        "qualified_name": qualified_name,
        "module_sha256": sha256_file(source_path),
    }


def _factory_source_path(factory: object) -> Path:
    """Return the inspectable source path already required by policy identity."""
    target = factory.func if isinstance(factory, partial) else factory
    try:
        source_path_raw = inspect.getsourcefile(cast(Any, target))
    except (OSError, TypeError) as exc:  # pragma: no cover - descriptor rejects first
        raise PairedEvaluationError(
            "evaluation policy factory must have an inspectable source module"
        ) from exc
    if source_path_raw is None:
        raise PairedEvaluationError(
            "evaluation policy factory must have an inspectable source module"
        )
    return Path(source_path_raw).resolve()


def _policy_source_scope(factory: object) -> Path:
    """Find the repository boundary for recursive runtime-config inspection."""
    source_path = _factory_source_path(factory)
    for parent in (source_path.parent, *source_path.parents):
        if (parent / ".git").exists():
            return parent
    return source_path.parent


def _inside_source_scope(factory: object, source_scope: Path) -> bool:
    try:
        _factory_source_path(factory).relative_to(source_scope)
    except ValueError:
        return False
    return True


def _tensor_identity(value: torch.Tensor) -> dict[str, object]:
    """Hash tensor values and layout metadata without serializing object reprs."""
    if value.layout != torch.strided:
        raise PairedEvaluationError(
            "policy identity only supports dense strided tensors"
        )
    tensor = value.detach().cpu().contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "value_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _module_identity(module: nn.Module, field_name: str) -> dict[str, object]:
    """Bind an in-memory checkpoint template to its class, state, and config."""
    public_attributes = {
        name: _closure_identity_value(value, f"{field_name}.{name}")
        for name, value in sorted(vars(module).items())
        if not name.startswith("_")
    }
    state = module.state_dict()
    if any(
        type(name) is not str or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise PairedEvaluationError("policy module state_dict is not tensor-valued")
    state_entries = [
        {
            "name": name,
            **_tensor_identity(cast(torch.Tensor, value)),
        }
        for name, value in sorted(state.items())
    ]
    return {
        "class": _factory_source_descriptor(type(module)),
        "public_attributes": public_attributes,
        "state_dict_sha256": _sha256_canonical(state_entries),
    }


def _tensor_state_sha256(state: Mapping[object, object]) -> str | None:
    """Return a reshape-tolerant tensor-state digest for loader attestation.

    Some legacy loaders squeeze a singleton normalizer dimension after loading.
    The checkpoint byte hash and the live module config still bind the exact
    layouts independently; this bridge therefore compares names, dtypes, element
    counts, and values while intentionally ignoring reshape-only differences.
    """
    if not state or any(
        type(name) is not str or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        return None
    entries = [
        {
            "name": cast(str, name),
            "dtype": str(cast(torch.Tensor, value).dtype),
            "numel": cast(torch.Tensor, value).numel(),
            "value_sha256": _tensor_identity(cast(torch.Tensor, value))["value_sha256"],
        }
        for name, value in sorted(state.items(), key=lambda item: cast(str, item[0]))
    ]
    return _sha256_canonical(entries)


def _checkpoint_tensor_state_hashes(value: object) -> set[str]:
    """Find tensor-valued state dictionaries inside one trusted checkpoint."""
    hashes: set[str] = set()
    seen: set[int] = set()

    def visit(item: object) -> None:
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, Mapping):
            digest = _tensor_state_sha256(cast(Mapping[object, object], item))
            if digest is not None:
                hashes.add(digest)
                return
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return hashes


def _factory_module_state_hashes(factory: object) -> set[str]:  # noqa: C901
    """Find all immutable module templates explicitly bound by a factory."""
    hashes: set[str] = set()
    seen: set[int] = set()

    def visit(item: object) -> None:  # noqa: C901
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, nn.Module):
            digest = _tensor_state_sha256(
                cast(Mapping[object, object], item.state_dict())
            )
            if digest is not None:
                hashes.add(digest)
            return
        if isinstance(item, partial):
            visit(item.func)
            visit(item.args)
            visit(item.keywords or {})
            return
        if inspect.isfunction(item):
            for cell in item.__closure__ or ():
                visit(cell.cell_contents)
            visit(item.__defaults__ or ())
            visit(item.__kwdefaults__ or {})
            return
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(factory)
    return hashes


def _agent_module_state_hashes(agent: object) -> set[str]:
    """Find modules loaded by the built agent's decision bytecode.

    ``code.co_names`` retains names from branches removed by the compiler.  Looking
    only at executed bytecode instructions avoids treating a checkpoint module
    mentioned exclusively in an ``if False`` branch as decision-referenced.
    """
    hashes: set[str] = set()
    decision = getattr(agent, "SelectAction", None)
    function = getattr(decision, "__func__", decision)
    code = getattr(function, "__code__", None)
    referenced_names = (
        {
            instruction.argval
            for instruction in dis.get_instructions(code)
            if instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
            and type(instruction.argval) is str
        }
        if code is not None
        else set()
    )
    attributes = vars(agent) if hasattr(agent, "__dict__") else {}
    for name, value in attributes.items():
        if name not in referenced_names or not isinstance(value, nn.Module):
            continue
        digest = _tensor_state_sha256(cast(Mapping[object, object], value.state_dict()))
        if digest is not None:
            hashes.add(digest)
    return hashes


def _attest_checkpoint_factory(candidate: CandidateSpec, checkpoint: Path) -> str:
    """Require both seat-specific agents to use one checkpoint-bound model state."""
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise PairedEvaluationError(
            "checkpoint policy artifact cannot be decoded for state attestation"
        ) from exc
    checkpoint_hashes = _checkpoint_tensor_state_hashes(payload)
    factory_hashes = _factory_module_state_hashes(candidate.factory)
    seat_matches: list[str] = []
    for seat in (0, 1):
        try:
            with isolated_seed(0):
                agent = candidate.build(seat)
        except Exception as exc:
            raise PairedEvaluationError(
                f"checkpoint policy factory could not build seat {seat} "
                "attestation agent"
            ) from exc
        agent_hashes = _agent_module_state_hashes(agent)
        matches = sorted(checkpoint_hashes & factory_hashes & agent_hashes)
        if len(matches) != 1:
            raise PairedEvaluationError(
                "checkpoint policy factory and each seat-specific built agent must "
                "expose exactly one decision-referenced model state present in "
                "its checkpoint"
            )
        seat_matches.append(matches[0])
    if len(set(seat_matches)) != 1:
        raise PairedEvaluationError(
            "checkpoint policy factory resolves to different model states by seat"
        )
    return seat_matches[0]


def _closure_identity_value(  # noqa: C901,PLR0911,PLR0912 - closed type surface
    value: object,
    field_name: str,
    *,
    callable_stack: frozenset[str] = frozenset(),
    source_scope: Path | None = None,
) -> object:
    """Canonicalize behavior-affecting values captured by a policy factory."""
    if isinstance(value, nn.Module):
        return {"torch_module": _module_identity(value, field_name)}
    if isinstance(value, torch.Tensor):
        return {"torch_tensor": _tensor_identity(value)}
    if isinstance(value, torch.device):
        return {"torch_device": str(value)}
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "numpy_array": {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "value_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        }
    if isinstance(value, np.bool_):
        return {
            "numpy_scalar": {
                "dtype": str(value.dtype),
                "value": bool(value),
            }
        }
    if isinstance(value, np.integer):
        return {
            "numpy_scalar": {
                "dtype": str(value.dtype),
                "value": int(value),
            }
        }
    if isinstance(value, np.floating):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise PairedEvaluationError(
                f"policy identity field {field_name} must be finite"
            )
        return {
            "numpy_scalar": {
                "dtype": str(value.dtype),
                "value": numeric,
            }
        }
    if isinstance(value, Path):
        resolved = value.resolve()
        return {
            "path": str(resolved),
            "file_sha256": sha256_file(resolved) if resolved.is_file() else None,
        }
    if callable(value):
        descriptor = _factory_source_descriptor(value)
        callable_key = _sha256_canonical(descriptor)
        if callable_key in callable_stack:
            return {
                "callable": descriptor,
                "runtime_config": {"recursive_reference": True},
            }
        scope = source_scope or _policy_source_scope(value)
        return {
            "callable": descriptor,
            "runtime_config": _factory_config_descriptor(
                value,
                allow_closure=True,
                callable_stack=callable_stack | {callable_key},
                source_scope=scope,
                include_globals=_inside_source_scope(value, scope),
            ),
        }
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PairedEvaluationError(
                f"policy identity field {field_name} must be finite"
            )
        return value
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": value.name,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": _closure_identity_value(
                asdict(value),
                field_name,
                callable_stack=callable_stack,
                source_scope=source_scope,
            ),
        }
    if isinstance(value, (tuple, list)):
        return [
            _closure_identity_value(
                item,
                f"{field_name}[{index}]",
                callable_stack=callable_stack,
                source_scope=source_scope,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise PairedEvaluationError(
                f"policy identity mapping {field_name} requires string keys"
            )
        return {
            key: _closure_identity_value(
                item,
                f"{field_name}.{key}",
                callable_stack=callable_stack,
                source_scope=source_scope,
            )
            for key, item in sorted(cast(Mapping[str, object], value).items())
        }
    raise PairedEvaluationError(
        f"policy closure field {field_name} has unsupported type {type(value).__name__}"
    )


def _global_reference_identity(
    value: object,
    field_name: str,
    *,
    callable_stack: frozenset[str],
    source_scope: Path,
) -> object:
    """Bind one referenced global, recursively within the policy source tree."""
    if isinstance(value, ModuleType):
        source_path_raw = getattr(value, "__file__", None)
        source_path = (
            Path(source_path_raw) if isinstance(source_path_raw, str) else None
        )
        return {
            "module": value.__name__,
            "source_sha256": (
                sha256_file(source_path)
                if source_path is not None and source_path.is_file()
                else None
            ),
        }
    return _closure_identity_value(
        value,
        field_name,
        callable_stack=callable_stack,
        source_scope=source_scope,
    )


def _class_runtime_config(
    target: type[object],
    *,
    callable_stack: frozenset[str],
    source_scope: Path,
) -> dict[str, object]:
    """Bind mutable class attributes and method defaults without graph explosion."""
    ignored = {
        "__annotations__",
        "__dict__",
        "__doc__",
        "__module__",
        "__slots__",
        "__weakref__",
    }
    runtime_attributes: dict[str, object] = {}
    method_config: dict[str, object] = {}
    for name, raw in sorted(vars(target).items()):
        if name in ignored:
            continue
        unwrapped = (
            raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
        )
        if inspect.isfunction(unwrapped):
            if not name.startswith("_") or name in {"__call__", "__init__", "__new__"}:
                descriptor = _factory_source_descriptor(unwrapped)
                key = _sha256_canonical(descriptor)
                if key in callable_stack:
                    runtime_config: object = {"recursive_reference": True}
                else:
                    runtime_config = _factory_config_descriptor(
                        unwrapped,
                        allow_closure=True,
                        callable_stack=callable_stack | {key},
                        source_scope=source_scope,
                        include_globals=False,
                    )
                method_config[name] = {
                    "callable": descriptor,
                    "runtime_config": runtime_config,
                }
            continue
        if isinstance(raw, property) or inspect.isroutine(raw):
            continue
        if name.startswith("__") and name.endswith("__"):
            continue
        runtime_attributes[name] = _closure_identity_value(
            raw,
            f"factory.class.{name}",
            callable_stack=callable_stack,
            source_scope=source_scope,
        )
    return {
        "runtime_attributes": runtime_attributes,
        "method_config": method_config,
    }


def _factory_config_descriptor(
    factory: object,
    *,
    allow_closure: bool,
    callable_stack: frozenset[str] = frozenset(),
    source_scope: Path | None = None,
    include_globals: bool = True,
) -> dict[str, object]:
    target = factory.func if isinstance(factory, partial) else factory
    if not (inspect.isfunction(target) or inspect.isclass(target)):
        raise PairedEvaluationError(
            "evaluation policy factory must be a class, function, or partial thereof"
        )
    scope = source_scope or _policy_source_scope(target)
    target_key = _sha256_canonical(_factory_source_descriptor(target))
    active_stack = callable_stack | {target_key}
    closure: dict[str, object] = {}
    defaults: object = []
    keyword_defaults: object = {}
    referenced_globals: dict[str, object] = {}
    if inspect.isfunction(target) and target.__closure__:
        if not allow_closure:
            raise PairedEvaluationError(
                "snapshotless evaluation policy factories cannot hide closure config"
            )
        closure = {
            name: _closure_identity_value(
                cell.cell_contents,
                f"factory.{name}",
                callable_stack=active_stack,
                source_scope=scope,
            )
            for name, cell in zip(
                target.__code__.co_freevars,
                target.__closure__,
                strict=True,
            )
        }
    if inspect.isfunction(target):
        defaults = _closure_identity_value(
            target.__defaults__ or (),
            "factory.defaults",
            callable_stack=active_stack,
            source_scope=scope,
        )
        keyword_defaults = _closure_identity_value(
            target.__kwdefaults__ or {},
            "factory.keyword_defaults",
            callable_stack=active_stack,
            source_scope=scope,
        )
        if include_globals:
            referenced_globals = {
                name: _global_reference_identity(
                    value,
                    f"factory.global.{name}",
                    callable_stack=active_stack,
                    source_scope=scope,
                )
                for name, value in sorted(
                    inspect.getclosurevars(target).globals.items()
                )
            }
    class_runtime = (
        _class_runtime_config(
            target,
            callable_stack=active_stack,
            source_scope=scope,
        )
        if inspect.isclass(target)
        else {"runtime_attributes": {}, "method_config": {}}
    )
    shared = {
        "closure": closure,
        "defaults": defaults,
        "keyword_defaults": keyword_defaults,
        "referenced_globals": referenced_globals,
        "globals_mode": (
            "recursive-source-scope" if include_globals else "source-only"
        ),
        "class_runtime": class_runtime,
    }
    if not isinstance(factory, partial):
        return {
            "kind": "callable",
            "bound_args": [],
            "bound_kwargs": {},
            **shared,
        }
    return {
        "kind": "partial",
        "bound_args": _closure_identity_value(
            factory.args,
            "factory.args",
            callable_stack=active_stack,
            source_scope=scope,
        ),
        "bound_kwargs": _closure_identity_value(
            factory.keywords or {},
            "factory.keywords",
            callable_stack=active_stack,
            source_scope=scope,
        ),
        **shared,
    }


def policy_source_sha256(candidate: CandidateSpec) -> str:
    """Hash the actual source module and callable selected by a policy spec."""
    return _sha256_canonical(
        {
            "protocol": POLICY_IDENTITY_PROTOCOL,
            "source": _factory_source_descriptor(candidate.factory),
        }
    )


def policy_config_sha256(candidate: CandidateSpec) -> str:
    """Hash all snapshot-independent policy configuration."""
    return _sha256_canonical(
        {
            "protocol": POLICY_IDENTITY_PROTOCOL,
            "name": candidate.name,
            "candidate_role": candidate.role,
            "feature_version": candidate.feature_version,
            "snapshot_kind": (
                "checkpoint" if candidate.snapshot is not None else "source-only"
            ),
            "factory": _factory_config_descriptor(
                candidate.factory,
                allow_closure=candidate.snapshot is not None,
            ),
        }
    )


def snapshotless_policy_sha256(candidate: CandidateSpec) -> str:
    """Return the only accepted identity for a source/config-only policy."""
    if candidate.snapshot is not None:
        raise PairedEvaluationError(
            "snapshotless policy identity cannot be used for a checkpoint"
        )
    return _sha256_canonical(
        {
            "protocol": POLICY_IDENTITY_PROTOCOL,
            "source_sha256": policy_source_sha256(candidate),
            "config_sha256": policy_config_sha256(candidate),
        }
    )


@dataclass(frozen=True)
class EvaluationPolicy:
    """A policy implementation bound to a stable source/checkpoint digest."""

    policy_id: str
    policy_sha256: str
    candidate: CandidateSpec
    role: PolicyRole
    treatment_id: str | None = None
    replicate_id: int | None = None
    model_seed: int | None = None
    checkpoint_sha256: str | None = None
    source_sha256: str = dataclass_field(init=False)
    config_sha256: str = dataclass_field(init=False)
    checkpoint_state_sha256: str | None = dataclass_field(init=False)

    def __post_init__(self) -> None:  # noqa: C901,PLR0912 - fail closed
        _require_string(self.policy_id, "policy_id")
        _require_sha256(self.policy_sha256, "policy_sha256")
        if self.role not in {
            "control",
            "treatment",
            "official",
            "frozen-baseline",
            "opponent",
            "ci-fixture",
        }:
            raise PairedEvaluationError("evaluation policy role is invalid")
        if self.treatment_id is not None:
            _require_string(self.treatment_id, "treatment_id")
        if self.replicate_id is not None:
            _require_nonnegative_int(self.replicate_id, "replicate_id")
        if self.model_seed is not None:
            _require_nonnegative_int(self.model_seed, "model_seed")
        checkpoint_state_sha256: str | None = None
        if self.checkpoint_sha256 is not None:
            _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
            if self.checkpoint_sha256 != self.policy_sha256:
                raise PairedEvaluationError(
                    "checkpoint policies must use their checkpoint SHA-256 as identity"
                )
            if self.candidate.snapshot is None:
                raise PairedEvaluationError(
                    "checkpoint policy must expose its snapshot path"
                )
            snapshot_path = Path(self.candidate.snapshot)
            if not snapshot_path.is_file():
                raise PairedEvaluationError(
                    f"checkpoint snapshot does not exist: {snapshot_path}"
                )
            if sha256_file(snapshot_path) != self.checkpoint_sha256:
                raise PairedEvaluationError("checkpoint snapshot SHA-256 mismatch")
            checkpoint_state_sha256 = _attest_checkpoint_factory(
                self.candidate, snapshot_path
            )
        elif self.role in {"control", "treatment", "official", "frozen-baseline"}:
            raise PairedEvaluationError(
                f"{self.role} policy requires a checkpoint SHA-256"
            )
        elif self.policy_sha256 != snapshotless_policy_sha256(self.candidate):
            raise PairedEvaluationError(
                "snapshotless policy identity must equal its source/config SHA-256"
            )
        if self.role in {"control", "treatment"} and (
            self.treatment_id is None
            or self.replicate_id is None
            or self.model_seed is None
        ):
            raise PairedEvaluationError(
                "control/treatment policies require treatment_id, replicate_id, "
                "and model_seed"
            )
        if self.role == "official" and (
            self.treatment_id is None
            or self.replicate_id is None
            or self.model_seed is None
        ):
            raise PairedEvaluationError(
                "official policy requires its treatment, replicate, and model seed"
            )
        if self.role in {"frozen-baseline", "ci-fixture"} and any(
            value is not None
            for value in (self.treatment_id, self.replicate_id, self.model_seed)
        ):
            raise PairedEvaluationError(
                f"{self.role} policy cannot carry treatment or replicate identity"
            )
        if self.role == "opponent" and (
            self.treatment_id is not None
            or self.replicate_id is not None
            or self.model_seed is not None
        ):
            raise PairedEvaluationError(
                "fixed opponents cannot carry treatment or replicate identity"
            )
        if self.candidate.snapshot is not None and self.checkpoint_sha256 is None:
            raise PairedEvaluationError(
                "a policy snapshot must be bound by checkpoint_sha256"
            )
        object.__setattr__(self, "source_sha256", policy_source_sha256(self.candidate))
        object.__setattr__(self, "config_sha256", policy_config_sha256(self.candidate))
        object.__setattr__(self, "checkpoint_state_sha256", checkpoint_state_sha256)

    def metadata(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_state_sha256": self.checkpoint_state_sha256,
            "snapshot": self.candidate.snapshot,
            "role": self.role,
            "treatment_id": self.treatment_id,
            "replicate_id": self.replicate_id,
            "model_seed": self.model_seed,
            "feature_version": self.candidate.feature_version,
        }


@dataclass(frozen=True, order=True)
class EpisodeKey:
    """Minimum uniqueness key required by the Task-1 statistical protocol."""

    candidate_sha256: str
    opponent_sha256: str
    scenario_id: str
    seat: int
    replicate_id: int | None

    def __post_init__(self) -> None:
        _require_sha256(self.candidate_sha256, "episode candidate_sha256")
        _require_sha256(self.opponent_sha256, "episode opponent_sha256")
        _require_sha256(self.scenario_id, "episode scenario_id")
        if self.seat not in (0, 1):
            raise PairedEvaluationError("episode seat must be 0 or 1")
        if self.replicate_id is not None:
            _require_nonnegative_int(self.replicate_id, "episode replicate_id")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "opponent_sha256": self.opponent_sha256,
            "scenario_id": self.scenario_id,
            "seat": self.seat,
            "replicate_id": self.replicate_id,
        }


@dataclass(frozen=True)
class EvaluationScheduleRow:
    """One declared candidate x opponent x scenario x seat game."""

    candidate_id: str
    candidate_sha256: str
    opponent_id: str
    opponent_sha256: str
    scenario_id: str
    seat: int
    replicate_id: int | None

    @property
    def key(self) -> EpisodeKey:
        return EpisodeKey(
            candidate_sha256=self.candidate_sha256,
            opponent_sha256=self.opponent_sha256,
            scenario_id=self.scenario_id,
            seat=self.seat,
            replicate_id=self.replicate_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "opponent_id": self.opponent_id,
            **self.key.to_dict(),
        }


@dataclass(frozen=True)
class PairedEvaluationSpec:
    """Immutable cartesian schedule and provenance for one evaluation batch."""

    experiment_id: str
    phase: str
    batch_id: str
    code_sha256: str
    manifest_declaration_sha256: str
    candidates: tuple[EvaluationPolicy, ...]
    opponents: tuple[EvaluationPolicy, ...]
    scenario_bank: ScenarioBank
    episodes_filename: str
    seats: tuple[int, ...] = (0, 1)

    def __post_init__(self) -> None:  # noqa: C901,PLR0912 - full spec gate
        for field in ("experiment_id", "phase", "batch_id"):
            _require_string(getattr(self, field), field)
        if Path(
            self.episodes_filename
        ).name != self.episodes_filename or not self.episodes_filename.endswith(
            (".jsonl", ".jsonl.zst")
        ):
            raise PairedEvaluationError(
                "episodes_filename must be a basename ending in .jsonl[.zst]"
            )
        for field in (
            "code_sha256",
            "manifest_declaration_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if self.seats != (0, 1):
            raise PairedEvaluationError("paired evaluation requires seats (0, 1)")
        if (
            not self.candidates
            or not self.opponents
            or not self.scenario_bank.scenarios
        ):
            raise PairedEvaluationError(
                "paired evaluation needs candidates, opponents, and scenarios"
            )
        if self.scenario_bank.selection_kind not in {
            "iid",
            "stress-balanced",
            "ci-fixture",
        }:
            raise PairedEvaluationError(
                "paired evaluation accepts only IID, stress, or CI banks; "
                "natural-deal SRSWOR banks are training-only"
            )
        for label, policies in (
            ("candidate", self.candidates),
            ("opponent", self.opponents),
        ):
            ids = [policy.policy_id for policy in policies]
            digests = [policy.policy_sha256 for policy in policies]
            if len(ids) != len(set(ids)):
                raise PairedEvaluationError(f"{label} policy IDs must be unique")
            if len(digests) != len(set(digests)):
                raise PairedEvaluationError(
                    f"{label} checkpoints/sources must be deduplicated by SHA-256"
                )
        if any(policy.role == "opponent" for policy in self.candidates):
            raise PairedEvaluationError("opponent-role policy cannot be a candidate")
        if any(policy.role != "opponent" for policy in self.opponents):
            raise PairedEvaluationError("evaluation opponents require role='opponent'")
        public_manifest = inspect_scenario_bank(self.scenario_bank.artifact_path)
        if (
            public_manifest["payload_sha256"] != self.scenario_bank.payload_sha256
            or public_manifest["state_set_sha256"]
            != self.scenario_bank.state_set_sha256
            or public_manifest["logical_split"] != self.scenario_bank.logical_split
            or self.scenario_bank.scenario_count != len(self.scenario_bank.scenarios)
        ):
            raise PairedEvaluationError(
                "formal evaluation requires a fully materialized verified bank"
            )
        if (
            scenario_collection_sha256(self.scenario_bank.scenarios)
            != public_manifest["scenario_collection_sha256"]
            or scenario_state_set_sha256(self.scenario_bank.scenarios)
            != public_manifest["state_set_sha256"]
        ):
            raise PairedEvaluationError("evaluation scenarios do not match the bank")
        scenario_ids: list[str] = []
        for scenario in self.scenario_bank.scenarios:
            validate_scenario(scenario)
            scenario_ids.append(scenario.scenario_id)
        if len(scenario_ids) != len(set(scenario_ids)):
            raise PairedEvaluationError("evaluation scenarios must be unique")

    def schedule(self) -> tuple[EvaluationScheduleRow, ...]:
        """Return a stable order independent of declaration container order."""
        rows = [
            EvaluationScheduleRow(
                candidate_id=candidate.policy_id,
                candidate_sha256=candidate.policy_sha256,
                opponent_id=opponent.policy_id,
                opponent_sha256=opponent.policy_sha256,
                scenario_id=scenario.scenario_id,
                seat=seat,
                replicate_id=candidate.replicate_id,
            )
            for candidate in sorted(
                self.candidates, key=lambda policy: policy.policy_sha256
            )
            for opponent in sorted(
                self.opponents, key=lambda policy: policy.policy_sha256
            )
            for scenario in sorted(
                self.scenario_bank.scenarios, key=lambda item: item.scenario_id
            )
            for seat in self.seats
        ]
        keys = [row.key for row in rows]
        if len(keys) != len(set(keys)):  # pragma: no cover - constructor guards
            raise PairedEvaluationError(
                "paired evaluation schedule contains duplicates"
            )
        return tuple(rows)

    @property
    def schedule_sha256(self) -> str:
        return _sha256_canonical([row.to_dict() for row in self.schedule()])

    def manifest_binding(self) -> dict[str, object]:
        return {
            "protocol": PAIRED_EVALUATION_PROTOCOL,
            "batch_id": self.batch_id,
            "episodes_filename": self.episodes_filename,
            "code_sha256": self.code_sha256,
            "scenario_bank_sha256": self.scenario_bank.payload_sha256,
            "scenario_bank_split": self.scenario_bank.logical_split,
            "candidates": [
                policy.metadata()
                for policy in sorted(
                    self.candidates, key=lambda item: item.policy_sha256
                )
            ],
            "opponents": [
                policy.metadata()
                for policy in sorted(
                    self.opponents, key=lambda item: item.policy_sha256
                )
            ],
            "scenario_count": len(self.scenario_bank.scenarios),
            "seats": list(self.seats),
            "scheduled_games": len(self.schedule()),
            "schedule_sha256": self.schedule_sha256,
        }


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PairedEvaluationError(f"value is not canonical-JSON safe: {exc}") from exc


def _sha256_canonical(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _lineage_dict(lineage: SeedLineage) -> dict[str, object]:
    return lineage.as_dict()


def _candidate_lineage(  # noqa: PLR0913 - semantic key fields stay explicit
    spec: PairedEvaluationSpec,
    candidate: EvaluationPolicy,
    opponent: EvaluationPolicy,
    scenario: ScenarioV1,
    seat: int,
    stream_name: str,
    *,
    focal_step: int | None = None,
) -> SeedLineage:
    return derive_seed(
        RngKey(
            experiment_id=spec.experiment_id,
            phase=spec.phase,
            coupling_group=f"candidate:{candidate.policy_sha256}",
            stream_name=stream_name,
            replicate_id=candidate.replicate_id,
            treatment_id=candidate.treatment_id or candidate.policy_id,
            scenario_id=scenario.scenario_id,
            seat=seat,
            opponent_id=opponent.policy_sha256,
            focal_step=focal_step,
        )
    )


def _opponent_lineage(  # noqa: PLR0913 - semantic key fields stay explicit
    spec: PairedEvaluationSpec,
    opponent: EvaluationPolicy,
    scenario: ScenarioV1,
    seat: int,
    stream_name: str,
    *,
    opponent_step: int | None = None,
) -> SeedLineage:
    # Deliberately omit candidate/treatment/replicate identity.  A random
    # opponent therefore consumes the same event variate for every candidate
    # on one (scenario, focal seat, opponent, opponent-step) coordinate.
    return derive_seed(
        RngKey(
            experiment_id=spec.experiment_id,
            phase=spec.phase,
            coupling_group=f"evaluation:{spec.batch_id}:opponent-common",
            stream_name=stream_name,
            scenario_id=scenario.scenario_id,
            seat=seat,
            opponent_id=opponent.policy_sha256,
            opponent_step=opponent_step,
        )
    )


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean_seconds": 0.0, "p95_seconds": 0.0, "max_seconds": 0.0}
    return {
        "mean_seconds": float(np.mean(values)),
        "p95_seconds": float(np.quantile(values, 0.95)),
        "max_seconds": float(max(values)),
    }


def _agent_inventory(state: SplendorState, seat: int) -> dict[str, object]:
    agent = state.agents[seat]
    bought = [
        card.code
        for colour, cards in agent.cards.items()
        if colour != "yellow"
        for card in cards
    ]
    reserved = [card.code for card in agent.cards["yellow"]]
    nobles = [str(noble[0]) for noble in agent.nobles]
    return {
        "bought_card_codes": bought,
        "reserved_card_codes": reserved,
        "noble_codes": nobles,
        "gem_counts": {str(key): int(value) for key, value in agent.gems.items()},
    }


def _empty_inventory() -> dict[str, object]:
    return {
        "bought_card_codes": [],
        "reserved_card_codes": [],
        "noble_codes": [],
        "gem_counts": {},
    }


def _action_payload(action: Mapping[str, object]) -> dict[str, object]:
    """Serialize the executed engine action without object repr identities."""
    card = action.get("card")
    noble = action.get("noble")
    card_position = action.get("card_position")
    if card is not None and type(getattr(card, "code", None)) is not str:
        raise PairedEvaluationError("executed action card has no stable code")
    if noble is not None and (
        not isinstance(noble, (tuple, list)) or not noble or type(noble[0]) is not str
    ):
        raise PairedEvaluationError("executed action noble has no stable code")
    if card_position is not None and (
        not isinstance(card_position, (tuple, list))
        or len(card_position) != CARD_POSITION_SIZE
        or any(type(value) is not int for value in card_position)
    ):
        raise PairedEvaluationError("executed action card_position is invalid")

    def gems(name: str) -> dict[str, int]:
        raw = action.get(name)
        if raw is None:
            return {}
        if not isinstance(raw, Mapping) or any(
            type(key) is not str or type(value) is not int or value < 0
            for key, value in raw.items()
        ):
            raise PairedEvaluationError(f"executed action {name} is invalid")
        return dict(sorted(cast(Mapping[str, int], raw).items()))

    return {
        "type": _require_string(action.get("type"), "executed action type"),
        "collected_gems": gems("collected_gems"),
        "returned_gems": gems("returned_gems"),
        "card_code": None if card is None else str(cast(Any, card).code),
        "card_position": None if card_position is None else list(card_position),
        "noble_code": None if noble is None else str(noble[0]),
    }


def _state_audit_payload(rule: object) -> dict[str, object]:
    """Serialize every gameplay-relevant primitive after an episode trace."""
    state = cast(Any, rule).current_game_state
    board = state.board
    return {
        "current_agent_index": int(cast(Any, rule).current_agent_index),
        "action_counter": int(cast(Any, rule).action_counter),
        "game_ends": bool(cast(Any, rule).gameEnds()),
        "state_agent_to_move": int(state.agent_to_move),
        "board": {
            "gems": {str(key): int(value) for key, value in sorted(board.gems.items())},
            "nobles": [str(noble[0]) for noble in board.nobles],
            "dealt": [
                [None if card is None else str(card.code) for card in tier]
                for tier in board.dealt
            ],
            "decks": [[str(card.code) for card in tier] for tier in board.decks],
        },
        "agents": [
            {
                "id": int(agent.id),
                "score": float(agent.score),
                "gems": {
                    str(key): int(value) for key, value in sorted(agent.gems.items())
                },
                "cards": {
                    str(colour): [str(card.code) for card in cards]
                    for colour, cards in sorted(agent.cards.items())
                },
                "nobles": [str(noble[0]) for noble in agent.nobles],
                "passed": bool(agent.passed),
            }
            for agent in state.agents
        ],
    }


def _failure(side: str, exc: BaseException) -> dict[str, str]:
    return {
        "side": side,
        "type": type(exc).__name__,
        "message": str(exc) or "<no message>",
    }


def play_paired_evaluation_game(  # noqa: C901,PLR0912,PLR0915 - row builder
    spec: PairedEvaluationSpec,
    candidate: EvaluationPolicy,
    opponent: EvaluationPolicy,
    scenario: ScenarioV1,
    seat: int,
) -> dict[str, object]:
    """Run one scheduled game and return a strict, failure-retaining row."""
    if candidate not in spec.candidates or opponent not in spec.opponents:
        raise PairedEvaluationError("game policy is absent from evaluation spec")
    if scenario not in spec.scenario_bank.scenarios or seat not in spec.seats:
        raise PairedEvaluationError("game scenario/seat is absent from evaluation spec")

    rule = rule_from_scenario(scenario)
    candidate_init = _candidate_lineage(
        spec, candidate, opponent, scenario, seat, "candidate_init"
    )
    opponent_init = _opponent_lineage(spec, opponent, scenario, seat, "opponent_init")
    target: Any | None = None
    rival: Any | None = None
    failure: dict[str, str] | None = None
    began = time.perf_counter()
    try:
        with isolated_seed(candidate_init.seed63):
            target = candidate.candidate.build(seat)
    except Exception as exc:  # policy construction failure remains a game row
        failure = _failure("candidate_init", exc)
    if failure is None:
        try:
            with isolated_seed(opponent_init.seed63):
                rival = opponent.candidate.build(1 - seat)
        except Exception as exc:
            failure = _failure("opponent_init", exc)

    candidate_latencies: list[float] = []
    opponent_latencies: list[float] = []
    candidate_lineages: list[dict[str, object]] = []
    opponent_lineages: list[dict[str, object]] = []
    candidate_queries = 0
    opponent_queries = 0
    candidate_illegal = 0
    opponent_illegal = 0
    candidate_search_nodes = 0
    opponent_search_nodes = 0
    candidate_actions: Counter[str] = Counter()
    opponent_actions: Counter[str] = Counter()
    action_trace: list[dict[str, object]] = []

    ended = False
    while failure is None:
        try:
            ended = rule.gameEnds()
            if ended:
                break
            state = rule.current_game_state
            turn = rule.current_agent_index
            actions = rule.getLegalActions(state, turn)
        except Exception as exc:
            failure = _failure("engine_query", exc)
            break
        if turn == seat:
            candidate_queries += 1
            lineage = _candidate_lineage(
                spec,
                candidate,
                opponent,
                scenario,
                seat,
                "candidate_action",
                focal_step=candidate_queries - 1,
            )
            candidate_lineages.append(_lineage_dict(lineage))
            try:
                assert target is not None
                decision = select_action(
                    target,
                    actions,
                    state,
                    rule,
                    rng_lineage=lineage,
                )
            except TeacherDecisionError as exc:
                candidate_latencies.append(exc.elapsed_seconds)
                candidate_search_nodes += exc.search_nodes
                candidate_illegal += int(exc.illegal)
                failure = _failure("candidate", exc)
                continue
            except Exception as exc:
                failure = _failure("candidate", exc)
                continue
            candidate_latencies.append(decision.elapsed_seconds)
            candidate_search_nodes += decision.search_nodes
            action_type = str(decision.action["type"])
        else:
            opponent_queries += 1
            lineage = _opponent_lineage(
                spec,
                opponent,
                scenario,
                seat,
                "opponent_action",
                opponent_step=opponent_queries - 1,
            )
            opponent_lineages.append(_lineage_dict(lineage))
            try:
                assert rival is not None
                decision = select_action(
                    rival,
                    actions,
                    state,
                    rule,
                    rng_lineage=lineage,
                )
            except TeacherDecisionError as exc:
                opponent_latencies.append(exc.elapsed_seconds)
                opponent_search_nodes += exc.search_nodes
                opponent_illegal += int(exc.illegal)
                failure = _failure("opponent", exc)
                continue
            except Exception as exc:
                failure = _failure("opponent", exc)
                continue
            opponent_latencies.append(decision.elapsed_seconds)
            opponent_search_nodes += decision.search_nodes
            action_type = str(decision.action["type"])
        try:
            action_payload = _action_payload(decision.action)
            rule.update(decision.action)
        except Exception as exc:
            failure = _failure("engine_update", exc)
            continue
        if turn == seat:
            candidate_actions[action_type] += 1
        else:
            opponent_actions[action_type] += 1
        action_trace.append(
            {
                "ply": rule.action_counter - 1,
                "seat": turn,
                "side": "candidate" if turn == seat else "opponent",
                "action_index": decision.action_index,
                "action_type": action_type,
                "action": action_payload,
                "action_sha256": _sha256_canonical(action_payload),
                "rng_digest": lineage.digest_hex,
            }
        )

    completed = failure is None and ended
    state = rule.current_game_state
    try:
        candidate_raw = float(state.agents[seat].score)
        opponent_raw = float(state.agents[1 - seat].score)
        candidate_cal = float(rule.calScore(state, seat))
        opponent_cal = float(rule.calScore(state, 1 - seat))
        candidate_inventory = _agent_inventory(state, seat)
        opponent_inventory = _agent_inventory(state, 1 - seat)
        final_state_payload = _state_audit_payload(rule)
        final_state_sha256 = _sha256_canonical(final_state_payload)
    except Exception as exc:
        if failure is None:
            failure = _failure("finalization", exc)
        else:
            failure = {
                **failure,
                "message": (
                    f"{failure['message']}; finalization also raised "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        completed = False
        candidate_raw = 0.0
        opponent_raw = 0.0
        candidate_cal = 0.0
        opponent_cal = 0.0
        candidate_inventory = _empty_inventory()
        opponent_inventory = _empty_inventory()
        final_state_sha256 = _sha256_canonical(
            {
                "status": "unavailable-after-failure",
                "plies": int(getattr(rule, "action_counter", len(action_trace))),
            }
        )
    outcome = (
        int(candidate_cal > opponent_cal) - int(candidate_cal < opponent_cal)
        if completed
        else None
    )
    score_rate = None if outcome is None else (outcome + 1) / 2
    key = EpisodeKey(
        candidate_sha256=candidate.policy_sha256,
        opponent_sha256=opponent.policy_sha256,
        scenario_id=scenario.scenario_id,
        seat=seat,
        replicate_id=candidate.replicate_id,
    )
    row: dict[str, object] = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "protocol": PAIRED_EVALUATION_PROTOCOL,
        "experiment_id": spec.experiment_id,
        "phase": spec.phase,
        "batch_id": spec.batch_id,
        "manifest_declaration_sha256": spec.manifest_declaration_sha256,
        "code_sha256": spec.code_sha256,
        "scenario_bank_sha256": spec.scenario_bank.payload_sha256,
        "scenario_bank_split": spec.scenario_bank.logical_split,
        "schedule_sha256": spec.schedule_sha256,
        "episode_key": key.to_dict(),
        "episode_key_sha256": _sha256_canonical(key.to_dict()),
        "candidate": candidate.metadata(),
        "opponent": opponent.metadata(),
        "scenario_id": scenario.scenario_id,
        "canonical_state_sha256": scenario.canonical_state_sha256,
        "scenario_source_segment": scenario.source_segment,
        "scenario_source_seed": scenario.source_seed,
        "scenario_selection_kind": scenario.selection_kind,
        "scenario_inclusion_probability": scenario.inclusion_probability,
        "scenario_selection_stratum": scenario.selection_stratum,
        "scenario_selection_design_sha256": scenario.selection_design_sha256,
        "seat": seat,
        "status": "completed" if completed else "failed",
        "failure": failure,
        "outcome": outcome,
        "score_rate": score_rate,
        "candidate_raw_score": candidate_raw,
        "opponent_raw_score": opponent_raw,
        "candidate_cal_score": candidate_cal,
        "opponent_cal_score": opponent_cal,
        "plies": rule.action_counter,
        "completed_rounds": rule.action_counter // 2,
        "candidate_queries": candidate_queries,
        "opponent_queries": opponent_queries,
        "candidate_illegal_actions": candidate_illegal,
        "opponent_illegal_actions": opponent_illegal,
        "candidate_search_nodes": candidate_search_nodes,
        "opponent_search_nodes": opponent_search_nodes,
        "candidate_action_counts": dict(sorted(candidate_actions.items())),
        "opponent_action_counts": dict(sorted(opponent_actions.items())),
        "candidate_inventory": candidate_inventory,
        "opponent_inventory": opponent_inventory,
        "candidate_latency": _latency_summary(candidate_latencies),
        "opponent_latency": _latency_summary(opponent_latencies),
        "elapsed_seconds": time.perf_counter() - began,
        "rng_lineage": {
            "candidate_init": _lineage_dict(candidate_init),
            "opponent_init": _lineage_dict(opponent_init),
            "candidate_actions": candidate_lineages,
            "opponent_actions": opponent_lineages,
        },
        "action_trace": action_trace,
        "action_trace_sha256": _sha256_canonical(action_trace),
        "final_state_sha256": final_state_sha256,
    }
    validate_episode_record(row)
    return row


_EPISODE_FIELDS = {
    "schema_version",
    "protocol",
    "experiment_id",
    "phase",
    "batch_id",
    "manifest_declaration_sha256",
    "code_sha256",
    "scenario_bank_sha256",
    "scenario_bank_split",
    "schedule_sha256",
    "episode_key",
    "episode_key_sha256",
    "candidate",
    "opponent",
    "scenario_id",
    "canonical_state_sha256",
    "scenario_source_segment",
    "scenario_source_seed",
    "scenario_selection_kind",
    "scenario_inclusion_probability",
    "scenario_selection_stratum",
    "scenario_selection_design_sha256",
    "seat",
    "status",
    "failure",
    "outcome",
    "score_rate",
    "candidate_raw_score",
    "opponent_raw_score",
    "candidate_cal_score",
    "opponent_cal_score",
    "plies",
    "completed_rounds",
    "candidate_queries",
    "opponent_queries",
    "candidate_illegal_actions",
    "opponent_illegal_actions",
    "candidate_search_nodes",
    "opponent_search_nodes",
    "candidate_action_counts",
    "opponent_action_counts",
    "candidate_inventory",
    "opponent_inventory",
    "candidate_latency",
    "opponent_latency",
    "elapsed_seconds",
    "rng_lineage",
    "action_trace",
    "action_trace_sha256",
    "final_state_sha256",
}


def _parse_policy_metadata(  # noqa: C901,PLR0912 - role schema fails closed
    raw: object, field: str
) -> Mapping[str, object]:
    required = {
        "policy_id",
        "policy_sha256",
        "source_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "checkpoint_state_sha256",
        "snapshot",
        "role",
        "treatment_id",
        "replicate_id",
        "model_seed",
        "feature_version",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise PairedEvaluationError(f"episode {field} metadata schema mismatch")
    _require_string(raw["policy_id"], f"episode {field}.policy_id")
    _require_sha256(raw["policy_sha256"], f"episode {field}.policy_sha256")
    _require_sha256(raw["source_sha256"], f"episode {field}.source_sha256")
    _require_sha256(raw["config_sha256"], f"episode {field}.config_sha256")
    checkpoint = raw["checkpoint_sha256"]
    checkpoint_state = raw["checkpoint_state_sha256"]
    if checkpoint is not None:
        _require_sha256(checkpoint, f"episode {field}.checkpoint_sha256")
        _require_sha256(checkpoint_state, f"episode {field}.checkpoint_state_sha256")
        if checkpoint != raw["policy_sha256"]:
            raise PairedEvaluationError(
                f"episode {field} checkpoint/identity digest mismatch"
            )
    elif checkpoint_state is not None:
        raise PairedEvaluationError(
            f"episode {field} snapshotless policy carries checkpoint state"
        )
    for optional in ("snapshot", "treatment_id"):
        if raw[optional] is not None:
            _require_string(raw[optional], f"episode {field}.{optional}")
    replicate = raw["replicate_id"]
    if replicate is not None:
        _require_nonnegative_int(replicate, f"episode {field}.replicate_id")
    model_seed = raw["model_seed"]
    if model_seed is not None:
        _require_nonnegative_int(model_seed, f"episode {field}.model_seed")
    role = _require_string(raw["role"], f"episode {field}.role")
    if role not in {
        "control",
        "treatment",
        "official",
        "frozen-baseline",
        "opponent",
        "ci-fixture",
    }:
        raise PairedEvaluationError(f"episode {field}.role is invalid")
    if role in {"control", "treatment", "official"} and (
        checkpoint is None
        or raw["treatment_id"] is None
        or replicate is None
        or model_seed is None
    ):
        raise PairedEvaluationError(
            f"episode {field} training policy metadata is incomplete"
        )
    if role == "frozen-baseline" and checkpoint is None:
        raise PairedEvaluationError(
            f"episode {field} frozen baseline lacks a checkpoint"
        )
    if role in {"frozen-baseline", "ci-fixture", "opponent"} and any(
        value is not None for value in (raw["treatment_id"], replicate, model_seed)
    ):
        raise PairedEvaluationError(
            f"episode {field} fixed policy carries replicate metadata"
        )
    snapshot = raw["snapshot"]
    if (snapshot is None) != (checkpoint is None):
        raise PairedEvaluationError(
            f"episode {field} snapshot/checkpoint binding is incomplete"
        )
    _require_string(raw["feature_version"], f"episode {field}.feature_version")
    return cast(Mapping[str, object], raw)


def validate_paired_evaluation_binding(  # noqa: C901,PLR0912 - full binding gate
    raw: object,
) -> None:
    """Validate the canonical evaluation declaration embedded in manifest v2."""
    required = {
        "protocol",
        "batch_id",
        "episodes_filename",
        "code_sha256",
        "scenario_bank_sha256",
        "scenario_bank_split",
        "candidates",
        "opponents",
        "scenario_count",
        "seats",
        "scheduled_games",
        "schedule_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise PairedEvaluationError("paired evaluation manifest schema mismatch")
    if raw["protocol"] != PAIRED_EVALUATION_PROTOCOL:
        raise PairedEvaluationError("paired evaluation protocol identifier is invalid")
    _require_string(raw["batch_id"], "paired evaluation batch_id")
    episodes_filename = _require_string(
        raw["episodes_filename"], "paired evaluation episodes_filename"
    )
    if Path(
        episodes_filename
    ).name != episodes_filename or not episodes_filename.endswith(
        (".jsonl", ".jsonl.zst")
    ):
        raise PairedEvaluationError("paired evaluation episodes filename is invalid")
    for field in (
        "code_sha256",
        "scenario_bank_sha256",
        "schedule_sha256",
    ):
        _require_sha256(raw[field], f"paired evaluation {field}")
    _require_string(raw["scenario_bank_split"], "paired evaluation split")
    if raw["seats"] != [0, 1]:
        raise PairedEvaluationError("paired evaluation seats must be [0, 1]")
    scenario_count = _require_nonnegative_int(
        raw["scenario_count"], "paired evaluation scenario_count"
    )
    scheduled_games = _require_nonnegative_int(
        raw["scheduled_games"], "paired evaluation scheduled_games"
    )
    if scenario_count == 0:
        raise PairedEvaluationError("paired evaluation scenario_count must be positive")
    parsed: dict[str, list[Mapping[str, object]]] = {}
    for field in ("candidates", "opponents"):
        values = raw[field]
        if not isinstance(values, list) or not values:
            raise PairedEvaluationError(f"paired evaluation {field} must be a list")
        rows = [
            _parse_policy_metadata(value, f"paired evaluation {field}")
            for value in values
        ]
        ids = [str(row["policy_id"]) for row in rows]
        hashes = [str(row["policy_sha256"]) for row in rows]
        if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
            raise PairedEvaluationError(
                f"paired evaluation {field} identities must be unique"
            )
        if hashes != sorted(hashes):
            raise PairedEvaluationError(
                f"paired evaluation {field} must use canonical hash order"
            )
        parsed[field] = rows
    if any(row["role"] == "opponent" for row in parsed["candidates"]):
        raise PairedEvaluationError("paired evaluation candidate role is invalid")
    if any(row["role"] != "opponent" for row in parsed["opponents"]):
        raise PairedEvaluationError("paired evaluation opponent role is invalid")
    expected_games = (
        len(parsed["candidates"]) * len(parsed["opponents"]) * scenario_count * 2
    )
    if scheduled_games != expected_games:
        raise PairedEvaluationError("paired evaluation scheduled denominator mismatch")


def _parse_episode_key(raw: object) -> EpisodeKey:
    required = {
        "candidate_sha256",
        "opponent_sha256",
        "scenario_id",
        "seat",
        "replicate_id",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise PairedEvaluationError("episode key schema mismatch")
    seat = raw["seat"]
    if type(seat) is not int:
        raise PairedEvaluationError("episode key seat must be an integer")
    replicate = raw["replicate_id"]
    if replicate is not None and type(replicate) is not int:
        raise PairedEvaluationError("episode key replicate_id is invalid")
    return EpisodeKey(
        candidate_sha256=_require_sha256(
            raw["candidate_sha256"], "episode key candidate_sha256"
        ),
        opponent_sha256=_require_sha256(
            raw["opponent_sha256"], "episode key opponent_sha256"
        ),
        scenario_id=_require_sha256(raw["scenario_id"], "episode key scenario_id"),
        seat=seat,
        replicate_id=cast(int | None, replicate),
    )


def _validate_lineage(raw: object, field: str) -> None:
    if not isinstance(raw, Mapping):
        raise PairedEvaluationError(f"episode RNG lineage {field} is not a mapping")
    required = {"key", "canonical_key_json", "digest_hex", "seed63", "u53"}
    if set(raw) != required or not isinstance(raw.get("key"), Mapping):
        raise PairedEvaluationError(f"episode RNG lineage {field} schema mismatch")
    expected = derive_seed(cast(Mapping[str, object], raw["key"])).as_dict()
    if dict(raw) != expected:
        raise PairedEvaluationError(f"episode RNG lineage {field} was tampered")


def _validate_counts(raw: object, field: str) -> int:
    if not isinstance(raw, Mapping) or any(
        type(key) is not str or not key or type(value) is not int or value < 0
        for key, value in raw.items()
    ):
        raise PairedEvaluationError(f"episode {field} must be non-negative counts")
    return sum(cast(Mapping[str, int], raw).values())


def _validate_inventory(raw: object, field: str) -> None:
    required = {
        "bought_card_codes",
        "reserved_card_codes",
        "noble_codes",
        "gem_counts",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise PairedEvaluationError(f"episode {field} schema mismatch")
    for name in ("bought_card_codes", "reserved_card_codes", "noble_codes"):
        values = raw[name]
        if not isinstance(values, list) or any(
            type(value) is not str for value in values
        ):
            raise PairedEvaluationError(f"episode {field}.{name} is invalid")
    _validate_counts(raw["gem_counts"], f"{field}.gem_counts")


def _validate_latency(raw: object, field: str) -> None:
    required = {"mean_seconds", "p95_seconds", "max_seconds"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise PairedEvaluationError(f"episode {field} schema mismatch")
    values = [
        _require_finite_number(raw[name], f"episode {field}.{name}")
        for name in required
    ]
    if any(value < 0 for value in values):
        raise PairedEvaluationError(f"episode {field} values are invalid")


def _validate_action_payload(raw: object) -> Mapping[str, object]:
    required = {
        "type",
        "collected_gems",
        "returned_gems",
        "card_code",
        "card_position",
        "noble_code",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise PairedEvaluationError("episode action payload schema mismatch")
    _require_string(raw["type"], "episode action payload type")
    for field in ("collected_gems", "returned_gems"):
        _validate_counts(raw[field], f"action payload {field}")
    for field in ("card_code", "noble_code"):
        value = raw[field]
        if value is not None:
            _require_string(value, f"episode action payload {field}")
    position = raw["card_position"]
    if position is not None and (
        not isinstance(position, list)
        or len(position) != CARD_POSITION_SIZE
        or any(type(value) is not int for value in position)
    ):
        raise PairedEvaluationError("episode action payload card_position is invalid")
    return cast(Mapping[str, object], raw)


def _validate_action_trace(  # noqa: C901,PLR0912 - full trace schema gate
    raw: object,
    *,
    focal_seat: int,
    candidate_lineages: Sequence[object],
    opponent_lineages: Sequence[object],
) -> tuple[Counter[str], Counter[str]]:
    if not isinstance(raw, list):
        raise PairedEvaluationError("episode action_trace must be a list")
    required = {
        "ply",
        "seat",
        "side",
        "action_index",
        "action_type",
        "action",
        "action_sha256",
        "rng_digest",
    }
    candidate_counts: Counter[str] = Counter()
    opponent_counts: Counter[str] = Counter()
    lineage_offsets = {"candidate": 0, "opponent": 0}
    lineage_by_side = {
        "candidate": candidate_lineages,
        "opponent": opponent_lineages,
    }
    initial_actor_seat: int | None = None
    for expected_ply, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != required:
            raise PairedEvaluationError("episode action_trace row schema mismatch")
        if item["ply"] != expected_ply:
            raise PairedEvaluationError("episode action_trace ply order is invalid")
        actor_seat = item["seat"]
        if type(actor_seat) is not int or actor_seat not in (0, 1):
            raise PairedEvaluationError("episode action_trace seat is invalid")
        if initial_actor_seat is None:
            initial_actor_seat = actor_seat
        if actor_seat != (initial_actor_seat + expected_ply) % 2:
            raise PairedEvaluationError("episode action_trace turn order is invalid")
        side = item["side"]
        expected_side = "candidate" if actor_seat == focal_seat else "opponent"
        if side != expected_side:
            raise PairedEvaluationError("episode action_trace side/seat mismatch")
        action_index = item["action_index"]
        if (
            type(action_index) is not int
            or action_index < 0
            or action_index >= ACTION_SPACE_SIZE
        ):
            raise PairedEvaluationError("episode action_trace action index is invalid")
        action_type = _require_string(
            item["action_type"], "episode action_trace action_type"
        )
        action_payload = _validate_action_payload(item["action"])
        if action_payload["type"] != action_type:
            raise PairedEvaluationError(
                "episode action trace type disagrees with its payload"
            )
        action_sha256 = _require_sha256(
            item["action_sha256"], "episode action_trace action_sha256"
        )
        if action_sha256 != _sha256_canonical(action_payload):
            raise PairedEvaluationError("episode action payload SHA-256 mismatch")
        digest = _require_sha256(item["rng_digest"], "episode action_trace rng_digest")
        offset = lineage_offsets[expected_side]
        side_lineages = lineage_by_side[expected_side]
        if offset >= len(side_lineages):
            raise PairedEvaluationError("episode action_trace exceeds RNG lineage")
        lineage = side_lineages[offset]
        if not isinstance(lineage, Mapping) or lineage.get("digest_hex") != digest:
            raise PairedEvaluationError("episode action_trace RNG digest mismatch")
        lineage_offsets[expected_side] += 1
        if expected_side == "candidate":
            candidate_counts[action_type] += 1
        else:
            opponent_counts[action_type] += 1
    return candidate_counts, opponent_counts


def validate_episode_record(  # noqa: C901,PLR0912,PLR0915
    record: Mapping[str, object],
) -> EpisodeKey:
    """Validate one row completely and return its canonical uniqueness key."""
    if set(record) != _EPISODE_FIELDS:
        raise PairedEvaluationError("episode record fields mismatch")
    if (
        record.get("schema_version") != EPISODE_SCHEMA_VERSION
        or record.get("protocol") != PAIRED_EVALUATION_PROTOCOL
    ):
        raise PairedEvaluationError("episode protocol/schema mismatch")
    for field in (
        "experiment_id",
        "phase",
        "batch_id",
        "scenario_bank_split",
        "scenario_source_segment",
    ):
        _require_string(record[field], f"episode {field}")
    for field in (
        "manifest_declaration_sha256",
        "code_sha256",
        "scenario_bank_sha256",
        "schedule_sha256",
        "scenario_id",
        "canonical_state_sha256",
        "action_trace_sha256",
        "final_state_sha256",
        "episode_key_sha256",
    ):
        _require_sha256(record[field], f"episode {field}")
    if record["scenario_id"] != record["canonical_state_sha256"]:
        raise PairedEvaluationError("episode scenario/state identity mismatch")
    seat = record["seat"]
    if type(seat) is not int or seat not in (0, 1):
        raise PairedEvaluationError("episode seat must be 0 or 1")
    _require_nonnegative_int(record["scenario_source_seed"], "scenario_source_seed")
    selection_kind = record["scenario_selection_kind"]
    if selection_kind not in {"iid", "stress-balanced", "ci-fixture"}:
        raise PairedEvaluationError("episode scenario selection kind is invalid")
    inclusion = _require_finite_number(
        record["scenario_inclusion_probability"],
        "episode scenario_inclusion_probability",
    )
    if not 0.0 < inclusion <= 1.0:
        raise PairedEvaluationError("episode scenario inclusion probability is invalid")
    stratum = record["scenario_selection_stratum"]
    design_hash = record["scenario_selection_design_sha256"]
    if selection_kind in {"iid", "ci-fixture"} and (
        inclusion != 1.0 or stratum is not None or design_hash is not None
    ):
        raise PairedEvaluationError("IID episode carries stress selection metadata")
    if selection_kind == "stress-balanced" and (
        type(stratum) is not str or not stratum or not _is_sha256(design_hash)
    ):
        raise PairedEvaluationError("stress episode selection metadata is incomplete")
    key = _parse_episode_key(record["episode_key"])
    if record["episode_key_sha256"] != _sha256_canonical(key.to_dict()):
        raise PairedEvaluationError("episode key SHA-256 mismatch")
    candidate = _parse_policy_metadata(record["candidate"], "candidate")
    opponent = _parse_policy_metadata(record["opponent"], "opponent")
    if (
        key.candidate_sha256 != candidate["policy_sha256"]
        or key.opponent_sha256 != opponent["policy_sha256"]
        or key.scenario_id != record["scenario_id"]
        or key.seat != seat
        or key.replicate_id != candidate["replicate_id"]
    ):
        raise PairedEvaluationError("episode key does not bind row identities")
    if opponent["role"] != "opponent" or opponent["replicate_id"] is not None:
        raise PairedEvaluationError("episode opponent role/replicate is invalid")

    status = record["status"]
    failure = record["failure"]
    outcome = record["outcome"]
    score_rate = record["score_rate"]
    if status == "completed":
        if failure is not None or type(outcome) is not int or outcome not in (-1, 0, 1):
            raise PairedEvaluationError("completed episode outcome/failure is invalid")
        if score_rate != (outcome + 1) / 2:
            raise PairedEvaluationError("episode score_rate disagrees with W/D/L")
    elif status == "failed":
        if (
            not isinstance(failure, Mapping)
            or set(failure) != {"side", "type", "message"}
            or any(
                type(failure[name]) is not str or not failure[name] for name in failure
            )
            or failure["side"]
            not in {
                "candidate_init",
                "opponent_init",
                "engine_query",
                "candidate",
                "opponent",
                "engine_update",
                "finalization",
            }
            or outcome is not None
            or score_rate is not None
        ):
            raise PairedEvaluationError("failed episode must retain a failure record")
    else:
        raise PairedEvaluationError("episode status must be completed or failed")

    for field in (
        "candidate_raw_score",
        "opponent_raw_score",
        "candidate_cal_score",
        "opponent_cal_score",
        "elapsed_seconds",
    ):
        _require_finite_number(record[field], f"episode {field}")
    elapsed_seconds = _require_finite_number(
        record["elapsed_seconds"], "episode elapsed_seconds"
    )
    if elapsed_seconds < 0:
        raise PairedEvaluationError("episode elapsed_seconds must be non-negative")
    candidate_cal_score = _require_finite_number(
        record["candidate_cal_score"], "episode candidate_cal_score"
    )
    opponent_cal_score = _require_finite_number(
        record["opponent_cal_score"], "episode opponent_cal_score"
    )
    expected_outcome = int(candidate_cal_score > opponent_cal_score) - int(
        candidate_cal_score < opponent_cal_score
    )
    if status == "completed" and record["outcome"] != expected_outcome:
        raise PairedEvaluationError("episode outcome disagrees with calScore")
    for field in (
        "plies",
        "completed_rounds",
        "candidate_queries",
        "opponent_queries",
        "candidate_illegal_actions",
        "opponent_illegal_actions",
        "candidate_search_nodes",
        "opponent_search_nodes",
    ):
        _require_nonnegative_int(record[field], f"episode {field}")
    if status == "completed" and (
        record["candidate_illegal_actions"] != 0
        or record["opponent_illegal_actions"] != 0
    ):
        raise PairedEvaluationError("completed episode cannot contain illegal actions")
    if record["completed_rounds"] != cast(int, record["plies"]) // 2:
        raise PairedEvaluationError("episode round/ply counts disagree")
    candidate_action_count = _validate_counts(
        record["candidate_action_counts"], "candidate_action_counts"
    )
    opponent_action_count = _validate_counts(
        record["opponent_action_counts"], "opponent_action_counts"
    )
    if candidate_action_count + opponent_action_count != record["plies"]:
        raise PairedEvaluationError("episode action counts do not add to plies")
    if candidate_action_count > cast(int, record["candidate_queries"]) or (
        opponent_action_count > cast(int, record["opponent_queries"])
    ):
        raise PairedEvaluationError("episode actions exceed policy queries")
    unanswered_queries = (
        cast(int, record["candidate_queries"])
        - candidate_action_count
        + cast(int, record["opponent_queries"])
        - opponent_action_count
    )
    if status == "completed" and unanswered_queries != 0:
        raise PairedEvaluationError("completed episode has an unanswered policy query")
    if status == "failed" and unanswered_queries not in (0, 1):
        raise PairedEvaluationError("failed episode has inconsistent policy queries")
    _validate_inventory(record["candidate_inventory"], "candidate_inventory")
    _validate_inventory(record["opponent_inventory"], "opponent_inventory")
    _validate_latency(record["candidate_latency"], "candidate_latency")
    _validate_latency(record["opponent_latency"], "opponent_latency")

    lineage = record["rng_lineage"]
    required_lineage = {
        "candidate_init",
        "opponent_init",
        "candidate_actions",
        "opponent_actions",
    }
    if not isinstance(lineage, Mapping) or set(lineage) != required_lineage:
        raise PairedEvaluationError("episode RNG lineage schema mismatch")
    _validate_lineage(lineage["candidate_init"], "candidate_init")
    _validate_lineage(lineage["opponent_init"], "opponent_init")
    for side in ("candidate", "opponent"):
        rows = lineage[f"{side}_actions"]
        if not isinstance(rows, list):
            raise PairedEvaluationError(f"episode {side} action lineage is invalid")
        for index, row in enumerate(rows):
            _validate_lineage(row, f"{side}_actions[{index}]")
        if len(rows) != record[f"{side}_queries"]:
            raise PairedEvaluationError(f"episode {side} query lineage is incomplete")
    candidate_trace_counts, opponent_trace_counts = _validate_action_trace(
        record["action_trace"],
        focal_seat=seat,
        candidate_lineages=cast(Sequence[object], lineage["candidate_actions"]),
        opponent_lineages=cast(Sequence[object], lineage["opponent_actions"]),
    )
    if len(cast(list[object], record["action_trace"])) != record["plies"]:
        raise PairedEvaluationError("episode action_trace length disagrees with plies")
    if (
        dict(sorted(candidate_trace_counts.items()))
        != record["candidate_action_counts"]
        or dict(sorted(opponent_trace_counts.items()))
        != record["opponent_action_counts"]
    ):
        raise PairedEvaluationError("episode action_trace disagrees with action counts")
    if record["action_trace_sha256"] != _sha256_canonical(record["action_trace"]):
        raise PairedEvaluationError("episode action_trace SHA-256 mismatch")
    return key


@contextmanager
def _payload_stream(path: Path) -> Iterator[IO[bytes]]:
    if path.name.endswith(".jsonl"):
        with path.open("rb") as stream:
            yield stream
        return
    if not path.name.endswith(".jsonl.zst"):
        raise PairedEvaluationError("episodes path must end in .jsonl or .jsonl.zst")
    executable = shutil.which("zstd")
    if executable is None:
        raise PairedEvaluationError("reading .jsonl.zst episodes requires zstd")
    process = subprocess.Popen(
        [executable, "--quiet", "--decompress", "--stdout", "--", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        yield process.stdout
    except BaseException:
        process.stdout.close()
        process.kill()
        process.wait()
        process.stderr.close()
        raise
    else:
        process.stdout.close()
        error = process.stderr.read()
        return_code = process.wait()
        process.stderr.close()
        if return_code != 0:
            message = error.decode("utf-8", errors="replace").strip()
            raise PairedEvaluationError(f"episode decompression failed: {message}")


def load_episode_records(path: Path) -> tuple[dict[str, object], ...]:
    """Load canonical rows and reject duplicate scheduled identities."""
    artifact = Path(path)
    if not artifact.exists():
        return ()
    records: list[dict[str, object]] = []
    keys: set[EpisodeKey] = set()
    with _payload_stream(artifact) as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                raw: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PairedEvaluationError(
                    f"episode row {line_number} is invalid JSON"
                ) from exc
            if not isinstance(raw, dict):
                raise PairedEvaluationError("episode JSONL row must be an object")
            record = cast(dict[str, object], raw)
            if line != _canonical_bytes(record) + b"\n":
                raise PairedEvaluationError("episode artifact is not canonical JSONL")
            key = validate_episode_record(record)
            if key in keys:
                raise PairedEvaluationError("episode artifact contains a duplicate key")
            keys.add(key)
            records.append(record)
    return tuple(records)


def _compress_frame(payload: bytes) -> bytes:
    executable = shutil.which("zstd")
    if executable is None:
        raise PairedEvaluationError("writing .jsonl.zst episodes requires zstd")
    result = subprocess.run(
        [executable, "--quiet", "--compress", "--stdout", "--threads=1", "-3"],
        input=payload,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise PairedEvaluationError(f"episode compression failed: {message}")
    return result.stdout


class EpisodeAppender:
    """Single-writer, fsync-on-row append session with cached duplicate keys."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock_stream: IO[str] | None = None
        self._keys: set[EpisodeKey] = set()
        self.records: list[dict[str, object]] = []

    def __enter__(self) -> EpisodeAppender:
        if self._lock_stream is not None:
            raise PairedEvaluationError("episode appender is already open")
        if not (
            self.path.name.endswith(".jsonl") or self.path.name.endswith(".jsonl.zst")
        ):
            raise PairedEvaluationError(
                "episodes path must end in .jsonl or .jsonl.zst"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        self._lock_stream = lock_path.open("a+", encoding="utf-8")
        fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX)
        self.records = list(load_episode_records(self.path))
        self._keys = {validate_episode_record(record) for record in self.records}
        return self

    def append(self, record: Mapping[str, object]) -> None:
        if self._lock_stream is None:
            raise PairedEvaluationError("episode appender is not open")
        canonical_record = cast(dict[str, object], dict(record))
        key = validate_episode_record(canonical_record)
        if key in self._keys:
            raise PairedEvaluationError("episode key was already recorded")
        payload = _canonical_bytes(canonical_record) + b"\n"
        encoded = (
            _compress_frame(payload) if self.path.name.endswith(".zst") else payload
        )
        with self.path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        self._keys.add(key)
        self.records.append(canonical_record)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback
        assert self._lock_stream is not None
        fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
        self._lock_stream.close()
        self._lock_stream = None


def append_episode_records(
    path: Path,
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Append a batch without accepting any duplicate game coordinate."""
    if not records:
        raise PairedEvaluationError("cannot append an empty episode batch")
    with EpisodeAppender(path) as appender:
        for record in records:
            appender.append(record)
        return tuple(appender.records)


def _record_matches_spec(
    record: Mapping[str, object],
    spec: PairedEvaluationSpec,
) -> EpisodeKey:
    key = validate_episode_record(record)
    expected_metadata = {
        "experiment_id": spec.experiment_id,
        "phase": spec.phase,
        "batch_id": spec.batch_id,
        "manifest_declaration_sha256": spec.manifest_declaration_sha256,
        "code_sha256": spec.code_sha256,
        "scenario_bank_sha256": spec.scenario_bank.payload_sha256,
        "scenario_bank_split": spec.scenario_bank.logical_split,
        "schedule_sha256": spec.schedule_sha256,
    }
    if any(record[field] != value for field, value in expected_metadata.items()):
        raise PairedEvaluationError("existing episode row belongs to another spec")
    candidate_by_hash = {policy.policy_sha256: policy for policy in spec.candidates}
    opponent_by_hash = {policy.policy_sha256: policy for policy in spec.opponents}
    scenario_by_id = {
        scenario.scenario_id: scenario for scenario in spec.scenario_bank.scenarios
    }
    if (
        key.candidate_sha256 not in candidate_by_hash
        or key.opponent_sha256 not in opponent_by_hash
        or key.scenario_id not in scenario_by_id
    ):
        raise PairedEvaluationError("episode identity is absent from its declared spec")
    candidate = candidate_by_hash[key.candidate_sha256]
    opponent = opponent_by_hash[key.opponent_sha256]
    scenario = scenario_by_id[key.scenario_id]
    if (
        record["candidate"] != candidate.metadata()
        or record["opponent"] != opponent.metadata()
        or record["canonical_state_sha256"] != scenario.canonical_state_sha256
        or record["scenario_source_segment"] != scenario.source_segment
        or record["scenario_source_seed"] != scenario.source_seed
        or record["scenario_selection_kind"] != scenario.selection_kind
        or record["scenario_inclusion_probability"] != scenario.inclusion_probability
        or record["scenario_selection_stratum"] != scenario.selection_stratum
        or record["scenario_selection_design_sha256"]
        != scenario.selection_design_sha256
    ):
        raise PairedEvaluationError("episode policy/scenario metadata was relabelled")
    lineage = cast(Mapping[str, object], record["rng_lineage"])
    expected_candidate_init = _lineage_dict(
        _candidate_lineage(
            spec, candidate, opponent, scenario, key.seat, "candidate_init"
        )
    )
    expected_opponent_init = _lineage_dict(
        _opponent_lineage(spec, opponent, scenario, key.seat, "opponent_init")
    )
    expected_candidate_actions = [
        _lineage_dict(
            _candidate_lineage(
                spec,
                candidate,
                opponent,
                scenario,
                key.seat,
                "candidate_action",
                focal_step=index,
            )
        )
        for index in range(cast(int, record["candidate_queries"]))
    ]
    expected_opponent_actions = [
        _lineage_dict(
            _opponent_lineage(
                spec,
                opponent,
                scenario,
                key.seat,
                "opponent_action",
                opponent_step=index,
            )
        )
        for index in range(cast(int, record["opponent_queries"]))
    ]
    if dict(lineage) != {
        "candidate_init": expected_candidate_init,
        "opponent_init": expected_opponent_init,
        "candidate_actions": expected_candidate_actions,
        "opponent_actions": expected_opponent_actions,
    }:
        raise PairedEvaluationError("episode RNG lineage does not match its schedule")
    if record["status"] == "completed":
        _replay_completed_record(record, scenario)
    return key


def _replay_completed_record(  # noqa: C901 - independent replay audit
    record: Mapping[str, object], scenario: ScenarioV1
) -> None:
    """Rebuild a completed outcome solely from ScenarioV1 and action indices."""
    if record.get("status") != "completed":
        raise PairedEvaluationError("only completed episodes can be replay-certified")
    rule = rule_from_scenario(scenario)
    trace = record["action_trace"]
    assert isinstance(trace, list)  # validated before this spec-level audit
    try:
        for item_raw in trace:
            item = cast(Mapping[str, object], item_raw)
            if rule.gameEnds():
                raise PairedEvaluationError(
                    "episode trace continues after the replayed game ended"
                )
            state = rule.current_game_state
            turn = rule.current_agent_index
            if item["seat"] != turn:
                raise PairedEvaluationError("episode trace seat disagrees with replay")
            legal_actions = rule.getLegalActions(state, turn)
            mapping = create_action_mapping(legal_actions, state, turn)
            action_index = cast(int, item["action_index"])
            if action_index not in mapping:
                raise PairedEvaluationError(
                    "episode trace contains an action illegal in replay"
                )
            action = mapping[action_index]
            payload = _action_payload(action)
            if payload != item["action"]:
                raise PairedEvaluationError(
                    "episode action payload disagrees with ScenarioV1 replay"
                )
            if action["type"] != item["action_type"]:
                raise PairedEvaluationError(
                    "episode action type disagrees with ScenarioV1 replay"
                )
            rule.update(action)
    except PairedEvaluationError:
        raise
    except Exception as exc:
        raise PairedEvaluationError(
            f"episode ScenarioV1 replay raised {type(exc).__name__}: {exc}"
        ) from exc

    if rule.action_counter != record["plies"]:
        raise PairedEvaluationError("episode ply count disagrees with replay")
    if not rule.gameEnds():
        raise PairedEvaluationError("completed episode replay has not ended")
    state = rule.current_game_state
    seat = cast(int, record["seat"])
    replay_values: dict[str, object] = {
        "candidate_raw_score": float(state.agents[seat].score),
        "opponent_raw_score": float(state.agents[1 - seat].score),
        "candidate_cal_score": float(rule.calScore(state, seat)),
        "opponent_cal_score": float(rule.calScore(state, 1 - seat)),
        "candidate_inventory": _agent_inventory(state, seat),
        "opponent_inventory": _agent_inventory(state, 1 - seat),
        "final_state_sha256": _sha256_canonical(_state_audit_payload(rule)),
    }
    if any(record[field] != value for field, value in replay_values.items()):
        raise PairedEvaluationError(
            "episode score/inventory/final-state digest disagrees with replay"
        )


def audit_episode_batch(
    spec: PairedEvaluationSpec,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Audit scheduled/completed denominators; any failure invalidates the batch."""
    expected = {row.key for row in spec.schedule()}
    observed: dict[EpisodeKey, Mapping[str, object]] = {}
    for record in records:
        key = _record_matches_spec(record, spec)
        if key in observed:
            raise PairedEvaluationError("episode batch contains a duplicate key")
        if key not in expected:
            raise PairedEvaluationError("episode batch contains an unscheduled game")
        observed[key] = record
    missing = sorted(expected - set(observed))
    failures = [record for record in observed.values() if record["status"] == "failed"]
    completed = [
        record for record in observed.values() if record["status"] == "completed"
    ]
    wins = sum(record["outcome"] == 1 for record in completed)
    draws = sum(record["outcome"] == 0 for record in completed)
    losses = sum(record["outcome"] == -1 for record in completed)
    completed_score_rate = (wins + 0.5 * draws) / len(completed) if completed else None
    return {
        "protocol": PAIRED_EVALUATION_PROTOCOL,
        "batch_id": spec.batch_id,
        "schedule_sha256": spec.schedule_sha256,
        "scheduled_games": len(expected),
        "recorded_games": len(observed),
        "completed_games": len(completed),
        "failed_games": len(failures),
        "missing_games": len(missing),
        "missing_episode_keys": [key.to_dict() for key in missing],
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "completed_score_rate": completed_score_rate,
        "scheduled_score_numerator": wins + 0.5 * draws,
        "status": "valid" if not failures and not missing else "invalid",
        "failure_policy": "retain-in-scheduled-denominator-and-invalidate-batch",
    }


def require_paired_evaluation_manifest(  # noqa: C901,PLR0912 - fail-closed gate
    source_manifest: Path,
    spec: PairedEvaluationSpec,
) -> None:
    """Fail closed unless a running manifest binds this exact evaluation."""
    from .manifest import MANIFEST_SCHEMA_V2, load_manifest  # noqa: PLC0415
    from .protocol import (  # noqa: PLC0415
        capture_code_revision,
        require_reproducible_code,
    )

    manifest = load_manifest(Path(source_manifest))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_V2:
        raise PairedEvaluationError("formal evaluation requires manifest schema v2")
    if manifest.get("status") != "running":
        raise PairedEvaluationError("formal evaluation requires a running manifest")
    if manifest.get("declaration_sha256") != spec.manifest_declaration_sha256:
        raise PairedEvaluationError("evaluation manifest declaration hash mismatch")
    declaration = manifest.get("declaration")
    provenance = manifest.get("provenance")
    if not isinstance(declaration, Mapping) or not isinstance(provenance, Mapping):
        raise PairedEvaluationError("evaluation manifest structure is invalid")
    if (
        declaration.get("experiment_id") != spec.experiment_id
        or declaration.get("phase") != spec.phase
        or declaration.get("paired_evaluation") != spec.manifest_binding()
    ):
        raise PairedEvaluationError("manifest does not bind the evaluation spec")
    code = provenance.get("code")
    if not isinstance(code, Mapping):
        raise PairedEvaluationError("evaluation manifest code provenance is missing")
    require_reproducible_code(code)
    if _sha256_canonical(dict(code)) != spec.code_sha256:
        raise PairedEvaluationError("evaluation code SHA-256 disagrees with provenance")
    repository = code.get("repository")
    if type(repository) is not str or capture_code_revision(Path(repository)) != dict(
        code
    ):
        raise PairedEvaluationError(
            "evaluation code changed after manifest provenance was captured"
        )
    for policy in (*spec.candidates, *spec.opponents):
        if policy.source_sha256 != policy_source_sha256(
            policy.candidate
        ) or policy.config_sha256 != policy_config_sha256(policy.candidate):
            raise PairedEvaluationError(
                f"evaluation policy {policy.policy_id!r} source/config changed"
            )
        if policy.checkpoint_sha256 is not None:
            snapshot = policy.candidate.snapshot
            if (
                snapshot is None
                or sha256_file(Path(snapshot)) != policy.checkpoint_sha256
            ):
                raise PairedEvaluationError(
                    f"evaluation policy {policy.policy_id!r} checkpoint changed"
                )
            if (
                _attest_checkpoint_factory(policy.candidate, Path(snapshot))
                != policy.checkpoint_state_sha256
            ):
                raise PairedEvaluationError(
                    f"evaluation policy {policy.policy_id!r} checkpoint state changed"
                )
    artifact_contract = declaration.get("artifact_contract")
    if not isinstance(artifact_contract, Mapping):
        raise PairedEvaluationError("evaluation artifact contract is missing")
    scenario_banks = artifact_contract.get("scenario_banks")
    if (
        not isinstance(scenario_banks, Mapping)
        or scenario_banks.get(spec.scenario_bank.logical_split)
        != spec.scenario_bank.payload_sha256
    ):
        raise PairedEvaluationError("evaluation bank is not bound by artifact contract")
    if artifact_contract.get("episodes") != spec.episodes_filename:
        raise PairedEvaluationError(
            "evaluation episodes filename is not manifest-bound"
        )


def run_paired_evaluation(
    spec: PairedEvaluationSpec,
    episodes_path: Path,
    *,
    source_manifest: Path,
) -> dict[str, object]:
    """Resume only missing declared rows and fsync every completed attempt."""
    require_paired_evaluation_manifest(source_manifest, spec)
    if Path(episodes_path).name != spec.episodes_filename:
        raise PairedEvaluationError(
            "episodes output does not match the declared filename"
        )
    candidates = {policy.policy_sha256: policy for policy in spec.candidates}
    opponents = {policy.policy_sha256: policy for policy in spec.opponents}
    scenarios = {
        scenario.scenario_id: scenario for scenario in spec.scenario_bank.scenarios
    }
    expected = {row.key for row in spec.schedule()}
    with EpisodeAppender(episodes_path) as appender:
        existing_keys = {
            _record_matches_spec(record, spec) for record in appender.records
        }
        if not existing_keys <= expected:
            raise PairedEvaluationError("episode store contains unscheduled games")
        for schedule_row in spec.schedule():
            if schedule_row.key in existing_keys:
                continue
            record = play_paired_evaluation_game(
                spec,
                candidates[schedule_row.candidate_sha256],
                opponents[schedule_row.opponent_sha256],
                scenarios[schedule_row.scenario_id],
                schedule_row.seat,
            )
            appender.append(record)
            existing_keys.add(schedule_row.key)
        records = tuple(appender.records)
    audit = audit_episode_batch(spec, records)
    audit["episodes_path"] = str(episodes_path)
    audit["episodes_artifact_sha256"] = sha256_file(Path(episodes_path))
    return audit


__all__ = [
    "EPISODE_SCHEMA_VERSION",
    "PAIRED_EVALUATION_PROTOCOL",
    "POLICY_IDENTITY_PROTOCOL",
    "EpisodeAppender",
    "EpisodeKey",
    "EvaluationPolicy",
    "EvaluationScheduleRow",
    "PairedEvaluationError",
    "PairedEvaluationSpec",
    "append_episode_records",
    "audit_episode_batch",
    "load_episode_records",
    "play_paired_evaluation_game",
    "policy_config_sha256",
    "policy_source_sha256",
    "require_paired_evaluation_manifest",
    "run_paired_evaluation",
    "snapshotless_policy_sha256",
    "validate_episode_record",
    "validate_paired_evaluation_binding",
]
