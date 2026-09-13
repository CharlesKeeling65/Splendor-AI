"""Reproducibility seed-segment registry (roadmap A2).

Single authoritative source for which integer seeds belong to which
experimental segment.  Every runner that draws game seeds from a range must
resolve that range through this module so manifests are auditable and sealed
test segments cannot be silently reused for tuning.

The human-readable counterpart is ``docs/seed_registry.md``; when amending a
segment, edit both files in the same commit.

Segment conventions
-------------------
* A segment is a half-open interval ``[start, end)`` of integers.
* ``sealed=True`` segments are frozen: they may only be consumed for the
  final reported evaluation of a declared experiment, never for tuning.
* Historical segments are kept for reference; the ranges they occupied must
  never be re-allocated (see ``FORBIDDEN_RANGES``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedSegment:
    """One declared, contiguous block of reproducibility seeds."""

    name: str
    start: int
    end: int
    purpose: str
    sealed: bool = False


#: Historical segments from the 2026-09-08 PPO stabilization protocol.
#: Kept so manifests that still reference them remain auditable.
STABILIZATION_TRAINING = SeedSegment(
    "stabilization_training", 820_000, 822_000, "PPO stabilization training", False
)
STABILIZATION_VALIDATION = SeedSegment(
    "stabilization_validation", 822_000, 823_000, "PPO stabilization validation", False
)
STABILIZATION_TEST = SeedSegment(
    "stabilization_test", 823_000, 824_000, "PPO stabilization test", True
)

#: Current segments declared by docs/IMPROVEMENT_ROADMAP_20260912.md §A2.
TRAINING = SeedSegment("training", 826_000, 828_000, "training runs", False)
VALIDATION = SeedSegment("validation", 828_000, 829_000, "model selection", False)
INDEPENDENT_TEST = SeedSegment(
    "independent_test", 829_000, 829_050, "final reported evaluation", True
)

#: Small scratch segment for offline CI smoke tests; never used for results.
CI_SMOKE = SeedSegment("ci_smoke", 825_000, 826_000, "CI smoke tests", False)

ALL_SEGMENTS: tuple[SeedSegment, ...] = (
    STABILIZATION_TRAINING,
    STABILIZATION_VALIDATION,
    STABILIZATION_TEST,
    TRAINING,
    VALIDATION,
    INDEPENDENT_TEST,
    CI_SMOKE,
)

SEGMENTS_BY_NAME: dict[str, SeedSegment] = {
    segment.name: segment for segment in ALL_SEGMENTS
}

#: Ranges that were burned by pre-registry experiments; no new seed may fall
#: inside them even though they are not declared segments themselves.
FORBIDDEN_RANGES: tuple[tuple[int, int], ...] = (
    (800_000, 820_000),
    (900_000, 940_000),
)


class SeedRegistryError(ValueError):
    """Raised when a requested seed range violates the registry."""


def resolve_segment(name: str) -> SeedSegment:
    """Return the declared segment called ``name``."""
    segment = SEGMENTS_BY_NAME.get(name)
    if segment is None:
        known = ", ".join(sorted(SEGMENTS_BY_NAME))
        raise SeedRegistryError(f"unknown seed segment {name!r}; known: {known}")
    return segment


def allocate_seeds(
    segment: SeedSegment, count: int, *, wrap: bool = False
) -> list[int]:
    """Allocate ``count`` seeds from ``segment``.

    With ``wrap=False`` (default) the call fails when the segment cannot
    satisfy the request without repetition; extending a sealed segment is an
    explicit registry amendment (edit this module and docs/seed_registry.md).
    With ``wrap=True`` seeds cycle, which is acceptable across *different*
    matchups (distinct agents re-randomize the same deal) but must be recorded
    in the manifest.
    """
    if count <= 0:
        raise SeedRegistryError(f"seed count must be positive, got {count}")
    capacity = segment.end - segment.start
    if not wrap and count > capacity:
        raise SeedRegistryError(
            f"segment {segment.name!r} holds {capacity} seeds "
            f"[{segment.start}, {segment.end}) but {count} were requested; "
            "either reduce the request or amend the registry explicitly"
        )
    return [segment.start + index % capacity for index in range(count)]


def check_seeds_declared(seeds: list[int]) -> SeedSegment:
    """Verify every seed lies inside exactly one declared segment.

    Returns the owning segment so callers can stamp it into a manifest.
    """
    if not seeds:
        raise SeedRegistryError("empty seed list")
    owners: dict[int, list[SeedSegment]] = {seed: [] for seed in seeds}
    for segment in ALL_SEGMENTS:
        for seed in seeds:
            if segment.start <= seed < segment.end:
                owners[seed].append(segment)
    for seed, hit in owners.items():
        if not hit:
            raise SeedRegistryError(
                f"seed {seed} is not covered by any declared segment; "
                "amend docs/seed_registry.md before use"
            )
        if len(hit) > 1:  # pragma: no cover - segments are disjoint today
            raise SeedRegistryError(
                f"seed {seed} belongs to multiple segments: {[s.name for s in hit]}"
            )
    first = owners[seeds[0]][0]
    if any(owners[seed][0] is not first for seed in seeds):
        raise SeedRegistryError("seeds span multiple segments; split the request")
    return first


def forbid_reserved(seeds: list[int]) -> None:
    """Reject any seed that falls inside a historically burned range."""
    for seed in seeds:
        for start, end in FORBIDDEN_RANGES:
            if start <= seed < end:
                raise SeedRegistryError(
                    f"seed {seed} lies inside the reserved range "
                    f"[{start}, {end}); it must never be reused"
                )
