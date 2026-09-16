"""Lightweight launcher for the read-only Task-1 pilot monitor.

The policy-imitation package intentionally exposes many training APIs from its
``__init__`` and therefore imports Torch eagerly.  Monitoring must not allocate
that training stack beside a live pilot.  The implementation itself uses only
the standard library, so this launcher executes that source file directly
without importing the heavyweight package initializer.
"""

from __future__ import annotations

import runpy
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import cast


@cache
def _monitor_main() -> Callable[[Sequence[str] | None], None]:
    implementation = (
        Path(__file__).resolve().parent
        / "agents"
        / "our_agents"
        / "policy_imitation"
        / "pilot_monitor.py"
    )
    namespace: Mapping[str, object] = runpy.run_path(
        str(implementation),
        run_name="splendor._standalone_pilot_monitor",
    )
    entrypoint = namespace.get("main")
    if not callable(entrypoint):  # pragma: no cover - packaging integrity guard
        raise RuntimeError("pilot monitor implementation has no main entrypoint")
    return cast(Callable[[Sequence[str] | None], None], entrypoint)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the stdlib-only monitor without importing the training package."""
    _monitor_main()(argv)


if __name__ == "__main__":
    main()
