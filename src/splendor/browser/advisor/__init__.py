"""
Browser advisor (plan phase-7): a read-only panel that watches the seat the
human plays in and ranks their legal moves.

Hard invariants (plan phase-7 §6):

* **Zero clicks** - this package never imports ``action_executor`` and never
  calls a ``BrowserDriver`` mutating method; the only driver interaction is
  the passive ``extract_snapshot`` evaluate. A source-level test
  (``tests/test_advisor_observer.py``) enforces the import ban.
* **Single rule source** - legality and evaluation always go through the
  engine (``getLegalActions`` on pseudo/determinized states, card registry);
  the tracker only remembers *observed events*, it never re-derives rules.
* **Honest degradation** - a missed observation (e.g. a transient opponent
  reserve reveal) degrades to an explicit "unknown" bucket, never to a guess.

Modules:

* :mod:`observer` - debounced read-only poll loop over one browser driver.
* :mod:`tracker` - opponent-reservation memory (the advisory-side memory
  reconstruction, plan phase-7 §3.2).
* :mod:`engine` - determinized state reconstruction + GA top-k / minimax
  deep advice + deck-composition histogram (plan phase-7 §3.3/§3.4).
"""

from .observer import AdvisorFrame, AdvisorSession, Phase, classify_phase
from .tracker import ReservationTracker, ReservedEvent, TrackedReserved, TrackerDelta

__all__ = [
    "AdvisorFrame",
    "AdvisorSession",
    "Phase",
    "ReservationTracker",
    "ReservedEvent",
    "TrackedReserved",
    "TrackerDelta",
    "classify_phase",
]
